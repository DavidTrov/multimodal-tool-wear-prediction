/**
 * sensor_pipeline.cpp — CWT + TFLite Micro inference for FRDM-MCXN947.
 *
 * Build with: -DARM_MATH_CM33 -DDESKTOP_SHIM -DTF_LITE_STATIC_MEMORY -O2 -ffast-math
 *
 * Memory layout inside the caller-provided pool (≥391 KB in main SRAM):
 *
 *   Offset 0:                signal buffer  (MAX_SIGNAL_SAMPLES × 4 bytes = 384 KB)
 *   Offset SIGNAL_BYTES:     CWT workspace  (2×2048 × 4 bytes            =  16 KB)
 *
 * The scalogram (5×64×64 × 4 bytes = 80 KB) is held in a SEPARATE buffer
 * passed by the caller — place it in SRAMX to keep it out of main SRAM:
 *   __attribute__((section(".bss.$SRAMX")))
 *   static float g_scalogram[SENSOR_PIPELINE_SCALOGRAM_FLOATS];
 *
 * After CWT completes, the entire pool is reused as the TFLite tensor arena
 * (391 KB available; ~370 KB required at peak for the fusion model).
 */

#include "sensor_pipeline.h"
#include "cwt_mcu.h"

/* TFLite Micro — include paths set by MCUXpresso SDK eIQ middleware */
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <cstdio>
#include <cstring>
#include <cmath>

/* Debug output — SDK debug console macro; replace with printf on non-NXP builds */
#ifndef PRINTF
#  define PRINTF printf
#endif

/* ── Pool layout ─────────────────────────────────────────────────────── */

static constexpr size_t MAX_SIGNAL_SAMPLES = SENSOR_PIPELINE_MAX_SAMPLES;
static constexpr size_t SIGNAL_BYTES  = MAX_SIGNAL_SAMPLES * sizeof(float); /* 384 KB */
static constexpr size_t WKSP_FLOATS   = 2u * CWT_MCU_N_KER;
static constexpr size_t WKSP_BYTES    = WKSP_FLOATS * sizeof(float);        /*  16 KB */

/* Pool = signal + workspace; scalogram lives in a separate caller-provided buffer */
static constexpr size_t OFF_SIGNAL    = 0;
static constexpr size_t OFF_WKSP      = OFF_SIGNAL + SIGNAL_BYTES;

/* After CWT, the full pool becomes the TFLite arena */
static constexpr size_t OFF_ARENA     = 0;
static constexpr size_t ARENA_BYTES   = SIGNAL_BYTES + WKSP_BYTES;          /* 391 KB */

static inline float  *signal_buf(uint8_t *pool)   { return reinterpret_cast<float*>(pool + OFF_SIGNAL); }
static inline float  *wksp_buf(uint8_t *pool)      { return reinterpret_cast<float*>(pool + OFF_WKSP);   }
static inline uint8_t*arena_buf(uint8_t *pool)     { return pool + OFF_ARENA; }

/* ── SensorPipeline struct ───────────────────────────────────────────── */

struct SensorPipeline {
    uint8_t                        *pool;
    float                          *scalogram;      /* caller-owned, in SRAMX */
    const tflite::Model            *model;
    tflite::MicroInterpreter       *interpreter;
    TfLiteTensor                   *input_tensor;   /* input(0) = image [1,3,224,224] INT8 NCHW */
    TfLiteTensor                   *scalo_tensor;   /* input(1) = scalogram [1,5,64,64] INT8 NCHW */
    TfLiteTensor                   *output_tensor;
};

/* Static storage for the handle and the op resolver.
 * TFLite Micro requires static lifetime for the interpreter. */
static SensorPipeline s_pipeline;

/* Op resolver — list every builtin op present in the TFLite model.
 *
 * Determined by enumerating the flatbuffer operator codes (20 ops):
 *   QUANTIZE, CONV_2D, RESHAPE, TRANSPOSE, DEQUANTIZE, RSQRT, MEAN,
 *   SQUARED_DIFFERENCE, ADD, SUB, MUL, RELU, CONCATENATION, MAX_POOL_2D,
 *   FULLY_CONNECTED, LOGISTIC, PAD, PADV2, SQUARE, TANH
 * (SQUARE/TANH and the MULs come from the Pow-free GELU-tanh approximation;
 *  PAD/PADV2 from ONNX→TFLite conv padding.)
 */
static tflite::MicroMutableOpResolver<20> s_resolver;

static tflite::MicroInterpreter *s_interpreter_storage = nullptr;
/* Use placement-new to construct interpreter into static storage without heap. */
alignas(tflite::MicroInterpreter)
static uint8_t s_interpreter_buf[sizeof(tflite::MicroInterpreter)];

/* ── Image input accessor ────────────────────────────────────────────── */

int8_t *sensor_pipeline_image_input_ptr(SensorPipeline *pipeline)
{
    if (!pipeline || !pipeline->input_tensor) return nullptr;
    return pipeline->input_tensor->data.int8;
}

/* ── CWT ─────────────────────────────────────────────────────────────── */

SensorPipelineStatus sensor_pipeline_cwt_channel(uint8_t *pool,
                                                  float   *scalogram,
                                                  float   *signal,
                                                  int      n_samples,
                                                  int      ch_idx)
{
    if (!pool || !scalogram || !signal) return SENSOR_PIPELINE_ERR_POOL;
    if (n_samples < 64 || n_samples > static_cast<int>(MAX_SIGNAL_SAMPLES))
        return SENSOR_PIPELINE_ERR_INPUT;

    /* Scalogram slice for this channel: ch_idx × 64 × 64 floats */
    float *scalo_slice = scalogram + ch_idx * CWT_MCU_CH_SIZE;
    float *wksp        = wksp_buf(pool);

    int ret = cwt_mcu_process_channel(signal, n_samples, ch_idx, scalo_slice, wksp);
    return (ret == 0) ? SENSOR_PIPELINE_OK : SENSOR_PIPELINE_ERR_INPUT;
}

/* ── TFLite init ─────────────────────────────────────────────────────── */

SensorPipeline *sensor_pipeline_init(uint8_t       *pool,
                                     size_t         pool_bytes,
                                     float         *scalogram,
                                     const uint8_t *model_data)
{
    if (!pool || pool_bytes < SENSOR_PIPELINE_POOL_BYTES) {
        PRINTF("[pipeline] ERROR: pool too small (%u < %u bytes)\r\n",
               (unsigned)pool_bytes, (unsigned)SENSOR_PIPELINE_POOL_BYTES);
        return nullptr;
    }
    if (!scalogram) {
        PRINTF("[pipeline] ERROR: scalogram pointer is NULL\r\n");
        return nullptr;
    }
    if (!model_data) {
        PRINTF("[pipeline] ERROR: model_data is NULL\r\n");
        return nullptr;
    }

    /* Verify TFLite model magic */
    s_pipeline.model = tflite::GetModel(model_data);
    if (s_pipeline.model->version() != TFLITE_SCHEMA_VERSION) {
        PRINTF("[pipeline] ERROR: model schema version mismatch (%d vs %d)\r\n",
               (int)s_pipeline.model->version(), TFLITE_SCHEMA_VERSION);
        return nullptr;
    }

    /* Register ops — exactly the 20 ops present in the model */
    s_resolver.AddQuantize();
    s_resolver.AddConv2D();
    s_resolver.AddReshape();
    s_resolver.AddTranspose();
    s_resolver.AddDequantize();
    s_resolver.AddRsqrt();
    s_resolver.AddMean();
    s_resolver.AddSquaredDifference();
    s_resolver.AddAdd();
    s_resolver.AddSub();
    s_resolver.AddMul();
    s_resolver.AddRelu();
    s_resolver.AddConcatenation();
    s_resolver.AddMaxPool2D();
    s_resolver.AddFullyConnected();
    s_resolver.AddLogistic();
    s_resolver.AddPad();
    s_resolver.AddPadV2();
    s_resolver.AddSquare();
    s_resolver.AddTanh();

    /* Construct interpreter in-place using the full pool as tensor arena.
     * 391 KB available; ~370 KB needed at peak. */
    uint8_t *arena = arena_buf(pool);
    s_interpreter_storage = new (s_interpreter_buf) tflite::MicroInterpreter(
        s_pipeline.model, s_resolver, arena, ARENA_BYTES);

    TfLiteStatus status = s_interpreter_storage->AllocateTensors();
    if (status != kTfLiteOk) {
        PRINTF("[pipeline] ERROR: AllocateTensors failed (arena too small?)\r\n");
        PRINTF("[pipeline]        Arena: %u KB, required: check arena_used_bytes()\r\n",
               (unsigned)(ARENA_BYTES / 1024));
        return nullptr;
    }

    PRINTF("[pipeline] Arena used: %u KB / %u KB\r\n",
           (unsigned)(s_interpreter_storage->arena_used_bytes() / 1024),
           (unsigned)(ARENA_BYTES / 1024));

    s_pipeline.pool           = pool;
    s_pipeline.scalogram      = scalogram;
    s_pipeline.interpreter    = s_interpreter_storage;
    s_pipeline.input_tensor   = s_interpreter_storage->input(0);  /* image [1,224,224,3] */
    s_pipeline.scalo_tensor   = s_interpreter_storage->input(1);  /* scalogram [1,64,64,5] */
    s_pipeline.output_tensor  = s_interpreter_storage->output(0);

    /* Validate expected I/O shapes */
    /* input(0): image [1, 3, 224, 224] INT8 NCHW */
    TfLiteIntArray *img_dims = s_pipeline.input_tensor->dims;
    if (img_dims->size != 4 ||
        img_dims->data[0] != 1 || img_dims->data[1] != 3 ||
        img_dims->data[2] != 224 || img_dims->data[3] != 224) {
        PRINTF("[pipeline] ERROR: unexpected image input shape (expected [1,3,224,224])\r\n");
        return nullptr;
    }
    /* input(1): scalogram [1, 5, 64, 64] INT8 NCHW */
    TfLiteIntArray *scalo_dims = s_pipeline.scalo_tensor->dims;
    if (scalo_dims->size != 4 ||
        scalo_dims->data[0] != 1 || scalo_dims->data[1] != 5 ||
        scalo_dims->data[2] != 64 || scalo_dims->data[3] != 64) {
        PRINTF("[pipeline] ERROR: unexpected scalogram input shape (expected [1,5,64,64])\r\n");
        return nullptr;
    }

    PRINTF("[pipeline] Init OK  image=%s scalo=%s output=%s\r\n",
           TfLiteTypeGetName(s_pipeline.input_tensor->type),
           TfLiteTypeGetName(s_pipeline.scalo_tensor->type),
           TfLiteTypeGetName(s_pipeline.output_tensor->type));
    PRINTF("[pipeline] Image input:     scale=%.7f  zero_point=%ld\r\n",
           (double)s_pipeline.input_tensor->params.scale,
           s_pipeline.input_tensor->params.zero_point);
    PRINTF("[pipeline] Scalogram input: scale=%.7f  zero_point=%ld\r\n",
           (double)s_pipeline.scalo_tensor->params.scale,
           s_pipeline.scalo_tensor->params.zero_point);
    PRINTF("[pipeline] Output:          scale=%.7f  zero_point=%ld\r\n",
           (double)s_pipeline.output_tensor->params.scale,
           s_pipeline.output_tensor->params.zero_point);

    return &s_pipeline;
}

/* ── Inference ───────────────────────────────────────────────────────── */

SensorPipelineStatus sensor_pipeline_infer(SensorPipeline *pipeline,
                                            float          *scalogram,
                                            float          *wear_um_out)
{
    if (!pipeline || !scalogram || !wear_um_out) return SENSOR_PIPELINE_ERR_INPUT;

    TfLiteTensor *scalo_in = pipeline->scalo_tensor;
    TfLiteTensor *out      = pipeline->output_tensor;

    /* input(0) = image: already populated by the caller via
     * sensor_pipeline_image_input_ptr() before this call. */

    /* input(1) = scalogram [1,5,64,64] NCHW. The CWT buffer is already stored
     * as [ch][h][w] = NCHW, identical to the model layout, so the copy is a
     * straight element-wise pass (no transpose). */
    static constexpr int SCALO_FLOATS =
        SENSOR_PIPELINE_SCALOGRAM_FLOATS;  /* 5 * 64 * 64 */

    if (scalo_in->type == kTfLiteFloat32) {
        memcpy(scalo_in->data.f, scalogram, SCALO_FLOATS * sizeof(float));
    } else if (scalo_in->type == kTfLiteInt8) {
        const float scale = scalo_in->params.scale;
        const int   zp    = scalo_in->params.zero_point;
        for (int i = 0; i < SCALO_FLOATS; i++) {
            int q = static_cast<int>(roundf(scalogram[i] / scale)) + zp;
            scalo_in->data.int8[i] =
                static_cast<int8_t>(q < -128 ? -128 : (q > 127 ? 127 : q));
        }
    } else {
        PRINTF("[pipeline] ERROR: unsupported scalogram tensor type %d\r\n",
               scalo_in->type);
        return SENSOR_PIPELINE_ERR_INVOKE;
    }

    /* Run inference */
    TfLiteStatus status = pipeline->interpreter->Invoke();
    if (status != kTfLiteOk) {
        PRINTF("[pipeline] ERROR: Invoke() failed\r\n");
        return SENSOR_PIPELINE_ERR_INVOKE;
    }

    /* Read output — INT8 dequantised to float (µm) */
    if (out->type == kTfLiteFloat32) {
        *wear_um_out = out->data.f[0];
    } else if (out->type == kTfLiteInt8) {
        *wear_um_out = static_cast<float>(
            out->data.int8[0] - out->params.zero_point) * out->params.scale;
    } else {
        PRINTF("[pipeline] ERROR: unsupported output tensor type %d\r\n", out->type);
        return SENSOR_PIPELINE_ERR_INVOKE;
    }

    return SENSOR_PIPELINE_OK;
}
