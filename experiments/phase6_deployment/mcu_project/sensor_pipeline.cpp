/**
 * sensor_pipeline.cpp — CWT + TFLite Micro inference for FRDM-MCXN947.
 *
 * Build with: -DARM_MATH_CM33 -DTF_LITE_STATIC_MEMORY -O2 -ffast-math
 * Do NOT define DESKTOP_SHIM — cwt_mcu.c uses real arm_math.h here.
 *
 * Memory layout inside the caller-provided pool (≥508 KB):
 *
 *   Offset 0:                signal buffer  (MAX_SIGNAL_SAMPLES × 4 bytes = 396 KB)
 *   Offset SIGNAL_BYTES:     scalogram      (5×64×64 × 4 bytes  =  80 KB)
 *   Offset SIGNAL_BYTES
 *         + SCALO_BYTES:     CWT workspace  (2×2048 × 4 bytes   =  16 KB)
 *
 * After CWT completes, the signal buffer + workspace region is reused as the
 * TFLite tensor arena (412 KB available, ~384 KB required at peak).
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
static constexpr size_t SIGNAL_BYTES  = MAX_SIGNAL_SAMPLES * sizeof(float); /* 396 KB */
static constexpr size_t SCALO_FLOATS  = SENSOR_PIPELINE_SCALOGRAM_FLOATS;
static constexpr size_t SCALO_BYTES   = SCALO_FLOATS * sizeof(float);        /*  80 KB */
static constexpr size_t WKSP_FLOATS   = 2u * CWT_MCU_N_KER;
static constexpr size_t WKSP_BYTES    = WKSP_FLOATS * sizeof(float);         /*  16 KB */

/* Offsets within pool */
static constexpr size_t OFF_SIGNAL    = 0;
static constexpr size_t OFF_SCALO     = OFF_SIGNAL + SIGNAL_BYTES;
static constexpr size_t OFF_WKSP      = OFF_SCALO  + SCALO_BYTES;
static constexpr size_t OFF_ARENA     = 0;    /* arena reuses signal region */
static constexpr size_t ARENA_BYTES   = SIGNAL_BYTES + WKSP_BYTES; /* 412 KB */

static inline float  *signal_buf(uint8_t *pool)   { return reinterpret_cast<float*>(pool + OFF_SIGNAL); }
static inline float  *scalo_buf(uint8_t *pool)     { return reinterpret_cast<float*>(pool + OFF_SCALO);  }
static inline float  *wksp_buf(uint8_t *pool)      { return reinterpret_cast<float*>(pool + OFF_WKSP);   }
static inline uint8_t*arena_buf(uint8_t *pool)     { return pool + OFF_ARENA; }

/* ── SensorPipeline struct ───────────────────────────────────────────── */

struct SensorPipeline {
    uint8_t                        *pool;
    const tflite::Model            *model;
    tflite::MicroInterpreter       *interpreter;
    TfLiteTensor                   *input_tensor;
    TfLiteTensor                   *output_tensor;
};

/* Static storage for the handle and the op resolver.
 * TFLite Micro requires static lifetime for the interpreter. */
static SensorPipeline s_pipeline;

/* Op resolver — list every builtin op present in the TFLite model.
 *
 * Determined by parsing the flatbuffer operator_codes table (19 ops):
 *   QUANTIZE, CONV_2D, RESHAPE, TRANSPOSE, DEQUANTIZE, RSQRT, MEAN,
 *   SQUARED_DIFFERENCE, ADD, SUB, MUL, RELU, CONCATENATION, MAX_POOL_2D,
 *   AVERAGE_POOL_2D, FULLY_CONNECTED, LOGISTIC, REDUCE_MAX, SUM
 *
 * Unlisted ops are not linked, saving flash space. Unused ops in the
 * resolver are harmless but waste a few bytes each.
 */
static tflite::MicroMutableOpResolver<19> s_resolver;

static tflite::MicroInterpreter *s_interpreter_storage = nullptr;
/* Use placement-new to construct interpreter into static storage without heap. */
alignas(tflite::MicroInterpreter)
static uint8_t s_interpreter_buf[sizeof(tflite::MicroInterpreter)];

/* ── CWT ─────────────────────────────────────────────────────────────── */

SensorPipelineStatus sensor_pipeline_cwt_channel(uint8_t *pool,
                                                  float   *signal,
                                                  int      n_samples,
                                                  int      ch_idx)
{
    if (!pool || !signal) return SENSOR_PIPELINE_ERR_POOL;
    if (n_samples < 64 || n_samples > static_cast<int>(MAX_SIGNAL_SAMPLES))
        return SENSOR_PIPELINE_ERR_INPUT;

    float *scalo   = scalo_buf(pool) + ch_idx * CWT_MCU_CH_SIZE;
    float *wksp    = wksp_buf(pool);

    int ret = cwt_mcu_process_channel(signal, n_samples, ch_idx, scalo, wksp);
    return (ret == 0) ? SENSOR_PIPELINE_OK : SENSOR_PIPELINE_ERR_INPUT;
}

float *sensor_pipeline_scalogram_ptr(uint8_t *pool)
{
    return scalo_buf(pool);
}

/* ── TFLite init ─────────────────────────────────────────────────────── */

SensorPipeline *sensor_pipeline_init(uint8_t      *pool,
                                     size_t        pool_bytes,
                                     const uint8_t *model_data)
{
    if (!pool || pool_bytes < SENSOR_PIPELINE_POOL_BYTES) {
        PRINTF("[pipeline] ERROR: pool too small (%u < %u bytes)\r\n",
               (unsigned)pool_bytes, (unsigned)SENSOR_PIPELINE_POOL_BYTES);
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

    /* Register ops — exactly the 19 ops present in the model */
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
    s_resolver.AddAveragePool2D();
    s_resolver.AddFullyConnected();
    s_resolver.AddLogistic();
    s_resolver.AddReduceMax();
    s_resolver.AddSum();

    /* Construct interpreter in-place using the arena that previously held
     * the signal buffer.  412 KB available; ~384 KB needed at peak. */
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
    s_pipeline.interpreter    = s_interpreter_storage;
    s_pipeline.input_tensor   = s_interpreter_storage->input(0);
    s_pipeline.output_tensor  = s_interpreter_storage->output(0);

    /* Validate expected I/O shapes */
    TfLiteIntArray *in_dims = s_pipeline.input_tensor->dims;
    if (in_dims->size != 4 ||
        in_dims->data[0] != 1 || in_dims->data[1] != 5 ||
        in_dims->data[2] != 64 || in_dims->data[3] != 64) {
        PRINTF("[pipeline] ERROR: unexpected input shape\r\n");
        return nullptr;
    }

    PRINTF("[pipeline] Init OK  input=%s output=%s\r\n",
           TfLiteTypeGetName(s_pipeline.input_tensor->type),
           TfLiteTypeGetName(s_pipeline.output_tensor->type));

    return &s_pipeline;
}

/* ── Inference ───────────────────────────────────────────────────────── */

SensorPipelineStatus sensor_pipeline_infer(SensorPipeline *pipeline,
                                            float          *wear_um_out)
{
    if (!pipeline || !wear_um_out) return SENSOR_PIPELINE_ERR_INPUT;

    TfLiteTensor *in  = pipeline->input_tensor;
    TfLiteTensor *out = pipeline->output_tensor;
    float        *scalogram = scalo_buf(pipeline->pool);

    /* Copy scalogram into the TFLite input tensor.
     *
     * The model was exported with float32 I/O (--keep-io-tensors-format).
     * Internal layers are INT8; the first op (QUANTIZE) converts the float32
     * input to INT8 at inference time.  No manual quantization needed here. */
    if (in->type == kTfLiteFloat32) {
        memcpy(in->data.f, scalogram, SCALO_BYTES);
    } else if (in->type == kTfLiteInt8) {
        /* Fallback: manual quantization if model was re-exported with INT8 I/O */
        const float scale = in->params.scale;
        const int   zp    = in->params.zero_point;
        for (size_t i = 0; i < SCALO_FLOATS; i++) {
            int q = static_cast<int>(roundf(scalogram[i] / scale)) + zp;
            in->data.int8[i] = static_cast<int8_t>(
                q < -128 ? -128 : (q > 127 ? 127 : q));
        }
    } else {
        PRINTF("[pipeline] ERROR: unsupported input tensor type %d\r\n", in->type);
        return SENSOR_PIPELINE_ERR_INVOKE;
    }

    /* Run inference */
    TfLiteStatus status = pipeline->interpreter->Invoke();
    if (status != kTfLiteOk) {
        PRINTF("[pipeline] ERROR: Invoke() failed\r\n");
        return SENSOR_PIPELINE_ERR_INVOKE;
    }

    /* Read output */
    if (out->type == kTfLiteFloat32) {
        *wear_um_out = out->data.f[0];
    } else if (out->type == kTfLiteInt8) {
        *wear_um_out = (out->data.int8[0] - out->params.zero_point)
                       * out->params.scale;
    } else {
        PRINTF("[pipeline] ERROR: unsupported output tensor type %d\r\n", out->type);
        return SENSOR_PIPELINE_ERR_INVOKE;
    }

    return SENSOR_PIPELINE_OK;
}
