/**
 * sensor_pipeline.h — CWT + CNN inference pipeline for FRDM-MCXN947.
 *
 * Plain C API so it can be called from both C and C++ translation units.
 * The implementation (sensor_pipeline.cpp) uses TFLite Micro internally.
 *
 * Memory contract
 * ───────────────
 * The caller provides:
 *   (a) A contiguous pool in main SRAM — signal buffer + CWT workspace, later
 *       reused as the TFLite tensor arena.
 *   (b) A separate scalogram buffer (ideally placed in SRAMX to save SRAM):
 *           __attribute__((section(".bss.$SRAMX")))
 *           static float g_scalogram[SENSOR_PIPELINE_SCALOGRAM_FLOATS];
 *
 * Keeping the scalogram (80 KB) out of the pool lets the pool be smaller,
 * which is critical on the MCXN947 with only 416 KB of contiguous SRAM.
 *
 * Phase 1 (CWT): signal lives in pool; scalogram slices written to g_scalogram.
 * Phase 2 (inference): pool becomes the TFLite tensor arena (391 KB).
 *
 *  pool (≥391 KB in SRAM):      ┌─ signal buffer ─┬─ CWT workspace ─┐
 *  CWT phase:                   │   384 KB         │   16 KB         │
 *  Infer phase (arena):         │◄────── 391 KB tensor arena ───────►│
 *                               └──────────────────┴─────────────────┘
 *
 *  scalogram (80 KB in SRAMX):  ┌─ ch0 64×64 ─┬─ ch1 ─┬─...─┬─ ch4 ─┐
 *                               └─────────────┴────────┴─────┴───────┘
 */

#ifndef SENSOR_PIPELINE_H
#define SENSOR_PIPELINE_H

#include <stddef.h>   /* size_t */
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Constants ──────────────────────────────────────────────────────────── */

#define SENSOR_PIPELINE_N_CHANNELS   5
#define SENSOR_PIPELINE_MAX_SAMPLES  96000    /* ~59 s at 1625 Hz per channel */
#define SENSOR_PIPELINE_SCALOGRAM_W  64
#define SENSOR_PIPELINE_SCALOGRAM_H  64
#define SENSOR_PIPELINE_SCALOGRAM_FLOATS \
    (SENSOR_PIPELINE_N_CHANNELS * SENSOR_PIPELINE_SCALOGRAM_W * SENSOR_PIPELINE_SCALOGRAM_H)

/* CWT workspace size in bytes (must match 2 × CWT_MCU_N_KER × sizeof(float)
 * where CWT_MCU_N_KER = 2048 in cwt_mcu.h). */
#define SENSOR_PIPELINE_WKSP_BYTES   (2u * 2048u * 4u)  /* 16 KB */

/* Pool = signal buffer (becomes TFLite arena after CWT) + CWT workspace.
 * Scalogram is NOT in the pool — allocate it separately (see file header). */
#define SENSOR_PIPELINE_POOL_BYTES   \
    (SENSOR_PIPELINE_MAX_SAMPLES * 4u + SENSOR_PIPELINE_WKSP_BYTES)  /* ~391 KB */

/* Image input dimensions (INT8, NCHW layout — ONNX-origin model).
 * The fusion model expects input(0) = image [1, 3, 224, 224] INT8. */
#define SENSOR_PIPELINE_IMG_H      224u
#define SENSOR_PIPELINE_IMG_W      224u
#define SENSOR_PIPELINE_IMG_C      3u
#define SENSOR_PIPELINE_IMG_BYTES  (SENSOR_PIPELINE_IMG_H * SENSOR_PIPELINE_IMG_W \
                                    * SENSOR_PIPELINE_IMG_C)  /* 150,528 bytes */

/* ── Result codes ───────────────────────────────────────────────────────── */

typedef enum {
    SENSOR_PIPELINE_OK           =  0,
    SENSOR_PIPELINE_ERR_POOL     = -1,  /* pool too small or NULL */
    SENSOR_PIPELINE_ERR_MODEL    = -2,  /* TFLite model invalid   */
    SENSOR_PIPELINE_ERR_ARENA    = -3,  /* tensor arena too small */
    SENSOR_PIPELINE_ERR_INVOKE   = -4,  /* inference failed       */
    SENSOR_PIPELINE_ERR_INPUT    = -5,  /* bad signal length      */
} SensorPipelineStatus;

/* ── Handle ─────────────────────────────────────────────────────────────── */

/* Opaque handle — do not access fields directly. */
typedef struct SensorPipeline SensorPipeline;

/* ── API ────────────────────────────────────────────────────────────────── */

/**
 * sensor_pipeline_cwt_channel — Compute the CWT scalogram for one channel.
 *
 * Must be called for channels 0–4 (in any order) before sensor_pipeline_init.
 * Do NOT call sensor_pipeline_init before all five channels are processed.
 *
 * @param pool         Caller-allocated pool (signal buffer region used as scratch).
 * @param scalogram    Caller-allocated float array [SENSOR_PIPELINE_SCALOGRAM_FLOATS].
 *                     Ideally placed in SRAMX (separate from main SRAM pool).
 *                     Slice for ch_idx is written at offset ch_idx × 64 × 64.
 * @param signal       float32 array of n_samples. Modified in-place (HPF for
 *                     force channels).
 * @param n_samples    Number of samples (must be ≥ 64 and ≤ SENSOR_PIPELINE_MAX_SAMPLES).
 * @param ch_idx       Channel index 0–4  (acc=0, acoustic=1, fx=2, fy=3, fz=4).
 * @return             SENSOR_PIPELINE_OK or error code.
 */
SensorPipelineStatus sensor_pipeline_cwt_channel(uint8_t *pool,
                                                  float   *scalogram,
                                                  float   *signal,
                                                  int      n_samples,
                                                  int      ch_idx);

/**
 * sensor_pipeline_init — Initialise the TFLite Micro interpreter.
 *
 * Call AFTER all five sensor_pipeline_cwt_channel() calls.
 * The pool's signal region is reused as the TFLite tensor arena here.
 *
 * @param pool         Same pool used in cwt_channel calls.
 * @param pool_bytes   sizeof(pool) — must be ≥ SENSOR_PIPELINE_POOL_BYTES.
 * @param scalogram    Same scalogram pointer used in cwt_channel calls.
 *                     Must remain valid for the lifetime of the pipeline.
 * @param model_data   Pointer to the TFLite flatbuffer (g_model_data[]).
 *                     Must remain valid for the lifetime of the pipeline.
 * @return             Pointer to the initialised handle, or NULL on error.
 */
SensorPipeline *sensor_pipeline_init(uint8_t       *pool,
                                     size_t         pool_bytes,
                                     float         *scalogram,
                                     const uint8_t *model_data);

/**
 * sensor_pipeline_image_input_ptr — Return pointer to the image input tensor.
 *
 * Valid only AFTER sensor_pipeline_init() returns non-NULL.
 * Write SENSOR_PIPELINE_IMG_BYTES of INT8 pixel data into this buffer before
 * calling sensor_pipeline_infer().
 *
 * Pixel layout: NCHW (channels × height × width, RGB), INT8.
 * Quantisation: q = clamp(round(x_normalised / scale) + zero_point, -128, 127)
 *   where x_normalised = (pixel_float - mean) / std  with ImageNet statistics.
 *   scale and zero_point are printed by sensor_pipeline_init() over UART.
 */
int8_t *sensor_pipeline_image_input_ptr(SensorPipeline *pipeline);

/**
 * sensor_pipeline_infer — Run fusion inference (image + scalogram → wear).
 *
 * Call after:
 *   1. sensor_pipeline_cwt_channel() for all 5 channels
 *   2. sensor_pipeline_init()
 *   3. Writing the image bytes into sensor_pipeline_image_input_ptr()
 *
 * @param pipeline     Handle from sensor_pipeline_init.
 * @param scalogram    Same float array passed to cwt_channel / init.
 * @param wear_um_out  Output: predicted tool wear in µm.
 * @return             SENSOR_PIPELINE_OK or error code.
 */
SensorPipelineStatus sensor_pipeline_infer(SensorPipeline *pipeline,
                                            float          *scalogram,
                                            float          *wear_um_out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SENSOR_PIPELINE_H */
