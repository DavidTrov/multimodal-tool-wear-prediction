/**
 * sensor_pipeline.h — CWT + CNN inference pipeline for FRDM-MCXN947.
 *
 * Plain C API so it can be called from both C and C++ translation units.
 * The implementation (sensor_pipeline.cpp) uses TFLite Micro internally.
 *
 * Memory contract
 * ───────────────
 * The caller provides a single contiguous memory pool from which this module
 * carves its buffers.  The pool must be at least SENSOR_PIPELINE_POOL_BYTES.
 *
 * Phase 1 (CWT): signal + scalogram + workspace live inside the pool.
 * Phase 2 (inference): TFLite tensor arena reuses the signal region.
 * The scalogram (80 KB) persists across both phases.
 *
 *                   ┌────────────── pool (≥508 KB) ─────────────────┐
 *   CWT phase:      │  signal (1ch)  │ scalogram │ workspace │ ...  │
 *                   │   ~396 KB      │   80 KB   │   16 KB   │      │
 *   Infer phase:    │  tensor arena  │ scalogram │           │ ...  │
 *                   │   412 KB       │ (persists)│           │      │
 *                   └────────────────┴───────────┴───────────┴──────┘
 */

#ifndef SENSOR_PIPELINE_H
#define SENSOR_PIPELINE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Constants ──────────────────────────────────────────────────────────── */

#define SENSOR_PIPELINE_N_CHANNELS   5
#define SENSOR_PIPELINE_MAX_SAMPLES  99000    /* ~61 s at 1625 Hz */
#define SENSOR_PIPELINE_SCALOGRAM_W  64
#define SENSOR_PIPELINE_SCALOGRAM_H  64
#define SENSOR_PIPELINE_SCALOGRAM_FLOATS \
    (SENSOR_PIPELINE_N_CHANNELS * SENSOR_PIPELINE_SCALOGRAM_W * SENSOR_PIPELINE_SCALOGRAM_H)

/* Minimum pool size in bytes.
 * signal (396 KB) + scalogram (80 KB) + workspace (16 KB) + 16 KB stack margin */
#define SENSOR_PIPELINE_POOL_BYTES   (508u * 1024u)

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
 * sensor_pipeline_init — Initialise the pipeline.
 *
 * @param pool         Caller-allocated memory pool (must be static or persistent).
 *                     Minimum size: SENSOR_PIPELINE_POOL_BYTES.
 *                     Recommended: static uint8_t pool[512*1024] placed in main SRAM.
 * @param pool_bytes   Size of pool in bytes.
 * @param model_data   Pointer to the TFLite flatbuffer (g_model_data[] from
 *                     phase4_multiscale_sgdm_best_25_model_data.h).
 *                     Must remain valid for the lifetime of the pipeline.
 * @return             Pointer to the initialised handle, or NULL on error.
 *                     Errors are printed via PRINTF (SDK debug console).
 *
 * Must be called once after TFLite arena is available (i.e. after CWT completes
 * and the signal buffer is freed). Call sensor_pipeline_init AFTER the last
 * sensor_pipeline_cwt_channel call.
 */
SensorPipeline *sensor_pipeline_init(uint8_t *pool, size_t pool_bytes,
                                     const uint8_t *model_data);

/**
 * sensor_pipeline_cwt_channel — Compute the CWT scalogram for one channel.
 *
 * Must be called for channels 0–4 (in any order) before sensor_pipeline_infer.
 * Do NOT call sensor_pipeline_init before all five channels are processed.
 *
 * @param pipeline     Handle returned by a previous sensor_pipeline_init, OR
 *                     NULL before init — the function accesses the pool directly
 *                     using an internal layout assumption. Pass a non-NULL handle
 *                     if init has already been called.
 * @param pool         The same pool passed to sensor_pipeline_init (used when
 *                     pipeline == NULL).
 * @param signal       float32 array of n_samples cutting-segment values.
 *                     Will be modified in-place (HPF for force channels).
 * @param n_samples    Number of samples (must be ≥ 64).
 * @param ch_idx       Channel index 0–4  (acc=0, acoustic=1, fx=2, fy=3, fz=4).
 * @return             SENSOR_PIPELINE_OK or error code.
 */
SensorPipelineStatus sensor_pipeline_cwt_channel(uint8_t *pool,
                                                  float   *signal,
                                                  int      n_samples,
                                                  int      ch_idx);

/**
 * sensor_pipeline_infer — Run CNN inference on the computed scalogram.
 *
 * Call after sensor_pipeline_cwt_channel has been called for all 5 channels
 * and sensor_pipeline_init has been called.
 *
 * @param pipeline     Handle from sensor_pipeline_init.
 * @param wear_um_out  Output: predicted tool wear in µm.
 * @return             SENSOR_PIPELINE_OK or error code.
 */
SensorPipelineStatus sensor_pipeline_infer(SensorPipeline *pipeline,
                                            float          *wear_um_out);

/**
 * sensor_pipeline_scalogram_ptr — Return pointer to the (5,64,64) float32
 * scalogram buffer inside the pool (for debug inspection).
 */
float *sensor_pipeline_scalogram_ptr(uint8_t *pool);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SENSOR_PIPELINE_H */
