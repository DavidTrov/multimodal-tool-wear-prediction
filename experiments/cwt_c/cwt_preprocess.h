/*
 * cwt_preprocess.h
 *
 * CWT scalogram preprocessing for MATWI tool-wear sensor data.
 *
 * Pipeline (matches Python reference in experiments/phase3_fusion/precompute_scalograms.py):
 *   1. Zero-phase 4th-order Butterworth HPF at 15 Hz on force channels (fx, fy, fz)
 *   2. Sequential complex Morlet CWT (cmor1.5-1.0) over 64 log-spaced scales
 *   3. Subsample each scale's power row to 64 time bins
 *   4. Global min-max normalisation across the full (5 x 64 x 64) tensor
 *
 * Output: (5, 64, 64) float32 scalogram, values in [0, 1].
 *
 * Channel order: acc=0, acoustic=1, fx=2, fy=3, fz=4  (SENSOR_COLS in Python)
 * HPF is applied to channels 2, 3, 4 (force channels).
 */

#ifndef CWT_PREPROCESS_H
#define CWT_PREPROCESS_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * cwt_compute_scalogram
 *
 * Params
 *   signal     : input sensor data, row-major [n_channels][n_samples] (float)
 *   n_channels : must be 5
 *   n_samples  : number of samples per channel (after aircut gating)
 *   out        : caller-allocated output buffer, [5 * 64 * 64] floats (= 20480 floats)
 *
 * Returns 0 on success, -1 on memory allocation failure.
 *
 * The caller is responsible for allocating `out` (20480 * sizeof(float) = 80 KB).
 * All internal working memory is heap-allocated and freed before return.
 */
int cwt_compute_scalogram(
    const float *signal,
    int          n_channels,
    int          n_samples,
    float       *out
);

#define CWT_N_CHANNELS 5
#define CWT_N_SCALES   64
#define CWT_N_TIME     64
#define CWT_OUT_SIZE   (CWT_N_CHANNELS * CWT_N_SCALES * CWT_N_TIME)  /* 20480 */

#ifdef __cplusplus
}
#endif

#endif /* CWT_PREPROCESS_H */
