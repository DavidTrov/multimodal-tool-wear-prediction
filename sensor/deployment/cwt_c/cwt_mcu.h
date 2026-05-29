/*
 * cwt_mcu.h — Memory-optimised CWT scalogram for MCU deployment.
 *
 * Designed for NXP FRDM-MCXN947 (Cortex-M33, 512 KB SRAM, 2 MB flash).
 * Uses CMSIS-DSP on MCU; compiles on desktop via cmsis_shim.h (DESKTOP_SHIM).
 *
 * Architecture: one channel at a time, direct time-domain CWT (no large FFT
 * buffers).  Peak RAM ~508 KB for 99K-sample signals.
 *
 * Pipeline (per channel):
 *   1. Optional zero-phase HPF (force channels 2-4)
 *   2. Compute wavelet kernel via 2048-point IFFT (reused across time bins)
 *   3. Direct convolution at 64 subsampled time bins per scale
 *   4. Per-channel min-max normalisation to [0,1]
 *
 * Output layout: [5][64][64] float32, row-major.
 * Channel order: acc=0, acoustic=1, fx=2, fy=3, fz=4.
 */

#ifndef CWT_MCU_H
#define CWT_MCU_H

#ifdef __cplusplus
extern "C" {
#endif

#define CWT_MCU_N_CHANNELS 5
#define CWT_MCU_N_SCALES   64
#define CWT_MCU_N_TIME     64
#define CWT_MCU_CH_SIZE    (CWT_MCU_N_SCALES * CWT_MCU_N_TIME)   /* 4096 */
#define CWT_MCU_OUT_SIZE   (CWT_MCU_N_CHANNELS * CWT_MCU_CH_SIZE) /* 20480 */

/* Wavelet kernel IFFT length — 2048 resolves all 64 scales adequately. */
#define CWT_MCU_N_KER      2048

/*
 * cwt_mcu_process_channel — compute one channel's (64x64) scalogram.
 *
 * Parameters:
 *   signal     float32[n_samples] — one channel's cutting-segment data.
 *              Modified in-place if HPF is applied (ch_idx 2-4).
 *   n_samples  Signal length (must be >= 64).
 *   ch_idx     Channel index 0-4.  Channels 2-4 get HPF applied.
 *   out        float32[64*64] — caller-allocated output, normalised to [0,1].
 *   workspace  float32[2 * CWT_MCU_N_KER] — caller-allocated temp (16 KB).
 *              Reused across scales; contents undefined on return.
 *
 * Returns 0 on success, -1 on failure.
 */
int cwt_mcu_process_channel(
    float       *signal,
    int          n_samples,
    int          ch_idx,
    float       *out,
    float       *workspace
);

#ifdef __cplusplus
}
#endif

#endif /* CWT_MCU_H */
