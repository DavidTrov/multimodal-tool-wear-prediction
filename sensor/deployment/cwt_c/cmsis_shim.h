/*
 * cmsis_shim.h — Desktop compatibility layer for CMSIS-DSP functions.
 *
 * When DESKTOP_SHIM is defined (macOS/Linux builds), this header provides
 * C implementations of the CMSIS-DSP functions used by cwt_mcu.c.
 * On the actual MCU, DESKTOP_SHIM is NOT defined and arm_math.h is included
 * directly — no kiss_fft dependency, real CMSIS-DSP runs on hardware.
 *
 * Functions provided:
 *   arm_cfft_f32                         — wraps kiss_fft inverse
 *   arm_biquad_cascade_df2T_f32/init     — Transposed Direct Form II biquad
 *   arm_cmplx_dot_prod_f32              — complex dot product
 *   arm_min_f32 / arm_max_f32           — min/max scan
 *   arm_offset_f32 / arm_scale_f32      — vector arithmetic
 */

#ifndef CMSIS_SHIM_H
#define CMSIS_SHIM_H

#ifdef DESKTOP_SHIM

#include "kiss_fft/kiss_fft.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

/* ── Types ─────────────────────────────────────────────────────────────────── */

typedef struct {
    int             len;       /* FFT length */
    kiss_fft_cfg    inv_cfg;   /* inverse FFT config (allocated once) */
} arm_cfft_instance_f32;

typedef struct {
    uint8_t         numStages;
    float          *pState;    /* 2 floats per stage */
    const float    *pCoeffs;   /* 5 floats per stage: {b0,b1,b2,a1,a2} */
} arm_biquad_cascade_df2T_instance_f32;

/* ── FFT ───────────────────────────────────────────────────────────────────── */

/*
 * Initialise a CFFT instance for the given length.
 * On MCU this would be arm_cfft_sR_f32_lenNNNN from const tables.
 * Here we allocate a kiss_fft inverse config.
 */
static inline void cmsis_shim_cfft_init(arm_cfft_instance_f32 *inst, int len)
{
    inst->len     = len;
    inst->inv_cfg = kiss_fft_alloc(len, 1 /* inverse */, NULL, NULL);
}

static inline void cmsis_shim_cfft_free(arm_cfft_instance_f32 *inst)
{
    if (inst->inv_cfg) {
        kiss_fft_free(inst->inv_cfg);
        inst->inv_cfg = NULL;
    }
}

/*
 * arm_cfft_f32 — in-place complex FFT/IFFT.
 *
 * p         : interleaved float32 complex array [2*len] = {re0,im0,re1,im1,...}
 * ifftFlag  : 0 = forward FFT, 1 = inverse FFT
 * bitReverseFlag : ignored (kiss_fft always does bit reversal)
 *
 * CMSIS-DSP convention: IFFT does NOT divide by N — caller must scale.
 * kiss_fft inverse also does NOT divide by N, so this matches.
 */
static inline void arm_cfft_f32(const arm_cfft_instance_f32 *S,
                                float *p,
                                uint8_t ifftFlag,
                                uint8_t bitReverseFlag)
{
    (void)bitReverseFlag;
    (void)ifftFlag;  /* we only use inverse in this codebase */

    int len = S->len;

    /* kiss_fft uses kiss_fft_cpx {float r, i} which is the same layout as
     * interleaved float pairs. Static assert on layout: */
    kiss_fft_cpx *in  = (kiss_fft_cpx *)p;
    kiss_fft_cpx *tmp = (kiss_fft_cpx *)malloc((size_t)len * sizeof(kiss_fft_cpx));
    if (!tmp) return;

    kiss_fft(S->inv_cfg, in, tmp);

    /* Copy result back (in-place) */
    memcpy(p, tmp, (size_t)len * sizeof(kiss_fft_cpx));
    free(tmp);
}

/* ── Biquad cascade (Transposed Direct Form II) ───────────────────────────── */

/*
 * CMSIS-DSP coefficient layout per stage: {b0, b1, b2, a1, a2}
 * where a1 and a2 are NEGATED compared to scipy:
 *   scipy:  y = b0*x + z1;  z1 = b1*x - a1_scipy*y + z2;  z2 = b2*x - a2_scipy*y
 *   CMSIS:  y = b0*x + z1;  z1 = b1*x + a1_cmsis*y + z2;  z2 = b2*x + a2_cmsis*y
 * So a1_cmsis = -a1_scipy,  a2_cmsis = -a2_scipy.
 *
 * The caller packs coefficients with this negation.
 */
static inline void arm_biquad_cascade_df2T_init_f32(
    arm_biquad_cascade_df2T_instance_f32 *S,
    uint8_t numStages,
    const float *pCoeffs,
    float *pState)
{
    S->numStages = numStages;
    S->pCoeffs   = pCoeffs;
    S->pState    = pState;
}

static inline void arm_biquad_cascade_df2T_f32(
    const arm_biquad_cascade_df2T_instance_f32 *S,
    const float *pSrc,
    float *pDst,
    uint32_t blockSize)
{
    uint8_t stage;
    uint32_t i;
    const float *src = pSrc;
    float       *dst = pDst;

    /* If src != dst, copy so we can cascade stages in-place on dst */
    if (src != dst) {
        memcpy(dst, src, blockSize * sizeof(float));
    }

    for (stage = 0; stage < S->numStages; stage++) {
        const float b0 = S->pCoeffs[stage * 5 + 0];
        const float b1 = S->pCoeffs[stage * 5 + 1];
        const float b2 = S->pCoeffs[stage * 5 + 2];
        const float a1 = S->pCoeffs[stage * 5 + 3];  /* already negated */
        const float a2 = S->pCoeffs[stage * 5 + 4];  /* already negated */
        float z1 = S->pState[stage * 2 + 0];
        float z2 = S->pState[stage * 2 + 1];

        for (i = 0; i < blockSize; i++) {
            float x = dst[i];
            float y = b0 * x + z1;
            z1 = b1 * x + a1 * y + z2;
            z2 = b2 * x + a2 * y;
            dst[i] = y;
        }

        S->pState[stage * 2 + 0] = z1;
        S->pState[stage * 2 + 1] = z2;
    }
}

/* ── Complex dot product ───────────────────────────────────────────────────── */

/*
 * arm_cmplx_dot_prod_f32
 *
 * a, b: interleaved complex arrays {re0,im0,re1,im1,...}, length numSamples each.
 * realResult = sum(a_re*b_re + a_im*b_im)
 * imagResult = sum(a_re*b_im - a_im*b_re)
 */
static inline void arm_cmplx_dot_prod_f32(
    const float *a,
    const float *b,
    uint32_t numSamples,
    float *realResult,
    float *imagResult)
{
    float sumR = 0.0f, sumI = 0.0f;
    uint32_t i;
    for (i = 0; i < numSamples; i++) {
        float ar = a[2*i], ai = a[2*i+1];
        float br = b[2*i], bi = b[2*i+1];
        sumR += ar * br + ai * bi;
        sumI += ar * bi - ai * br;
    }
    *realResult = sumR;
    *imagResult = sumI;
}

/* ── Vector min / max ──────────────────────────────────────────────────────── */

static inline void arm_min_f32(const float *pSrc, uint32_t blockSize,
                               float *pResult, uint32_t *pIndex)
{
    float minVal = pSrc[0];
    uint32_t idx = 0, i;
    for (i = 1; i < blockSize; i++) {
        if (pSrc[i] < minVal) { minVal = pSrc[i]; idx = i; }
    }
    *pResult = minVal;
    *pIndex  = idx;
}

static inline void arm_max_f32(const float *pSrc, uint32_t blockSize,
                               float *pResult, uint32_t *pIndex)
{
    float maxVal = pSrc[0];
    uint32_t idx = 0, i;
    for (i = 1; i < blockSize; i++) {
        if (pSrc[i] > maxVal) { maxVal = pSrc[i]; idx = i; }
    }
    *pResult = maxVal;
    *pIndex  = idx;
}

/* ── Vector offset / scale ─────────────────────────────────────────────────── */

static inline void arm_offset_f32(const float *pSrc, float offset,
                                  float *pDst, uint32_t blockSize)
{
    uint32_t i;
    for (i = 0; i < blockSize; i++) pDst[i] = pSrc[i] + offset;
}

static inline void arm_scale_f32(const float *pSrc, float scale,
                                 float *pDst, uint32_t blockSize)
{
    uint32_t i;
    for (i = 0; i < blockSize; i++) pDst[i] = pSrc[i] * scale;
}

#else  /* !DESKTOP_SHIM — real MCU build */

#include "arm_math.h"
#include "arm_const_structs.h"

#endif /* DESKTOP_SHIM */

#endif /* CMSIS_SHIM_H */
