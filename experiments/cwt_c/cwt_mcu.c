/*
 * cwt_mcu.c — Memory-optimised CWT scalogram for MCU deployment.
 *
 * Uses CMSIS-DSP functions throughout (biquad HPF, IFFT, complex dot product,
 * vector min/max/offset/scale).  On desktop, cmsis_shim.h provides equivalent
 * C implementations wrapping kiss_fft.  On MCU, real CMSIS-DSP runs natively.
 *
 * Key difference from cwt_preprocess.c (FFT-based):
 *   Instead of FFT-ing the full signal (up to 128K points), we compute the
 *   wavelet kernel via a small 2048-point IFFT and convolve directly at only
 *   the 64 needed output time bins.  This eliminates all large FFT buffers.
 *
 * Memory budget (worst case 99K samples):
 *   Signal buffer:  396 KB  (one channel, float32)
 *   Output buffer:   80 KB  (5 x 64 x 64, float32) — caller-owned
 *   Workspace:       16 KB  (2048 complex = 4096 floats)
 *   HPF extension:  ~120 B  (30 floats for odd-extension padding)
 *   Stack/overhead:  ~16 KB
 *   Peak:           ~508 KB — fits in 512 KB SRAM
 */

#include "cwt_mcu.h"
#include "cmsis_shim.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ── Physical / wavelet constants ──────────────────────────────────────────── */

#define CWT_BW   1.5f
#define CWT_FC   1.0f
#define PI_F     3.14159265358979323846f

/*
 * Noise-floor gate: after computing all CWT power values for a channel, any
 * value below NOISE_GATE_REL * max_power is set to zero BEFORE min-max
 * normalisation.  This prevents normalisation from amplifying meaningless
 * noise-floor differences between float32/float64 or FFT/direct CWT
 * discretisations into large normalised differences.
 *
 * 1e-3 means power values below 0.1% of the channel's peak are zeroed.
 */
#define NOISE_GATE_REL  1e-3f

/* Force HPF channel indices */
#define HPF_CH_MIN 2
#define HPF_CH_MAX 4

/* HPF padding: 3 * (2*N_SOS_SECTIONS + 1) = 15  (scipy 1.17+) */
#define N_SOS_SECTIONS 2
#define HPF_PADLEN     15

/* ── HPF SOS coefficients (CMSIS layout: {b0, b1, b2, -a1, -a2} per stage) ─
 *
 * From scipy: butter(4, 15.0, 'high', fs=1625.0, output='sos')
 * CMSIS negates a1,a2 compared to scipy convention.
 * Float32 — Cortex-M33 has no hardware double.
 */
static const float HPF_COEFFS[N_SOS_SECTIONS * 5] = {
    /* Section 0: {b0, b1, b2, -a1, -a2} */
     9.27009640e-01f, -1.85401928e+00f,  9.27009640e-01f,
     1.89514504e+00f, -8.98337002e-01f,
    /* Section 1 */
     1.00000000e+00f, -2.00000000e+00f,  1.00000000e+00f,
     1.95330752e+00f, -9.56597435e-01f,
};

/*
 * sosfilt_zi initial conditions (from scipy.signal.sosfilt_zi(HPF_SOS)).
 * Section 1 ICs are zero because the HPF blocks DC.
 * Float32 for MCU.
 */
static const float HPF_SOS_ZI[N_SOS_SECTIONS][2] = {
    { -9.27009640e-01f,  9.27009640e-01f },
    {  0.00000000e+00f,  0.00000000e+00f },
};

/* ── 64 log-spaced scales: numpy.geomspace(1.0, 64.0, num=64) ─────────────── */
static const float SCALES[CWT_MCU_N_SCALES] = {
    1.00000000e+00f, 1.06824169e+00f, 1.14114031e+00f, 1.21901365e+00f,
    1.30220121e+00f, 1.39106562e+00f, 1.48599429e+00f, 1.58740105e+00f,
    1.69572798e+00f, 1.81144733e+00f, 1.93506356e+00f, 2.06711557e+00f,
    2.20817903e+00f, 2.35886890e+00f, 2.51984210e+00f, 2.69180039e+00f,
    2.87549339e+00f, 3.07172193e+00f, 3.28134142e+00f, 3.50526571e+00f,
    3.74447097e+00f, 4.00000000e+00f, 4.27296676e+00f, 4.56456124e+00f,
    4.87605462e+00f, 5.20880483e+00f, 5.56426248e+00f, 5.94397716e+00f,
    6.34960421e+00f, 6.78291194e+00f, 7.24578931e+00f, 7.74025423e+00f,
    8.26846226e+00f, 8.83271611e+00f, 9.43547559e+00f, 1.00793684e+01f,
    1.07672015e+01f, 1.15019736e+01f, 1.22868877e+01f, 1.31253657e+01f,
    1.40210628e+01f, 1.49778839e+01f, 1.60000000e+01f, 1.70918671e+01f,
    1.82582450e+01f, 1.95042185e+01f, 2.08352193e+01f, 2.22570499e+01f,
    2.37759086e+01f, 2.53984168e+01f, 2.71316477e+01f, 2.89831573e+01f,
    3.09610169e+01f, 3.30738491e+01f, 3.53308644e+01f, 3.77419024e+01f,
    4.03174736e+01f, 4.30688062e+01f, 4.60078943e+01f, 4.91475508e+01f,
    5.25014628e+01f, 5.60842514e+01f, 5.99115355e+01f, 6.40000000e+01f,
};

/* ── Internal helpers ──────────────────────────────────────────────────────── */

/*
 * reverse_f32 — reverse a float array in-place.
 */
static void reverse_f32(float *buf, int n)
{
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        float tmp = buf[lo];
        buf[lo]   = buf[hi];
        buf[hi]   = tmp;
        lo++; hi--;
    }
}

/*
 * highpass_force_f32 — zero-phase HPF via CMSIS biquad cascade.
 *
 * Replicates scipy.signal.sosfiltfilt:
 *   1. Odd-extension padding (padlen=15 each side)
 *   2. Forward biquad cascade with sosfilt_zi ICs
 *   3. Reverse + backward biquad cascade with ICs from last sample
 *   4. Reverse back, extract unpadded signal
 *
 * All float32 arithmetic — suitable for Cortex-M33 SP FPU.
 *
 * signal:   float32[n], modified in-place with filtered result.
 * n:        signal length (must be > HPF_PADLEN).
 * ext_buf:  caller-allocated float32[n + 2*HPF_PADLEN] temp buffer.
 */
static void highpass_force_f32(float *signal, int n, float *ext_buf)
{
    const int padlen = HPF_PADLEN;
    const int n_ext  = n + 2 * padlen;
    int i, s;

    arm_biquad_cascade_df2T_instance_f32 hpf_inst;
    float hpf_state[2 * N_SOS_SECTIONS];

    /* ── Build odd extension ──────────────────────────────────────────────── */
    for (i = 0; i < padlen; i++)
        ext_buf[i] = 2.0f * signal[0] - signal[padlen - i];

    memcpy(ext_buf + padlen, signal, (size_t)n * sizeof(float));

    for (i = 0; i < padlen; i++)
        ext_buf[n + padlen + i] = 2.0f * signal[n - 1] - signal[n - 2 - i];

    /* ── Forward pass ─────────────────────────────────────────────────────── */
    {
        float x0 = ext_buf[0];
        for (s = 0; s < N_SOS_SECTIONS; s++) {
            hpf_state[s * 2 + 0] = HPF_SOS_ZI[s][0] * x0;
            hpf_state[s * 2 + 1] = HPF_SOS_ZI[s][1] * x0;
        }
    }
    arm_biquad_cascade_df2T_init_f32(&hpf_inst, N_SOS_SECTIONS,
                                     HPF_COEFFS, hpf_state);
    arm_biquad_cascade_df2T_f32(&hpf_inst, ext_buf, ext_buf, (uint32_t)n_ext);

    /* ── Backward pass ────────────────────────────────────────────────────── */
    {
        float y_last = ext_buf[n_ext - 1];
        reverse_f32(ext_buf, n_ext);
        for (s = 0; s < N_SOS_SECTIONS; s++) {
            hpf_state[s * 2 + 0] = HPF_SOS_ZI[s][0] * y_last;
            hpf_state[s * 2 + 1] = HPF_SOS_ZI[s][1] * y_last;
        }
    }
    arm_biquad_cascade_df2T_init_f32(&hpf_inst, N_SOS_SECTIONS,
                                     HPF_COEFFS, hpf_state);
    arm_biquad_cascade_df2T_f32(&hpf_inst, ext_buf, ext_buf, (uint32_t)n_ext);
    reverse_f32(ext_buf, n_ext);

    /* ── Extract unpadded result ──────────────────────────────────────────── */
    memcpy(signal, ext_buf + padlen, (size_t)n * sizeof(float));
}

/*
 * compute_kernel — build the time-domain wavelet kernel for a given scale
 * using a CWT_MCU_N_KER-point IFFT.
 *
 * 1. Fill freq-domain Morlet: Psi[k] = sqrt(s) * exp(-Bw*pi^2*(s*k/N_ker - Fc)^2)
 *    as interleaved complex {re, 0} (wavelet is real in freq domain).
 * 2. Zero negative frequencies (k > N_ker/2).
 * 3. IFFT via arm_cfft_f32 (does NOT divide by N — we handle scaling in convolution).
 * 4. Extract truncated kernel: kernel[m] = psi_t[(-m) mod N_ker] for m in [-half_M, +half_M].
 *    6-sigma truncation: half_M = ceil(6 * s * sqrt(Bw/2)).
 *
 * workspace:  float32[2 * N_ker] — used for IFFT, then kernel stored at start.
 * kernel_out: set to point into workspace where kernel starts (interleaved complex).
 * kernel_len: set to 2*half_M + 1.
 *
 * Returns half_M.
 */
static int compute_kernel(float scale, float *workspace,
                          const arm_cfft_instance_f32 *cfft_inst,
                          float **kernel_out, int *kernel_len)
{
    const int N_ker = CWT_MCU_N_KER;
    const float inv_nk = 1.0f / (float)N_ker;
    const float sqrt_s = sqrtf(scale);
    int k;

    /* Fill frequency-domain Morlet wavelet (interleaved complex) */
    for (k = 0; k <= N_ker / 2; k++) {
        float freq = scale * (float)k * inv_nk;
        float arg  = freq - CWT_FC;
        float psi  = sqrt_s * expf(-CWT_BW * PI_F * PI_F * arg * arg);
        workspace[2 * k]     = psi;   /* real */
        workspace[2 * k + 1] = 0.0f;  /* imag */
    }
    /* Zero negative frequencies */
    for (k = N_ker / 2 + 1; k < N_ker; k++) {
        workspace[2 * k]     = 0.0f;
        workspace[2 * k + 1] = 0.0f;
    }

    /* IFFT in-place (CMSIS convention: no 1/N scaling) */
    arm_cfft_f32(cfft_inst, workspace, 1 /* ifft */, 1 /* bitrev */);

    /* Determine kernel half-width: 6-sigma truncation */
    int half_M = (int)ceilf(6.0f * scale * sqrtf(CWT_BW / 2.0f));
    if (half_M > N_ker / 2 - 1)
        half_M = N_ker / 2 - 1;

    int M = 2 * half_M + 1;

    /* Extract time-domain kernel: psi_t[(-m) mod N_ker]
     * We need m = -half_M .. +half_M.
     * psi_t[(-m) mod N_ker] = psi_t[(m) mod N_ker] for time reversal.
     * Actually: kernel[m + half_M] = workspace[m mod N_ker] (complex).
     *
     * For the convolution W(tau) = sum_m signal[tau-m] * kernel[m],
     * the kernel sample at offset m corresponds to IFFT bin (m mod N_ker).
     *
     * We'll store the kernel contiguously after the workspace IFFT area.
     * But workspace IS the IFFT area — so we extract into a separate region.
     * Since M_max = 2*667+1 = 1335 < 2048, the kernel fits in 1335*2 = 2670 floats.
     * We use the second half of the workspace (floats [2048..4095]) as kernel storage.
     */
    /* Extract kernel from IFFT output into a temp buffer, then copy back
     * to workspace start. We can't extract in-place because source and
     * destination ranges overlap for small m values. */
    float *kern_buf = (float *)malloc((size_t)(2 * M) * sizeof(float));
    if (!kern_buf) {
        *kernel_out = NULL;
        *kernel_len = 0;
        return -1;
    }

    {
        int j;
        for (j = 0; j < M; j++) {
            int m = -half_M + j;
            int bin = ((m % N_ker) + N_ker) % N_ker;
            kern_buf[2 * j]     = workspace[2 * bin]     * inv_nk;
            kern_buf[2 * j + 1] = workspace[2 * bin + 1] * inv_nk;
        }
    }

    /* Copy kernel back to start of workspace */
    memcpy(workspace, kern_buf, (size_t)(2 * M) * sizeof(float));
    free(kern_buf);

    *kernel_out = workspace;
    *kernel_len = M;
    return half_M;
}

/* ── Public API ────────────────────────────────────────────────────────────── */

int cwt_mcu_process_channel(
    float       *signal,
    int          n_samples,
    int          ch_idx,
    float       *out,
    float       *workspace)
{
    int si, ti;
    const int N = n_samples;

    if (N < CWT_MCU_N_TIME)
        return -1;

    /* ── Step 1: HPF for force channels ───────────────────────────────────── */
    if (ch_idx >= HPF_CH_MIN && ch_idx <= HPF_CH_MAX) {
        const int n_ext = N + 2 * HPF_PADLEN;
        float *ext_buf = (float *)malloc((size_t)n_ext * sizeof(float));
        if (!ext_buf) return -1;
        highpass_force_f32(signal, N, ext_buf);
        free(ext_buf);
    }

    /* ── Step 2: Compute subsampling indices ──────────────────────────────── */
    int sub_idx[CWT_MCU_N_TIME];
    for (ti = 0; ti < CWT_MCU_N_TIME; ti++) {
        sub_idx[ti] = (int)((float)ti * (float)(N - 1) / (float)(CWT_MCU_N_TIME - 1));
    }

    /* ── Step 3: Initialise IFFT instance ─────────────────────────────────── */
#ifdef DESKTOP_SHIM
    arm_cfft_instance_f32 cfft_inst;
    cmsis_shim_cfft_init(&cfft_inst, CWT_MCU_N_KER);
#else
    /* On MCU: use the pre-built const twiddle table for 2048-point FFT */
    const arm_cfft_instance_f32 *cfft_inst_ptr = &arm_cfft_sR_f32_len2048;
    #define cfft_inst (*cfft_inst_ptr)
#endif

    /* ── Step 4: Per-scale CWT via direct convolution ─────────────────────── */

    /* Allocate signal-as-complex buffer for arm_cmplx_dot_prod_f32.
     * For each time bin, we need a segment of the signal as interleaved
     * complex {x[j], 0, x[j+1], 0, ...}.  We allocate for the largest
     * possible kernel length to avoid per-bin allocation.
     *
     * Max kernel length: 2*ceil(6*64*sqrt(1.5/2))+1 = 2*667+1 = 1335
     * So we need 1335 * 2 * 4 = 10.7 KB for the complex signal segment.
     */
    int max_half_M = (int)ceilf(6.0f * SCALES[CWT_MCU_N_SCALES - 1] * sqrtf(CWT_BW / 2.0f));
    if (max_half_M > CWT_MCU_N_KER / 2 - 1)
        max_half_M = CWT_MCU_N_KER / 2 - 1;
    int max_M = 2 * max_half_M + 1;

    float *sig_cpx = (float *)malloc((size_t)(2 * max_M) * sizeof(float));
    if (!sig_cpx) {
#ifdef DESKTOP_SHIM
        cmsis_shim_cfft_free(&cfft_inst);
#endif
        return -1;
    }

    for (si = 0; si < CWT_MCU_N_SCALES; si++) {
        float scale = SCALES[si];
        float *kernel;
        int kernel_len;
        int half_M;

        half_M = compute_kernel(scale, workspace, &cfft_inst, &kernel, &kernel_len);
        if (half_M < 0) {
            free(sig_cpx);
#ifdef DESKTOP_SHIM
            cmsis_shim_cfft_free(&cfft_inst);
#endif
            return -1;
        }

        /* For each output time bin, direct convolution */
        for (ti = 0; ti < CWT_MCU_N_TIME; ti++) {
            int tau = sub_idx[ti];
            int m;
            float wr, wi;

            /* Build complex signal segment: {x[tau + m - half_M], 0, ...}
             * with zero-padding at boundaries */
            for (m = 0; m < kernel_len; m++) {
                int idx = tau + m - half_M;
                if (idx >= 0 && idx < N) {
                    sig_cpx[2 * m]     = signal[idx];
                } else {
                    sig_cpx[2 * m]     = 0.0f;
                }
                sig_cpx[2 * m + 1] = 0.0f;  /* imaginary = 0 */
            }

            arm_cmplx_dot_prod_f32(sig_cpx, kernel,
                                   (uint32_t)kernel_len, &wr, &wi);

            out[si * CWT_MCU_N_TIME + ti] = wr * wr + wi * wi;
        }
    }

    free(sig_cpx);

#ifdef DESKTOP_SHIM
    cmsis_shim_cfft_free(&cfft_inst);
#endif

    /* ── Step 5: Noise-floor gate + per-channel min-max normalisation ────── */
    /*
     * Zero any CWT power value below NOISE_GATE_REL * max_power.  This
     * eliminates noise-floor regions that would otherwise be amplified
     * by normalisation into meaningless large values.  Applied identically
     * in the desktop FFT version (cwt_preprocess.c) and the Python reference,
     * making all three produce identical output in these regions.
     */
    {
        float lo, hi, gate;
        uint32_t idx_dummy;
        int i;

        /* Find max power (= hi, since power >= 0) */
        arm_max_f32(out, CWT_MCU_CH_SIZE, &hi, &idx_dummy);
        gate = NOISE_GATE_REL * hi;

        /* Zero values below gate */
        for (i = 0; i < CWT_MCU_CH_SIZE; i++) {
            if (out[i] < gate)
                out[i] = 0.0f;
        }

        /* Min-max normalise */
        arm_min_f32(out, CWT_MCU_CH_SIZE, &lo, &idx_dummy);
        if (hi > lo) {
            arm_offset_f32(out, -lo, out, CWT_MCU_CH_SIZE);
            arm_scale_f32(out, 1.0f / (hi - lo), out, CWT_MCU_CH_SIZE);
        }
    }

    return 0;
}
