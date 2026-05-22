/*
 * cwt_preprocess.c
 *
 * CWT scalogram preprocessing — portable C99 implementation.
 * Uses kiss_fft for frequency-domain convolution.
 *
 * Pipeline (matches Python reference in precompute_scalograms.py):
 *   1. Zero-phase 4th-order Butterworth HPF at 15 Hz on force channels (2,3,4)
 *   2. Sequential complex Morlet CWT (cmor1.5-1.0) over 64 log-spaced scales
 *   3. Subsample each scale's power row to 64 time bins
 *   4. Per-channel min-max normalisation (each 64×64 slice independently → [0,1])
 *
 * Normalisation note:
 *   The header comment says "global" normalisation — that is INCORRECT.
 *   The existing .pt reference files use per-channel normalisation (each of the
 *   5 channels is independently scaled to [0,1]).  This implementation matches
 *   Python's compute_scalogram_sequential().
 */

#include "cwt_preprocess.h"
#include "kiss_fft/kiss_fft.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ── Physical constants ─────────────────────────────────────────────────────── */

#define CWT_BW   1.5   /* cmor bandwidth parameter */
#define CWT_FC   1.0   /* cmor centre frequency in cycles/sample (pywt convention) */
/* Note: CWT_FC = 1.0 means the wavelet is centred at 1 cycle/sample = Nyquist.
 * Scale s maps the centre to 1/s cycles/sample = FS/s Hz.
 * Scales 1–64 therefore span FS/64 (≈25 Hz) to FS/1 (>Nyquist, ~0 power). */

/*
 * sosfilt_zi — initial conditions for a unit-step input.
 * Precomputed from Python: scipy.signal.sosfilt_zi(HPF_SOS)
 * shape: [N_SOS_SECTIONS][2]
 * Usage: initial state for section s = HPF_SOS_ZI[s][j] * x[0]
 */
static const double HPF_SOS_ZI[2][2] = {
    { -9.27009639993301519e-01,  9.27009639993304669e-01 },  /* section 0 */
    {  0.00000000000000000e+00,  0.00000000000000000e+00 },  /* section 1 */
};

/* ── Force HPF channel indices (fx=2, fy=3, fz=4) ──────────────────────────── */
/* channels 0 (acc) and 1 (acoustic) are NOT filtered */
#define HPF_CH_MIN 2
#define HPF_CH_MAX 4

/* ── 4th-order Butterworth HPF SOS coefficients ─────────────────────────────
 * scipy.signal.butter(4, 15.0, 'high', fs=1625.0, output='sos')
 * Format per row: [b0, b1, b2, a0(=1.0), a1, a2]
 * Double precision to match scipy's float64 biquad arithmetic.
 */
static const double HPF_SOS[2][6] = {
    { 9.27009639993332413e-01, -1.85401927998666483e+00,  9.27009639993332413e-01,
      1.00000000000000000e+00, -1.89514504496437919e+00,  8.98337002438857835e-01 },
    { 1.00000000000000000e+00, -2.00000000000000000e+00,  1.00000000000000000e+00,
      1.00000000000000000e+00, -1.95330751594092811e+00,  9.56597435381147054e-01 },
};
#define N_SOS_SECTIONS 2

/* ── 64 log-spaced scales: numpy.geomspace(1.0, 64.0, num=64) ──────────────── */
static const double SCALES[CWT_N_SCALES] = {
    1.00000000000000000e+00, 1.06824169081440212e+00, 1.14114030999401295e+00, 1.21901365420447538e+00,
    1.30220120709323184e+00, 1.39106561924582950e+00, 1.48599428913694842e+00, 1.58740105196819936e+00,
    1.69572798375507006e+00, 1.81144732852781321e+00, 1.93506355704778321e+00, 2.06711556601405544e+00,
    2.20817902734762450e+00, 2.35886889779472853e+00, 2.51984209978974638e+00, 2.69180038526471233e+00,
    2.87549339489003541e+00, 3.07172192608297712e+00, 3.28134142403055140e+00, 3.50526571094573480e+00,
    3.74447096981441918e+00, 4.00000000000000000e+00, 4.27296676325760849e+00, 4.56456123997605179e+00,
    4.87605461681790153e+00, 5.20880482837292735e+00, 5.56426247698331800e+00, 5.94397715654779368e+00,
    6.34960420787279745e+00, 6.78291193502028111e+00, 7.24578931411125282e+00, 7.74025422819113373e+00,
    8.26846226405622176e+00, 8.83271610939049800e+00, 9.43547559117891410e+00, 1.00793683991589855e+01,
    1.07672015410588457e+01, 1.15019735795601399e+01, 1.22868877043319085e+01, 1.31253656961222092e+01,
    1.40210628437829357e+01, 1.49778838792576767e+01, 1.60000000000000000e+01, 1.70918670530304375e+01,
    1.82582449599042036e+01, 1.95042184672716061e+01, 2.08352193134917094e+01, 2.22570499079332755e+01,
    2.37759086261911712e+01, 2.53984168314911898e+01, 2.71316477400811245e+01, 2.89831572564450042e+01,
    3.09610169127645278e+01, 3.30738490562248870e+01, 3.53308644375619991e+01, 3.77419023647156493e+01,
    4.03174735966359350e+01, 4.30688061642353972e+01, 4.60078943182405737e+01, 4.91475508173276197e+01,
    5.25014627844888224e+01, 5.60842513751317568e+01, 5.99115355170307211e+01, 6.40000000000000000e+01,
};

/* ── Internal helpers ───────────────────────────────────────────────────────── */

/*
 * sosfilt_inplace — single forward pass of all SOS sections (in-place).
 *
 * Implements Transposed Direct Form II, matching scipy.signal.sosfilt:
 *
 *   y[n] = b0*x[n] + z1
 *   z1'  = b1*x[n] - a1*y[n] + z2
 *   z2'  = b2*x[n] - a2*y[n]
 *
 * Processes all N_SOS_SECTIONS sections sequentially; output of each section
 * feeds into the next (standard SOS cascade).
 *
 * buf    : double array of length n (modified in-place)
 * z1_init: initial z1 state per section [N_SOS_SECTIONS]  (may be NULL → zeros)
 * z2_init: initial z2 state per section [N_SOS_SECTIONS]  (may be NULL → zeros)
 *
 * Using HPF_SOS_ZI[s][j] * x0 as initial conditions matches scipy's sosfiltfilt
 * steady-state initialisation, eliminating the large startup transient that
 * arises from zero initial conditions (section 0 has DC gain ≈ 0.003, so the
 * error is amplified ~300× with zero ICs).
 */
static void sosfilt_inplace(double *buf, int n,
                             const double *z1_init,
                             const double *z2_init)
{
    int s;
    for (s = 0; s < N_SOS_SECTIONS; s++) {
        const double b0 = HPF_SOS[s][0];
        const double b1 = HPF_SOS[s][1];
        const double b2 = HPF_SOS[s][2];
        /* HPF_SOS[s][3] == 1.0 always */
        const double a1 = HPF_SOS[s][4];
        const double a2 = HPF_SOS[s][5];
        double z1 = z1_init ? z1_init[s] : 0.0;
        double z2 = z2_init ? z2_init[s] : 0.0;
        int i;
        for (i = 0; i < n; i++) {
            double x = buf[i];
            double y = b0 * x + z1;
            z1 = b1 * x - a1 * y + z2;
            z2 = b2 * x - a2 * y;
            buf[i] = y;
        }
    }
}

/*
 * reverse_buf — reverse a double array in-place.
 */
static void reverse_buf(double *buf, int n)
{
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        double tmp = buf[lo];
        buf[lo]    = buf[hi];
        buf[hi]    = tmp;
        lo++;
        hi--;
    }
}

/*
 * highpass_force — zero-phase HPF via two-pass SOS (forward + reverse).
 *
 * Replicates scipy.signal.sosfiltfilt including the odd-extension padding:
 *
 *   padlen = 3 * (2 * N_SOS_SECTIONS + 1) = 15 samples on each side
 *   (matches scipy ≥1.17 formula: 3 * ntaps where ntaps = 2*n_sections+1).
 *
 *   1. Build odd-extension:
 *        prepend : ext[i] = 2*x[0] - x[padlen - i]   for i = 0..padlen-1
 *        signal  : ext[padlen + i] = x[i]              for i = 0..n-1
 *        append  : ext[n+padlen+i] = 2*x[n-1] - x[n-2-i]  for i = 0..padlen-1
 *   2. Forward sosfilt on ext with IC = HPF_SOS_ZI[s] * ext[0]
 *   3. Reverse ext
 *   4. Backward sosfilt on ext with IC = HPF_SOS_ZI[s] * ext[0] (= forward's last)
 *   5. Reverse ext back
 *   6. Extract out[0..n-1] = ext[padlen..padlen+n-1]
 *
 * signal : float input,  length n  (n must be >= padlen + 1 = 10)
 * out    : double output, length n (caller-allocated)
 */
#define HPF_PADLEN  15  /* 3 * (2 * N_SOS_SECTIONS + 1) — matches scipy 1.17 */

static void highpass_force(const float *signal, int n, double *out)
{
    int i, s;
    double z1_init[N_SOS_SECTIONS], z2_init[N_SOS_SECTIONS];
    const int padlen = HPF_PADLEN;
    const int n_ext  = n + 2 * padlen;
    double *ext;

    ext = (double *)malloc((size_t)n_ext * sizeof(double));
    if (!ext) {
        /* Fallback: filter without padding (lossy but non-fatal) */
        for (i = 0; i < n; i++) out[i] = (double)signal[i];
        sosfilt_inplace(out, n, NULL, NULL);
        reverse_buf(out, n);
        sosfilt_inplace(out, n, NULL, NULL);
        reverse_buf(out, n);
        return;
    }

    /* ── Build odd extension ──────────────────────────────────────────────── */
    /* Prepend: reflect x[1..padlen] around x[0] */
    for (i = 0; i < padlen; i++)
        ext[i] = 2.0 * (double)signal[0] - (double)signal[padlen - i];

    /* Original signal */
    for (i = 0; i < n; i++)
        ext[padlen + i] = (double)signal[i];

    /* Append: reflect x[n-2..n-2-padlen+1] around x[n-1] */
    for (i = 0; i < padlen; i++)
        ext[n + padlen + i] = 2.0 * (double)signal[n - 1] - (double)signal[n - 2 - i];

    /* ── Forward pass ─────────────────────────────────────────────────────── */
    {
        double x0 = ext[0];
        for (s = 0; s < N_SOS_SECTIONS; s++) {
            z1_init[s] = HPF_SOS_ZI[s][0] * x0;
            z2_init[s] = HPF_SOS_ZI[s][1] * x0;
        }
    }
    sosfilt_inplace(ext, n_ext, z1_init, z2_init);

    /* ── Backward pass ────────────────────────────────────────────────────── */
    {
        double y_last = ext[n_ext - 1];
        reverse_buf(ext, n_ext);
        for (s = 0; s < N_SOS_SECTIONS; s++) {
            z1_init[s] = HPF_SOS_ZI[s][0] * y_last;
            z2_init[s] = HPF_SOS_ZI[s][1] * y_last;
        }
    }
    sosfilt_inplace(ext, n_ext, z1_init, z2_init);
    reverse_buf(ext, n_ext);

    /* ── Extract unpadded result ──────────────────────────────────────────── */
    for (i = 0; i < n; i++)
        out[i] = ext[padlen + i];

    free(ext);
}

/* ── next_pow2 ──────────────────────────────────────────────────────────────── */

/*
 * Return the smallest power of 2 >= n.
 * Zero-padding the signal to this length before FFT avoids large-prime-factor
 * slowdowns in kiss_fft (an N with a large prime factor is O(N²) for that factor).
 */
static int next_pow2(int n)
{
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

/*
 * cwt_channel — compute CWT scalogram for one signal channel.
 *
 * Key design choices:
 *   • Zero-pads signal to N_pad = next_pow2(n_samples) for fast FFT.
 *   • Normalized frequency: f_k = k / N_pad  (cycles/sample, pywt convention).
 *     pywt does NOT use Hz; it uses cycles per sample where Fc=1.0 ≈ Nyquist.
 *   • Negative frequency bins (k > N_pad/2) zeroed → analytic (cmor) wavelet.
 *   • Subsampling indices drawn from [0, n_samples-1] (original signal range),
 *     matching np.linspace(0, n_samples-1, 64, dtype=int) with truncation.
 *   • Per-channel min-max normalization after all 64 scales.
 *
 * signal_d  : double array, length n_samples
 * n_samples : number of signal samples (before padding)
 * row_out   : float output [CWT_N_SCALES × CWT_N_TIME] (caller-allocated)
 *
 * Returns 0 on success, -1 on allocation failure.
 */
static int cwt_channel(const double *signal_d, int n_samples, float *row_out)
{
    const int    N_sig = n_samples;
    const int    N_pad = next_pow2(n_samples);  /* zero-padded FFT length */
    const double PI    = M_PI;
    const double invNp = 1.0 / (double)N_pad;
    int si, k, ti, ret = 0;

    kiss_fft_cpx *fft_in   = (kiss_fft_cpx *)malloc((size_t)N_pad * sizeof(kiss_fft_cpx));
    kiss_fft_cpx *fft_sig  = (kiss_fft_cpx *)malloc((size_t)N_pad * sizeof(kiss_fft_cpx));
    kiss_fft_cpx *prod     = (kiss_fft_cpx *)malloc((size_t)N_pad * sizeof(kiss_fft_cpx));
    kiss_fft_cpx *ifft_out = (kiss_fft_cpx *)malloc((size_t)N_pad * sizeof(kiss_fft_cpx));

    if (!fft_in || !fft_sig || !prod || !ifft_out) {
        ret = -1;
        goto cleanup_bufs;
    }

    {
        kiss_fft_cfg fwd_cfg = kiss_fft_alloc(N_pad, 0, NULL, NULL);
        kiss_fft_cfg inv_cfg = kiss_fft_alloc(N_pad, 1, NULL, NULL);

        if (!fwd_cfg || !inv_cfg) {
            ret = -1;
            kiss_fft_free(fwd_cfg);
            kiss_fft_free(inv_cfg);
            goto cleanup_bufs;
        }

        /* Signal → complex, zero-padded */
        for (k = 0; k < N_sig; k++) {
            fft_in[k].r = (kiss_fft_scalar)signal_d[k];
            fft_in[k].i = 0.0f;
        }
        for (k = N_sig; k < N_pad; k++) {
            fft_in[k].r = 0.0f;
            fft_in[k].i = 0.0f;
        }
        kiss_fft(fwd_cfg, fft_in, fft_sig);

        /* ── Per-scale CWT ──────────────────────────────────────────────── */
        for (si = 0; si < CWT_N_SCALES; si++) {
            const double s      = SCALES[si];
            const double sqrt_s = sqrt(s);
            const int    nyq    = N_pad / 2;

            for (k = 0; k < N_pad; k++) {
                if (k > nyq) {
                    /* Negative frequency → zero (analytic wavelet) */
                    prod[k].r = 0.0f;
                    prod[k].i = 0.0f;
                } else {
                    /* Normalized linear frequency: f = k / N_pad (cycles/sample).
                     * Matches pywt which uses fftfreq(N) = k/N, with Fc in
                     * cycles/sample units (cmor1.5-1.0 → Fc=1.0 cyc/sample). */
                    double freq = (double)k * invNp;
                    double arg  = s * freq - CWT_FC;
                    double psi  = sqrt_s * exp(-CWT_BW * PI * PI * arg * arg);
                    prod[k].r = (kiss_fft_scalar)((double)fft_sig[k].r * psi);
                    prod[k].i = (kiss_fft_scalar)((double)fft_sig[k].i * psi);
                }
            }

            kiss_fft(inv_cfg, prod, ifft_out);

            /* Power + subsample — indices from [0, N_sig-1] only */
            {
                float *row = row_out + (size_t)si * CWT_N_TIME;
                for (ti = 0; ti < CWT_N_TIME; ti++) {
                    /* np.linspace(0, N_sig-1, CWT_N_TIME, dtype=int) — truncation */
                    int idx = (int)((double)ti * (double)(N_sig - 1)
                                    / (double)(CWT_N_TIME - 1));
                    double wr = (double)ifft_out[idx].r * invNp;
                    double wi = (double)ifft_out[idx].i * invNp;
                    row[ti]   = (float)(wr * wr + wi * wi);
                }
            }
        }

        kiss_fft_free(fwd_cfg);
        kiss_fft_free(inv_cfg);
    }

    /* ── Noise-floor gate + per-channel min-max normalisation ────────────── */
    /*
     * Zero any CWT power value below 1e-3 * max_power.  This eliminates
     * noise-floor regions that would otherwise be amplified by normalisation.
     * Applied identically in the MCU version and Python reference.
     */
    {
        int total = CWT_N_SCALES * CWT_N_TIME;
        float lo =  FLT_MAX;
        float hi = -FLT_MAX;
        float gate;
        int i;
        /* Find max first */
        for (i = 0; i < total; i++) {
            if (row_out[i] > hi) hi = row_out[i];
        }
        gate = 1e-3f * hi;
        /* Zero values below gate, then find min */
        for (i = 0; i < total; i++) {
            if (row_out[i] < gate) row_out[i] = 0.0f;
            if (row_out[i] < lo) lo = row_out[i];
        }
        if (hi > lo) {
            float inv_range = 1.0f / (hi - lo);
            for (i = 0; i < total; i++)
                row_out[i] = (row_out[i] - lo) * inv_range;
        }
    }

cleanup_bufs:
    free(fft_in);
    free(fft_sig);
    free(prod);
    free(ifft_out);
    return ret;
}

/* ── Public API ─────────────────────────────────────────────────────────────── */

int cwt_compute_scalogram(
    const float *signal,
    int          n_channels,
    int          n_samples,
    float       *out
)
{
    double *sig_d;
    int ch, ret = 0;

    if (n_channels != CWT_N_CHANNELS)
        return -1;

    sig_d = (double *)malloc((size_t)n_samples * sizeof(double));
    if (!sig_d)
        return -1;

    for (ch = 0; ch < CWT_N_CHANNELS && ret == 0; ch++) {
        const float *ch_in  = signal + (size_t)ch * n_samples;
        float       *ch_out = out    + (size_t)ch * CWT_N_SCALES * CWT_N_TIME;

        if (ch >= HPF_CH_MIN && ch <= HPF_CH_MAX) {
            /* Force channel: apply zero-phase HPF before CWT */
            highpass_force(ch_in, n_samples, sig_d);
        } else {
            /* Acc / acoustic: no filter */
            int i;
            for (i = 0; i < n_samples; i++)
                sig_d[i] = (double)ch_in[i];
        }

        ret = cwt_channel(sig_d, n_samples, ch_out);
    }

    free(sig_d);
    return ret;
}
