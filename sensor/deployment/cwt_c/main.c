/*
 * main.c — CLI wrapper for cwt_compute_scalogram()
 *
 * Usage:
 *   ./cwt_preprocess <sensor.csv> <output.bin>
 *
 * Input CSV:
 *   - 5 columns: acc, acoustic, fx, fy, fz
 *   - No header row
 *   - One sample per row (comma-separated floats)
 *   - Must contain only the CUTTING SEGMENT (post aircut-gating)
 *
 * Output binary:
 *   - 5 × 64 × 64 × 4 bytes = 81,920 bytes
 *   - Row-major float32: out[ch][scale][time_bin]
 *   - Values in [0, 1]
 *
 * Example (load in Python):
 *   import numpy as np
 *   t = np.fromfile('output.bin', dtype=np.float32).reshape(5, 64, 64)
 */

#include "cwt_preprocess.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_COLS 8

/* ── CSV loader ─────────────────────────────────────────────────────────────── */

/*
 * read_csv_5col — read a no-header 5-column CSV into per-channel float arrays.
 *
 * Allocates a single block: *data = malloc(CWT_N_CHANNELS * n_rows * sizeof(float))
 * The caller must free(*data).
 *
 * Returns the number of rows read on success, 0 on failure.
 */
static int read_csv_5col(const char *path, float **data)
{
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Error: cannot open '%s'\n", path);
        return 0;
    }

    /* First pass: count rows */
    int n_rows = 0;
    char line[4096];
    while (fgets(line, sizeof(line), fp)) {
        /* Skip blank lines */
        const char *p = line;
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
        if (*p == '\0') continue;
        n_rows++;
    }

    if (n_rows == 0) {
        fprintf(stderr, "Error: no data rows in '%s'\n", path);
        fclose(fp);
        return 0;
    }

    /* Allocate row-major block: [channel][sample] */
    *data = (float *)malloc((size_t)CWT_N_CHANNELS * (size_t)n_rows * sizeof(float));
    if (!*data) {
        fprintf(stderr, "Error: out of memory\n");
        fclose(fp);
        return 0;
    }

    /* Second pass: read values */
    rewind(fp);
    int row = 0;
    while (fgets(line, sizeof(line), fp) && row < n_rows) {
        /* Skip blank lines */
        const char *p = line;
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
        if (*p == '\0') continue;

        float vals[MAX_COLS] = {0};
        int   col = 0;
        char *tok = strtok(line, ",\t \r\n");
        while (tok && col < CWT_N_CHANNELS) {
            vals[col++] = (float)atof(tok);
            tok = strtok(NULL, ",\t \r\n");
        }
        if (col < CWT_N_CHANNELS) {
            fprintf(stderr, "Warning: row %d has only %d columns (expected %d)\n",
                    row + 1, col, CWT_N_CHANNELS);
        }

        /* Store in channel-major order */
        int ch;
        for (ch = 0; ch < CWT_N_CHANNELS; ch++)
            (*data)[ch * n_rows + row] = vals[ch];

        row++;
    }

    fclose(fp);
    return n_rows;
}

/* ── main ───────────────────────────────────────────────────────────────────── */

int main(int argc, char *argv[])
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <sensor.csv> <output.bin>\n", argv[0]);
        fprintf(stderr, "  sensor.csv : 5-column (acc,acoustic,fx,fy,fz), no header\n");
        fprintf(stderr, "  output.bin : %d float32 values (5×64×64)\n", CWT_OUT_SIZE);
        return 1;
    }

    const char *csv_path = argv[1];
    const char *bin_path = argv[2];

    /* Load CSV */
    float *signal  = NULL;
    int    n_samp  = read_csv_5col(csv_path, &signal);
    if (n_samp == 0)
        return 1;

    if (n_samp < 64) {
        fprintf(stderr, "Error: signal too short (%d samples, minimum 64)\n", n_samp);
        free(signal);
        return 1;
    }

    /* Allocate output buffer */
    float *out = (float *)malloc((size_t)CWT_OUT_SIZE * sizeof(float));
    if (!out) {
        fprintf(stderr, "Error: out of memory for output buffer\n");
        free(signal);
        return 1;
    }

    /* Compute scalogram */
    int ret = cwt_compute_scalogram(signal, CWT_N_CHANNELS, n_samp, out);
    free(signal);

    if (ret != 0) {
        fprintf(stderr, "Error: cwt_compute_scalogram failed (code %d)\n", ret);
        free(out);
        return 1;
    }

    /* Write binary output */
    FILE *fp = fopen(bin_path, "wb");
    if (!fp) {
        fprintf(stderr, "Error: cannot open output file '%s'\n", bin_path);
        free(out);
        return 1;
    }
    size_t written = fwrite(out, sizeof(float), (size_t)CWT_OUT_SIZE, fp);
    fclose(fp);
    free(out);

    if ((int)written != CWT_OUT_SIZE) {
        fprintf(stderr, "Error: wrote %zu / %d floats\n", written, CWT_OUT_SIZE);
        return 1;
    }

    fprintf(stdout, "OK  %d samples → %s  (%d floats, %.1f KB)\n",
            n_samp, bin_path, CWT_OUT_SIZE,
            (double)CWT_OUT_SIZE * sizeof(float) / 1024.0);
    return 0;
}
