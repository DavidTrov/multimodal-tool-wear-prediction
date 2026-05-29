/*
 * main_mcu.c — CLI wrapper for cwt_mcu_process_channel()
 *
 * Usage:
 *   ./cwt_mcu <sensor.csv> <output.bin>
 *
 * Reads one channel at a time from the CSV to minimise peak memory.
 * Same input/output format as the desktop cwt_preprocess binary.
 *
 * Input CSV:
 *   - 5 columns: acc, acoustic, fx, fy, fz
 *   - No header row, comma-separated floats
 *   - Must contain only the CUTTING SEGMENT (post aircut-gating)
 *
 * Output binary:
 *   - 5 x 64 x 64 x 4 bytes = 81,920 bytes
 *   - Row-major float32: out[ch][scale][time_bin]
 *   - Values in [0, 1]
 */

#include "cwt_mcu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── CSV helpers ───────────────────────────────────────────────────────────── */

/*
 * count_csv_rows — count non-blank rows in a CSV file.
 */
static int count_csv_rows(const char *path)
{
    FILE *fp = fopen(path, "r");
    if (!fp) return 0;

    int n = 0;
    char line[4096];
    while (fgets(line, sizeof(line), fp)) {
        const char *p = line;
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
        if (*p != '\0') n++;
    }
    fclose(fp);
    return n;
}

/*
 * read_csv_column — read a single column from a 5-column CSV.
 *
 * col_idx:  0-based column to extract (0=acc, 1=acoustic, 2=fx, 3=fy, 4=fz)
 * buf:      pre-allocated float[n_rows]
 *
 * Returns the number of rows read.
 */
static int read_csv_column(const char *path, int col_idx, float *buf, int n_rows)
{
    FILE *fp = fopen(path, "r");
    if (!fp) return 0;

    int row = 0;
    char line[4096];
    while (fgets(line, sizeof(line), fp) && row < n_rows) {
        const char *p = line;
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
        if (*p == '\0') continue;

        /* Tokenise and extract the target column */
        char *tok = strtok(line, ",\t \r\n");
        int col = 0;
        float val = 0.0f;
        while (tok) {
            if (col == col_idx) {
                val = (float)atof(tok);
                break;
            }
            col++;
            tok = strtok(NULL, ",\t \r\n");
        }
        buf[row++] = val;
    }
    fclose(fp);
    return row;
}

/* ── main ──────────────────────────────────────────────────────────────────── */

int main(int argc, char *argv[])
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <sensor.csv> <output.bin>\n", argv[0]);
        fprintf(stderr, "  sensor.csv : 5-column (acc,acoustic,fx,fy,fz), no header\n");
        fprintf(stderr, "  output.bin : %d float32 values (5x64x64)\n", CWT_MCU_OUT_SIZE);
        return 1;
    }

    const char *csv_path = argv[1];
    const char *bin_path = argv[2];

    /* Count rows */
    int n_rows = count_csv_rows(csv_path);
    if (n_rows == 0) {
        fprintf(stderr, "Error: no data rows in '%s'\n", csv_path);
        return 1;
    }
    if (n_rows < CWT_MCU_N_TIME) {
        fprintf(stderr, "Error: signal too short (%d samples, minimum %d)\n",
                n_rows, CWT_MCU_N_TIME);
        return 1;
    }

    /* Allocate buffers */
    float *signal    = (float *)malloc((size_t)n_rows * sizeof(float));
    float *out       = (float *)calloc((size_t)CWT_MCU_OUT_SIZE, sizeof(float));
    float *workspace = (float *)malloc((size_t)(2 * CWT_MCU_N_KER) * sizeof(float));

    if (!signal || !out || !workspace) {
        fprintf(stderr, "Error: out of memory\n");
        free(signal); free(out); free(workspace);
        return 1;
    }

    /* Process each channel */
    int ch;
    for (ch = 0; ch < CWT_MCU_N_CHANNELS; ch++) {
        int rows_read = read_csv_column(csv_path, ch, signal, n_rows);
        if (rows_read != n_rows) {
            fprintf(stderr, "Error: column %d read %d/%d rows\n", ch, rows_read, n_rows);
            free(signal); free(out); free(workspace);
            return 1;
        }

        float *ch_out = out + (size_t)ch * CWT_MCU_CH_SIZE;
        int ret = cwt_mcu_process_channel(signal, n_rows, ch, ch_out, workspace);
        if (ret != 0) {
            fprintf(stderr, "Error: cwt_mcu_process_channel failed for ch %d (code %d)\n",
                    ch, ret);
            free(signal); free(out); free(workspace);
            return 1;
        }
    }

    free(signal);
    free(workspace);

    /* Write output */
    FILE *fp = fopen(bin_path, "wb");
    if (!fp) {
        fprintf(stderr, "Error: cannot open output file '%s'\n", bin_path);
        free(out);
        return 1;
    }
    size_t written = fwrite(out, sizeof(float), (size_t)CWT_MCU_OUT_SIZE, fp);
    fclose(fp);
    free(out);

    if ((int)written != CWT_MCU_OUT_SIZE) {
        fprintf(stderr, "Error: wrote %zu / %d floats\n", written, CWT_MCU_OUT_SIZE);
        return 1;
    }

    fprintf(stdout, "OK  %d samples -> %s  (%d floats, %.1f KB)\n",
            n_rows, bin_path, CWT_MCU_OUT_SIZE,
            (double)CWT_MCU_OUT_SIZE * sizeof(float) / 1024.0);
    return 0;
}
