/*
 * eval_compare.c — Read positions from a gen-evals file, re-evaluate with RAVE,
 * and compare to the stored value.
 *
 * Compile: cc -O2 -o eval_compare game2.c rave.c eval_compare.c -lm
 * Usage:   ./eval_compare <file> [n_positions] [eval_playouts]
 */
#include "game2.h"
#include "rave.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Parse coordinate string (e.g. "b2") to flat index. "pass" → PASS. */
static int32_t parse_move(const char *s) {
    if (s[0] == 'p') return PASS;
    int x = s[0] - 'a';
    int y = atoi(s + 1) - 1;
    return y * BOARD_SIZE + x;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <file> [n_positions] [eval_playouts]\n", argv[0]);
        return 1;
    }
    const char *filename = argv[1];
    int n_positions  = argc > 2 ? atoi(argv[2]) : 20;
    int eval_playouts = argc > 3 ? atoi(argv[3]) : 200000;

    g2_seed((uint32_t)time(NULL));
    g2_init_topology();

    FILE *f = fopen(filename, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", filename); return 1; }

    /* Read all lines */
    char **lines = NULL;
    int nlines = 0, cap_lines = 0;
    char buf[8192];
    while (fgets(buf, sizeof(buf), f)) {
        if (buf[0] == '\n' || buf[0] == '\0') continue;
        if (nlines >= cap_lines) {
            cap_lines = cap_lines ? cap_lines * 2 : 256;
            lines = realloc(lines, cap_lines * sizeof(char *));
        }
        lines[nlines++] = strdup(buf);
    }
    fclose(f);

    if (nlines == 0) { fprintf(stderr, "no lines in file\n"); return 1; }

    /* Sample n_positions evenly spaced lines */
    RaveState *s = rave_create();

    printf("%5s  %8s  %8s  %8s\n", "pos", "file_val", "c_val", "delta");

    float total_abs_delta = 0;
    int step = nlines / n_positions;
    if (step < 1) step = 1;

    for (int li = 0; li < nlines && li / step < n_positions; li += step) {
        char *line = lines[li];

        /* Parse: "<size> <moves> <value>" */
        int size;
        char moves_str[8192];
        double file_value;
        if (sscanf(line, "%d %s %lf", &size, moves_str, &file_value) != 3) continue;
        if (size != BOARD_SIZE) continue;

        /* Replay moves */
        Game2 g;
        g2_new(&g);
        char *tok = strtok(moves_str, ",");
        int ok = 1;
        while (tok) {
            int32_t m = parse_move(tok);
            if (!g2_play(&g, m)) { ok = 0; break; }
            tok = strtok(NULL, ",");
        }
        if (!ok || g.game_over) continue;

        /* Evaluate */
        RaveResult r = rave_search(s, &g, eval_playouts, 0);
        float c_value = r.win_ratio;
        float delta = c_value - (float)file_value;

        printf("%5d  %8.4f  %8.4f  %+8.4f\n", li, (float)file_value, c_value, delta);
        total_abs_delta += (delta < 0 ? -delta : delta);
    }

    int evaluated = n_positions < nlines / step ? n_positions : nlines / step;
    if (evaluated > 0)
        printf("\nmean |delta|: %.4f\n", total_abs_delta / evaluated);

    rave_destroy(s);
    for (int i = 0; i < nlines; i++) free(lines[i]);
    free(lines);
    return 0;
}
