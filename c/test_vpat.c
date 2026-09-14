#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vpat.h"

/* Differential test against the JS twin.
 *
 *   ./test_vpat.bin <model.js> <positions.txt>
 *
 * positions.txt lines (from the JS harness): N nMoves m1 .. mN expectedVal
 * Each line replays from the standard start (g2_new: free centre stone) and
 * compares vpat_evaluate against the JS evaluate() value.
 */

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s <model.js> <positions.txt>\n", argv[0]); return 1; }
    vpat_load(argv[1]);
    FILE *f = fopen(argv[2], "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }

    static Game2 g;
    int cur_n = 0;
    long lines = 0, bad = 0;
    double max_diff = 0.0;
    int n, n_moves;
    while (fscanf(f, "%d %d", &n, &n_moves) == 2) {
        if (n != cur_n) { g2_init_topology(n); cur_n = n; }
        g2_new(&g, n);
        for (int i = 0; i < n_moves; i++) {
            int mv;
            if (fscanf(f, "%d", &mv) != 1) { fprintf(stderr, "truncated line %ld\n", lines); return 1; }
            if (!g2_play(&g, mv)) { fprintf(stderr, "illegal replay move %d on line %ld\n", mv, lines); return 1; }
        }
        double expect;
        if (fscanf(f, "%lf", &expect) != 1) { fprintf(stderr, "missing value on line %ld\n", lines); return 1; }
        const double got = vpat_evaluate(&g);
        const double d = fabs(got - expect);
        if (d > max_diff) max_diff = d;
        if (d > 1e-9) {
            if (bad < 5) fprintf(stderr, "line %ld: C %.12f JS %.12f (diff %.3g)\n", lines, got, expect, d);
            bad++;
        }
        lines++;
    }
    fclose(f);
    printf("test_vpat: %ld positions, %ld beyond 1e-9, max |diff| %.3g\n", lines, bad, max_diff);
    return bad ? 1 : 0;
}
