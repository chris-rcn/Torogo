#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "fpol.h"

/* Differential test against featurepol-lib.js (stone8AdjLib3).
 *
 *   node c/test_fpol_gen.js ref/ref-fp-fast.js > /tmp/fpol-fixture.txt
 *   ./test_fpol.bin ref/ref-fp-fast.js /tmp/fpol-fixture.txt
 *
 * Each fixture line replays a position from g2_new and gives featurepol's
 * candidates with their logits and temperature-1 probabilities.  The C twin
 * must produce the same candidate set, the same logits bit for bit, and the
 * same probabilities to within libm's exp; then sample and rank are checked
 * for consistency on every position.
 */
int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s <model.js> <fixture.txt>\n", argv[0]); return 1; }
    fpol_load(argv[1]);
    FILE *f = fopen(argv[2], "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    static Game2 g;
    static FpolState st;
    static int order[MAX_CAP];
    static int32_t jm[MAX_CAP];
    static double jl[MAX_CAP], jp[MAX_CAP];
    int cur_n = 0, n, n_moves;
    long lines = 0, bad = 0, cands = 0;
    double max_pdiff = 0;
    Rng rng; rng_seed(&rng, 12345);
    while (fscanf(f, "%d %d", &n, &n_moves) == 2) {
        if (n != cur_n) { g2_init_topology(n); cur_n = n; }
        g2_new(&g, n);
        for (int i = 0; i < n_moves; i++) {
            int mv;
            if (fscanf(f, "%d", &mv) != 1) { fprintf(stderr, "truncated line %ld\n", lines); return 1; }
            if (!g2_play(&g, mv)) { fprintf(stderr, "illegal replay move %d on line %ld\n", mv, lines); return 1; }
        }
        int nc;
        if (fscanf(f, "%d", &nc) != 1) { fprintf(stderr, "truncated line %ld\n", lines); return 1; }
        for (int i = 0; i < nc; i++)
            if (fscanf(f, "%d %lf %lf", &jm[i], &jl[i], &jp[i]) != 3) { fprintf(stderr, "truncated line %ld\n", lines); return 1; }
        fpol_eval(&g, &st, 1.0);
        lines++; cands += nc;
        int ok = st.n == nc;
        for (int i = 0; ok && i < nc; i++) {
            int k = -1;
            for (int j = 0; j < st.n; j++) if (st.moves[j] == jm[i]) { k = j; break; }
            if (k < 0) { ok = 0; fprintf(stderr, "line %ld: JS candidate %d missing in C\n", lines, jm[i]); break; }
            if (st.logits[k] != jl[i]) { ok = 0; fprintf(stderr, "line %ld move %d: logit C %.17g JS %.17g\n", lines, jm[i], st.logits[k], jl[i]); break; }
            const double d = fabs(st.probs[k] - jp[i]);
            if (d > max_pdiff) max_pdiff = d;
            if (d > 1e-12) { ok = 0; fprintf(stderr, "line %ld move %d: prob C %.17g JS %.17g\n", lines, jm[i], st.probs[k], jp[i]); break; }
        }
        if (!ok && st.n != nc) fprintf(stderr, "line %ld: %d candidates in C, %d in JS\n", lines, st.n, nc);
        /* sample returns a candidate; rank is sorted and a permutation */
        if (ok && st.n > 0) {
            const int32_t s = fpol_sample(&st, &rng);
            int found = 0; for (int j = 0; j < st.n; j++) if (st.moves[j] == s) found = 1;
            fpol_rank(&st, order);
            for (int j = 1; j < st.n; j++) if (st.logits[order[j - 1]] < st.logits[order[j]]) found = 0;
            if (!found) { ok = 0; fprintf(stderr, "line %ld: sample/rank inconsistent\n", lines); }
        }
        if (!ok) bad++;
    }
    fclose(f);
    printf("%ld positions, %ld candidates, %ld mismatched; max |prob diff| %.3g\n", lines, cands, bad, max_pdiff);
    return (bad == 0 && lines > 0) ? 0 : 1;
}
