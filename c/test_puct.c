#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "fpol.h"
#include "puct.h"

/* Differential test against ai/puct-ppat-fp.js with uniform playouts.
 *
 *   node c/test_puct_gen.js ref/ref-fp-fast.js > /tmp/puct-fixture.txt
 *   ./test_puct.bin ref/ref-fp-fast.js /tmp/puct-fixture.txt [ppat-model.js]
 *
 * Each fixture line replays a position, gives the playout count and seed, and
 * the JS search's move, root win ratio and every root edge's visits and wins.
 * The C search on the same seed must match all of them exactly.
 *
 * With a ppat model, every position is searched again with ppat playouts at
 * PPAT_MIN_PHASE 0.6 (not bit-comparable to JS: C ppat samples in float) and
 * the ms per simulation of both passes is printed.
 */
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
int main(int argc, char **argv) {
    if (argc != 3 && argc != 4) { fprintf(stderr, "usage: %s <fpol-model.js> <fixture.txt> [ppat-model.js]\n", argv[0]); return 1; }
    fpol_load(argv[1]);
    FILE *f = fopen(argv[2], "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    const PuctCfg cfg = puct_default_cfg();
    PuctSearch *s = puct_new(&cfg);
    PuctSearch *sp = NULL;
    if (argc == 4) {
        PuctCfg pc = puct_default_cfg();
        pc.ppat_w = ppat_load_weights(argv[3], &pc.early_pass, &pc.pass_weight);
        if (!pc.ppat_w) { fprintf(stderr, "cannot load ppat model %s\n", argv[3]); return 1; }
        sp = puct_new(&pc);
    }
    double ms_uniform = 0, ms_ppat = 0, wr_ppat = 0;
    static Game2 g;
    int cur_n = 0, n, n_moves;
    long lines = 0, bad = 0, sims = 0;
    while (fscanf(f, "%d %d", &n, &n_moves) == 2) {
        if (n != cur_n) { g2_init_topology(n); cur_n = n; }
        g2_new(&g, n);
        for (int i = 0; i < n_moves; i++) {
            int mv;
            if (fscanf(f, "%d", &mv) != 1 || !g2_play(&g, mv)) { fprintf(stderr, "bad replay on line %ld\n", lines); return 1; }
        }
        int playouts, jmove, m; long seed; double jwr;
        if (fscanf(f, "%d %ld %d %lf %d", &playouts, &seed, &jmove, &jwr, &m) != 5) { fprintf(stderr, "truncated line %ld\n", lines); return 1; }
        Rng rng; rng_seed(&rng, seed);
        double t0 = now_ms();
        const PuctResult r = puct_get_move(s, &g, playouts, &rng);
        ms_uniform += now_ms() - t0;
        const int32_t *cm; const float *cv, *cw;
        const int cn = puct_root_stats(s, &cm, &cv, &cw);
        lines++; sims += playouts;
        int ok = 1;
        if (r.move != jmove) { ok = 0; fprintf(stderr, "line %ld: move C %d JS %d\n", lines, r.move, jmove); }
        if (r.root_win_ratio != jwr) { ok = 0; fprintf(stderr, "line %ld: rootWinRatio C %.17g JS %.17g\n", lines, r.root_win_ratio, jwr); }
        if (cn != m) { ok = 0; fprintf(stderr, "line %ld: %d root edges in C, %d in JS\n", lines, cn, m); }
        int edge_ok = 1;
        for (int i = 0; i < m; i++) {
            int mv; double v, w;
            if (fscanf(f, "%d %lf %lf", &mv, &v, &w) != 3) { fprintf(stderr, "truncated line %ld\n", lines); return 1; }
            int k = -1;
            for (int j = 0; j < cn; j++) if (cm[j] == mv) { k = j; break; }
            if (k < 0) { if (edge_ok) fprintf(stderr, "line %ld: JS edge %d missing in C\n", lines, mv); ok = edge_ok = 0; continue; }
            if (cv[k] != (float)v || cw[k] != (float)w) {
                if (edge_ok) fprintf(stderr, "line %ld move %d: C visits %.9g wins %.9g, JS %.9g %.9g\n", lines, mv, cv[k], cw[k], v, w);
                ok = edge_ok = 0;
            }
        }
        if (!ok) bad++;
        if (sp) {
            rng_seed(&rng, seed);
            t0 = now_ms();
            const PuctResult rp = puct_get_move(sp, &g, playouts, &rng);
            ms_ppat += now_ms() - t0;
            if (rp.playouts != playouts || !(rp.root_win_ratio >= 0 && rp.root_win_ratio <= 1)) {
                fprintf(stderr, "line %ld: ppat search ran %d of %d, root win ratio %g\n", lines, rp.playouts, playouts, rp.root_win_ratio);
                bad++;
            }
            wr_ppat += rp.root_win_ratio;
        }
    }
    fclose(f);
    puct_free(s); puct_free(sp);
    printf("%ld positions, %ld simulations, %ld mismatched\n", lines, sims, bad);
    printf("uniform playouts: %.4f ms/sim\n", ms_uniform / sims);
    if (sp) printf("ppat playouts:    %.4f ms/sim, mean root win ratio %.4f\n", ms_ppat / sims, wr_ppat / lines);
    return (bad == 0 && lines > 0) ? 0 : 1;
}
