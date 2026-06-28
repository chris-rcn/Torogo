/* test_fp_train.c — verify the C REINFORCE update + serialize/deserialize against JS.
 *
 *   node ../fp-train-oracle.js --spec S --size N --steps K --seed s --save js.js > train.txt
 *   ./test_fp_train.bin <spec> <size> <js-model> <c-model-out> < train.txt
 *
 * Replays the oracle's 'P' positions through the same deterministic REINFORCE run,
 * then (1) compares every weight to the oracle's 'W' lines, (2) loads the JS-saved
 * model and checks it dequantizes to the trained weights, (3) re-serializes to
 * <c-model-out> for a JS-side byte/qval comparison.  Exit 0 iff all checks pass.
 */
#include "featurepol.h"
#include "game2.h"
#include "game3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LINEMAX (1 << 20)
#define MAXMOVE 1024

static void play_pos(Game2 *g, Game3 **g3p, const FpSpec *spec, int size, char *csv) {
    g2_new(g, size);
    Game3 *g3 = NULL;
    if (spec->needs_ladder) { g3 = g3_new(size); }
    int moves[MAXMOVE], nm = 0;
    char *sv = NULL;
    for (char *m = strtok_r(csv, ",", &sv); m && nm < MAXMOVE; m = strtok_r(NULL, ",", &sv))
        moves[nm++] = atoi(m);
    for (int i = 0; i < nm; i++) g2_play(g, moves[i]);
    if (g3) { g3_from_game2(g3, g); }   /* rebuild mirror at the final position */
    *g3p = g3;
}

static int cmp_int(const void *a, const void *b) { return *(const int *)a - *(const int *)b; }

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s <spec> <size> <js-model> <c-model-out>\n", argv[0]); return 2; }
    const char *spec_str = argv[1];
    int size = atoi(argv[2]);
    const char *js_model = argv[3], *c_model = argv[4];
    const double LR = 0.05, WD = 0.000002;

    FpSpec spec; char err[256];
    if (!fp_parse_spec(spec_str, &spec, err, sizeof(err))) { fprintf(stderr, "spec: %s\n", err); return 2; }
    g2_init_topology(size);
    FpWeights *w = fp_create_weights(&spec);
    FpState *st = fp_create_state(size, &spec);

    char *line = malloc(LINEMAX);
    int step = 0;
    double wabs = 0; long wcount = 0;

    /* phase 1: train on P lines */
    long pos_after_p = 0;
    while (fgets(line, LINEMAX, stdin)) {
        if (line[0] != 'P') { pos_after_p = ftell(stdin); break; }
        char *csv = line + 2; csv[strcspn(csv, "\r\n")] = 0;
        Game2 g; Game3 *g3;
        play_pos(&g, &g3, &spec, size, csv);
        fp_extract(&g, g3, st, w);
        if (g3) g3_free(g3);
        int n = st->count;
        if (n > 0) {
            fp_softmax(st, w, 1.0);
            int sorted[MAXMOVE];
            for (int k = 0; k < n; k++) sorted[k] = st->moves[k];
            qsort(sorted, n, sizeof(int), cmp_int);
            int chosen_move = sorted[step % n], ci = 0;
            for (int k = 0; k < n; k++) if (st->moves[k] == chosen_move) { ci = k; break; }
            double adv = (step % 2 ? 1 : -1) * 0.5;
            fp_reinforce_update(st, ci, adv, w, LR, WD, &wabs, &wcount);
        }
        step++;
    }

    /* phase 2: compare W lines to trained weights (reuse the line already read) */
    int wmism = 0; double maxwdiff = 0; long ncompared = 0;
    do {
        if (line[0] != 'W') continue;
        uint32_t hash; double val;
        if (sscanf(line + 2, "%u %lf", &hash, &val) != 2) continue;
        int32_t idx = fp_lookup(w, hash);
        double cval = (idx >= 0) ? w->vals[idx] : 0.0;
        double d = fabs(cval - val);
        if (d > maxwdiff) maxwdiff = d;
        if (d > 1e-6) wmism++;
        ncompared++;
    } while (fgets(line, LINEMAX, stdin));
    (void)pos_after_p;

    /* phase 3: load the JS-saved model and check dequantization ≈ trained weights */
    double ema; long tu;
    FpWeights *wl = fp_load_model(js_model, &ema, &tu, err, sizeof(err));
    int loadmism = 0; double maxqdiff = 0;
    if (!wl) { fprintf(stderr, "load failed: %s\n", err); }
    else {
        /* quantization step = 1/scale; recover scale from a nonzero pair */
        for (int i = 0; i < w->size; i++) {
            int32_t li = fp_lookup(wl, w->key_hash[i]);
            double lv = (li >= 0) ? wl->vals[li] : 0.0;
            double d = fabs(lv - w->vals[i]);
            if (d > maxqdiff) maxqdiff = d;
        }
        /* tolerance: one quantization step.  maxAbs≈max|w|, step=maxAbs/32767. */
        double maxAbs = 0; for (int i = 0; i < w->size; i++) { double a = fabs(w->vals[i]); if (a > maxAbs) maxAbs = a; }
        double qstep = maxAbs / 32767.0;
        if (maxqdiff > qstep + 1e-9) loadmism = 1;
        fp_free_weights(wl);
    }

    /* phase 4: serialize C-trained weights for the JS-side comparison */
    fp_serialize(w, c_model, ema, tu);

    printf("steps: %d  weights: %d  W-compared: %ld  W-mismatch(>1e-6): %d  maxWdiff: %.3e\n",
           step, w->size, ncompared, wmism, maxwdiff);
    printf("deserialize: maxQdiff: %.3e  within-1-qstep: %s\n", maxqdiff, loadmism ? "NO" : "yes");

    free(line); fp_free_state(st); fp_free_weights(w);
    return (wmism || loadmism) ? 1 : 0;
}
