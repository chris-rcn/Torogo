/* train_featurepol_reinforce.c — C port of train-featurepol-reinforce.js.
 *
 * Self-play REINFORCE trainer for the featurepol policy.  Same model file format as
 * the JS trainer (interoperable via featurepol.c serialize/load).  Randomness is a
 * seeded xorshift64 (matching xorshift.js), so runs are reproducible.
 *
 *   ./train_featurepol_reinforce.bin --spec '<spec>' [options]
 *     --spec S            feature spec (required unless --load supplies one)
 *     --size N / --train-size N   self-play board size (default 9)
 *     --eval-size N       evaluation board size (default 13)
 *     --lr F              learning rate (default 0.02)
 *     --reward-ema F      EMA baseline decay; 0 disables (default 0.99)
 *     --weight-decay F    decoupled L2 shrink (default 0.000002)
 *     --temperature F     sampling temperature (default 1)
 *     --eval-agent S      'random' or 'featurepol' (with --eval-model); omitted ⇒ no eval
 *     --eval-model P      reference model file for --eval-agent featurepol
 *     --eval-size N       eval board size (default 13)
 *     --load P            resume from saved weights (imported by hash into --spec)
 *     --save P            output path (default out/featurepol-c-<seed>.js)
 *     --seed N            RNG seed (default 1)
 */
#include "featurepol.h"
#include "game2.h"
#include "game3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* RNG (Rng, rng_seed/rng_next/rng_random) now comes from the shared rng.h. */
static int rng_legal_move(Rng *r, Game2 *g) {
    /* uniform among legal non-true-eye moves (matches game.randomLegalMove semantics) */
    int cap = g->cap, tries = 0;
    /* collect once for correctness */
    static int buf[MAX_CAP];
    int nb = 0;
    for (int i = 0; i < cap; i++) if (g->cells[i] == EMPTY && g2_is_legal(g, i) && !g2_is_true_eye(g, i)) buf[nb++] = i;
    (void)tries;
    if (nb == 0) return PASS;
    return buf[(int)(rng_random(r) * nb)];
}

/* ── CLI ───────────────────────────────────────────────────────────────────── */
static const char *argval(int argc, char **argv, const char *key) {
    for (int i = 1; i < argc - 1; i++) if (strcmp(argv[i], key) == 0) return argv[i + 1];
    return NULL;
}

/* ── Greedy move for a (weights,state) policy at a position ────────────────── */
static int greedy_move(Game2 *g, Game3 *g3, FpState *st, FpWeights *w) {
    fp_extract(g, g3, st, w);
    int n = st->count;
    if (n == 0) return PASS;
    int best = 0; double bs = -INFINITY;
    for (int i = 0; i < n; i++) {
        double s = 0;
        for (int k = st->key_off[i]; k < st->key_off[i + 1]; k++) s += w->vals[st->keys[k]];
        if (s > bs) { bs = s; best = i; }
    }
    return st->moves[best];
}

/* ── One self-play game + REINFORCE update ─────────────────────────────────── */
typedef struct { int player, ci, count, nk; int32_t *keys, *key_off; double *probs; } Step;

static int train_game(int N, FpWeights *w, FpState *st, Rng *rng, double lr, double ema_decay,
                      double wd, double temp, double *ema, double *wabs, long *wcount,
                      double *maxprob_sum, int *maxprob_n) {
    Game2 g; g2_new(&g, N);
    Game3 *g3 = w->spec.needs_ladder ? g3_new(N) : NULL;
    if (g3) g3_from_game2(g3, &g);
    int maxMoves = N * N * 4;

    static Step steps[4 * MAX_CAP];
    int nsteps = 0;

    int moves = 0;
    while (!g.game_over && moves < maxMoves) {
        int player = g.current;
        fp_extract(&g, g3, st, w);
        int n = st->count, chosen_idx = -1, move;
        if (n == 0) { move = PASS; }
        else {
            fp_softmax(st, w, temp);
            double r = rng_random(rng), acc = 0; (void)acc;
            int chosen = n - 1;
            for (int i = 0; i < n; i++) { r -= st->probs[i]; if (r <= 0) { chosen = i; break; } }
            chosen_idx = chosen; move = st->moves[chosen];
            /* record step: copy keys/key_off/probs */
            int nk = st->key_off[n];
            Step *s = &steps[nsteps++];
            s->player = player; s->ci = chosen_idx; s->count = n; s->nk = nk;
            s->keys = malloc(nk * sizeof(int32_t)); memcpy(s->keys, st->keys, nk * sizeof(int32_t));
            s->key_off = malloc((n + 1) * sizeof(int32_t)); memcpy(s->key_off, st->key_off, (n + 1) * sizeof(int32_t));
            s->probs = malloc(n * sizeof(double)); memcpy(s->probs, st->probs, n * sizeof(double));
            double mx = 0; for (int i = 0; i < n; i++) if (st->probs[i] > mx) mx = st->probs[i];
            *maxprob_sum += mx; (*maxprob_n)++;
        }
        g2_play(&g, move);
        if (g3) g3_play(g3, move);
        moves++;
    }

    int8_t winner = g2_estimate_winner(&g);
    int outcome_black = (winner == BLACK) ? 1 : -1;

    int updates = 0;
    FpState tmp = *st;   /* reuse scratch (touched) + buffers via overridden pointers */
    for (int i = 0; i < nsteps; i++) {
        Step *s = &steps[i];
        double R = (s->player == BLACK) ? outcome_black : -outcome_black;
        double adv = ema_decay > 0 ? (R - *ema) : R;
        tmp.keys = s->keys; tmp.key_off = s->key_off; tmp.probs = s->probs; tmp.count = s->count;
        tmp.touched = st->touched;
        updates += fp_reinforce_update(&tmp, s->ci, adv, w, lr, wd, wabs, wcount);
        free(s->keys); free(s->key_off); free(s->probs);
    }
    if (ema_decay > 0 && nsteps > 0) *ema = ema_decay * (*ema) + (1 - ema_decay) * outcome_black;
    if (g3) g3_free(g3);
    return updates;
}

/* ── Eval vs reference (greedy featurepol vs random|featurepol) ─────────────── */
static int eval_games(int N, FpWeights *w, FpState *st, Rng *rng,
                      int ref_is_fp, FpWeights *wref, FpState *stref, int ngames) {
    int wins = 0;
    for (int gi = 0; gi < ngames; gi++) {
        int policy_is_black = (gi % 2 == 0);
        Game2 g; g2_new(&g, N);
        Game3 *g3 = w->spec.needs_ladder ? g3_new(N) : NULL;
        Game3 *g3r = (ref_is_fp && wref->spec.needs_ladder) ? g3_new(N) : NULL;
        if (g3) g3_from_game2(g3, &g);
        if (g3r) g3_from_game2(g3r, &g);
        for (int r = 0; r < 3 && !g.game_over; r++) {
            int mv = rng_legal_move(rng, &g);
            g2_play(&g, mv); if (g3) g3_play(g3, mv); if (g3r) g3_play(g3r, mv);
        }
        int m = 0;
        while (!g.game_over && m++ < N * N * 4) {
            int idx;
            if ((g.current == BLACK) == policy_is_black) idx = greedy_move(&g, g3, st, w);
            else if (ref_is_fp)                          idx = greedy_move(&g, g3r, stref, wref);
            else                                         idx = rng_legal_move(rng, &g);
            g2_play(&g, idx); if (g3) g3_play(g3, idx); if (g3r) g3_play(g3r, idx);
        }
        int8_t winner = g2_estimate_winner(&g);
        if ((winner == BLACK) == policy_is_black) wins++;
        if (g3) g3_free(g3);
        if (g3r) g3_free(g3r);
    }
    return wins;
}

static double now_s(void) { return (double)clock() / CLOCKS_PER_SEC; }

int main(int argc, char **argv) {
    const char *spec_str = argval(argc, argv, "--spec");
    const char *load_path = argval(argc, argv, "--load");
    const char *save_path = argval(argc, argv, "--save");
    const char *eval_agent = argval(argc, argv, "--eval-agent");
    const char *eval_model = argval(argc, argv, "--eval-model");
    int train_size = atoi(argval(argc, argv, "--train-size") ? argval(argc, argv, "--train-size")
                          : (argval(argc, argv, "--size") ? argval(argc, argv, "--size") : "9"));
    int eval_size = atoi(argval(argc, argv, "--eval-size") ? argval(argc, argv, "--eval-size") : "13");
    double lr = atof(argval(argc, argv, "--lr") ? argval(argc, argv, "--lr") : "0.02");
    double ema_decay = atof(argval(argc, argv, "--reward-ema") ? argval(argc, argv, "--reward-ema") : "0.99");
    double wd = atof(argval(argc, argv, "--weight-decay") ? argval(argc, argv, "--weight-decay") : "0.000002");
    double temp = atof(argval(argc, argv, "--temperature") ? argval(argc, argv, "--temperature") : "1");
    const char *seed_arg = argval(argc, argv, "--seed");
    long seed = seed_arg ? atol(seed_arg) : (long)time(NULL);   /* default: time-seeded (varies per run) */

    char err[256];
    double ema = 0; long total_updates = 0;
    FpWeights *w = NULL;

    /* load (optional) then build the training weights */
    FpWeights *loaded = NULL;
    if (load_path) { loaded = fp_load_model(load_path, &ema, &total_updates, err, sizeof(err));
                     if (!loaded) fprintf(stderr, "warning: --load failed: %s\n", err); }
    const char *use_spec = spec_str ? spec_str : (loaded ? loaded->spec.str : NULL);
    if (!use_spec) { fprintf(stderr, "Error: --spec required unless --load supplies one.\n"); return 1; }
    FpSpec spec;
    if (!fp_parse_spec(use_spec, &spec, err, sizeof(err))) { fprintf(stderr, "Error: %s\n", err); return 1; }
    w = fp_create_weights(&spec);
    if (loaded) {
        int imported = 0;
        for (int i = 0; i < loaded->size; i++) { w->vals[fp_intern(w, loaded->key_hash[i])] = loaded->vals[i]; imported++; }
        fprintf(stderr, "Resumed: imported %d weights, ema=%.3f\n", imported, ema);
        fp_free_weights(loaded);
    }

    /* reference policy */
    int ref_is_fp = 0; FpWeights *wref = NULL; FpState *stref = NULL;
    if (eval_agent && strcmp(eval_agent, "featurepol") == 0) {
        if (!eval_model) { fprintf(stderr, "Error: --eval-agent featurepol needs --eval-model\n"); return 1; }
        double re; long rt;
        wref = fp_load_model(eval_model, &re, &rt, err, sizeof(err));
        if (!wref) { fprintf(stderr, "Error: eval model: %s\n", err); return 1; }
        ref_is_fp = 1; stref = fp_create_state(eval_size, &wref->spec);
    }
    int do_eval = (eval_agent != NULL);

    char default_save[256];
    if (!save_path) { snprintf(default_save, sizeof(default_save), "out/featurepol-c-%ld.js", seed); save_path = default_save; }

    Rng rng; rng_seed(&rng, seed);
    FpState *st = NULL;          /* train-size state */
    FpState *st_eval = NULL;     /* eval-size state (for the training policy) */

    g2_init_topology(train_size);
    st = fp_create_state(train_size, &spec);

    fprintf(stderr, "spec='%s' spaces=%d needsLadder=%d\n", spec.str, spec.n_spaces, spec.needs_ladder);
    char evalinfo[320];
    if (do_eval && ref_is_fp) snprintf(evalinfo, sizeof(evalinfo), "  eval=featurepol eval-size=%d eval-model=%s", eval_size, eval_model);
    else if (do_eval)         snprintf(evalinfo, sizeof(evalinfo), "  eval=random eval-size=%d", eval_size);
    else                      snprintf(evalinfo, sizeof(evalinfo), "  (no eval)");
    fprintf(stderr, "lr=%g reward-ema=%g weight-decay=%g temperature=%g train-size=%d%s seed=%ld\n",
            lr, ema_decay, wd, temp, train_size, evalinfo, seed);
    fprintf(stderr, "save: %s\n", save_path);
    printf("%8s  %7s  %8s  %6s  %7s  %6s  %s\n", "elapsed", "game", "tGm", "nWts", "avgW", "maxP", do_eval ? "winRatio" : "");

    double t0 = now_s(), next_print = 1.0;
    double wabs = 0; long wcount = 0;
    double maxp_sum = 0; int maxp_n = 0;
    long g = 0, last_g = 0;
    double train_acc = 0;   /* CPU time spent in train_game this interval */
    /* rolling eval history (recent-half win ratio) */
    static char evalhist[1 << 16]; long evalhist_n = 0;

    for (;;) {
        g++;
        double tg0 = now_s();
        total_updates += train_game(train_size, w, st, &rng, lr, ema_decay, wd, temp,
                                    &ema, &wabs, &wcount, &maxp_sum, &maxp_n);
        train_acc += now_s() - tg0;
        double el = now_s() - t0;
        if (el >= next_print) {
            int nz = 0; for (int i = 0; i < w->size; i++) if (w->vals[i] != 0) nz++;
            double avgw = wcount > 0 ? wabs / wcount : 0;
            double maxp = maxp_n > 0 ? maxp_sum / maxp_n : 0;
            char ev[64] = "";
            if (do_eval) {
                if (!st_eval) { g2_init_topology(eval_size); st_eval = fp_create_state(eval_size, &spec);
                                if (ref_is_fp && !stref) stref = fp_create_state(eval_size, &wref->spec); }
                g2_init_topology(eval_size);
                /* eval games = half this cycle's training games, capped at 1000 */
                long eval_target = (g - last_g) / 2;
                if (eval_target > 1000) eval_target = 1000;
                int ew = 0, eg = 0;
                for (; eg < eval_target; eg++) {
                    int w1 = eval_games(eval_size, w, st_eval, &rng, ref_is_fp, wref, stref, 1);
                    if (evalhist_n < (long)sizeof(evalhist)) evalhist[evalhist_n++] = (char)w1;
                    ew += w1;
                }
                long half = evalhist_n / 2; if (half < 1) half = 1;
                long hsum = 0; for (long i = evalhist_n - half; i < evalhist_n; i++) hsum += evalhist[i];
                snprintf(ev, sizeof(ev), "%.4f(%4d)/%.4f(%4ld)", eg ? (double)ew / eg : 0, eg, (double)hsum / half, half);
                g2_init_topology(train_size);
            }
            char tgm[16];
            snprintf(tgm, sizeof(tgm), "%.3fms", g > last_g ? train_acc / (g - last_g) * 1000.0 : 0.0);
            printf("%7.1fs  %7ld  %8s  %6d  %7.4f  %6.4f  %s\n", el, g, tgm, nz, avgw, maxp, ev);
            fflush(stdout);
            fp_serialize(w, save_path, ema, total_updates);
            maxp_sum = 0; maxp_n = 0; train_acc = 0; last_g = g;
            next_print = el * 1.4;
        }
    }
    return 0;
}
