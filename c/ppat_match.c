/* ppat_match — the ppat match metric as a standalone run (the C twin of
 *   FPOL_DATA=ref/ref-fp-fast.js PLAYOUTS=100 P1_PPAT_MIN_PHASE=1 P2_PPAT_DATA=<model>
 *   node selfplay.js --p1 puct-ppat-fp --min-phase 0.5 --fallback rff --size 13 --limit N).
 * See match.h for the games.  Pair i is seeded seed+i, so two runs with one
 * --seed play the same openings and seat streams: their difference is paired. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "fpol.h"
#include "ppat.h"
#include "match.h"

static const char *arg_str(int argc, char **argv, const char *flag, const char *def) {
    for (int i = 1; i < argc - 1; i++) if (!strcmp(argv[i], flag)) return argv[i + 1];
    return def;
}
static double arg_num(int argc, char **argv, const char *flag, double def) {
    const char *s = arg_str(argc, argv, flag, NULL);
    return s ? atof(s) : def;
}
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

/* ppat.h keeps one active encoding, so two ppat sides must share it. */

static void load_side(PuctCfg *c, const char *path, double min_phase, const char *who) {
    *c = puct_default_cfg();
    c->ppat_min_phase = min_phase;
    if (!path) return;                              /* uniform playouts */
    c->ppat_w = ppat_load_model(path, &c->early_pass, &c->pass_weight);   /* the file defines its features */
    if (!c->ppat_w) { fprintf(stderr, "ppat_match: cannot load %s ppat model %s\n", who, path); exit(1); }
}

int main(int argc, char **argv) {
    const char *p2_path = arg_str(argc, argv, "--p2-ppat", NULL);
    if (!p2_path || arg_str(argc, argv, "--help", NULL)) {
        fprintf(stderr,
"usage: ppat_match.bin --p2-ppat FILE [options]\n"
"  --p2-ppat FILE        P2's ppat playout model (required)\n"
"  --p1-ppat FILE        P1's ppat playout model (default: uniform playouts)\n"
"  --p1-ppat-min-phase F / --p2-ppat-min-phase F\n"
"                        uniform playouts below this fullness (default 0.6)\n"
"  --fpol FILE           prior / top-K / rff model (default ref/ref-fp-fast.js)\n"
"  --playouts N          simulations per search move (default 100)\n"
"  --min-phase F         rff plays both sides below this phase (default 0.5)\n"
"  --mirror-pairs N      random mirror pairs opening each game (default 3)\n"
"  --size N              board size (default 13)\n"
"  --games N             games, even: N/2 colour-swapped pairs (default 100)\n"
"  --seed S              pair i is seeded S+i (default 1)\n");
        return 1;
    }
    MatchCfg cfg;
    cfg.size         = (int)arg_num(argc, argv, "--size", 13);
    cfg.mirror_pairs = (int)arg_num(argc, argv, "--mirror-pairs", 3);
    cfg.min_phase    = arg_num(argc, argv, "--min-phase", 0.5);
    cfg.playouts     = (int)arg_num(argc, argv, "--playouts", 100);
    const int games  = (int)arg_num(argc, argv, "--games", 100);
    const long seed  = (long)arg_num(argc, argv, "--seed", 1);
    const char *fpol_path = arg_str(argc, argv, "--fpol", "ref/ref-fp-fast.js");
    const char *p1_path   = arg_str(argc, argv, "--p1-ppat", NULL);
    if (games < 2 || games % 2) { fprintf(stderr, "ppat_match: --games must be even and >= 2\n"); return 1; }
    if (cfg.size > MAX_BOARD_SIZE) { fprintf(stderr, "ppat_match: --size above %d\n", MAX_BOARD_SIZE); return 1; }
    if (cfg.playouts < 1) { fprintf(stderr, "ppat_match: --playouts must be >= 1\n"); return 1; }

    fpol_load(fpol_path);
    load_side(&cfg.side[0], p1_path, arg_num(argc, argv, "--p1-ppat-min-phase", 0.6), "P1");
    const PpatEncoding e1 = ppat_get_encoding();
    load_side(&cfg.side[1], p2_path, arg_num(argc, argv, "--p2-ppat-min-phase", 0.6), "P2");
    if (p1_path) {
        const PpatEncoding e2 = ppat_get_encoding();
        if (memcmp(&e1, &e2, sizeof e1)) { fprintf(stderr, "ppat_match: P1 and P2 ppat models use different encodings\n"); return 1; }
    }
    printf("p1: %s  p2: %s  fpol: %s\n", p1_path ? p1_path : "uniform playouts", p2_path, fpol_path);
    printf("size: %d  playouts: %d  min-phase: %g  mirror-pairs: %d  games: %d  seed: %ld\n",
           cfg.size, cfg.playouts, cfg.min_phase, cfg.mirror_pairs, games, seed);
    printf("%5s  %8s  %5s  %6s  %8s  %8s  %6s  %5s\n", "games", "elapsed", "blkWR", "avgLen", "P1ms/mv", "P2ms/mv", "P2WR", "2se");
    fflush(stdout);

    Match *m = match_new(&cfg);
    MatchStats st = {0};
    const double t0 = now_ms();
    double next_print = t0 + 1000, period = 1000;
    for (int p = 0; p < games / 2; p++) {
        match_play_pair(m, (uint64_t)(seed + p), &st);
        const double t = now_ms();
        if (t >= next_print || p == games / 2 - 1) {
            printf("%5d  %7.1fs  %5.3f  %6.1f  %8.2f  %8.2f  %6.4f  %5.3f\n", st.games, (t - t0) / 1000,
                   (double)st.black_wins / st.games, (double)st.total_moves / st.games,
                   st.moves[0] ? st.ms[0] / st.moves[0] : 0, st.moves[1] ? st.ms[1] / st.moves[1] : 0,
                   match_p2_ratio(&st), 2 * match_p2_se(&st));
            fflush(stdout);
            period *= 1.5; next_print = t + period;
        }
    }
    match_free(m);
    return 0;
}
