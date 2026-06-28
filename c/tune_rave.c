/*
 * tune_rave.c — Head-to-head RAVE config tuning, entirely in C (fast).
 *
 * Plays config A (candidate) vs config B (anchor), alternating colors in swapped
 * pairs from a shared random opening, and early-stops (SPRT-style) once P(A truly
 * better than 50%) crosses 1-tol or tol.  Reports A's win%.
 *
 * Configs come from env vars so the rolling-pool bash harness reuses unchanged:
 *   A: RAVE_K   EXPLORATION_C   N_EXPAND   RAVE_INHERIT   PLAYOUTS
 *   B: RAVE_K_B EXPLORATION_C_B N_EXPAND_B RAVE_INHERIT_B PLAYOUTS_B(=PLAYOUTS)
 * Unset → the rave.c defaults (the oracle config) / PLAYOUTS=1000.
 *
 * Flags: --size N --limit N --stop-tol a --stop-min n --rand-moves n
 *
 * Compile: see build.sh (game2.c rave.c tune_rave.c).
 */
#include "game2.h"
#include "rave.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <limits.h>

/* ── env helpers ───────────────────────────────────────────────────────────── */
static float env_f(const char *name, float def) {
    const char *v = getenv(name); return v && *v ? (float)atof(v) : def;
}
static int env_i(const char *name, int def) {
    const char *v = getenv(name); return v && *v ? atoi(v) : def;
}

/* ── P(player win-rate > 0.5 | w wins of n), Beta(w+1,n-w+1) posterior ──────── */
static double log_gamma(double z) {
    static const double cof[6] = {
        76.18009172947146, -86.50532032941677, 24.01409824083091,
        -1.231739572450155, 0.001208650973866179, -0.000005395239384953 };
    double x = z, y = z, tmp = x + 5.5;
    tmp -= (x + 0.5) * log(tmp);
    double ser = 1.000000000190015;
    for (int j = 0; j < 6; j++) { y += 1; ser += cof[j] / y; }
    return -tmp + log(2.5066282746310005 * ser / x);
}
static double beta_cf(double x, double a, double b) {
    const int MAX_ITER = 100; const double EPS = 1e-12;
    double am = 1, bm = 1, az = 1;
    double qab = a + b, qap = a + 1, qam = a - 1;
    double bz = 1 - qab * x / qap;
    for (int m = 1; m <= MAX_ITER; m++) {
        double em = m, tem = em + em;
        double d = em * (b - em) * x / ((qam + tem) * (a + tem));
        double ap = az + d * am, bp = bz + d * bm;
        d = -(a + em) * (qab + em) * x / ((a + tem) * (qap + tem));
        double app = ap + d * az, bpp = bp + d * bz;
        double aold = az;
        am = ap / bpp; bm = bp / bpp; az = app / bpp; bz = 1;
        if (fabs(az - aold) < EPS * fabs(az)) return az;
    }
    return az;
}
static double reg_inc_beta(double x, double a, double b) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    double bt = exp(log_gamma(a + b) - log_gamma(a) - log_gamma(b)
                    + a * log(x) + b * log(1 - x));
    if (x < (a + 1) / (a + b + 2)) return bt * beta_cf(x, a, b) / a;
    return 1 - bt * beta_cf(1 - x, b, a) / b;
}
static double prob_better(int w, int n) {
    if (n <= 0) return 0.5;
    return 1.0 - reg_inc_beta(0.5, w + 1, n - w + 1);
}

/* ── one game from `opening`; A plays a_color.  +1 = A win, -1 = B win, 0 draw ─ */
static int play_game(const Game2 *opening, int8_t a_color, RaveState *s,
                     const RaveConfig *ca, const RaveConfig *cb, int po_a, int po_b,
                     Rng *rng) {
    Game2 g; g2_clone(&g, opening);
    while (!g.game_over) {
        int po;
        if (g.current == a_color) { rave_cfg = *ca; po = po_a; }
        else                      { rave_cfg = *cb; po = po_b; }
        RaveResult r = rave_search(s, &g, po, 0, rng);
        g2_play(&g, r.move);
    }
    int8_t winner = g2_estimate_winner(&g);
    if (winner == a_color)  return  1;
    if (winner == -a_color) return -1;
    return 0;
}

int main(int argc, char **argv) {
    int   size      = 9;
    long  limit     = LONG_MAX;
    double stop_tol  = -1;        /* <0 = no early stop */
    int   stop_min  = 0;
    int   rand_moves = 0;
    for (int i = 1; i < argc - 1; i++) {
        if      (!strcmp(argv[i], "--size"))       size       = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--limit"))      limit      = atol(argv[++i]);
        else if (!strcmp(argv[i], "--stop-tol"))   stop_tol   = atof(argv[++i]);
        else if (!strcmp(argv[i], "--stop-min"))   stop_min   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rand-moves")) rand_moves = atoi(argv[++i]);
    }

    int po_a = env_i("PLAYOUTS", 1000);
    int po_b = env_i("PLAYOUTS_B", po_a);
    RaveConfig ca = {
        .exploration_c = env_f("EXPLORATION_C", rave_cfg.exploration_c),
        .k             = env_f("RAVE_K",        rave_cfg.k),
        .n_expand      = env_i("N_EXPAND",      rave_cfg.n_expand),
        .inherit       = env_f("RAVE_INHERIT",  rave_cfg.inherit),
    };
    RaveConfig cb = {
        .exploration_c = env_f("EXPLORATION_C_B", rave_cfg.exploration_c),
        .k             = env_f("RAVE_K_B",        rave_cfg.k),
        .n_expand      = env_i("N_EXPAND_B",      rave_cfg.n_expand),
        .inherit       = env_f("RAVE_INHERIT_B",  rave_cfg.inherit),
    };

    Rng rng;
    rng_seed_entropy(&rng);
    g2_init_topology(size);
    RaveState *s = rave_create();

    printf("A: K=%.0f C=%.2f N=%d INH=%.2f po=%d   B: K=%.0f C=%.2f N=%d INH=%.2f po=%d   "
           "size=%d tol=%.3g min=%d\n",
           ca.k, ca.exploration_c, ca.n_expand, ca.inherit, po_a,
           cb.k, cb.exploration_c, cb.n_expand, cb.inherit, po_b,
           size, stop_tol, stop_min);
    printf("%6s  %8s  %7s  %9s\n", "games", "elapsed", "Awin%", "P(A>B)");
    fflush(stdout);

    long games = 0, winsA = 0, draws = 0;
    clock_t t0 = clock();
    double next_print = 0.5;
    int decided = 0;

    while (games < limit && !decided) {
        Game2 opening; g2_new(&opening, size);
        for (int i = 0; i < rand_moves && !opening.game_over; i++)
            g2_play(&opening, g2_random_legal_move(&opening, &rng));

        for (int swap = 0; swap < 2 && games < limit; swap++) {
            int8_t a_color = (swap == 0) ? BLACK : WHITE;
            int res = play_game(&opening, a_color, s, &ca, &cb, po_a, po_b, &rng);
            if      (res > 0) winsA++;
            else if (res == 0) draws++;
            games++;

            double pA = prob_better(winsA, games);
            double el = (double)(clock() - t0) / CLOCKS_PER_SEC;
            if (el >= next_print) {
                printf("%6ld  %7.1fs  %6.1f  %9.4f\n", games, el, 100.0 * winsA / games, pA);
                fflush(stdout);
                next_print = el * 1.5 + 0.5;
            }
            if (stop_tol >= 0 && games >= stop_min &&
                (pA >= 1 - stop_tol || pA <= stop_tol)) { decided = 1; break; }
        }
    }

    double pA = prob_better(winsA, games);
    double el = (double)(clock() - t0) / CLOCKS_PER_SEC;
    const char *verdict = (pA >= 1 - (stop_tol < 0 ? 0.02 : stop_tol)) ? "BETTER"
                        : (pA <= (stop_tol < 0 ? 0.02 : stop_tol))     ? "worse" : "tie";
    printf("RESULT games=%ld Awin=%.1f pbetter=%.4f draws=%ld elapsed=%.1fs verdict=%s\n",
           games, 100.0 * winsA / games, pA, draws, el, verdict);
    rave_destroy(s);
    return 0;
}
