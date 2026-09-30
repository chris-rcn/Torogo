/* match.c — the ppat match metric's games (see match.h). */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "match.h"
#include "fpol.h"

struct Match {
    MatchCfg    cfg;
    PuctSearch *search[2];
    FpolState   fp;
    Game2       opening, game, trial;
};

Match *match_new(const MatchCfg *cfg) {
    Match *m = calloc(1, sizeof *m);
    m->cfg = *cfg;
    for (int i = 0; i < 2; i++) m->search[i] = puct_new(&cfg->side[i]);
    return m;
}

void match_free(Match *m) {
    if (!m) return;
    for (int i = 0; i < 2; i++) puct_free(m->search[i]);
    free(m);
}

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static double phase_of(const Game2 *g) { return 1.0 - (double)g->empty_count / (g->N * g->N); }

/* selfplay.js mirror180: the torus's point reflection, an exact automorphism. */
static int mirror180(int idx, int N) {
    const int x = idx % N, y = idx / N;
    return ((N - y) % N) * N + (N - x) % N;
}

static bool is_antisymmetric(const Game2 *g) {
    for (int i = 0; i < g->cap; i++)
        if (g->cells[i] != -g->cells[mirror180(i, g->N)]) return false;
    return true;
}

/* selfplay.js playMirrorPair: a random move and its rotation, validated on a
 * trial copy (a fixed point, an illegal mirror, or interacting captures redraw). */
static bool play_mirror_pair(Match *m, Game2 *g, Rng *rng) {
    for (int tries = 0; tries < 32; tries++) {
        const int32_t idx = g2_random_legal_move_js(g, rng);
        if (idx == PASS) return false;
        const int mi = mirror180(idx, g->N);
        if (mi == idx) continue;
        g2_clone(&m->trial, g);
        if (!g2_play(&m->trial, idx) || !g2_play(&m->trial, mi)) continue;
        if (!is_antisymmetric(&m->trial)) continue;
        g2_play(g, idx);
        g2_play(g, mi);
        return true;
    }
    return false;
}

static void play_or_die(Game2 *g, int32_t mv, const char *who) {
    if (!g2_play(g, mv)) { fprintf(stderr, "match: illegal %s move %d\n", who, mv); exit(1); }
}

static void build_opening(Match *m, Rng *rng, MatchStats *st) {
    const MatchCfg *c = &m->cfg;
    Game2 *g = &m->opening;
    if (c->mirror_pairs > 0) g2_new_empty(g, c->size); else g2_new(g, c->size);
    for (int i = 0; i < c->mirror_pairs && !g->game_over; i++)
        if (!play_mirror_pair(m, g, rng)) break;
    /* rff, both sides, up to min-phase */
    while (!g->game_over && phase_of(g) < c->min_phase) {
        fpol_eval(g, &m->fp, 1.0);
        play_or_die(g, fpol_sample(&m->fp, rng), "rff");
    }
    st->opening_moves += g->move_count;
}

/* One game from the opening; returns true when P2 wins. */
static bool play_game(Match *m, bool p1_black, const uint32_t seat_seed[2], MatchStats *st) {
    Game2 *g = &m->game;
    g2_clone(g, &m->opening);
    Rng seat[2];                                   /* [black, white] */
    for (int i = 0; i < 2; i++) rng_seed(&seat[i], seat_seed[i]);
    while (!g->game_over) {
        const bool black = g->current == BLACK;
        const int side = (black == p1_black) ? 0 : 1;
        const double t0 = now_ms();
        const PuctResult r = puct_get_move(m->search[side], g, m->cfg.playouts, &seat[black ? 0 : 1]);
        st->ms[side] += now_ms() - t0;
        st->moves[side]++;
        play_or_die(g, r.move, side ? "P2" : "P1");
    }
    st->total_moves += g->move_count;
    const bool black_won = g2_estimate_winner(g) == BLACK;
    st->black_wins += black_won;
    return black_won != p1_black;
}

int match_play_pair(Match *m, uint64_t seed, MatchStats *st) {
    g2_init_topology(m->cfg.size);
    Rng rng; rng_seed(&rng, (long)seed);
    build_opening(m, &rng, st);
    const uint32_t seat_seed[2] = { rng_next(&rng), rng_next(&rng) };
    int wins = 0;
    for (int swap = 0; swap < 2; swap++) {
        wins += play_game(m, swap == 0, seat_seed, st);
        st->games++;
    }
    st->p2_wins += wins;
    const double ps = wins / 2.0;
    st->pair_sum += ps; st->pair_sq += ps * ps; st->pairs++;
    return wins;
}

double match_p2_ratio(const MatchStats *st) { return st->games ? (double)st->p2_wins / st->games : 0; }

double match_p2_se(const MatchStats *st) {
    if (st->pairs < 2) return 0;
    const double mean = st->pair_sum / st->pairs;
    const double var = (st->pair_sq - st->pairs * mean * mean) / (st->pairs - 1);
    return var > 0 ? sqrt(var / st->pairs) : 0;
}
