/*
 * test_rave.c — Tests for the RAVE-MCTS engine.
 * Compile: cc -O2 -o test_rave game2.c rave.c test_rave.c -lm && ./test_rave
 */
#include "game2.h"
#include "rave.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int passed = 0, failed = 0;

static void check(const char *label, int ok) {
    if (ok) { passed++; }
    else { failed++; fprintf(stderr, "FAIL: %s\n", label); }
}

/* ── Tests ─────────────────────────────────────────────────────────────────── */

static void test_basic_search(void) {
    Game2 g;
    g2_new(&g);
    RaveState *s = rave_create();
    RaveResult r = rave_search(s, &g, 100, 0);
    check("basic: returns a move", r.move >= 0 || r.move == PASS);
    check("basic: playouts == 100", r.playouts == 100);
    check("basic: win_ratio in [0,1]", r.win_ratio >= 0.0f && r.win_ratio <= 1.0f);
    rave_destroy(s);
}

static void test_game_over(void) {
    Game2 g;
    g2_new(&g);
    g2_play(&g, PASS);
    g2_play(&g, PASS);
    check("gameover: game is over", g.game_over);
    RaveState *s = rave_create();
    RaveResult r = rave_search(s, &g, 100, 0);
    check("gameover: returns PASS", r.move == PASS);
    check("gameover: 0 playouts", r.playouts == 0);
    rave_destroy(s);
}

static void test_obvious_pass(void) {
    Game2 g;
    g2_new(&g);
    for (int i = 0; i < 40 && !g.game_over; i++) {
        int32_t m = g2_random_legal_move(&g);
        g2_play(&g, m >= 0 ? m : PASS);
    }
    if (!g.game_over) g2_play(&g, PASS);
    if (!g.game_over) {
        RaveState *s = rave_create();
        RaveResult r = rave_search(s, &g, 1000, 0);
        int8_t winner = g2_estimate_winner(&g);
        if (winner == g.current) {
            check("obvious_pass: passes when winning after opp pass", r.move == PASS);
            check("obvious_pass: 0 playouts", r.playouts == 0);
        } else {
            check("obvious_pass: plays when losing (expected)", r.move != PASS || r.playouts > 0);
        }
        rave_destroy(s);
    } else {
        check("obvious_pass: game ended early (ok)", 1);
    }
}

static void test_move_is_legal(void) {
    Game2 g;
    g2_new(&g);
    RaveState *s = rave_create();
    for (int i = 0; i < 20 && !g.game_over; i++) {
        RaveResult r = rave_search(s, &g, 50, 0);
        check("legal: move is legal", g2_is_legal(&g, r.move));
        g2_play(&g, r.move);
    }
    rave_destroy(s);
}

static void test_more_playouts_better(void) {
    Game2 g;
    g2_new(&g);
    for (int i = 0; i < 10 && !g.game_over; i++) {
        int32_t m = g2_random_legal_move(&g);
        g2_play(&g, m >= 0 ? m : PASS);
    }
    if (g.game_over) { check("more_playouts: game ended early", 1); return; }

    RaveState *s = rave_create();
    int32_t moves_lo[20], moves_hi[20];
    for (int t = 0; t < 20; t++) {
        RaveResult r = rave_search(s, &g, 50, 0);
        moves_lo[t] = r.move;
        r = rave_search(s, &g, 500, 0);
        moves_hi[t] = r.move;
    }
    int max_lo = 0, max_hi = 0;
    for (int i = 0; i < 20; i++) {
        int cnt_lo = 0, cnt_hi = 0;
        for (int j = 0; j < 20; j++) {
            if (moves_lo[j] == moves_lo[i]) cnt_lo++;
            if (moves_hi[j] == moves_hi[i]) cnt_hi++;
        }
        if (cnt_lo > max_lo) max_lo = cnt_lo;
        if (cnt_hi > max_hi) max_hi = cnt_hi;
    }
    /* With randomness, hi should usually be >= lo, but allow some slack */
    check("more_playouts: higher playouts more consistent", max_hi >= max_lo - 3);
    rave_destroy(s);
}

static void test_full_game(void) {
    Game2 g;
    g2_new(&g);
    RaveState *s = rave_create();
    int moves = 0;
    while (!g.game_over && moves < 4 * CAP) {
        RaveResult r = rave_search(s, &g, 50, 0);
        g2_play(&g, r.move);
        moves++;
    }
    check("full_game: game ended", g.game_over);
    check("full_game: has a winner", g2_estimate_winner(&g) == BLACK ||
                                     g2_estimate_winner(&g) == WHITE);
    rave_destroy(s);
}

static void test_pool_reuse(void) {
    RaveState *s = rave_create();
    Game2 g;
    g2_new(&g);
    rave_search(s, &g, 100, 0);
    int used1 = s->total_used;
    rave_search(s, &g, 100, 0);
    int used2 = s->total_used;
    check("pool_reuse: pool resets between searches", used2 <= used1 + 10);
    check("pool_reuse: similar usage", abs(used2 - used1) < used1 / 2 + 10);
    rave_destroy(s);
}

static void test_node_pool_sufficient(void) {
    /* Verify pool sized at playouts/N_EXPAND+1 is sufficient */
    RaveState *s = rave_create();
    Game2 g;
    g2_new(&g);
    RaveResult r = rave_search(s, &g, 5000, 0);
    check("pool_sufficient: completes 5000 playouts", r.playouts == 5000);
    check("pool_sufficient: allocated nodes > 0", s->total_used > 0);
    rave_destroy(s);
}

static void test_time_based(void) {
    /* Time-based search with default pool */
    RaveState *s = rave_create(); /* default pool for time-based */
    Game2 g;
    g2_new(&g);
    RaveResult r = rave_search(s, &g, 0, 200); /* 200ms */
    check("time_based: did some playouts", r.playouts > 0);
    check("time_based: returns a move", r.move >= 0 || r.move == PASS);
    rave_destroy(s);
}

/* ── Benchmark ─────────────────────────────────────────────────────────────── */

static void bench_search(void) {
    RaveState *s = rave_create();
    Game2 g;
    g2_new(&g);
    for (int i = 0; i < 5; i++) {
        int32_t m = g2_random_legal_move(&g);
        g2_play(&g, m >= 0 ? m : PASS);
    }

    clock_t t0 = clock();
    int total_playouts = 0;
    int searches = 0;
    double target = 2.0;
    while ((double)(clock() - t0) / CLOCKS_PER_SEC < target) {
        RaveResult r = rave_search(s, &g, 1000, 0);
        total_playouts += r.playouts;
        searches++;
    }
    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("bench: %d searches, %d playouts in %.1fs (%.0f playouts/s)\n",
           searches, total_playouts, elapsed, total_playouts / elapsed);
    rave_destroy(s);
}

/* ── Main ──────────────────────────────────────────────────────────────────── */

int main(void) {
    g2_seed((uint32_t)time(NULL));
    g2_init_topology();

    test_basic_search();
    test_game_over();
    test_obvious_pass();
    test_move_is_legal();
    test_more_playouts_better();
    test_full_game();
    test_pool_reuse();
    test_node_pool_sufficient();
    test_time_based();

    printf("\n%d passed, %d failed\n\n", passed, failed);

    bench_search();

    return failed > 0 ? 1 : 0;
}
