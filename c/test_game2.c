/*
 * test_game2.c — Tests for the C Game2 engine.
 * Compile: cc -O2 -o test_game2 game2.c test_game2.c && ./test_game2
 */
#include "game2.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int passed = 0, failed = 0;

static void check(const char *label, int ok) {
    if (ok) { passed++; }
    else { failed++; fprintf(stderr, "FAIL: %s\n", label); }
}

/* ── Helpers ───────────────────────────────────────────────────────────────── */

/* Play a sequence of moves given as flat indices; -1 = PASS. Returns true if all succeed. */
static bool play_seq(Game2 *g, const int32_t *moves, int n) {
    for (int i = 0; i < n; i++)
        if (!g2_play(g, moves[i])) return false;
    return true;
}

/* ── Tests ─────────────────────────────────────────────────────────────────── */

static void test_init(void) {
    Game2 g;
    g2_new(&g);
    int center = (BOARD_SIZE >> 1) * BOARD_SIZE + (BOARD_SIZE >> 1);
    check("init: center stone is BLACK", g.cells[center] == BLACK);
    check("init: current is WHITE", g.current == WHITE);
    check("init: move_count is 1", g.move_count == 1);
    check("init: empty_count is CAP-1", g.empty_count == CAP - 1);
    check("init: not game_over", !g.game_over);
    check("init: ko is PASS", g.ko == PASS);
}

static void test_init_empty(void) {
    Game2 g;
    g2_new_empty(&g);
    check("init_empty: current is BLACK", g.current == BLACK);
    check("init_empty: move_count is 0", g.move_count == 0);
    check("init_empty: empty_count is CAP", g.empty_count == CAP);
    int any_stone = 0;
    for (int i = 0; i < CAP; i++) if (g.cells[i] != EMPTY) any_stone = 1;
    check("init_empty: no stones", !any_stone);
}

static void test_toroidal_neighbors(void) {
    /* Top-left corner (0,0) = cell 0 */
    int N = BOARD_SIZE;
    check("topo: 0 N wraps", g2_nbr[0*4+0] == (N-1)*N + 0);
    check("topo: 0 W wraps", g2_nbr[0*4+2] == 0*N + (N-1));
    /* Bottom-right corner */
    int br = (N-1)*N + (N-1);
    check("topo: BR S wraps", g2_nbr[br*4+1] == 0*N + (N-1));
    check("topo: BR E wraps", g2_nbr[br*4+3] == (N-1)*N + 0);
}

static void test_play_and_capture(void) {
    Game2 g;
    g2_new_empty(&g);
    int N = BOARD_SIZE;
    /* Place B stone at center, surround with W, then W captures */
    int c = (N/2)*N + (N/2);
    int n0 = g2_nbr[c*4+0]; /* N */
    int n1 = g2_nbr[c*4+1]; /* S */
    int n2 = g2_nbr[c*4+2]; /* W */
    int n3 = g2_nbr[c*4+3]; /* E */

    check("play: B@center", g2_play(&g, c));       /* B */
    check("play: W@north",  g2_play(&g, n0));      /* W */
    check("play: B passes", g2_play(&g, PASS));    /* B pass */
    check("play: W@south",  g2_play(&g, n1));      /* W */
    check("play: B passes", g2_play(&g, PASS));    /* B pass */
    check("play: W@west",   g2_play(&g, n2));      /* W */
    check("play: B passes", g2_play(&g, PASS));    /* B pass — 3 consecutive? No, alternating */

    /* B at center now has 1 liberty (E). W plays E to capture. */
    check("pre-capture: B has 1 lib", g.ls[g.gid[c]] == 1);
    check("play: W@east captures", g2_play(&g, n3)); /* W captures B@center */
    check("capture: center is empty", g.cells[c] == EMPTY);
    check("capture: 1 stone captured", g.last_capture_count == 1);
}

static void test_suicide_illegal(void) {
    Game2 g;
    g2_new_empty(&g);
    int N = BOARD_SIZE;
    int c = (N/2)*N + (N/2);
    int n0 = g2_nbr[c*4+0];
    int n1 = g2_nbr[c*4+1];
    int n2 = g2_nbr[c*4+2];
    int n3 = g2_nbr[c*4+3];

    /* Surround center with W stones, then B tries to play center = suicide */
    g2_play(&g, 0);   /* B dummy */
    g2_play(&g, n0);  /* W */
    g2_play(&g, 1);   /* B dummy */
    g2_play(&g, n1);  /* W */
    g2_play(&g, 2);   /* B dummy */
    g2_play(&g, n2);  /* W */
    g2_play(&g, 3);   /* B dummy */
    g2_play(&g, n3);  /* W */

    /* Now B to play. Center is empty, surrounded by 4 W stones with >1 liberty each. */
    check("suicide: B@center is illegal", !g2_is_legal(&g, c));
    check("suicide: play returns false", !g2_play(&g, c));
}

static void test_ko(void) {
    /* Build a ko position.
     * On 9x9:  B@31, W@40, B@49, W@32, B@39, W@50, B@0(dummy), W@42
     *          B@41 captures W@40 → ko at 40.
     */
    Game2 g;
    g2_new_empty(&g);
#if BOARD_SIZE == 9
    int32_t seq[] = {31, 40, 49, 32, 39, 50, 0, 42};
    play_seq(&g, seq, 8);
    check("ko setup: W@40 in atari", g.ls[g.gid[40]] == 1);
    check("ko: B@41 captures", g2_play(&g, 41));
    check("ko: ko point set", g.ko == 40);
    check("ko: cell 40 empty", g.cells[40] == EMPTY);
    /* W cannot recapture at 40 immediately */
    check("ko: W@40 is illegal", !g2_is_legal(&g, 40));
    /* W plays elsewhere, then ko clears */
    g2_play(&g, 10); /* W elsewhere */
    g2_play(&g, 11); /* B elsewhere */
    check("ko: ko cleared after other moves", g.ko == PASS);
    check("ko: 40 now legal", g2_is_legal(&g, 40));
#else
    /* Simplified ko test for non-9x9 boards */
    (void)seq;
    check("ko: skipped (not 9x9)", 1);
#endif
}

static void test_consecutive_passes(void) {
    Game2 g;
    g2_new_empty(&g);
    g2_play(&g, PASS);
    check("pass: 1 consecutive", g.consecutive_passes == 1);
    check("pass: not game_over", !g.game_over);
    g2_play(&g, PASS);
    check("pass: 2 consecutive", g.consecutive_passes == 2);
    check("pass: game_over", g.game_over);
}

static void test_true_eye(void) {
    Game2 g;
    g2_new_empty(&g);
    int N = BOARD_SIZE;
    int c = (N/2)*N + (N/2);
    int n_n = g2_nbr[c*4+0];  /* N of center */
    int n_s = g2_nbr[c*4+1];  /* S */
    int n_w = g2_nbr[c*4+2];  /* W */
    int n_e = g2_nbr[c*4+3];  /* E */

    /* Build a connected B group around center by placing a ring.
     * Place N, then E of N (=NE of center), then E, then connect via S, W. */
    /* Place 4 ortho + 4 diag of center to ensure all same group. */
    int d0 = g2_dnbr[c*4+0]; /* NW */
    int d1 = g2_dnbr[c*4+1]; /* NE */
    int d2 = g2_dnbr[c*4+2]; /* SW */
    int d3 = g2_dnbr[c*4+3]; /* SE */

    /* All 8 neighbors form a connected ring on the toroidal board */
    int32_t bstones[] = {n_n, d1, n_e, d3, n_s, d2, n_w, d0};
    for (int i = 0; i < 8; i++) {
        g2_play(&g, bstones[i]); /* B */
        g2_play(&g, PASS);       /* W */
    }

    /* Verify all 4 ortho are BLACK and same group */
    int all_b = (g.cells[n_n] == BLACK && g.cells[n_s] == BLACK &&
                 g.cells[n_w] == BLACK && g.cells[n_e] == BLACK);
    check("eye setup: 4 B neighbors", all_b);
    int same = (g.gid[n_n] == g.gid[n_s] && g.gid[n_s] == g.gid[n_w] &&
                g.gid[n_w] == g.gid[n_e]);
    check("eye setup: all same group", same);
    check("eye: center is true eye", g2_is_true_eye(&g, c));
}

static void test_clone(void) {
    Game2 g, g2;
    g2_new(&g);
    g2_play(&g, 0);
    g2_play(&g, 1);
    g2_clone(&g2, &g);
    check("clone: same current", g2.current == g.current);
    check("clone: same move_count", g2.move_count == g.move_count);
    check("clone: same cells[0]", g2.cells[0] == g.cells[0]);
    check("clone: same empty_count", g2.empty_count == g.empty_count);
    /* Mutate clone, original unchanged */
    g2_play(&g2, 5);
    check("clone: original unchanged", g.cells[5] != g2.cells[5] || g.move_count != g2.move_count);
}

static void test_random_legal_move(void) {
    Game2 g;
    g2_new(&g);
    /* Play many random moves; game should eventually end */
    int moves = 0;
    while (!g.game_over && moves < 4 * CAP) {
        int32_t m = g2_random_legal_move(&g);
        g2_play(&g, m >= 0 ? m : PASS);
        moves++;
    }
    check("random: game ends", g.game_over);
    check("random: played some moves", moves > 0);
}

static void test_scoring(void) {
    Game2 g;
    g2_new_empty(&g);
    /* Play until game over, then check scoring doesn't crash */
    while (!g.game_over) {
        int32_t m = g2_random_legal_move(&g);
        g2_play(&g, m >= 0 ? m : PASS);
    }
    Score s = g2_estimate_score(&g);
    check("score: black >= 0", s.black >= 0);
    check("score: white >= 0", s.white >= 0);
    int8_t w = g2_estimate_winner(&g);
    check("score: winner is BLACK or WHITE", w == BLACK || w == WHITE);
}

static void test_is_capture(void) {
    Game2 g;
    g2_new_empty(&g);
    int N = BOARD_SIZE;
    int c = (N/2)*N + (N/2);
    int n0 = g2_nbr[c*4+0];
    int n1 = g2_nbr[c*4+1];
    int n2 = g2_nbr[c*4+2];
    int n3 = g2_nbr[c*4+3];

    /* B@center, surround with W on 3 sides */
    g2_play(&g, c);    /* B */
    g2_play(&g, n0);   /* W */
    g2_play(&g, PASS); /* B */
    g2_play(&g, n1);   /* W */
    g2_play(&g, PASS); /* B */
    g2_play(&g, n2);   /* W */
    /* B@center has 1 lib at n3. It's B's turn. */
    g2_play(&g, PASS); /* B */
    /* W's turn. n3 captures B@center. */
    check("is_capture: W@n3 captures", g2_is_capture(&g, n3));
}

static void test_group_tracking(void) {
    Game2 g;
    g2_new_empty(&g);
    int N = BOARD_SIZE;
    int c = (N/2)*N + (N/2);
    int e = g2_nbr[c*4+3]; /* E of center */

    /* B@center, W pass, B@east → should merge into one group */
    g2_play(&g, c);       /* B */
    g2_play(&g, PASS);    /* W */
    g2_play(&g, e);       /* B */
    check("group: same gid", g.gid[c] == g.gid[e]);
    check("group: size is 2", g.ss[g.gid[c]] == 2);
}

static void test_move_limit(void) {
    Game2 g;
    g2_new_empty(&g);
    /* Force game to move limit */
    int limit = 4 * CAP;
    for (int i = 0; i < limit && !g.game_over; i++) {
        int32_t m = g2_random_legal_move(&g);
        g2_play(&g, m >= 0 ? m : PASS);
    }
    check("move_limit: game_over", g.game_over);
}

static void test_consistency_random_games(void) {
    /* Play 100 full random games and verify invariants after each move */
    int ok = 1;
    for (int trial = 0; trial < 100 && ok; trial++) {
        Game2 g;
        g2_new(&g);
        while (!g.game_over) {
            int32_t m = g2_random_legal_move(&g);
            g2_play(&g, m >= 0 ? m : PASS);

            /* Check: empty_count matches actual empties */
            int actual_empty = 0;
            for (int i = 0; i < CAP; i++) if (g.cells[i] == EMPTY) actual_empty++;
            if (actual_empty != g.empty_count) { ok = 0; break; }

            /* Check: every stone has a valid gid */
            for (int i = 0; i < CAP; i++) {
                if (g.cells[i] != EMPTY && g.gid[i] == -1) { ok = 0; break; }
                if (g.cells[i] == EMPTY && g.gid[i] != -1) { ok = 0; break; }
            }
            if (!ok) break;

            /* Check: every group's liberty count matches its bitset */
            for (int i = 0; i < CAP; i++) {
                if (g.cells[i] == EMPTY) continue;
                int32_t gid = g.gid[i];
                int lc = 0;
                for (int wi = 0; wi < BW; wi++) lc += g2_popcount(g.lw[gid * BW + wi]);
                if (lc != g.ls[gid]) { ok = 0; break; }
            }
        }
    }
    check("consistency: 100 random games pass invariants", ok);
}

/* ── Performance benchmark ─────────────────────────────────────────────────── */

static void bench_random_games(void) {
    clock_t t0 = clock();
    int games = 0;
    int total_moves = 0;
    double target = 2.0; /* seconds */
    while ((double)(clock() - t0) / CLOCKS_PER_SEC < target) {
        Game2 g;
        g2_new(&g);
        while (!g.game_over) {
            int32_t m = g2_random_legal_move(&g);
            g2_play(&g, m >= 0 ? m : PASS);
            total_moves++;
        }
        games++;
    }
    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("bench: %d games, %d moves in %.1fs (%.0f moves/s, %.1f ms/game)\n",
           games, total_moves, elapsed,
           total_moves / elapsed,
           1000.0 * elapsed / games);
}

/* ── Main ──────────────────────────────────────────────────────────────────── */

int main(void) {
    g2_seed((uint32_t)time(NULL));
    g2_init_topology();

    test_init();
    test_init_empty();
    test_toroidal_neighbors();
    test_play_and_capture();
    test_suicide_illegal();
    test_ko();
    test_consecutive_passes();
    test_true_eye();
    test_clone();
    test_random_legal_move();
    test_scoring();
    test_is_capture();
    test_group_tracking();
    test_move_limit();
    test_consistency_random_games();

    printf("\n%d passed, %d failed\n\n", passed, failed);

    bench_random_games();

    return failed > 0 ? 1 : 0;
}
