/*
 * test_ladder2.c — Tests for ladder2.c (C port of ladder2.js).
 * Compile: cc -O2 -o test_ladder2 game2.c game3.c ladder2.c test_ladder2.c
 *          && ./test_ladder2
 */
#include "ladder2.h"
#include "game3.h"
#include "game2.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>   /* dup, dup2, close — for stderr-suppress in invalid-input test */

static int passed = 0, failed = 0;

static void check(const char *label, int ok) {
    if (ok) { passed++; }
    else    { failed++; fprintf(stderr, "FAIL: %s\n", label); }
}

/* ── Setup helpers ─────────────────────────────────────────────────────────── */

static bool play_seq(Game3 *g, const int32_t *moves, int n) {
    for (int i = 0; i < n; i++)
        if (!g3_play(g, moves[i])) return false;
    return true;
}

/* Light state-equality snapshot — enough to confirm ladder calls don't leak. */
typedef struct {
    int8_t  cells[256];
    int32_t op_count, move_count;
    int8_t  current;
    int32_t ko, last_move, empty_count;
} StateSnap;

static StateSnap snap(const Game3 *g) {
    StateSnap s;
    memcpy(s.cells, g->cells, g->cap);
    s.op_count    = g->op_count;
    s.move_count  = g->move_count;
    s.current     = g->current;
    s.ko          = g->ko;
    s.last_move   = g->last_move;
    s.empty_count = g->empty_count;
    return s;
}

static int snap_eq(const StateSnap *s, const Game3 *g) {
    if (memcmp(s->cells, g->cells, g->cap) != 0) return 0;
    return s->op_count    == g->op_count
        && s->move_count  == g->move_count
        && s->current     == g->current
        && s->ko          == g->ko
        && s->last_move   == g->last_move
        && s->empty_count == g->empty_count;
}

/* Build a 1-liberty BLACK group at 31 (lib=40), W stones surrounding except S.
 * After this call, `*mover_out` indicates who is to move next. */
static Game3 *make_atari_at_31(int8_t mover_should_be) {
    /* Plies:  31(B) 22(W) 0(B) 30(W) 1(B) 32(W) [optional 2(B) to flip mover] */
    Game3 *g = g3_new(9);
    int32_t base[] = { 31, 22, 0, 30, 1, 32 };
    play_seq(g, base, 6);
    /* After 6 plies, current = BLACK (defender to move). */
    if (mover_should_be == G3_WHITE) {
        g3_play(g, 2);  /* B plays elsewhere, now W to move */
    }
    return g;
}

/* ── ladder2_can_reach_3libs ───────────────────────────────────────────────── */

static void test_can_reach_3libs_already_safe(void) {
    /* B@40 on otherwise-empty board has 4 libs — trivially safe. */
    Game3 *g = g3_new(9);
    g3_play(g, 40);
    g3_play(g, G3_PASS);
    check("can_reach: 3+ libs trivially true", ladder2_can_reach_3libs(g, 40));
    g3_free(g);
}

static void test_can_reach_3libs_defender_escapes(void) {
    /* B@31 in atari (lib=40), B to move.  B@40 extends to 3 libs. */
    Game3 *g = make_atari_at_31(G3_BLACK);
    check("can_reach: defender extends out of atari",
          ladder2_can_reach_3libs(g, 31));
    g3_free(g);
}

static void test_can_reach_3libs_attacker_captures(void) {
    /* Same atari but W to move — attacker plays 40 and captures. */
    Game3 *g = make_atari_at_31(G3_WHITE);
    check("can_reach: attacker captures atari → false",
          !ladder2_can_reach_3libs(g, 31));
    g3_free(g);
}

static void test_can_reach_3libs_two_lib_defender_safe(void) {
    /* B@40 with W@31 and W@39 → libs {41,49}.  Defender (B) to move. */
    Game3 *g = g3_new(9);
    int32_t seq[] = { 40, 31, G3_PASS, 39 };
    play_seq(g, seq, 4);
    /* After 4 plies: current is B (defender), 2 libs.  Defender can extend
     * either way to multiple libs → escape = true. */
    check("can_reach: 2-lib defender extends to safety",
          ladder2_can_reach_3libs(g, 40));
    g3_free(g);
}

static void test_can_reach_3libs_state_restored(void) {
    Game3 *g = make_atari_at_31(G3_BLACK);
    StateSnap s = snap(g);
    (void)ladder2_can_reach_3libs(g, 31);
    check("can_reach: state restored", snap_eq(&s, g));
    g3_free(g);
}

/* ── ladder2_get_status ────────────────────────────────────────────────────── */

static void test_status_atari_defender_to_move(void) {
    Game3 *g = make_atari_at_31(G3_BLACK);
    Ladder2Status st = ladder2_get_status(g, 31);
    check("status atari-def: valid",      st.valid);
    check("status atari-def: lc=1",       st.lib_count == 1);
    check("status atari-def: lib0=40",    st.libs[0] == 40);
    check("status atari-def: mover_succeeds",
          st.mover_succeeds);
    check("status atari-def: urgent_count=1",
          st.urgent_count == 1);
    check("status atari-def: urgent_libs[0]=40",
          st.urgent_libs[0] == 40);
    g3_free(g);
}

static void test_status_atari_attacker_to_move(void) {
    /* W to move, B@31 in atari.  Attacker's winning move = 40 (captures).
     * From attacker's perspective: defender(B) cannot escape from atari with
     * attacker about to play → group is urgent → urgent_libs = [40]. */
    Game3 *g = make_atari_at_31(G3_WHITE);
    Ladder2Status st = ladder2_get_status(g, 31);
    check("status atari-atk: valid",      st.valid);
    check("status atari-atk: lc=1",       st.lib_count == 1);
    check("status atari-atk: mover_succeeds",
          st.mover_succeeds);
    check("status atari-atk: urgent={40}",
          st.urgent_count == 1 && st.urgent_libs[0] == 40);
    g3_free(g);
}

static void test_status_two_libs_not_urgent(void) {
    /* B@40 with W@31, W@39 → 2 libs.  Defender(B) to move.
     * Attacker playing first cannot kill (defender extends), so
     * defending == escape → mover_succeeds true, urgent_count = 0. */
    Game3 *g = g3_new(9);
    int32_t seq[] = { 40, 31, G3_PASS, 39 };
    play_seq(g, seq, 4);
    Ladder2Status st = ladder2_get_status(g, 40);
    check("status 2-lib: valid",          st.valid);
    check("status 2-lib: lc=2",           st.lib_count == 2);
    check("status 2-lib: mover_succeeds (not urgent)",
          st.mover_succeeds);
    check("status 2-lib: urgent_count=0", st.urgent_count == 0);
    g3_free(g);
}

static void test_status_invalid_too_many_libs(void) {
    Game3 *g = g3_new(9);
    g3_play(g, 40);            /* B@40 — 4 libs on empty board */
    g3_play(g, G3_PASS);
    /* Suppress the stderr warning during this test. */
    int saved_stderr = dup(fileno(stderr));
    FILE *devnull = freopen("/dev/null", "w", stderr);
    (void)devnull;
    Ladder2Status st = ladder2_get_status(g, 40);
    fflush(stderr);
    dup2(saved_stderr, fileno(stderr));
    close(saved_stderr);
    check("status invalid: valid=false on >2 libs", !st.valid);
    g3_free(g);
}

static void test_status_state_restored(void) {
    Game3 *g = make_atari_at_31(G3_BLACK);
    StateSnap s = snap(g);
    (void)ladder2_get_status(g, 31);
    check("status: state restored", snap_eq(&s, g));
    g3_free(g);
}

/* ── ladder2_get_all_statuses ──────────────────────────────────────────────── */

static void test_all_statuses_filters_by_libs(void) {
    /* Set up two groups:
     *   B@31 in atari (lib=40) — qualifies (1 lib)
     *   B@70 alone on otherwise-empty area — too many libs, skipped.
     * After 7 plies, W is to move; we want B to be to move so the
     * defender-to-move logic is exercised for B@31. */
    Game3 *g = g3_new(9);
    int32_t seq[] = { 31, 22, 70, 30, G3_PASS, 32 };
    play_seq(g, seq, 6);
    /* B@31 in atari (lib=40). B@70 has all 4 toroidal libs free. */
    check("setup: B@31 in atari", g->ls[g->gid[31]] == 1);
    check("setup: B@70 has 4 libs", g->ls[g->gid[70]] == 4);

    Ladder2Result out[81];
    int32_t n = ladder2_get_all_statuses(g, /*min*/1, out, 81);
    /* Expected qualifying groups:
     *   - B@31 (1 lib)
     *   - W@22 with 3 libs — too many, skipped.
     *   - W@30 with 3 libs — skipped.
     *   - W@32 with 3 libs — skipped.
     *   - B@70 with 4 libs — skipped.
     * So only B@31 should appear. */
    check("all: returned 1 result", n == 1);
    if (n >= 1) {
        check("all: color BLACK", out[0].color == G3_BLACK);
        check("all: gid matches B@31", out[0].gid == g->gid[31]);
        check("all: status valid",     out[0].status.valid);
        check("all: status lc=1",      out[0].status.lib_count == 1);
        check("all: status lib0=40",   out[0].status.libs[0] == 40);
    }
    g3_free(g);
}

static void test_all_statuses_min_chain_size(void) {
    /* Same setup as above but ask for min_chain_size=2: B@31 is a singleton
     * (size 1), so should be excluded. */
    Game3 *g = g3_new(9);
    int32_t seq[] = { 31, 22, 70, 30, G3_PASS, 32 };
    play_seq(g, seq, 6);
    Ladder2Result out[81];
    int32_t n = ladder2_get_all_statuses(g, /*min*/2, out, 81);
    check("all: min_chain_size=2 excludes singletons", n == 0);
    g3_free(g);
}

static void test_all_statuses_two_lib_group(void) {
    /* B@40 with W@31, W@39 → B has 2 libs (qualifies).
     * The W stones have many libs (skipped). */
    Game3 *g = g3_new(9);
    int32_t seq[] = { 40, 31, G3_PASS, 39 };
    play_seq(g, seq, 4);
    Ladder2Result out[81];
    int32_t n = ladder2_get_all_statuses(g, /*min*/1, out, 81);
    /* Only B@40 qualifies (2 libs). */
    check("all 2-lib: 1 result", n == 1);
    if (n >= 1) {
        check("all 2-lib: lc=2",      out[0].status.lib_count == 2);
        check("all 2-lib: BLACK",     out[0].color == G3_BLACK);
        check("all 2-lib: gid match", out[0].gid == g->gid[40]);
    }
    g3_free(g);
}

static void test_all_statuses_state_restored(void) {
    Game3 *g = g3_new(9);
    int32_t seq[] = { 31, 22, 70, 30, G3_PASS, 32 };
    play_seq(g, seq, 6);
    StateSnap s = snap(g);
    Ladder2Result out[81];
    (void)ladder2_get_all_statuses(g, 1, out, 81);
    check("all: state restored", snap_eq(&s, g));
    g3_free(g);
}

static void test_all_statuses_buffer_too_small(void) {
    /* The function must report the total count of qualifying groups even
     * when the caller's buffer is too small to hold them. */
    Game3 *g = g3_new(9);
    int32_t seq[] = { 31, 22, 0, 30, 1, 32 };   /* B@31 in atari, B to move */
    play_seq(g, seq, 6);
    Ladder2Result out[1];
    int32_t n = ladder2_get_all_statuses(g, /*min*/1, out, /*max*/0);
    check("all: max=0 still reports total", n == 1);
    g3_free(g);
}

/* ── Main ──────────────────────────────────────────────────────────────────── */

int main(void) {
    test_can_reach_3libs_already_safe();
    test_can_reach_3libs_defender_escapes();
    test_can_reach_3libs_attacker_captures();
    test_can_reach_3libs_two_lib_defender_safe();
    test_can_reach_3libs_state_restored();

    test_status_atari_defender_to_move();
    test_status_atari_attacker_to_move();
    test_status_two_libs_not_urgent();
    test_status_invalid_too_many_libs();
    test_status_state_restored();

    test_all_statuses_filters_by_libs();
    test_all_statuses_min_chain_size();
    test_all_statuses_two_lib_group();
    test_all_statuses_state_restored();
    test_all_statuses_buffer_too_small();

    printf("\n%d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
