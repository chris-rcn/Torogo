/*
 * test_game3.c — Tests for the C Game3 engine (reversible undo).
 * Compile: cc -O2 -o test_game3 game2.c game3.c test_game3.c && ./test_game3
 */
#include "game3.h"
#include "game2.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static Rng rng;
static int passed = 0, failed = 0;

static void check(const char *label, int ok) {
    if (ok) { passed++; }
    else    { failed++; fprintf(stderr, "FAIL: %s\n", label); }
}

/* ── Helpers ───────────────────────────────────────────────────────────────── */

static bool play_seq(Game3 *g, const int32_t *moves, int n) {
    for (int i = 0; i < n; i++)
        if (!g3_play(g, moves[i])) return false;
    return true;
}

/* Capture every byte of "live" state that should round-trip exactly through
 * play+undo: board, scalar state, group slots in [0, next_gid).  Slots past
 * next_gid are scratch and intentionally ignored. */
typedef struct {
    int32_t  cap, W, next_gid;
    int8_t  *cells;
    int32_t *gid;
    uint8_t *gc;
    int32_t *ss;
    int32_t *ls;
    uint32_t *sw;
    uint32_t *lw;
    int8_t   current;
    int32_t  ko, empty_count, move_count, last_move, consecutive_passes;
    bool     game_over;
} Snapshot;

static Snapshot *snap_take(const Game3 *g) {
    Snapshot *s = (Snapshot *)calloc(1, sizeof(Snapshot));
    s->cap      = g->cap;
    s->W        = g->W;
    s->next_gid = g->next_gid;
    s->cells    = (int8_t  *)malloc(g->cap * sizeof(int8_t));
    s->gid      = (int32_t *)malloc(g->cap * sizeof(int32_t));
    memcpy(s->cells, g->cells, g->cap * sizeof(int8_t));
    memcpy(s->gid,   g->gid,   g->cap * sizeof(int32_t));

    const size_t bw = (size_t)g->next_gid * g->W;
    s->gc = (uint8_t  *)malloc(g->next_gid * sizeof(uint8_t));
    s->ss = (int32_t  *)malloc(g->next_gid * sizeof(int32_t));
    s->ls = (int32_t  *)malloc(g->next_gid * sizeof(int32_t));
    s->sw = (uint32_t *)malloc(bw * sizeof(uint32_t));
    s->lw = (uint32_t *)malloc(bw * sizeof(uint32_t));
    if (g->next_gid > 0) {
        memcpy(s->gc, g->gc, g->next_gid * sizeof(uint8_t));
        memcpy(s->ss, g->ss, g->next_gid * sizeof(int32_t));
        memcpy(s->ls, g->ls, g->next_gid * sizeof(int32_t));
    }
    if (bw > 0) {
        memcpy(s->sw, g->sw, bw * sizeof(uint32_t));
        memcpy(s->lw, g->lw, bw * sizeof(uint32_t));
    }

    s->current             = g->current;
    s->ko                  = g->ko;
    s->empty_count         = g->empty_count;
    s->move_count          = g->move_count;
    s->last_move           = g->last_move;
    s->consecutive_passes  = g->consecutive_passes;
    s->game_over           = g->game_over;
    return s;
}

static void snap_free(Snapshot *s) {
    if (!s) return;
    free(s->cells); free(s->gid);
    free(s->gc); free(s->ss); free(s->ls);
    free(s->sw); free(s->lw);
    free(s);
}

/* Returns NULL if equal, else a static string describing the first mismatch. */
static const char *snap_diff(const Snapshot *s, const Game3 *g) {
    static char msg[128];
    if (g->next_gid != s->next_gid) {
        snprintf(msg, sizeof(msg), "next_gid %d != %d", g->next_gid, s->next_gid);
        return msg;
    }
    if (g->current != s->current)           return "current";
    if (g->ko != s->ko)                     return "ko";
    if (g->empty_count != s->empty_count)   return "empty_count";
    if (g->move_count != s->move_count)     return "move_count";
    if (g->last_move != s->last_move)       return "last_move";
    if (g->consecutive_passes != s->consecutive_passes) return "consecutive_passes";
    if (g->game_over != s->game_over)       return "game_over";
    for (int32_t i = 0; i < g->cap; i++) {
        if (g->cells[i] != s->cells[i]) {
            snprintf(msg, sizeof(msg), "cells[%d] %d != %d", i, g->cells[i], s->cells[i]);
            return msg;
        }
        if (g->gid[i] != s->gid[i]) {
            snprintf(msg, sizeof(msg), "gid[%d] %d != %d", i, g->gid[i], s->gid[i]);
            return msg;
        }
    }
    for (int32_t k = 0; k < g->next_gid; k++) {
        if (g->gc[k] != s->gc[k]) { snprintf(msg, sizeof(msg), "gc[%d]", k); return msg; }
        if (g->ss[k] != s->ss[k]) { snprintf(msg, sizeof(msg), "ss[%d] %d != %d", k, g->ss[k], s->ss[k]); return msg; }
        if (g->ls[k] != s->ls[k]) { snprintf(msg, sizeof(msg), "ls[%d] %d != %d", k, g->ls[k], s->ls[k]); return msg; }
        for (int32_t wi = 0; wi < g->W; wi++) {
            if (g->sw[k * g->W + wi] != s->sw[k * g->W + wi]) {
                snprintf(msg, sizeof(msg), "sw[gid=%d,wi=%d]", k, wi);
                return msg;
            }
            if (g->lw[k * g->W + wi] != s->lw[k * g->W + wi]) {
                snprintf(msg, sizeof(msg), "lw[gid=%d,wi=%d]", k, wi);
                return msg;
            }
        }
    }
    return NULL;
}

/* Compare-and-report. */
static void check_snap_eq(const char *label, const Snapshot *s, const Game3 *g) {
    const char *d = snap_diff(s, g);
    if (d) {
        failed++;
        fprintf(stderr, "FAIL: %s — diff at %s\n", label, d);
    } else {
        passed++;
    }
}

/* ── Lifecycle / construction ──────────────────────────────────────────────── */

static void test_init(void) {
    Game3 *g = g3_new(9);
    check("init: N",                g->N == 9);
    check("init: cap",              g->cap == 81);
    check("init: W",                g->W == 3);          /* (81+31)/32 = 3 */
    check("init: max_g",            g->max_g == 4*81 + 4);
    check("init: current is BLACK", g->current == G3_BLACK);
    check("init: ko is PASS",       g->ko == G3_PASS);
    check("init: empty_count",      g->empty_count == 81);
    check("init: move_count",       g->move_count == 0);
    check("init: last_move PASS",   g->last_move == G3_PASS);
    check("init: not game_over",    !g->game_over);
    check("init: consecutive_passes 0", g->consecutive_passes == 0);
    check("init: next_gid 0",       g->next_gid == 0);
    int any_stone = 0, any_gid = 0;
    for (int i = 0; i < 81; i++) {
        if (g->cells[i] != G3_EMPTY) any_stone = 1;
        if (g->gid[i]   != -1)       any_gid   = 1;
    }
    check("init: no stones", !any_stone);
    check("init: no group ids", !any_gid);
    g3_free(g);
}

static void test_toroidal_neighbors(void) {
    Game3 *g = g3_new(9);
    const int N = 9;
    check("topo: nbr[0] N wraps", g->nbr[0*4+0] == (N-1)*N + 0);
    check("topo: nbr[0] S",       g->nbr[0*4+1] == 1*N + 0);
    check("topo: nbr[0] W wraps", g->nbr[0*4+2] == 0*N + (N-1));
    check("topo: nbr[0] E",       g->nbr[0*4+3] == 0*N + 1);
    int br = (N-1)*N + (N-1);
    check("topo: BR S wraps", g->nbr[br*4+1] == 0*N + (N-1));
    check("topo: BR E wraps", g->nbr[br*4+3] == (N-1)*N + 0);
    /* dnbr order: NW, NE, SW, SE */
    check("topo: dnbr[0] NW wraps to br", g->dnbr[0*4+0] == (N-1)*N + (N-1));
    check("topo: dnbr[0] NE wraps", g->dnbr[0*4+1] == (N-1)*N + 1);
    g3_free(g);
}

static void test_reset(void) {
    Game3 *g = g3_new(9);
    g3_play(g, 40);
    g3_play(g, 31);
    g3_play(g, G3_PASS);
    g3_reset(g);
    check("reset: empty_count",     g->empty_count == 81);
    check("reset: move_count",      g->move_count == 0);
    check("reset: current BLACK",   g->current == G3_BLACK);
    check("reset: next_gid 0",      g->next_gid == 0);
    check("reset: op_count 0",      g->op_count == 0);
    check("reset: capture_count 0", g->capture_count == 0);
    int any = 0;
    for (int i = 0; i < 81; i++) if (g->cells[i] != G3_EMPTY) any = 1;
    check("reset: no stones", !any);
    g3_free(g);
}

/* ── Basic play ────────────────────────────────────────────────────────────── */

static void test_play_alternates(void) {
    Game3 *g = g3_new(9);
    g3_play(g, 40);
    check("alt: BLACK at 40",        g->cells[40] == G3_BLACK);
    check("alt: current WHITE",      g->current == G3_WHITE);
    check("alt: last_move 40",       g->last_move == 40);
    check("alt: move_count 1",       g->move_count == 1);
    g3_play(g, 31);
    check("alt: WHITE at 31",        g->cells[31] == G3_WHITE);
    check("alt: current BLACK",      g->current == G3_BLACK);
    check("alt: move_count 2",       g->move_count == 2);
    /* Different groups */
    check("alt: distinct gids", g->gid[40] != g->gid[31]);
    g3_free(g);
}

static void test_illegal_returns_false(void) {
    Game3 *g = g3_new(9);
    g3_play(g, 40);
    Snapshot *s = snap_take(g);
    bool ok = g3_play(g, 40);   /* not empty */
    check("illegal: returns false", !ok);
    check_snap_eq("illegal: state unchanged", s, g);
    snap_free(s);
    g3_free(g);
}

/* ── Suicide & capture rules ───────────────────────────────────────────────── */

static void test_single_suicide(void) {
    Game3 *g = g3_new(9);
    const int c = 40;
    /* Surround center with WHITE; then BLACK to play center = single suicide. */
    int32_t seq[] = {
        0, g->nbr[c*4+0], 1, g->nbr[c*4+1],
        2, g->nbr[c*4+2], 3, g->nbr[c*4+3],
    };
    play_seq(g, seq, 8);
    check("suicide: BLACK@center illegal", !g3_is_legal(g, c));
    check("suicide: play returns false", !g3_play(g, c));
    g3_free(g);
}

static void test_multi_suicide_legal_capture(void) {
    /* A move that fills our last liberty but simultaneously captures an
     * opponent group should be legal. */
    Game3 *g = g3_new(9);
    /* B@31, W@32, W@22, W@40, B@33, W@23, B@30 → playing W@21 would be both
     * adjacent-capture and self-atari avoidance — but let's just test that
     * capture-and-continue is legal via a simpler ko-shaped setup. */
    int32_t seq[] = {31, 40, 49, 32, 39, 50, 0, 42};
    play_seq(g, seq, 8);
    /* Now B@41 captures W@40. */
    check("multi-suicide: B@41 legal (captures)", g3_is_legal(g, 41));
    check("multi-suicide: B@41 plays",            g3_play(g, 41));
    check("multi-suicide: W@40 captured",         g->cells[40] == G3_EMPTY);
    g3_free(g);
}

static void test_capture_single_stone(void) {
    Game3 *g = g3_new(9);
    const int c = 40;
    /* Surround BLACK@c with WHITE except for E, then capture. */
    g3_play(g, c);                     /* B */
    g3_play(g, g->nbr[c*4+0]);         /* W N */
    g3_play(g, G3_PASS);               /* B pass */
    g3_play(g, g->nbr[c*4+1]);         /* W S */
    g3_play(g, G3_PASS);               /* B pass */
    g3_play(g, g->nbr[c*4+2]);         /* W W */
    g3_play(g, G3_PASS);               /* B pass */
    check("capture-setup: B has 1 lib", g->ls[g->gid[c]] == 1);
    check("capture: W@east plays", g3_play(g, g->nbr[c*4+3]));
    check("capture: center empty", g->cells[c] == G3_EMPTY);
    check("capture: center gid -1", g->gid[c] == -1);
    check("capture: empty_count back up", g->empty_count == 81 - 4);  /* 4 W stones placed */
    g3_free(g);
}

/* ── Ko ────────────────────────────────────────────────────────────────────── */

static void test_ko(void) {
    Game3 *g = g3_new(9);
    int32_t seq[] = {31, 40, 49, 32, 39, 50, 0, 42};
    play_seq(g, seq, 8);
    check("ko setup: W@40 in atari", g->ls[g->gid[40]] == 1);
    check("ko: B@41 captures",       g3_play(g, 41));
    check("ko: ko point set",        g->ko == 40);
    check("ko: cell 40 empty",       g->cells[40] == G3_EMPTY);
    check("ko: W@40 illegal",        !g3_is_legal(g, 40));
    /* W plays elsewhere, B plays elsewhere → ko clears */
    g3_play(g, 10);
    g3_play(g, 11);
    check("ko: cleared after intervening moves", g->ko == G3_PASS);
    check("ko: 40 legal again",                 g3_is_legal(g, 40));
    g3_free(g);
}

/* ── Pass / game over ──────────────────────────────────────────────────────── */

static void test_pass_and_game_over(void) {
    Game3 *g = g3_new(9);
    g3_play(g, G3_PASS);
    check("pass: cp=1",          g->consecutive_passes == 1);
    check("pass: not game_over", !g->game_over);
    g3_play(g, G3_PASS);
    check("pass: cp=2",          g->consecutive_passes == 2);
    check("pass: game_over",     g->game_over);
    g3_free(g);
}

/* ── True eye ──────────────────────────────────────────────────────────────── */

static void test_true_eye(void) {
    Game3 *g = g3_new(9);
    const int c = 40;
    int32_t bstones[] = {
        g->nbr[c*4+0],  g->dnbr[c*4+1], g->nbr[c*4+3], g->dnbr[c*4+3],
        g->nbr[c*4+1],  g->dnbr[c*4+2], g->nbr[c*4+2], g->dnbr[c*4+0],
    };
    for (int i = 0; i < 8; i++) {
        g3_play(g, bstones[i]);
        g3_play(g, G3_PASS);
    }
    check("eye: all-4-ortho is true eye", g3_is_true_eye(g, c));
    g3_free(g);
}

/* ── Group queries ─────────────────────────────────────────────────────────── */

static void test_group_queries(void) {
    Game3 *g = g3_new(9);
    g3_play(g, 40);                     /* B */
    g3_play(g, G3_PASS);                /* W pass */
    g3_play(g, 31);                     /* B — connects to 40? not adjacent vertically? */
    /* 31 is N-of-40 on 9x9 — yes adjacent, joins into one group. */
    check("group: 40 and 31 same gid", g->gid[40] == g->gid[31]);
    check("group: size 2",             g3_group_size(g, g->gid[40]) == 2);
    G3GroupLibs2 gl2 = g3_group_libs2(g, 40);
    check("group: libs2 count > 0",   gl2.count > 0);
    check("group: lib0 valid",        gl2.lib0 >= 0 && gl2.lib0 < g->cap);
    /* Liberties should equal what g3_group_libs returns. */
    int32_t buf[16];
    int32_t n = g3_group_libs(g, 40, buf, 16);
    check("group: libs count matches liberty_count",
          n == g3_group_liberty_count(g, g->gid[40]));
    g3_free(g);
}

/* ── Undo ──────────────────────────────────────────────────────────────────── */

static void test_undo_empty_stack(void) {
    Game3 *g = g3_new(9);
    check("undo: false on empty stack", !g3_undo(g));
    g3_free(g);
}

static void test_undo_pass(void) {
    Game3 *g = g3_new(9);
    Snapshot *s = snap_take(g);
    g3_play(g, G3_PASS);
    g3_undo(g);
    check_snap_eq("undo PASS round-trips empty state", s, g);
    snap_free(s);
    g3_free(g);
}

static void test_undo_double_pass_clears_game_over(void) {
    Game3 *g = g3_new(9);
    g3_play(g, G3_PASS);
    Snapshot *s = snap_take(g);
    g3_play(g, G3_PASS);
    check("undo: game_over set after 2nd pass", g->game_over);
    g3_undo(g);
    check("undo: game_over cleared on undo", !g->game_over);
    check_snap_eq("undo 2nd PASS restores 1-pass state", s, g);
    snap_free(s);
    g3_free(g);
}

static void test_undo_simple_play(void) {
    Game3 *g = g3_new(9);
    Snapshot *s = snap_take(g);
    g3_play(g, 40);
    g3_undo(g);
    check_snap_eq("undo single play", s, g);
    snap_free(s);
    g3_free(g);
}

static void test_undo_merge(void) {
    /* Place two BLACK stones in separate groups, then connect them; undo
     * the connecting move must split them back. */
    Game3 *g = g3_new(9);
    g3_play(g, 30);                     /* B@30 (group A) */
    g3_play(g, G3_PASS);                /* W pass */
    g3_play(g, 32);                     /* B@32 (group B, separate from 30) */
    g3_play(g, G3_PASS);                /* W pass */
    check("merge pre: 30 and 32 distinct gids", g->gid[30] != g->gid[32]);
    int32_t gid30_before = g->gid[30];
    int32_t gid32_before = g->gid[32];
    Snapshot *s = snap_take(g);
    g3_play(g, 31);                     /* B@31 connects them */
    check("merge: all three same gid",
          g->gid[30] == g->gid[31] && g->gid[31] == g->gid[32]);
    g3_undo(g);
    check("merge undo: 30 gid restored",  g->gid[30] == gid30_before);
    check("merge undo: 32 gid restored",  g->gid[32] == gid32_before);
    check_snap_eq("undo merge fully restores state", s, g);
    snap_free(s);
    g3_free(g);
}

static void test_undo_three_way_merge(void) {
    /* B at 31, 39, 41 — three distinct groups around 40.  Playing B@40 merges
     * all three (plus 49 from below).  Undo splits them back. */
    Game3 *g = g3_new(9);
    g3_play(g, 31);                     /* B@31 */
    g3_play(g, G3_PASS);                /* W pass */
    g3_play(g, 39);                     /* B@39 */
    g3_play(g, G3_PASS);                /* W pass */
    g3_play(g, 41);                     /* B@41 */
    g3_play(g, G3_PASS);                /* W pass */
    g3_play(g, 49);                     /* B@49 */
    g3_play(g, G3_PASS);                /* W pass */
    int32_t g31 = g->gid[31], g39 = g->gid[39], g41 = g->gid[41], g49 = g->gid[49];
    check("3-way pre: 4 distinct gids",
          g31 != g39 && g39 != g41 && g41 != g49 &&
          g31 != g41 && g31 != g49 && g39 != g49);
    Snapshot *s = snap_take(g);
    g3_play(g, 40);                     /* B@40 — merges all four */
    check("3-way: all merged",
          g->gid[40] == g->gid[31] && g->gid[31] == g->gid[39] &&
          g->gid[39] == g->gid[41] && g->gid[41] == g->gid[49]);
    g3_undo(g);
    check("3-way undo: 31 restored", g->gid[31] == g31);
    check("3-way undo: 39 restored", g->gid[39] == g39);
    check("3-way undo: 41 restored", g->gid[41] == g41);
    check("3-way undo: 49 restored", g->gid[49] == g49);
    check_snap_eq("undo 4-way merge restores state", s, g);
    snap_free(s);
    g3_free(g);
}

static void test_undo_capture(void) {
    Game3 *g = g3_new(9);
    int32_t seq[] = {31, 40, 49, 32, 39, 50, 0, 42};
    play_seq(g, seq, 8);
    Snapshot *s = snap_take(g);
    g3_play(g, 41);                     /* B@41 captures W@40, ko at 40 */
    check("cap pre-undo: 40 empty", g->cells[40] == G3_EMPTY);
    check("cap pre-undo: ko at 40", g->ko == 40);
    g3_undo(g);
    check_snap_eq("undo capture restores full state (incl. captured stone)", s, g);
    snap_free(s);
    g3_free(g);
}

static void test_undo_multi_capture(void) {
    /* Capture a 2-stone group.  Setup: W@31, W@40 (vertical pair), surrounded by B. */
    Game3 *g = g3_new(9);
    g3_play(g, G3_PASS);                /* B pass */
    g3_play(g, 31);                     /* W@31 */
    g3_play(g, G3_PASS);                /* B pass */
    g3_play(g, 40);                     /* W@40 — merges with 31 */
    /* Now B surrounds the W pair on 5 sides; last B move = capture */
    g3_play(g, 22);                     /* B@22 (N of 31) */
    g3_play(g, G3_PASS);                /* W pass */
    g3_play(g, 30);                     /* B@30 (W of 31) */
    g3_play(g, G3_PASS);                /* W pass */
    g3_play(g, 32);                     /* B@32 (E of 31) */
    g3_play(g, G3_PASS);                /* W pass */
    g3_play(g, 39);                     /* B@39 (W of 40) */
    g3_play(g, G3_PASS);                /* W pass */
    g3_play(g, 41);                     /* B@41 (E of 40) */
    g3_play(g, G3_PASS);                /* W pass */
    check("multi-cap setup: W pair has 1 lib", g->ls[g->gid[31]] == 1);
    Snapshot *s = snap_take(g);
    g3_play(g, 49);                     /* B@49 captures W{31,40} */
    check("multi-cap: 31 empty", g->cells[31] == G3_EMPTY);
    check("multi-cap: 40 empty", g->cells[40] == G3_EMPTY);
    g3_undo(g);
    check("multi-cap undo: 31 restored to WHITE", g->cells[31] == G3_WHITE);
    check("multi-cap undo: 40 restored to WHITE", g->cells[40] == G3_WHITE);
    check("multi-cap undo: 31 and 40 same gid", g->gid[31] == g->gid[40]);
    check_snap_eq("undo multi-stone capture restores full state", s, g);
    snap_free(s);
    g3_free(g);
}

static void test_undo_full_game(void) {
    /* Play a sequence of moves, snapshot at each step, undo all the way back,
     * and confirm we land on the initial state. */
    Game3 *g = g3_new(9);
    Snapshot *initial = snap_take(g);

    /* Play a varied sequence: stones, pass, capture, merge */
    int32_t moves[] = {
        40, 31,             /* B@40, W@31 */
        49, 50,             /* B@49 (joins 40), W@50 (joins 31) */
        39, G3_PASS,        /* B@39 (joins 40), W pass */
        32, G3_PASS,        /* B@32, W pass */
        41, G3_PASS,        /* B@41, W pass */
        G3_PASS, G3_PASS,   /* B pass, W pass — game over */
    };
    const int n = sizeof(moves)/sizeof(moves[0]);

    Snapshot *snaps[32] = {0};
    snaps[0] = snap_take(g);
    for (int i = 0; i < n; i++) {
        bool ok = g3_play(g, moves[i]);
        check("full game: play", ok);
        snaps[i+1] = snap_take(g);
    }

    /* Undo back to start, checking against the snapshots taken along the way. */
    for (int i = n; i > 0; i--) {
        bool ok = g3_undo(g);
        check("full game: undo", ok);
        const char *d = snap_diff(snaps[i-1], g);
        if (d) {
            failed++;
            fprintf(stderr, "FAIL: full game undo step %d — diff at %s\n", i, d);
        } else passed++;
    }
    check_snap_eq("full game undo lands on initial state", initial, g);

    for (int i = 0; i <= n; i++) snap_free(snaps[i]);
    snap_free(initial);
    g3_free(g);
}

/* ── from_game2 round-trip ─────────────────────────────────────────────────── */

static void test_from_game2(void) {
    /* Build a Game2 position, port it into Game3, and verify cell-by-cell. */
    Game2 g2;
    g2_init_topology(9);
    g2_new_empty(&g2, 9);
    int32_t seq[] = {31, 40, 49, 32, 39, 50, 41, 42};
    for (int i = 0; i < (int)(sizeof(seq)/sizeof(seq[0])); i++) g2_play(&g2, seq[i]);

    Game3 *g3 = g3_new(9);
    bool ok = g3_from_game2(g3, &g2);
    check("from_game2: returns true", ok);
    int matches = 1;
    for (int i = 0; i < 81; i++) if (g3->cells[i] != g2.cells[i]) { matches = 0; break; }
    check("from_game2: cells match", matches);
    check("from_game2: current matches", g3->current == g2.current);
    check("from_game2: ko matches",      g3->ko == g2.ko);
    check("from_game2: last_move",       g3->last_move == g2.last_move);
    g3_free(g3);
}

/* ── Stress: random play+undo cycle ────────────────────────────────────────── */

/* Tiny xorshift so the test is deterministic. */
static uint32_t test_rng = 0xc0ffee;
static uint32_t test_rand(void) {
    uint32_t x = test_rng;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return test_rng = x;
}

static void test_stress_random_undo(void) {
    /* Play up to 60 random legal-or-pass moves, snapshotting at each step;
     * then undo back and verify each intermediate state. */
    Game3 *g = g3_new(9);
    const int max_moves = 60;
    Snapshot **snaps = (Snapshot **)calloc(max_moves + 1, sizeof(Snapshot *));
    snaps[0] = snap_take(g);

    int played = 0;
    for (int i = 0; i < max_moves && !g->game_over; i++) {
        /* Try a few random cells; if none legal, pass. */
        int32_t move = G3_PASS;
        for (int t = 0; t < 16; t++) {
            int32_t cand = test_rand() % 81;
            if (g3_is_legal(g, cand)) { move = cand; break; }
        }
        if (!g3_play(g, move)) {
            /* PASS is always legal unless game_over (and we guarded that). */
            check("stress: pass always playable", g3_play(g, G3_PASS));
        }
        played++;
        snaps[played] = snap_take(g);
    }

    int all_ok = 1;
    for (int i = played; i > 0; i--) {
        if (!g3_undo(g)) { all_ok = 0; break; }
        if (snap_diff(snaps[i-1], g)) { all_ok = 0; break; }
    }
    check("stress: random play+undo cycle preserves state", all_ok);

    for (int i = 0; i <= played; i++) snap_free(snaps[i]);
    free(snaps);
    g3_free(g);
}

/* ── Main ──────────────────────────────────────────────────────────────────── */

int main(void) {
    rng_seed_entropy(&rng);

    test_init();
    test_toroidal_neighbors();
    test_reset();

    test_play_alternates();
    test_illegal_returns_false();

    test_single_suicide();
    test_multi_suicide_legal_capture();
    test_capture_single_stone();

    test_ko();

    test_pass_and_game_over();
    test_true_eye();
    test_group_queries();

    test_undo_empty_stack();
    test_undo_pass();
    test_undo_double_pass_clears_game_over();
    test_undo_simple_play();
    test_undo_merge();
    test_undo_three_way_merge();
    test_undo_capture();
    test_undo_multi_capture();
    test_undo_full_game();

    test_from_game2();

    test_stress_random_undo();

    printf("\n%d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
