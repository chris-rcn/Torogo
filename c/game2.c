/*
 * game2.c — Toroidal Go board engine (C port of game2.js).
 */
#include "game2.h"
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>

/* ── RNG ───────────────────────────────────────────────────────────────────── */

uint32_t g2_rng_state = 1;

void g2_seed(uint32_t seed) {
    g2_rng_state = seed ? seed : 1;
}

/* ── Topology ──────────────────────────────────────────────────────────────── */

int32_t g2_nbr[MAX_CAP * 4];
int32_t g2_dnbr[MAX_CAP * 4];

void g2_init_topology(int N) {
    if (N > MAX_BOARD_SIZE) {
        fprintf(stderr, "g2_init_topology: N=%d exceeds MAX_BOARD_SIZE=%d\n", N, MAX_BOARD_SIZE);
        exit(1);
    }
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            int i = y * N + x;
            g2_nbr[i*4+0] = ((y-1+N)%N)*N + x;
            g2_nbr[i*4+1] = ((y+1  )%N)*N + x;
            g2_nbr[i*4+2] = y*N + (x-1+N)%N;
            g2_nbr[i*4+3] = y*N + (x+1  )%N;
            g2_dnbr[i*4+0] = ((y-1+N)%N)*N + (x-1+N)%N;
            g2_dnbr[i*4+1] = ((y-1+N)%N)*N + (x+1  )%N;
            g2_dnbr[i*4+2] = ((y+1  )%N)*N + (x-1+N)%N;
            g2_dnbr[i*4+3] = ((y+1  )%N)*N + (x+1  )%N;
        }
    }
}

/* ── Komi ──────────────────────────────────────────────────────────────────── */

float g2_komi(int N) {
    switch (N) {
        case 5: return 24.5f;
        case 6: return 35.5f;
        case 7: return 48.5f;
        default: return 3.5f;
    }
}

/* ── Internal: place a stone, merge groups ─────────────────────────────────── */

static inline int32_t g2_alloc_gid(Game2 *g) {
    if (g->n_free_gids > 0)
        return (int32_t)g->free_gids[--g->n_free_gids];
    return g->next_gid++;
}

static inline void g2_free_gid(Game2 *g, int32_t gid) {
    g->free_gids[g->n_free_gids++] = (int16_t)gid;
}

static int32_t g2_place(Game2 *g, int32_t idx, int8_t color) {
    const int W = g->W;
    g->cells[idx] = color;
    g->empty_count--;

    /* Remove idx from empty-cell list (swap-and-pop) */
    {
        int32_t es = g->empty_slot[idx];
        int32_t el = g->empty_count;
        g->empty_slot[idx] = -1;
        if (es != el) {
            int32_t em = g->empty_cells[el];
            g->empty_cells[es] = em;
            g->empty_slot[em]  = es;
        }
    }

    /* Single pass over 4 neighbors: remove idx from liberties + collect groups and own liberties */
    int32_t sg[4] = {-1, -1, -1, -1};
    int     ns = 0;
    int32_t ml[4];
    int     nml = 0;
    {
        int base = idx * 4;
        uint32_t m  = 1u << (idx & 31);
        int      wi = idx >> 5;
        int32_t lib_seen[4] = {-1, -1, -1, -1};
        int nls = 0;
        for (int i = 0; i < 4; i++) {
            int32_t ni  = g2_nbr[base + i];
            int16_t gid = g->gid[ni];
            if (gid == -1) {
                /* EMPTY neighbor = own liberty */
                ml[nml++] = ni;
            } else {
                int8_t c = g->cells[ni];
                /* Remove idx from this group's liberties (dedup) */
                bool dup = false;
                for (int j = 0; j < nls; j++) if (lib_seen[j] == gid) { dup = true; break; }
                if (!dup) {
                    lib_seen[nls++] = gid;
                    int32_t lb = gid * W;
                    if (g->lw[lb + wi] & m) { g->lw[lb + wi] &= ~m; g->ls[gid]--; }
                }
                /* Collect same-color groups (dedup) */
                if (c == color) {
                    bool dup2 = false;
                    for (int j = 0; j < ns; j++) if (sg[j] == gid) { dup2 = true; break; }
                    if (!dup2) sg[ns++] = gid;
                }
            }
        }
    }

    if (ns == 0) {
        int32_t gid = g2_alloc_gid(g);
        g->gid[idx] = gid;
        g->gc[gid]  = color;
        /* Clear bitsets for reused slot */
        memset(&g->sw[gid * W], 0, W * sizeof(uint32_t));
        memset(&g->lw[gid * W], 0, W * sizeof(uint32_t));
        g->sw[gid * W + (idx >> 5)] |= (1u << (idx & 31));
        g->ss[gid] = 1;
        int32_t lb = gid * W;
        int lc = 0;
        for (int i = 0; i < nml; i++) {
            uint32_t m = 1u << (ml[i] & 31);
            int      w = ml[i] >> 5;
            if (!(g->lw[lb + w] & m)) { g->lw[lb + w] |= m; lc++; }
        }
        g->ls[gid] = lc;
        return gid;
    }

    int32_t main_gid = sg[0];
    for (int i = 1; i < ns; i++)
        if (g->ss[sg[i]] > g->ss[main_gid]) main_gid = sg[i];

    int32_t gb = main_gid * W;

    g->sw[gb + (idx >> 5)] |= (1u << (idx & 31));
    g->ss[main_gid]++;
    g->gid[idx] = main_gid;

    for (int i = 0; i < nml; i++) {
        uint32_t m = 1u << (ml[i] & 31);
        int      w = ml[i] >> 5;
        if (!(g->lw[gb + w] & m)) { g->lw[gb + w] |= m; g->ls[main_gid]++; }
    }

    bool need_recount = false;
    for (int si = 0; si < ns; si++) {
        if (sg[si] == main_gid) continue;
        need_recount = true;
        int32_t other = sg[si];
        int32_t ob = other * W;
        for (int wi = 0; wi < W; wi++) {
            uint32_t w = g->sw[ob + wi];
            while (w) {
                int bit = __builtin_ctz(w);
                g->gid[wi * 32 + bit] = main_gid;
                w &= w - 1;
            }
            g->sw[gb + wi] |= g->sw[ob + wi];
        }
        g->ss[main_gid] += g->ss[other];
        for (int wi = 0; wi < W; wi++)
            g->lw[gb + wi] |= g->lw[ob + wi];
        g2_free_gid(g, other);
    }

    if (need_recount) {
        int lc = 0;
        for (int wi = 0; wi < W; wi++) lc += g2_popcount(g->lw[gb + wi]);
        g->ls[main_gid] = lc;
    }

    return main_gid;
}

/* ── Internal: remove a group from the board ───────────────────────────────── */

static int32_t g2_remove(Game2 *g, int32_t gid, int32_t ncap0) {
    const int W = g->W;
    int32_t sb = gid * W;
    int32_t ncap = ncap0;

    for (int wi = 0; wi < W; wi++) {
        uint32_t w = g->sw[sb + wi];
        while (w) {
            int bit = __builtin_ctz(w);
            int32_t idx = wi * 32 + bit;
            g->last_captures[ncap++] = idx;
            g->cells[idx] = EMPTY;
            g->gid[idx] = -1;
            g->empty_slot[idx] = g->empty_count;
            g->empty_cells[g->empty_count++] = idx;
            int base = idx * 4;
            uint32_t m  = 1u << (idx & 31);
            int      nwi = idx >> 5;
            for (int i = 0; i < 4; i++) {
                int32_t ngid = g->gid[g2_nbr[base + i]];
                if (ngid != -1 && ngid != gid) {
                    int32_t nlb = ngid * W;
                    if (!(g->lw[nlb + nwi] & m)) { g->lw[nlb + nwi] |= m; g->ls[ngid]++; }
                }
            }
            w &= w - 1;
        }
    }
    return ncap;
}

/* ── Legality checks ──────────────────────────────────────────────────────── */

bool g2_is_single_suicide(const Game2 *g, int32_t idx, int8_t color) {
    int base = idx * 4;
    for (int i = 0; i < 4; i++) {
        int32_t ni = g2_nbr[base + i];
        int8_t  c  = g->cells[ni];
        if (c == EMPTY) return false;
        int32_t gid = g->gid[ni];
        if (c == color) { if (g->ls[gid] > 1) return false; }
        else            { if (g->ls[gid] == 1) return false; }
    }
    return true;
}

bool g2_is_multi_suicide(const Game2 *g, int32_t idx, int8_t color) {
    const int W = g->W;
    int base = idx * 4;
    bool has_friendly = false;
    int32_t seen[4] = {-1, -1, -1, -1};
    int ns = 0;
    for (int i = 0; i < 4; i++) {
        int32_t ni = g2_nbr[base + i];
        int8_t  c  = g->cells[ni];
        if (c == EMPTY) return false;
        int32_t gid = g->gid[ni];
        bool dup = false;
        for (int j = 0; j < ns; j++) if (seen[j] == gid) { dup = true; break; }
        if (dup) continue;
        seen[ns++] = gid;
        if (c == color) {
            has_friendly = true;
            if (g->ls[gid] > 1) return false;
            if (g->ls[gid] == 1 && !((g->lw[gid * W + (idx >> 5)] >> (idx & 31)) & 1))
                return false;
        } else {
            if (g->ls[gid] == 1 && ((g->lw[gid * W + (idx >> 5)] >> (idx & 31)) & 1))
                return false;
        }
    }
    return has_friendly;
}

bool g2_is_ko(const Game2 *g, int32_t idx, int8_t color) {
    if (idx != g->ko) return false;
    const int W = g->W;
    int8_t opp = -color;
    int base = idx * 4;
    int32_t seen[4] = {-1, -1, -1, -1};
    int ns = 0;
    int captured = 0;
    for (int i = 0; i < 4; i++) {
        int32_t ni = g2_nbr[base + i];
        if (g->cells[ni] != opp) continue;
        int32_t gid = g->gid[ni];
        bool dup = false;
        for (int j = 0; j < ns; j++) if (seen[j] == gid) { dup = true; break; }
        if (dup) continue;
        seen[ns++] = gid;
        if (g->ls[gid] == 1 && ((g->lw[gid * W + (idx >> 5)] >> (idx & 31)) & 1))
            captured += g->ss[gid];
    }
    return captured == 1;
}

bool g2_is_legal(const Game2 *g, int32_t idx) {
    if (idx == PASS) return true;
    if (g->cells[idx] != EMPTY) return false;
    int base = idx * 4;
    if (g->cells[g2_nbr[base]] == EMPTY || g->cells[g2_nbr[base+1]] == EMPTY ||
        g->cells[g2_nbr[base+2]] == EMPTY || g->cells[g2_nbr[base+3]] == EMPTY)
        return !g2_is_ko(g, idx, g->current);
    if (g2_is_single_suicide(g, idx, g->current)) return false;
    if (g2_is_multi_suicide(g, idx, g->current))  return false;
    if (g2_is_ko(g, idx, g->current))             return false;
    return true;
}

bool g2_is_capture(const Game2 *g, int32_t idx) {
    if (idx == PASS) return false;
    int8_t opp = -g->current;
    int base = idx * 4;
    for (int d = 0; d < 4; d++) {
        int32_t ni = g2_nbr[base + d];
        if (g->cells[ni] == opp && g->ls[g->gid[ni]] == 1) return true;
    }
    return false;
}

bool g2_is_true_eye(const Game2 *g, int32_t idx) {
    int8_t color = g->current;
    int base = idx * 4;
    int32_t first_gid = -2;
    int friend_count = 0, empty_count = 0, same_group = 0;
    for (int i = 0; i < 4; i++) {
        int32_t ni = g2_nbr[base + i];
        int8_t  c  = g->cells[ni];
        if (c == color) {
            friend_count++;
            int32_t gid = g->gid[ni];
            if (first_gid == -2) { first_gid = gid; same_group = 1; }
            else if (gid == first_gid) same_group++;
        } else if (c == EMPTY) {
            empty_count++;
        }
    }
    if (friend_count == 3 && empty_count == 1 && same_group == 3) return true;
    if (friend_count < 4) return false;
    if (same_group == 4) return true;
    int dc = 0;
    for (int i = 0; i < 4; i++)
        if (g->cells[g2_dnbr[base + i]] == color) dc++;
    return dc >= 3;
}

/* ── Initialization ────────────────────────────────────────────────────────── */

static void g2_init_common(Game2 *g, int N) {
    g->N   = N;
    g->cap = N * N;
    g->W   = (g->cap + 31) >> 5;
    memset(g->cells, 0, g->cap * sizeof(g->cells[0]));
    memset(g->gid, 0xFF, g->cap * sizeof(g->gid[0]));
    g->next_gid = 0;
    g->n_free_gids = 0;
    /* gc/ss/ls/sw/lw are zeroed per-group in g2_place, no bulk init needed */
    for (int i = 0; i < g->cap; i++) {
        g->empty_cells[i] = i;
        g->empty_slot[i]  = i;
    }
    g->empty_count = g->cap;
    g->current = BLACK;
    g->ko = PASS;
    g->ko_stone[0] = g->ko_stone[1] = g->ko_stone[2] = PASS;
    g->consecutive_passes = 0;
    g->game_over = false;
    g->move_count = 0;
    g->last_move = PASS;
    g->last_capture_count = 0;
}

void g2_new_empty(Game2 *g, int N) {
    g2_init_common(g, N);
}

void g2_new(Game2 *g, int N) {
    g2_init_common(g, N);
    int center = (N >> 1) * N + (N >> 1);
    g2_place(g, center, BLACK);
    g->current = WHITE;
    g->move_count = 1;
}

void g2_clone(Game2 *dst, const Game2 *src) {
    /* Copy everything up to (but not including) sw/lw */
    memcpy(dst, src, offsetof(Game2, sw));
    /* Copy only the used portion of sw and lw */
    int32_t used = src->next_gid * src->W;
    memcpy(dst->sw, src->sw, used * sizeof(uint32_t));
    memcpy(dst->lw, src->lw, used * sizeof(uint32_t));
}

/* ── Parse ASCII board ─────────────────────────────────────────────────────── */

void g2_parse_board(Game2 *g, const char *board, int8_t to_move) {
    /* First pass: count rows to determine board size */
    int size = 0;
    const char *p = board;
    while (*p) {
        while (*p == ' ' || *p == '\t') p++;
        if (*p >= '0' && *p <= '9') { while (*p >= '0' && *p <= '9') p++; while (*p == ' ') p++; }
        int col = 0;
        while (*p && *p != '\n') {
            if (*p == 'X' || *p == 'O' || *p == '.') col++;
            p++;
        }
        if (*p == '\n') p++;
        if (col > 0) size++;
    }

    g2_init_common(g, size);

    /* Second pass: place stones */
    p = board;
    int row = size - 1;
    while (*p && row >= 0) {
        while (*p == ' ' || *p == '\t') p++;
        if (*p >= '0' && *p <= '9') { while (*p >= '0' && *p <= '9') p++; while (*p == ' ') p++; }
        int col = 0;
        while (*p && *p != '\n' && col < size) {
            if (*p == 'X' || *p == 'O' || *p == '.') {
                if (*p == 'X') g2_place(g, row * size + col, BLACK);
                else if (*p == 'O') g2_place(g, row * size + col, WHITE);
                col++;
            }
            p++;
        }
        if (*p == '\n') p++;
        if (col > 0) row--;
    }
    g->current = to_move;
    g->ko = PASS;
    g->last_move = PASS;
}

/* ── Main move interface ───────────────────────────────────────────────────── */

bool g2_play(Game2 *g, int32_t idx) {
    if (g->game_over) return false;
    const int W = g->W;
    int8_t color = g->current;
    int8_t opp   = -color;

    if (idx == PASS) {
        g->consecutive_passes++;
        if (g->consecutive_passes >= 2) g->game_over = true;
        g->current = opp;
        g->ko = PASS;
        g->ko_stone[color + 1] = PASS;
        g->move_count++;
        g->last_move = PASS;
        g->last_capture_count = 0;
        return true;
    }

    if (!g2_is_legal(g, idx)) return false;

    g2_place(g, idx, color);

    int base = idx * 4;
    int32_t seen[4] = {-1, -1, -1, -1};
    int ns_seen = 0;
    int32_t captured_idx = PASS;
    int32_t ncap = 0, nchains = 0;

    for (int i = 0; i < 4; i++) {
        int32_t ni = g2_nbr[base + i];
        if (g->cells[ni] != opp) continue;
        int32_t gid = g->gid[ni];
        if (gid == -1) continue;
        bool dup = false;
        for (int j = 0; j < ns_seen; j++) if (seen[j] == gid) { dup = true; break; }
        if (dup) continue;
        seen[ns_seen++] = gid;
        if (g->ls[gid] == 0) {
            int32_t ncap_before = ncap;
            int32_t gsize = g->ss[gid];
            ncap = g2_remove(g, gid, ncap);
            if (gsize == 1) captured_idx = g->last_captures[ncap_before];
            nchains++;
            g2_free_gid(g, gid);
        }
    }
    g->last_capture_count = ncap;

    /* Ko detection */
    g->ko = PASS;
    g->ko_stone[color + 1] = PASS;
    if (ncap == 1 && captured_idx != PASS) {
        int32_t my_gid = g->gid[idx];
        if (g->ss[my_gid] == 1 && g->ls[my_gid] == 1 &&
            ((g->lw[my_gid * W + (captured_idx >> 5)] >> (captured_idx & 31)) & 1)) {
            g->ko = captured_idx;
            g->ko_stone[color + 1] = idx;
        }
    }

    g->consecutive_passes = 0;
    g->current = opp;
    g->move_count++;
    g->last_move = idx;
    if (g->move_count >= 4 * g->cap) g->game_over = true;
    return true;
}

void g2_play_unchecked(Game2 *g, int32_t idx) {
    const int W = g->W;
    int8_t color = g->current;
    int8_t opp   = -color;

    if (idx == PASS) {
        g->consecutive_passes++;
        if (g->consecutive_passes >= 2) g->game_over = true;
        g->current = opp;
        g->ko = PASS;
        g->ko_stone[color + 1] = PASS;
        g->move_count++;
        g->last_move = PASS;
        g->last_capture_count = 0;
        return;
    }

    g2_place(g, idx, color);

    int base = idx * 4;
    int32_t seen[4] = {-1, -1, -1, -1};
    int ns_seen = 0;
    int32_t captured_idx = PASS;
    int32_t ncap = 0, nchains = 0;

    for (int i = 0; i < 4; i++) {
        int32_t ni = g2_nbr[base + i];
        if (g->cells[ni] != opp) continue;
        int32_t gid = g->gid[ni];
        if (gid == -1) continue;
        bool dup = false;
        for (int j = 0; j < ns_seen; j++) if (seen[j] == gid) { dup = true; break; }
        if (dup) continue;
        seen[ns_seen++] = gid;
        if (g->ls[gid] == 0) {
            int32_t ncap_before = ncap;
            int32_t gsize = g->ss[gid];
            ncap = g2_remove(g, gid, ncap);
            if (gsize == 1) captured_idx = g->last_captures[ncap_before];
            nchains++;
            g2_free_gid(g, gid);
        }
    }
    g->last_capture_count = ncap;

    g->ko = PASS;
    g->ko_stone[color + 1] = PASS;
    if (ncap == 1 && captured_idx != PASS) {
        int32_t my_gid = g->gid[idx];
        if (g->ss[my_gid] == 1 && g->ls[my_gid] == 1 &&
            ((g->lw[my_gid * W + (captured_idx >> 5)] >> (captured_idx & 31)) & 1)) {
            g->ko = captured_idx;
            g->ko_stone[color + 1] = idx;
        }
    }

    g->consecutive_passes = 0;
    g->current = opp;
    g->move_count++;
    g->last_move = idx;
    if (g->move_count >= 4 * g->cap) g->game_over = true;
}

/* ── Random legal move ─────────────────────────────────────────────────────── */

int32_t g2_random_legal_move(Game2 *g) {
    int ec = g->empty_count;
    for (int end = ec - 1; end >= 0; end--) {
        int ri  = g2_rand_n(end + 1);
        int32_t idx = g->empty_cells[ri];
        if (!g2_is_true_eye(g, idx) && g2_is_legal(g, idx)) return idx;
        int32_t t = g->empty_cells[end];
        g->empty_cells[ri]  = t;
        g->empty_cells[end] = idx;
        g->empty_slot[t]   = ri;
        g->empty_slot[idx] = end;
    }
    return PASS;
}

/* ── Scoring ───────────────────────────────────────────────────────────────── */

Score g2_estimate_score(const Game2 *g) {
    const int cap = g->cap;
    float black = 0, white = 0;
    for (int i = 0; i < cap; i++) {
        int8_t c = g->cells[i];
        if (c == BLACK) { black++; continue; }
        if (c == WHITE) { white++; continue; }
        int base = i * 4;
        bool b_adj = false, w_adj = false;
        for (int k = 0; k < 4; k++) {
            int8_t nc = g->cells[g2_nbr[base + k]];
            if (nc == BLACK) b_adj = true;
            else if (nc == WHITE) w_adj = true;
        }
        if (b_adj && !w_adj) black++;
        else if (w_adj && !b_adj) white++;
    }
    white += g2_komi(g->N);
    return (Score){ black, white };
}

int8_t g2_estimate_winner(const Game2 *g) {
    Score s = g2_estimate_score(g);
    return s.black > s.white ? BLACK : WHITE;
}
