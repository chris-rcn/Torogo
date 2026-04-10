/*
 * game2.c — Toroidal Go board engine (C port of game2.js).
 */
#include "game2.h"
#include <stdlib.h>

/* ── RNG ───────────────────────────────────────────────────────────────────── */

uint32_t g2_rng_state = 1;

void g2_seed(uint32_t seed) {
    g2_rng_state = seed ? seed : 1;  /* state must never be 0 */
}

/* ── Topology ──────────────────────────────────────────────────────────────── */

int32_t g2_nbr[CAP * 4];
int32_t g2_dnbr[CAP * 4];

void g2_init_topology(void) {
    const int N = BOARD_SIZE;
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            int i = y * N + x;
            g2_nbr[i*4+0] = ((y-1+N)%N)*N + x;          /* N */
            g2_nbr[i*4+1] = ((y+1  )%N)*N + x;          /* S */
            g2_nbr[i*4+2] = y*N + (x-1+N)%N;            /* W */
            g2_nbr[i*4+3] = y*N + (x+1  )%N;            /* E */
            g2_dnbr[i*4+0] = ((y-1+N)%N)*N + (x-1+N)%N; /* NW */
            g2_dnbr[i*4+1] = ((y-1+N)%N)*N + (x+1  )%N; /* NE */
            g2_dnbr[i*4+2] = ((y+1  )%N)*N + (x-1+N)%N; /* SW */
            g2_dnbr[i*4+3] = ((y+1  )%N)*N + (x+1  )%N; /* SE */
        }
    }
}

/* ── Internal: place a stone, merge groups ─────────────────────────────────── */

static int32_t g2_place(Game2 *g, int32_t idx, int8_t color) {
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

    /* Remove idx from liberties of all adjacent groups */
    {
        int32_t seen[4] = {-1, -1, -1, -1};
        int ns = 0;
        int base = idx * 4;
        uint32_t m  = 1u << (idx & 31);
        int      wi = idx >> 5;
        for (int i = 0; i < 4; i++) {
            int32_t ni  = g2_nbr[base + i];
            int32_t gid = g->gid[ni];
            if (gid == -1) continue;
            /* deduplicate */
            bool dup = false;
            for (int j = 0; j < ns; j++) if (seen[j] == gid) { dup = true; break; }
            if (dup) continue;
            seen[ns++] = gid;
            int32_t lb = gid * BW;
            if (g->lw[lb + wi] & m) { g->lw[lb + wi] &= ~m; g->ls[gid]--; }
        }
    }

    /* Collect same-color neighbor groups and own liberties */
    int32_t sg[4] = {-1, -1, -1, -1};
    int     ns = 0;
    int32_t ml[4] = {-1, -1, -1, -1};
    int     nml = 0;
    {
        int base = idx * 4;
        for (int i = 0; i < 4; i++) {
            int32_t ni = g2_nbr[base + i];
            int8_t  c  = g->cells[ni];
            if (c == color) {
                int32_t gid = g->gid[ni];
                bool dup = false;
                for (int j = 0; j < ns; j++) if (sg[j] == gid) { dup = true; break; }
                if (!dup) sg[ns++] = gid;
            } else if (c == EMPTY) {
                ml[nml++] = ni;
            }
        }
    }

    if (ns == 0) {
        /* New singleton group */
        int32_t gid = g->next_gid++;
        g->gid[idx] = gid;
        g->gc[gid]  = color;
        g->sw[gid * BW + (idx >> 5)] |= (1u << (idx & 31));
        g->ss[gid] = 1;
        int32_t lb = gid * BW;
        int lc = 0;
        for (int i = 0; i < nml; i++) {
            uint32_t m = 1u << (ml[i] & 31);
            int      w = ml[i] >> 5;
            if (!(g->lw[lb + w] & m)) { g->lw[lb + w] |= m; lc++; }
        }
        g->ls[gid] = lc;
        return gid;
    }

    /* Merge into largest same-color neighbor group */
    int32_t main_gid = sg[0];
    for (int i = 1; i < ns; i++)
        if (g->ss[sg[i]] > g->ss[main_gid]) main_gid = sg[i];

    int32_t gb = main_gid * BW;

    g->sw[gb + (idx >> 5)] |= (1u << (idx & 31));
    g->ss[main_gid]++;
    g->gid[idx] = main_gid;

    /* Add own empty neighbors as liberties */
    for (int i = 0; i < nml; i++) {
        uint32_t m = 1u << (ml[i] & 31);
        int      w = ml[i] >> 5;
        if (!(g->lw[gb + w] & m)) { g->lw[gb + w] |= m; g->ls[main_gid]++; }
    }

    /* Merge smaller groups into main */
    bool need_recount = false;
    for (int si = 0; si < ns; si++) {
        if (sg[si] == main_gid) continue;
        need_recount = true;
        int32_t other = sg[si];
        int32_t ob = other * BW;
        for (int wi = 0; wi < BW; wi++) {
            uint32_t w = g->sw[ob + wi];
            while (w) {
                int bit = __builtin_ctz(w);
                g->gid[wi * 32 + bit] = main_gid;
                w &= w - 1;
            }
            g->sw[gb + wi] |= g->sw[ob + wi];
        }
        g->ss[main_gid] += g->ss[other];
        for (int wi = 0; wi < BW; wi++)
            g->lw[gb + wi] |= g->lw[ob + wi];
    }

    if (need_recount) {
        int lc = 0;
        for (int wi = 0; wi < BW; wi++) lc += g2_popcount(g->lw[gb + wi]);
        g->ls[main_gid] = lc;
    }

    return main_gid;
}

/* ── Internal: remove a group from the board ───────────────────────────────── */

static int32_t g2_remove(Game2 *g, int32_t gid, int32_t ncap0) {
    int32_t sb = gid * BW;
    int32_t ncap = ncap0;

    for (int wi = 0; wi < BW; wi++) {
        uint32_t w = g->sw[sb + wi];
        while (w) {
            int bit = __builtin_ctz(w);
            int32_t idx = wi * 32 + bit;
            g->last_captures[ncap++] = idx;
            g->cells[idx] = EMPTY;
            g->gid[idx] = -1;
            /* Add back to empty list */
            g->empty_slot[idx] = g->empty_count;
            g->empty_cells[g->empty_count++] = idx;
            /* Restore liberty to adjacent groups */
            int base = idx * 4;
            uint32_t m  = 1u << (idx & 31);
            int      nwi = idx >> 5;
            for (int i = 0; i < 4; i++) {
                int32_t ngid = g->gid[g2_nbr[base + i]];
                if (ngid != -1 && ngid != gid) {
                    int32_t nlb = ngid * BW;
                    if (!(g->lw[nlb + nwi] & m)) { g->lw[nlb + nwi] |= m; g->ls[ngid]++; }
                }
            }
            w &= w - 1;
        }
    }
    return ncap;
}

/* ── Legality checks ──────────────────────────────────────────────────────── */

static bool g2_is_single_suicide(const Game2 *g, int32_t idx, int8_t color) {
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

static bool g2_is_multi_suicide(const Game2 *g, int32_t idx, int8_t color) {
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
            if (g->ls[gid] == 1 && !((g->lw[gid * BW + (idx >> 5)] >> (idx & 31)) & 1))
                return false;
        } else {
            if (g->ls[gid] == 1 && ((g->lw[gid * BW + (idx >> 5)] >> (idx & 31)) & 1))
                return false;
        }
    }
    return has_friendly;
}

static bool g2_is_ko(const Game2 *g, int32_t idx, int8_t color) {
    if (idx != g->ko) return false;
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
        if (g->ls[gid] == 1 && ((g->lw[gid * BW + (idx >> 5)] >> (idx & 31)) & 1))
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

static void g2_init_common(Game2 *g) {
    memset(g->cells, 0, sizeof(g->cells));
    memset(g->gid, 0xFF, sizeof(g->gid));  /* fill with -1 */
    g->next_gid = 0;
    memset(g->gc, 0, sizeof(g->gc));
    memset(g->ss, 0, sizeof(g->ss));
    memset(g->ls, 0, sizeof(g->ls));
    memset(g->sw, 0, sizeof(g->sw));
    memset(g->lw, 0, sizeof(g->lw));
    for (int i = 0; i < CAP; i++) {
        g->empty_cells[i] = i;
        g->empty_slot[i]  = i;
    }
    g->empty_count = CAP;
    g->current = BLACK;
    g->ko = PASS;
    g->ko_stone[0] = g->ko_stone[1] = g->ko_stone[2] = PASS;
    g->consecutive_passes = 0;
    g->game_over = false;
    g->move_count = 0;
    g->last_move = PASS;
    g->last_capture_count = 0;
}

void g2_new_empty(Game2 *g) {
    g2_init_common(g);
}

void g2_new(Game2 *g) {
    g2_init_common(g);
    int center = (BOARD_SIZE >> 1) * BOARD_SIZE + (BOARD_SIZE >> 1);
    g2_place(g, center, BLACK);
    g->current = WHITE;
    g->move_count = 1;
}

void g2_clone(Game2 *dst, const Game2 *src) {
    memcpy(dst, src, sizeof(Game2));
}

/* ── Parse ASCII board ─────────────────────────────────────────────────────── */

void g2_parse_board(Game2 *g, const char *board, int8_t to_move) {
    g2_init_common(g);
    /* Parse rows top-to-bottom (row N down to row 1). */
    const char *p = board;
    int row = BOARD_SIZE - 1;
    while (*p && row >= 0) {
        /* Skip leading whitespace and optional row number */
        while (*p == ' ' || *p == '\t') p++;
        if (*p >= '0' && *p <= '9') { while (*p >= '0' && *p <= '9') p++; while (*p == ' ') p++; }
        int col = 0;
        while (*p && *p != '\n' && col < BOARD_SIZE) {
            if (*p == 'X' || *p == 'O' || *p == '.') {
                if (*p == 'X') g2_place(g, row * BOARD_SIZE + col, BLACK);
                else if (*p == 'O') g2_place(g, row * BOARD_SIZE + col, WHITE);
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
            ncap = g2_remove(g, gid, ncap);
            if (g->ss[gid] == 1) captured_idx = g->last_captures[ncap_before];
            nchains++;
        }
    }
    g->last_capture_count = ncap;

    /* Ko detection */
    g->ko = PASS;
    g->ko_stone[color + 1] = PASS;
    if (ncap == 1 && captured_idx != PASS) {
        int32_t my_gid = g->gid[idx];
        if (g->ss[my_gid] == 1 && g->ls[my_gid] == 1 &&
            ((g->lw[my_gid * BW + (captured_idx >> 5)] >> (captured_idx & 31)) & 1)) {
            g->ko = captured_idx;
            g->ko_stone[color + 1] = idx;
        }
    }

    g->consecutive_passes = 0;
    g->current = opp;
    g->move_count++;
    g->last_move = idx;
    if (g->move_count >= 4 * CAP) g->game_over = true;
    return true;
}

/* ── Random legal move ─────────────────────────────────────────────────────── */

int32_t g2_random_legal_move(Game2 *g) {
    int ec = g->empty_count;
    for (int end = ec - 1; end >= 0; end--) {
        int ri  = g2_rand_n(end + 1);
        int32_t idx = g->empty_cells[ri];
        if (!g2_is_true_eye(g, idx) && g2_is_legal(g, idx)) return idx;
        /* Swap to back */
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
    float black = 0, white = 0;
    for (int i = 0; i < CAP; i++) {
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
    white += g2_komi();
    return (Score){ black, white };
}

int8_t g2_estimate_winner(const Game2 *g) {
    Score s = g2_estimate_score(g);
    return s.black > s.white ? BLACK : WHITE;
}
