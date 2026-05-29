/*
 * game3.c — Fully reversible toroidal Go engine (C port of game3.js).
 *
 * Mirrors game3.js operation-for-operation.  See game3.h for the type layout
 * and game3.js for the original commentary on the merge/undo invariants.
 */
#include "game3.h"
#include "game2.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Internal helpers ──────────────────────────────────────────────────────── */

static inline int g3_pop32(uint32_t x) { return __builtin_popcount(x); }

/* Lowest set bit position (count of trailing zeros). */
static inline int g3_ctz32(uint32_t x) { return __builtin_ctz(x); }

static void g3_build_topology(Game3 *g) {
    const int N = g->N;
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            const int i = y * N + x;
            g->nbr [i*4+0] = ((y - 1 + N) % N) * N + x;
            g->nbr [i*4+1] = ((y + 1    ) % N) * N + x;
            g->nbr [i*4+2] =  y * N + (x - 1 + N) % N;
            g->nbr [i*4+3] =  y * N + (x + 1    ) % N;
            g->dnbr[i*4+0] = ((y - 1 + N) % N) * N + (x - 1 + N) % N;
            g->dnbr[i*4+1] = ((y - 1 + N) % N) * N + (x + 1    ) % N;
            g->dnbr[i*4+2] = ((y + 1    ) % N) * N + (x - 1 + N) % N;
            g->dnbr[i*4+3] = ((y + 1    ) % N) * N + (x + 1    ) % N;
        }
    }
}

/* ── Word-buffer pool ──────────────────────────────────────────────────────── */

static uint32_t *g3_acquire_word_buffer(Game3 *g) {
    if (g->pool_count > 0) {
        return g->pool[--g->pool_count];
    }
    return (uint32_t *)calloc(g->W, sizeof(uint32_t));
}

static void g3_release_word_buffer(Game3 *g, uint32_t *buf) {
    if (g->pool_count >= g->pool_capacity) {
        int32_t new_cap = g->pool_capacity > 0 ? g->pool_capacity * 2 : 16;
        g->pool = (uint32_t **)realloc(g->pool, new_cap * sizeof(uint32_t *));
        g->pool_capacity = new_cap;
    }
    g->pool[g->pool_count++] = buf;
}

/* ── Op-stack growth ───────────────────────────────────────────────────────── */

static G3Op *g3_push_op(Game3 *g) {
    if (g->op_count >= g->op_capacity) {
        int32_t new_cap = g->op_capacity > 0 ? g->op_capacity * 2 : 1024;
        g->ops = (G3Op *)realloc(g->ops, new_cap * sizeof(G3Op));
        g->op_capacity = new_cap;
    }
    return &g->ops[g->op_count++];
}

static void g3_push_capture(Game3 *g, int32_t idx, int32_t gid) {
    if (g->capture_count >= g->capture_capacity) {
        int32_t new_cap = g->capture_capacity > 0 ? g->capture_capacity * 2 : 64;
        g->captures = (G3Capture *)realloc(g->captures, new_cap * sizeof(G3Capture));
        g->capture_capacity = new_cap;
    }
    g->captures[g->capture_count].idx = idx;
    g->captures[g->capture_count].gid = gid;
    g->capture_count++;
}

/* ── Raw ops (don't push records) ──────────────────────────────────────────── */

static inline void g3_add_stone_raw(Game3 *g, int32_t idx, int32_t gid) {
    g->sw[gid * g->W + (idx >> 5)] |= 1u << (idx & 31);
    g->ss[gid]++;
}

static inline void g3_remove_stone_raw(Game3 *g, int32_t idx, int32_t gid) {
    g->sw[gid * g->W + (idx >> 5)] &= ~(1u << (idx & 31));
    g->ss[gid]--;
}

static inline void g3_add_liberty_raw(Game3 *g, int32_t gid, int32_t idx) {
    g->lw[gid * g->W + (idx >> 5)] |= 1u << (idx & 31);
    g->ls[gid]++;
}

static inline void g3_remove_liberty_raw(Game3 *g, int32_t gid, int32_t idx) {
    g->lw[gid * g->W + (idx >> 5)] &= ~(1u << (idx & 31));
    g->ls[gid]--;
}

/* ── Recording ops ─────────────────────────────────────────────────────────── */

static void g3_add_stone(Game3 *g, int32_t idx, int32_t gid) {
    g3_add_stone_raw(g, idx, gid);
    G3Op *op = g3_push_op(g);
    op->type = G3_OP_ADD_STONE;
    op->stone.idx = idx;
    op->stone.gid = gid;
}

static void g3_remove_stone(Game3 *g, int32_t idx, int32_t gid) {
    g3_remove_stone_raw(g, idx, gid);
    G3Op *op = g3_push_op(g);
    op->type = G3_OP_REMOVE_STONE;
    op->stone.idx = idx;
    op->stone.gid = gid;
}

static void g3_add_liberty(Game3 *g, int32_t gid, int32_t idx) {
    uint32_t m = 1u << (idx & 31);
    uint32_t wi = (uint32_t)idx >> 5;
    bool was_set = (g->lw[gid * g->W + wi] & m) != 0;
    if (!was_set) {
        g3_add_liberty_raw(g, gid, idx);
        G3Op *op = g3_push_op(g);
        op->type = G3_OP_ADD_LIBERTY;
        op->liberty.gid = gid;
        op->liberty.idx = idx;
    }
}

static void g3_remove_liberty(Game3 *g, int32_t gid, int32_t idx) {
    uint32_t m = 1u << (idx & 31);
    uint32_t wi = (uint32_t)idx >> 5;
    bool was_set = (g->lw[gid * g->W + wi] & m) != 0;
    if (was_set) {
        g3_remove_liberty_raw(g, gid, idx);
        G3Op *op = g3_push_op(g);
        op->type = G3_OP_REMOVE_LIBERTY;
        op->liberty.gid = gid;
        op->liberty.idx = idx;
    }
}

static int32_t g3_lib_popcount(const Game3 *g, int32_t gid) {
    int32_t count = 0;
    int32_t gb = gid * g->W;
    for (int wi = 0; wi < g->W; wi++) count += g3_pop32(g->lw[gb + wi]);
    return count;
}

/* Merge `other_id` into `main_gid`.  See game3.js _mergeGroups for the
 * invariants that make this reversible without snapshotting the other group. */
static void g3_merge_groups(Game3 *g, int32_t main_gid, int32_t other_id) {
    const int W = g->W;
    const int gb = main_gid * W;
    const int ob = other_id * W;

    uint32_t *main_libs = g3_acquire_word_buffer(g);
    for (int wi = 0; wi < W; wi++) main_libs[wi] = g->lw[gb + wi];

    /* Merge stones */
    for (int wi = 0; wi < W; wi++) {
        uint32_t w = g->sw[ob + wi];
        while (w) {
            int bit = g3_ctz32(w);
            g->gid[wi * 32 + bit] = main_gid;
            w &= w - 1;
        }
        g->sw[gb + wi] |= g->sw[ob + wi];
    }
    g->ss[main_gid] += g->ss[other_id];

    /* Merge liberties (OR; counts recomputed via popcount) */
    for (int wi = 0; wi < W; wi++) g->lw[gb + wi] |= g->lw[ob + wi];
    g->ls[main_gid] = g3_lib_popcount(g, main_gid);

    G3Op *op = g3_push_op(g);
    op->type = G3_OP_MERGE_GROUPS;
    op->merge.main_gid  = main_gid;
    op->merge.other_id  = other_id;
    op->merge.main_libs = main_libs;
}

/* ── Legality ──────────────────────────────────────────────────────────────── */

bool g3_is_single_suicide(const Game3 *g, int32_t idx, int8_t color) {
    const int32_t base = idx * 4;
    for (int i = 0; i < 4; i++) {
        int32_t ni = g->nbr[base + i];
        int8_t  c  = g->cells[ni];
        if (c == G3_EMPTY) return false;
        int32_t gid = g->gid[ni];
        if (c == color) { if (g->ls[gid] >  1) return false; }
        else            { if (g->ls[gid] == 1) return false; }
    }
    return true;
}

bool g3_is_multi_suicide(const Game3 *g, int32_t idx, int8_t color) {
    const int W = g->W;
    const int32_t base = idx * 4;
    bool has_friendly = false;
    int32_t s0 = -1, s1 = -1, s2 = -1, s3 = -1;
    uint32_t m  = 1u << (idx & 31);
    uint32_t wi = (uint32_t)idx >> 5;
    for (int i = 0; i < 4; i++) {
        int32_t ni = g->nbr[base + i];
        int8_t  c  = g->cells[ni];
        if (c == G3_EMPTY) return false;
        int32_t gid = g->gid[ni];
        if (gid == s0 || gid == s1 || gid == s2 || gid == s3) continue;
        if      (s0 == -1) s0 = gid;
        else if (s1 == -1) s1 = gid;
        else if (s2 == -1) s2 = gid;
        else               s3 = gid;
        if (c == color) {
            has_friendly = true;
            if (g->ls[gid] > 1) return false;
            if (g->ls[gid] == 1 && !(g->lw[gid * W + wi] & m)) return false;
        } else {
            if (g->ls[gid] == 1 && (g->lw[gid * W + wi] & m)) return false;
        }
    }
    return has_friendly;
}

bool g3_is_legal_for(const Game3 *g, int32_t idx, int8_t color) {
    if (idx == G3_PASS) return !g->game_over;
    if (g->cells[idx] != G3_EMPTY) return false;
    if (g->ko == idx) return false;
    if (g3_is_single_suicide(g, idx, color)) return false;
    if (g3_is_multi_suicide (g, idx, color)) return false;
    return true;
}

bool g3_is_legal(const Game3 *g, int32_t idx) {
    return g3_is_legal_for(g, idx, g->current);
}

bool g3_is_true_eye(const Game3 *g, int32_t idx) {
    const int8_t color = g->current;
    const int32_t base = idx * 4;
    int32_t first_gid = -2;
    int friend_count = 0, empty_count = 0, same_group = 0;
    for (int i = 0; i < 4; i++) {
        int32_t ni = g->nbr[base + i];
        int8_t  c  = g->cells[ni];
        if (c == color) {
            friend_count++;
            int32_t gid = g->gid[ni];
            if (first_gid == -2) { first_gid = gid; same_group = 1; }
            else if (gid == first_gid) same_group++;
        } else if (c == G3_EMPTY) {
            empty_count++;
        }
    }
    if (friend_count == 3 && empty_count == 1 && same_group == 3) return true;
    if (friend_count < 4) return false;
    if (same_group == 4) return true;
    int dc = 0;
    for (int i = 0; i < 4; i++) if (g->cells[g->dnbr[base + i]] == color) dc++;
    return dc >= 3;
}

bool g3_is_valid_move(const Game3 *g, int32_t idx) {
    return g3_is_legal(g, idx) && !g3_is_true_eye(g, idx);
}

/* ── Play ──────────────────────────────────────────────────────────────────── */

bool g3_play(Game3 *g, int32_t move) {
    if (move == G3_PASS) {
        G3Op *op = g3_push_op(g);
        op->type = G3_OP_PASS;
        op->pass.previous_current             = g->current;
        op->pass.previous_consecutive_passes  = g->consecutive_passes;
        g->consecutive_passes++;
        if (g->consecutive_passes >= 2) g->game_over = true;
        g->ko        = G3_PASS;
        g->current   = -g->current;
        g->move_count++;
        return true;
    }

    if (!g3_is_legal(g, move)) return false;

    const int8_t  color    = g->current;
    const int8_t  opp      = -color;
    const int32_t base     = move * 4;
    const int     W        = g->W;

    const int32_t op_count_before     = g->op_count;
    const int32_t captures_start      = g->capture_count;
    const int32_t previous_ko         = g->ko;
    const int32_t previous_empty      = g->empty_count;
    const int32_t previous_cp         = g->consecutive_passes;
    const int32_t previous_last_move  = g->last_move;
    const int32_t previous_next_gid   = g->next_gid;

    /* Step 1: place the stone (board mutation only — gid set below). */
    g->cells[move] = color;
    g->empty_count--;
    g->last_move   = move;

    /* Step 2: remove `move` from each adjacent opponent group's liberties. */
    int32_t opp_gids[4]; int n_opp = 0;
    for (int i = 0; i < 4; i++) {
        int32_t ni = g->nbr[base + i];
        if (g->cells[ni] == opp) {
            int32_t gid = g->gid[ni];
            if (gid != -1) {
                bool dup = false;
                for (int j = 0; j < n_opp; j++) if (opp_gids[j] == gid) { dup = true; break; }
                if (!dup) { opp_gids[n_opp++] = gid; g3_remove_liberty(g, gid, move); }
            }
        }
    }

    /* Step 3: collect same-color neighbor groups + empty neighbors. */
    int32_t same_gids[4];     int n_same = 0;
    int32_t empty_nbrs[4];    int n_empty = 0;
    for (int i = 0; i < 4; i++) {
        int32_t ni = g->nbr[base + i];
        int8_t  c  = g->cells[ni];
        if (c == color) {
            int32_t gid = g->gid[ni];
            bool dup = false;
            for (int j = 0; j < n_same; j++) if (same_gids[j] == gid) { dup = true; break; }
            if (!dup) same_gids[n_same++] = gid;
        } else if (c == G3_EMPTY) {
            bool dup = false;
            for (int j = 0; j < n_empty; j++) if (empty_nbrs[j] == ni) { dup = true; break; }
            if (!dup) empty_nbrs[n_empty++] = ni;
        }
    }

    /* Step 4: create or merge into a main group. */
    int32_t main_gid;
    if (n_same == 0) {
        main_gid = g->next_gid++;
        g->gc[main_gid] = (uint8_t)color;
        /* Reset the slot's stats so they start at 0 (next_gid is monotonic but
         * a slot may still hold stale data from before the previous undo). */
        g->ss[main_gid] = 0;
        g->ls[main_gid] = 0;
        for (int wi = 0; wi < W; wi++) {
            g->sw[main_gid * W + wi] = 0;
            g->lw[main_gid * W + wi] = 0;
        }
        g->gid[move] = main_gid;
        g3_add_stone(g, move, main_gid);
    } else {
        /* Pick the largest of the neighboring same-color groups as main. */
        main_gid = same_gids[0];
        for (int i = 1; i < n_same; i++) {
            if (g->ss[same_gids[i]] > g->ss[main_gid]) main_gid = same_gids[i];
        }
        g->gid[move] = main_gid;
        g3_add_stone(g, move, main_gid);
        for (int i = 0; i < n_same; i++) {
            int32_t other = same_gids[i];
            if (other != main_gid) g3_merge_groups(g, main_gid, other);
        }
        /* The placed cell is no longer a liberty of the merged group. */
        g3_remove_liberty(g, main_gid, move);
    }

    /* Step 5: add the placed stone's empty neighbors to our liberties. */
    for (int i = 0; i < n_empty; i++) g3_add_liberty(g, main_gid, empty_nbrs[i]);

    /* Step 6: capture any opponent groups now at 0 liberties. */
    int captured_total = 0;
    for (int oi = 0; oi < n_opp; oi++) {
        int32_t opp_gid = opp_gids[oi];
        if (g->ls[opp_gid] == 0) {
            int32_t gb = opp_gid * W;
            for (int wi = 0; wi < W; wi++) {
                uint32_t w = g->sw[gb + wi];
                while (w) {
                    int bit = g3_ctz32(w);
                    int32_t stone_idx = wi * 32 + bit;
                    g->cells[stone_idx] = G3_EMPTY;
                    g->gid[stone_idx]   = -1;
                    g3_remove_stone(g, stone_idx, opp_gid);
                    g3_push_capture(g, stone_idx, opp_gid);
                    g->empty_count++;
                    captured_total++;

                    /* Restore liberties to any adjacent surviving groups. */
                    const int32_t sb = stone_idx * 4;
                    for (int j = 0; j < 4; j++) {
                        int32_t ngid = g->gid[g->nbr[sb + j]];
                        if (ngid != -1 && ngid != opp_gid)
                            g3_add_liberty(g, ngid, stone_idx);
                    }
                    w &= w - 1;
                }
            }
        }
    }

    /* Step 7: detect ko (single stone captured, lone friendly stone with 1
     * liberty that is exactly the captured cell). */
    g->ko = G3_PASS;
    if (captured_total == 1) {
        int32_t cap_idx = g->captures[captures_start].idx;
        if (g->ss[main_gid] == 1 && g->ls[main_gid] == 1) {
            uint32_t m  = 1u << (cap_idx & 31);
            uint32_t wi = (uint32_t)cap_idx >> 5;
            if (g->lw[main_gid * W + wi] & m) g->ko = cap_idx;
        }
    }

    /* Wrap with a MOVE record. */
    G3Op *op = g3_push_op(g);
    op->type = G3_OP_MOVE;
    op->move.move                         = move;
    op->move.color                        = color;
    op->move.previous_current             = color;
    op->move.previous_ko                  = previous_ko;
    op->move.previous_empty_count         = previous_empty;
    op->move.previous_consecutive_passes  = previous_cp;
    op->move.previous_last_move           = previous_last_move;
    op->move.previous_next_gid            = previous_next_gid;
    op->move.ops_start                    = op_count_before;
    op->move.captures_start               = captures_start;
    op->move.captures_count               = captured_total;

    g->consecutive_passes = 0;
    g->current            = opp;
    g->move_count++;
    return true;
}

/* ── Undo ──────────────────────────────────────────────────────────────────── */

static void g3_undo_one(Game3 *g, const G3Op *op) {
    const int W = g->W;
    switch (op->type) {
        case G3_OP_ADD_STONE:
            g3_remove_stone_raw(g, op->stone.idx, op->stone.gid);
            break;
        case G3_OP_REMOVE_STONE:
            g3_add_stone_raw(g, op->stone.idx, op->stone.gid);
            break;
        case G3_OP_ADD_LIBERTY:
            g3_remove_liberty_raw(g, op->liberty.gid, op->liberty.idx);
            break;
        case G3_OP_REMOVE_LIBERTY:
            g3_add_liberty_raw(g, op->liberty.gid, op->liberty.idx);
            break;
        case G3_OP_MERGE_GROUPS: {
            const int gb = op->merge.main_gid * W;
            const int ob = op->merge.other_id * W;
            for (int wi = 0; wi < W; wi++) {
                /* Move other's stones' gid entries back. */
                uint32_t w = g->sw[ob + wi];
                while (w) {
                    int bit = g3_ctz32(w);
                    g->gid[wi * 32 + bit] = op->merge.other_id;
                    w &= w - 1;
                }
                /* Remove other's stones from main's bitset (groups disjoint). */
                g->sw[gb + wi] &= ~g->sw[ob + wi];
                /* Restore main's liberty bitset from the snapshot. */
                g->lw[gb + wi] = op->merge.main_libs[wi];
            }
            g->ss[op->merge.main_gid] -= g->ss[op->merge.other_id];
            g->ls[op->merge.main_gid] = g3_lib_popcount(g, op->merge.main_gid);
            g3_release_word_buffer(g, op->merge.main_libs);
            break;
        }
        default:
            /* MOVE and PASS markers are handled in g3_undo, not here. */
            break;
    }
}

bool g3_undo(Game3 *g) {
    if (g->op_count == 0) return false;

    while (g->op_count > 0) {
        G3Op op = g->ops[--g->op_count];

        if (op.type == G3_OP_PASS) {
            g->current             = op.pass.previous_current;
            g->consecutive_passes  = op.pass.previous_consecutive_passes;
            if (g->consecutive_passes < 2) g->game_over = false;
            g->move_count--;
            return true;
        }

        if (op.type == G3_OP_MOVE) {
            /* Undo all interior ops back to the move's boundary. */
            while (g->op_count > op.move.ops_start) {
                G3Op inner = g->ops[--g->op_count];
                g3_undo_one(g, &inner);
            }

            /* Restore the placed cell. */
            g->cells[op.move.move] = G3_EMPTY;
            g->gid  [op.move.move] = -1;

            /* Restore captured stones — they were the opposite color. */
            const int8_t cap_color = -op.move.color;
            for (int32_t i = 0; i < op.move.captures_count; i++) {
                const G3Capture *cap = &g->captures[op.move.captures_start + i];
                g->cells[cap->idx] = cap_color;
                g->gid  [cap->idx] = cap->gid;
            }
            g->capture_count = op.move.captures_start;

            g->current            = op.move.previous_current;
            g->ko                 = op.move.previous_ko;
            g->empty_count        = op.move.previous_empty_count;
            g->consecutive_passes = op.move.previous_consecutive_passes;
            g->last_move          = op.move.previous_last_move;
            g->next_gid           = op.move.previous_next_gid;
            g->move_count--;
            return true;
        }

        /* Loose op without a wrapping MOVE — shouldn't happen in well-formed
         * play sequences but unwind defensively. */
        g3_undo_one(g, &op);
    }
    return true;
}

/* ── Group queries ─────────────────────────────────────────────────────────── */

int32_t g3_group_id_at(const Game3 *g, int32_t idx) { return g->gid[idx]; }
int32_t g3_group_size(const Game3 *g, int32_t gid)  { return g->ss[gid]; }
int32_t g3_group_liberty_count(const Game3 *g, int32_t gid) { return g->ls[gid]; }

G3GroupLibs2 g3_group_libs2(const Game3 *g, int32_t idx) {
    G3GroupLibs2 r = { 0, -1, -1 };
    int32_t gid = g->gid[idx];
    if (gid == -1) return r;
    int32_t lc = g->ls[gid];
    if (lc == 0) return r;
    r.count = lc;
    const int W = g->W;
    const int32_t lb = gid * W;
    int found = 0;
    for (int wi = 0; wi < W && found < 2; wi++) {
        uint32_t w = g->lw[lb + wi];
        while (w && found < 2) {
            int bit = g3_ctz32(w);
            int32_t i = wi * 32 + bit;
            if (i < g->cap) {
                if (found == 0) r.lib0 = i;
                else            r.lib1 = i;
                found++;
            }
            w &= w - 1;
        }
    }
    return r;
}

int32_t g3_group_libs(const Game3 *g, int32_t idx, int32_t *out, int32_t max) {
    int32_t gid = g->gid[idx];
    if (gid == -1) return 0;
    int32_t lc = g->ls[gid];
    const int W = g->W;
    const int32_t lb = gid * W;
    int32_t found = 0;
    for (int wi = 0; wi < W && found < lc; wi++) {
        uint32_t w = g->lw[lb + wi];
        while (w && found < lc) {
            int bit = g3_ctz32(w);
            int32_t i = wi * 32 + bit;
            if (i < g->cap) {
                if (found < max) out[found] = i;
                found++;
            }
            w &= w - 1;
        }
    }
    return lc;
}

/* ── Display ───────────────────────────────────────────────────────────────── */

char *g3_to_string(const Game3 *g, int32_t mark_idx,
                   int32_t center_at, bool show_axes) {
    const int N = g->N;
    const int mark_x = (mark_idx != G3_PASS) ?  mark_idx % N : -1;
    const int mark_y = (mark_idx != G3_PASS) ?  mark_idx / N : -1;
    const int half   = N / 2;
    const int x0 = (center_at != G3_PASS) ? ((center_at % N) - half + N) % N : 0;
    const int y0 = (center_at != G3_PASS) ? ((center_at / N) - half + N) % N : 0;
    const int dmx = (mark_x >= 0) ? ((mark_x - x0 + N) % N) : -1;
    const int dmy = (mark_y >= 0) ? ((mark_y - y0 + N) % N) : -1;

    /* Conservative upper bound: each row is ~(N*2 + axis pad) chars + newline. */
    const size_t row_max  = (size_t)(N * 2 + 16);
    const size_t row_cnt  = (size_t)(N + (show_axes ? 2 : 0));
    const size_t buf_size = row_max * row_cnt + 1;
    char *buf = (char *)malloc(buf_size);
    if (!buf) return NULL;
    char *p = buf;

    if (show_axes) {
        p += sprintf(p, "    ");
        for (int dx = 0; dx < N; dx++) {
            int bx = (x0 + dx) % N;
            p += sprintf(p, "%c ", 'a' + bx);
        }
        *p++ = '\n';
    }

    for (int dy = N - 1; dy >= 0; dy--) {
        int by = (y0 + dy) % N;
        int mx = (dy == dmy) ? dmx : -1;

        if (show_axes) p += sprintf(p, "%2d ", by + 1);

        *p++ = (mx == 0) ? '(' : ' ';
        for (int dx = 0; dx < N; dx++) {
            int bx = (x0 + dx) % N;
            int8_t c = g->cells[by * N + bx];
            char ch = (c == G3_BLACK) ? 'X' : (c == G3_WHITE) ? 'O' : '.';
            if (dx > 0) *p++ = (dx == mx) ? '(' : (dx - 1 == mx) ? ')' : ' ';
            *p++ = ch;
        }
        *p++ = (mx == N - 1) ? ')' : ' ';

        if (show_axes) p += sprintf(p, " %d", by + 1);
        *p++ = '\n';
    }

    if (show_axes) {
        p += sprintf(p, "   ");
        for (int dx = 0; dx < N; dx++) {
            int bx = (x0 + dx) % N;
            p += sprintf(p, "%c ", 'a' + bx);
        }
        *p++ = '\n';
    }

    *p = '\0';
    return buf;
}

void g3_free_string(char *s) { free(s); }

/* ── Lifecycle ─────────────────────────────────────────────────────────────── */

Game3 *g3_new(int32_t N) {
    Game3 *g = (Game3 *)calloc(1, sizeof(Game3));
    if (!g) return NULL;
    g->N     = N;
    g->cap   = N * N;
    g->W     = (g->cap + 31) >> 5;
    g->max_g = 4 * g->cap + 4;

    g->cells = (int8_t  *)calloc(g->cap, sizeof(int8_t));
    g->gid   = (int32_t *)malloc(g->cap * sizeof(int32_t));
    for (int i = 0; i < g->cap; i++) g->gid[i] = -1;

    g->gc = (uint8_t  *)calloc(g->max_g, sizeof(uint8_t));
    g->ss = (int32_t  *)calloc(g->max_g, sizeof(int32_t));
    g->ls = (int32_t  *)calloc(g->max_g, sizeof(int32_t));
    g->sw = (uint32_t *)calloc((size_t)g->max_g * g->W, sizeof(uint32_t));
    g->lw = (uint32_t *)calloc((size_t)g->max_g * g->W, sizeof(uint32_t));

    g->nbr  = (int32_t *)malloc((size_t)g->cap * 4 * sizeof(int32_t));
    g->dnbr = (int32_t *)malloc((size_t)g->cap * 4 * sizeof(int32_t));
    g3_build_topology(g);

    g->current             = G3_BLACK;
    g->ko                  = G3_PASS;
    g->empty_count         = g->cap;
    g->move_count          = 0;
    g->last_move           = G3_PASS;
    g->game_over           = false;
    g->consecutive_passes  = 0;
    g->next_gid            = 0;
    return g;
}

void g3_free(Game3 *g) {
    if (!g) return;
    /* Release any pending merge snapshots still on the op stack. */
    for (int32_t i = 0; i < g->op_count; i++) {
        if (g->ops[i].type == G3_OP_MERGE_GROUPS)
            free(g->ops[i].merge.main_libs);
    }
    /* Free pooled buffers. */
    for (int32_t i = 0; i < g->pool_count; i++) free(g->pool[i]);
    free(g->pool);
    free(g->ops);
    free(g->captures);
    free(g->cells);
    free(g->gid);
    free(g->gc); free(g->ss); free(g->ls);
    free(g->sw); free(g->lw);
    free(g->nbr); free(g->dnbr);
    free(g);
}

void g3_reset(Game3 *g) {
    /* Release any merge snapshots still on the op stack back to the pool. */
    for (int32_t i = 0; i < g->op_count; i++) {
        if (g->ops[i].type == G3_OP_MERGE_GROUPS)
            g3_release_word_buffer(g, g->ops[i].merge.main_libs);
    }
    g->op_count       = 0;
    g->capture_count  = 0;

    memset(g->cells, 0, g->cap * sizeof(int8_t));
    for (int i = 0; i < g->cap; i++) g->gid[i] = -1;

    const size_t used = (size_t)g->next_gid * g->W;
    if (used > 0) {
        memset(g->sw, 0, used * sizeof(uint32_t));
        memset(g->lw, 0, used * sizeof(uint32_t));
    }
    if (g->next_gid > 0) {
        memset(g->ss, 0, g->next_gid * sizeof(int32_t));
        memset(g->ls, 0, g->next_gid * sizeof(int32_t));
        memset(g->gc, 0, g->next_gid * sizeof(uint8_t));
    }
    g->next_gid            = 0;

    g->current             = G3_BLACK;
    g->ko                  = G3_PASS;
    g->consecutive_passes  = 0;
    g->game_over           = false;
    g->move_count          = 0;
    g->last_move           = G3_PASS;
    g->empty_count         = g->cap;
}

/* ── Game2 → Game3 ─────────────────────────────────────────────────────────── */

bool g3_from_game2(Game3 *g3, const Game2 *g2) {
    const int N   = g2->N;
    const int cap = N * N;
    if (g3->N != N) {
        fprintf(stderr, "g3_from_game2: size mismatch (game2=%d, game3=%d)\n",
                N, g3->N);
        return false;
    }
    g3_reset(g3);

    int32_t stones_placed = 0;
    for (int32_t i = 0; i < cap; i++) {
        if (g2->cells[i] == G3_EMPTY) continue;
        g3->current = g2->cells[i];
        if (!g3_is_legal(g3, i)) {
            fprintf(stderr,
                    "g3_from_game2: cell %d (%s) is not legal in Game3\n",
                    i, g2->cells[i] == G3_BLACK ? "BLACK" : "WHITE");
            return false;
        }
        g3_play(g3, i);
        stones_placed++;
        if (g3->empty_count != cap - stones_placed) {
            fprintf(stderr,
                    "g3_from_game2: capture detected placing cell %d\n", i);
            return false;
        }
    }

    g3->current             = g2->current;
    g3->ko                  = g2->ko;
    g3->consecutive_passes  = g2->consecutive_passes;
    g3->game_over           = g2->game_over;
    g3->move_count          = g2->move_count;
    g3->last_move           = g2->last_move;
    return true;
}
