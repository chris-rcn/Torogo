/*
 * ppat.c — 3×3 pattern + previous-move feature library (C port of ppat-lib.js).
 */
#include "ppat.h"
#include <math.h>
#include <string.h>

/* ── Canon table ───────────────────────────────────────────────────────────── */

int32_t ppat_canon_id[PPAT_RAW_SIZE];
int32_t ppat_num_patterns = 0;

/* D4 permutations: perm[src] = dst */
static const int D4[8][8] = {
    {0,1,2,3,4,5,6,7},  /* Identity */
    {1,2,3,0,5,6,7,4},  /* Rot90CW */
    {2,3,0,1,6,7,4,5},  /* Rot180 */
    {3,0,1,2,7,4,5,6},  /* Rot270CW */
    {0,3,2,1,7,6,5,4},  /* FlipH */
    {2,1,0,3,5,4,7,6},  /* FlipV */
    {3,2,1,0,6,5,4,7},  /* TransposeMD */
    {1,0,3,2,4,7,6,5},  /* TransposeAD */
};

static int encode8(const int *v) {
    return v[0] + 5*(v[1] + 5*(v[2] + 5*(v[3] + 5*(v[4] + 3*(v[5] + 3*(v[6] + 3*v[7]))))));
}

void ppat_init(void) {
    /* Map: minVariant → assigned dense ID.
     * Use a simple scan: for each raw, compute min variant.  Track assigned IDs
     * in a flat array indexed by minVariant (only 50625 entries, fast enough). */
    int32_t id_of[PPAT_RAW_SIZE];
    memset(id_of, -1, sizeof(id_of));
    int next_id = 0;

    int v[8], tv[8];
    for (int raw = 0; raw < PPAT_RAW_SIZE; raw++) {
        int r = raw;
        v[0] = r % 5; r /= 5;
        v[1] = r % 5; r /= 5;
        v[2] = r % 5; r /= 5;
        v[3] = r % 5; r /= 5;
        v[4] = r % 3; r /= 3;
        v[5] = r % 3; r /= 3;
        v[6] = r % 3;
        v[7] = r / 3;

        int min_v = raw;
        for (int di = 0; di < 8; di++) {
            const int *p = D4[di];
            for (int i = 0; i < 8; i++) tv[p[i]] = v[i];
            int enc = encode8(tv);
            if (enc < min_v) min_v = enc;
        }

        if (id_of[min_v] == -1) id_of[min_v] = next_id++;
        ppat_canon_id[raw] = id_of[min_v];
    }
    ppat_num_patterns = next_id;
}

/* ── Internal helpers ──────────────────────────────────────────────────────── */

static inline int adj_val(int32_t ni, const Game2 *g, int8_t cur) {
    int8_t c = g->cells[ni];
    if (c == EMPTY) return 0;
    int atari = (g->ls[g->gid[ni]] == 1);
    return (c == cur) ? (atari ? 2 : 1) : (atari ? 4 : 3);
}

static inline int diag_val(int32_t ni, const Game2 *g, int8_t cur) {
    int8_t c = g->cells[ni];
    return (c == EMPTY) ? 0 : (c == cur ? 1 : 2);
}

static int32_t first_lib(int32_t gid, const Game2 *g) {
    int32_t lb = gid * BW;
    for (int wi = 0; wi < BW; wi++) {
        uint32_t w = g->lw[lb + wi];
        if (w) {
            int bit = __builtin_ctz(w);
            int idx = wi * 32 + bit;
            if (idx < CAP) return idx;
        }
    }
    return -1;
}

static bool can_save_by_capture(int32_t idx, const int32_t *atari_gids, int n_atari,
                                const Game2 *g, int8_t foe) {
    uint32_t m = 1u << (idx & 31);
    int wi = idx >> 5;
    for (int ai = 0; ai < n_atari; ai++) {
        int32_t sgid = atari_gids[ai];
        int32_t sb = sgid * BW;
        for (int swi = 0; swi < BW; swi++) {
            uint32_t w = g->sw[sb + swi];
            while (w) {
                int bit = __builtin_ctz(w);
                int si = swi * 32 + bit;
                if (si < CAP) {
                    int b4 = si * 4;
                    for (int d = 0; d < 4; d++) {
                        int32_t ni = g2_nbr[b4 + d];
                        if (g->cells[ni] != foe) continue;
                        int32_t egid = g->gid[ni];
                        if (g->ls[egid] == 1 && (g->lw[egid * BW + wi] & m))
                            return true;
                    }
                }
                w &= w - 1;
            }
        }
    }
    return false;
}

/* Find both liberties of a group with exactly 2 libs. */
static void two_libs(int32_t gid, const Game2 *g, int32_t *lib0, int32_t *lib1) {
    int32_t lb = gid * BW;
    int found = 0;
    for (int wi = 0; wi < BW && found < 2; wi++) {
        uint32_t w = g->lw[lb + wi];
        while (w && found < 2) {
            int bit = __builtin_ctz(w);
            int32_t cell = wi * 32 + bit;
            if (cell < CAP) {
                if (found == 0) *lib0 = cell; else *lib1 = cell;
                found++;
            }
            w &= w - 1;
        }
    }
}

/* Check if the other liberty of egid (not idx) connects to a same-color group
 * (excluding egid) with ≥2 liberties. If so, the opponent can save by joining. */
static bool opponent_can_save(int32_t idx, int32_t egid, const Game2 *g, int8_t foe) {
    int32_t l0 = -1, l1 = -1;
    two_libs(egid, g, &l0, &l1);
    int32_t other = (l0 == idx) ? l1 : l0;
    int ob4 = other * 4;
    for (int d = 0; d < 4; d++) {
        int32_t ni = g2_nbr[ob4 + d];
        if (g->cells[ni] != foe) continue;
        int32_t ngid = g->gid[ni];
        if (ngid == egid) continue;  /* same group, skip */
        if (g->ls[ngid] >= 2) return true;
    }
    return false;
}

static bool gives_2lib_atari(int32_t idx, const int32_t *two_lib_gids, int n_two,
                              const Game2 *g, int8_t foe) {
    uint32_t m = 1u << (idx & 31);
    int wi = idx >> 5;
    for (int ti = 0; ti < n_two; ti++) {
        int32_t sgid = two_lib_gids[ti];
        int32_t sb = sgid * BW;
        for (int swi = 0; swi < BW; swi++) {
            uint32_t w = g->sw[sb + swi];
            while (w) {
                int bit = __builtin_ctz(w);
                int si = swi * 32 + bit;
                if (si < CAP) {
                    int b4 = si * 4;
                    for (int d = 0; d < 4; d++) {
                        int32_t ni = g2_nbr[b4 + d];
                        if (g->cells[ni] != foe) continue;
                        int32_t egid = g->gid[ni];
                        if (g->ls[egid] == 2 && (g->lw[egid * BW + wi] & m)
                            && !opponent_can_save(idx, egid, g, foe))
                            return true;
                    }
                }
                w &= w - 1;
            }
        }
    }
    return false;
}

static bool not_self_atari_cheap(int32_t idx, int b4, const Game2 *g,
                                  int8_t cur) {
    int free = 0;
    uint32_t m = 1u << (idx & 31);
    int wi = idx >> 5;
    for (int d = 0; d < 4; d++) {
        int32_t ni = g2_nbr[b4 + d];
        int8_t c = g->cells[ni];
        if (c == EMPTY) {
            if (++free >= 2) return true;
        } else if (c == cur) {
            if (g->ls[g->gid[ni]] >= 3) return true;
        } else {
            int32_t egid = g->gid[ni];
            if (g->ls[egid] == 1 && (g->lw[egid * BW + wi] & m))
                if (++free >= 2) return true;
        }
    }
    return false;
}

/* ── Extract features ──────────────────────────────────────────────────────── */

void ppat_extract(const Game2 *g, PpatState *st) {
    int8_t cur = g->current;
    int8_t foe = -cur;
    int32_t prev = g->last_move;
    bool has_prev = (prev != PASS);
    int32_t my_ko_stone = g->ko_stone[cur + 1];  /* cell that created ko on our last turn */

    /* Pre-scan: build prevNeighborSet + find atari/2-lib friendly strings.
     * KNOWN LIMITATION (Features 2–5): We find strings that currently have 1 liberty
     * adjacent to prev, but don't verify that prev *caused* the atari. The spec says
     * "new atari" — the string should have had >1 liberty before the opponent's move.
     * KNOWN LIMITATION (Feature 7): Same issue — we find strings with 2 liberties but
     * don't verify prev reduced them to 2. */
    int32_t atari_gids[8];
    int n_atari = 0;
    int32_t atari_libs[8];  /* single liberty for each atari group */

    int32_t two_lib_gids[8];
    int n_two = 0;

    if (has_prev) {
        int pb4 = prev * 4;
        for (int d = 0; d < 4; d++) {
            st->prev_neighbor_set[g2_nbr[pb4 + d]]  = 1;
            st->prev_neighbor_set[g2_dnbr[pb4 + d]] = 1;
            /* Only orthogonal neighbors can have had a liberty removed by prev. */
            int32_t ni = g2_nbr[pb4 + d];
            if (g->cells[ni] != cur) continue;
            int32_t gid = g->gid[ni];
            int32_t ls  = g->ls[gid];
            if (ls == 1) {
                bool dup = false;
                for (int j = 0; j < n_atari; j++) if (atari_gids[j] == gid) { dup = true; break; }
                if (!dup && n_atari < 8) atari_gids[n_atari++] = gid;
            } else if (ls == 2) {
                bool dup = false;
                for (int j = 0; j < n_two; j++) if (two_lib_gids[j] == gid) { dup = true; break; }
                if (!dup && n_two < 8) two_lib_gids[n_two++] = gid;
            }
        }
        /* Cache single liberty for each atari group */
        for (int i = 0; i < n_atari; i++)
            atari_libs[i] = first_lib(atari_gids[i], g);
    }

    /* Feature 6 pre-scan: find liberty cells that would capture an enemy group
     * adjacent to our ko stone. */
    int32_t ko_solve_libs[4];
    int n_ko_solve = 0;
    if (my_ko_stone != PASS) {
        int ks4 = my_ko_stone * 4;
        for (int d = 0; d < 4; d++) {
            int32_t ni = g2_nbr[ks4 + d];
            if (g->cells[ni] != foe) continue;
            int32_t egid = g->gid[ni];
            if (g->ls[egid] == 1) {
                int32_t lib = first_lib(egid, g);
                if (lib >= 0) ko_solve_libs[n_ko_solve++] = lib;
            }
        }
    }

    int count = 0;

    for (int ei = 0; ei < g->empty_count; ei++) {
        int32_t idx = g->empty_cells[ei];
        if (!g2_is_legal(g, idx) || g2_is_true_eye(g, idx)) continue;

        /* ── 3×3 pattern ──────────────────────────────────────────────────── */
        int b4 = idx * 4;
        int vN  = adj_val(g2_nbr[b4],     g, cur);
        int vE  = adj_val(g2_nbr[b4 + 3], g, cur);
        int vS  = adj_val(g2_nbr[b4 + 1], g, cur);
        int vW  = adj_val(g2_nbr[b4 + 2], g, cur);
        int vNE = diag_val(g2_dnbr[b4 + 1], g, cur);
        int vSE = diag_val(g2_dnbr[b4 + 3], g, cur);
        int vSW = diag_val(g2_dnbr[b4 + 2], g, cur);
        int vNW = diag_val(g2_dnbr[b4],     g, cur);

        int raw = vN + 5*(vE + 5*(vS + 5*(vW + 5*(vNE + 3*(vSE + 3*(vSW + 3*vNW))))));

        st->moves[count]   = idx;
        st->pat_ids[count]  = ppat_canon_id[raw];

        /* ── Previous-move features ───────────────────────────────────────── */
        uint8_t mask = 0;

        /* Feature 1: 8-neighborhood of prev */
        if (has_prev && st->prev_neighbor_set[idx])
            mask = 1;

        /* Features 2–5: save atari by capture or extension.
         * Capture (F2/3) takes priority over extension (F4/5). */
        if (n_atari > 0) {
            bool feat2 = can_save_by_capture(idx, atari_gids, n_atari, g, foe);
            bool feat4 = false;
            if (!feat2) {
                for (int i = 0; i < n_atari; i++) {
                    if (atari_libs[i] == idx) { feat4 = true; break; }
                }
            }
            if (feat2 || feat4) {
                bool sa = false;
                if (!not_self_atari_cheap(idx, b4, g, cur)) {
                    Game2 cg;
                    g2_clone(&cg, g);
                    g2_play(&cg, idx);
                    int32_t cid = cg.gid[idx];
                    sa = (cid != -1 && cg.ls[cid] == 1);
                }
                if (feat2) mask |= sa ? 4 : 2;
                if (feat4) mask |= sa ? 16 : 8;
            }
        }

        /* Feature 7: 2-point semeai. Only fires if the atari likely kills
         * (opponent's other liberty doesn't join to a non-atari group). */
        if (n_two > 0 && gives_2lib_atari(idx, two_lib_gids, n_two, g, foe))
            mask |= 64;

        /* Feature 6: ko-solve capture */
        for (int ki = 0; ki < n_ko_solve; ki++) {
            if (idx == ko_solve_libs[ki]) { mask |= 32; break; }
        }

        /* Bit 0 piggyback: active for all features 2-7 */
        if (mask & 0x7E) mask |= 1;

        st->prev_masks[count] = mask;
        count++;
    }

    st->count = count;

    /* Clear prevNeighborSet for reuse */
    if (has_prev) {
        int pb4 = prev * 4;
        for (int d = 0; d < 4; d++) {
            st->prev_neighbor_set[g2_nbr[pb4 + d]]  = 0;
            st->prev_neighbor_set[g2_dnbr[pb4 + d]] = 0;
        }
    }
}

/* ── Policy move ───────────────────────────────────────────────────────────── */

static float logits_buf[CAP];
static float probs_buf[CAP];

int32_t ppat_policy_move(const Game2 *g, PpatState *st, const PpatWeights *w) {
    ppat_extract(g, st);
    int n = st->count;
    if (n == 0) return PASS;

    const float *pat  = w->pat;
    const float *prev = w->prev;

    for (int i = 0; i < n; i++) {
        float v = pat[st->pat_ids[i]];
        uint8_t m = st->prev_masks[i];
        for (int b = 0; m; b++, m >>= 1)
            if (m & 1) v += prev[b];
        logits_buf[i] = v;
    }

    /* Softmax */
    float mx = logits_buf[0];
    for (int i = 1; i < n; i++) if (logits_buf[i] > mx) mx = logits_buf[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { probs_buf[i] = expf(logits_buf[i] - mx); sum += probs_buf[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) probs_buf[i] *= inv;

    /* Sample */
    float r = g2_randf();
    int chosen = n - 1;
    for (int i = 0; i < n; i++) { r -= probs_buf[i]; if (r <= 0) { chosen = i; break; } }
    return st->moves[chosen];
}
