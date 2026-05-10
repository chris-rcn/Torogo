/*
 * ppat.c — 3×3 pattern + previous-move feature library (C port of ppat-lib.js).
 */
#include "ppat.h"
#include <math.h>
#include <string.h>

/* ── Canon table ───────────────────────────────────────────────────────────── */

int16_t ppat_canon_id[PPAT_RAW_SIZE];
int32_t ppat_num_patterns = 0;
int     ppat_phase_count = 1;

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
    int16_t id_of[PPAT_RAW_SIZE];
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
    int32_t lb = gid * g->W;
    for (int wi = 0; wi < g->W; wi++) {
        uint32_t w = g->lw[lb + wi];
        if (w) {
            int bit = __builtin_ctz(w);
            int idx = wi * 32 + bit;
            if (idx < g->cap) return idx;
        }
    }
    return -1;
}

/* Precompute the set of cells that save a friendly atari group by capture.
 * Returns the count of such cells, stored in sbc_out. */
static int precompute_save_by_capture(const int32_t *atari_gids, int n_atari,
                                       const Game2 *g, int8_t foe,
                                       int32_t *sbc_out) {
    int n_sbc = 0;
    const int W = g->W;
    int32_t seen[16];
    int n_seen = 0;
    for (int ai = 0; ai < n_atari; ai++) {
        int32_t sgid = atari_gids[ai];
        int32_t sb = sgid * W;
        for (int swi = 0; swi < W; swi++) {
            uint32_t w = g->sw[sb + swi];
            while (w) {
                int bit = __builtin_ctz(w);
                int si = swi * 32 + bit;
                if (si < g->cap) {
                    int b4 = si * 4;
                    for (int d = 0; d < 4; d++) {
                        int32_t ni = g2_nbr[b4 + d];
                        if (g->cells[ni] != foe) continue;
                        int32_t egid = g->gid[ni];
                        if (g->ls[egid] != 1) continue;
                        bool dup = false;
                        for (int j = 0; j < n_seen; j++)
                            if (seen[j] == egid) { dup = true; break; }
                        if (dup) continue;
                        if (n_seen < 16) seen[n_seen++] = egid;
                        int32_t lib = first_lib(egid, g);
                        if (lib >= 0 && n_sbc < MAX_CAP)
                            sbc_out[n_sbc++] = lib;
                    }
                }
                w &= w - 1;
            }
        }
    }
    return n_sbc;
}

/* Find both liberties of a group with exactly 2 libs. */
static void two_libs(int32_t gid, const Game2 *g, int32_t *lib0, int32_t *lib1) {
    int32_t lb = gid * g->W;
    int found = 0;
    for (int wi = 0; wi < g->W && found < 2; wi++) {
        uint32_t w = g->lw[lb + wi];
        while (w && found < 2) {
            int bit = __builtin_ctz(w);
            int32_t cell = wi * 32 + bit;
            if (cell < g->cap) {
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

/* A semeai candidate: cell + the enemy gid it would put in atari */
typedef struct { int32_t cell; int32_t egid; } SemeaiCandidate;

/* Precompute semeai candidates: cells that are liberties of 2-lib enemy groups
 * adjacent to our 2-lib groups. Returns count. */
static int precompute_semeai(const int32_t *two_lib_gids, int n_two,
                              const Game2 *g, int8_t foe,
                              SemeaiCandidate *out) {
    int n = 0;
    const int W = g->W;
    /* Track seen enemy gids with a small inline list (avoids static array + cleanup) */
    int32_t seen[16];
    int n_seen = 0;
    for (int ti = 0; ti < n_two; ti++) {
        int32_t sgid = two_lib_gids[ti];
        int32_t sb = sgid * W;
        for (int swi = 0; swi < W; swi++) {
            uint32_t w = g->sw[sb + swi];
            while (w) {
                int bit = __builtin_ctz(w);
                int si = swi * 32 + bit;
                if (si < g->cap) {
                    int b4 = si * 4;
                    for (int d = 0; d < 4; d++) {
                        int32_t ni = g2_nbr[b4 + d];
                        if (g->cells[ni] != foe) continue;
                        int32_t egid = g->gid[ni];
                        if (g->ls[egid] != 2) continue;
                        /* Check if already seen */
                        bool dup = false;
                        for (int j = 0; j < n_seen; j++)
                            if (seen[j] == egid) { dup = true; break; }
                        if (dup) continue;
                        if (n_seen < 16) seen[n_seen++] = egid;
                        int32_t l0 = -1, l1 = -1;
                        two_libs(egid, g, &l0, &l1);
                        if (l0 >= 0 && n < MAX_CAP) out[n++] = (SemeaiCandidate){l0, egid};
                        if (l1 >= 0 && n < MAX_CAP) out[n++] = (SemeaiCandidate){l1, egid};
                    }
                }
                w &= w - 1;
            }
        }
    }
    return n;
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
            if (g->ls[egid] == 1 && (g->lw[egid * g->W + wi] & m))
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
    int32_t my_ko_stone = g->ko_stone[cur + 1];

    const int phase = ppat_phase_count * (g->cap - g->empty_count) / g->cap;
    const int pat_offset = phase * ppat_num_patterns;
    const int prev_offset = ppat_phase_count * ppat_num_patterns + phase * 7;  /* cell that created ko on our last turn */

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

    /* Precompute save-by-capture cells */
    int32_t sbc_cells[MAX_CAP];
    int n_sbc = 0;
    if (n_atari > 0)
        n_sbc = precompute_save_by_capture(atari_gids, n_atari, g, foe, sbc_cells);

    /* Precompute semeai candidates */
    SemeaiCandidate sem_cells[MAX_CAP];
    int n_sem = 0;
    if (n_two > 0)
        n_sem = precompute_semeai(two_lib_gids, n_two, g, foe, sem_cells);

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
    int nf = 0;  /* index into st->feat[] */

    const int32_t ko = g->ko;

    for (int ei = 0; ei < g->empty_count; ei++) {
        int32_t idx = g->empty_cells[ei];
        int b4 = idx * 4;

        /* ── Single pass over 4 orthogonal neighbors ─────────────────────── */
        int32_t ni0 = g2_nbr[b4], ni1 = g2_nbr[b4+1], ni2 = g2_nbr[b4+2], ni3 = g2_nbr[b4+3];
        int8_t c0 = g->cells[ni0], c1 = g->cells[ni1], c2 = g->cells[ni2], c3 = g->cells[ni3];

        /* Fast legality: if any neighbor is empty, legal unless ko */
        int any_empty = (c0 == EMPTY) | (c1 == EMPTY) | (c2 == EMPTY) | (c3 == EMPTY);
        if (any_empty) {
            if (idx == ko && g2_is_ko(g, idx, cur)) continue;
        } else {
            if (g2_is_single_suicide(g, idx, cur)) continue;
            if (g2_is_multi_suicide(g, idx, cur)) continue;
            if (idx == ko && g2_is_ko(g, idx, cur)) continue;
        }

        /* Combined true-eye check + adj_val computation.
         * Reads gid and ls only once per neighbor. */
        int vN, vS, vW, vE;
        {
            int friend_count = 0, empty_count_e = 0;
            int32_t first_gid = -2, same_group = 0;
            #define CHECK_AND_ADJ(c, ni, vout) do { \
                if ((c) == EMPTY) { \
                    empty_count_e++; \
                    vout = 0; \
                } else { \
                    int32_t _gid = g->gid[(ni)]; \
                    int _atari = (g->ls[_gid] == 1); \
                    if ((c) == cur) { \
                        friend_count++; \
                        if (first_gid == -2) { first_gid = _gid; same_group = 1; } \
                        else if (_gid == first_gid) same_group++; \
                        vout = _atari ? 2 : 1; \
                    } else { \
                        vout = _atari ? 4 : 3; \
                    } \
                } \
            } while(0)
            CHECK_AND_ADJ(c0, ni0, vN);
            CHECK_AND_ADJ(c1, ni1, vS);
            CHECK_AND_ADJ(c2, ni2, vW);
            CHECK_AND_ADJ(c3, ni3, vE);
            #undef CHECK_AND_ADJ
            if (friend_count == 3 && empty_count_e == 1 && same_group == 3) continue;
            if (friend_count == 4) {
                if (same_group == 4) continue;
                int dc = 0;
                for (int i = 0; i < 4; i++)
                    if (g->cells[g2_dnbr[b4 + i]] == cur) dc++;
                if (dc >= 3) continue;
            }
        }

        int vNE = diag_val(g2_dnbr[b4 + 1], g, cur);
        int vSE = diag_val(g2_dnbr[b4 + 3], g, cur);
        int vSW = diag_val(g2_dnbr[b4 + 2], g, cur);
        int vNW = diag_val(g2_dnbr[b4],     g, cur);

        int raw = vN + 5*(vE + 5*(vS + 5*(vW + 5*(vNE + 3*(vSE + 3*(vSW + 3*vNW))))));

        st->moves[count] = idx;
        st->feat_start[count] = nf;

        /* Pattern feature */
        st->feat[nf++] = pat_offset + ppat_canon_id[raw];

        /* ── Previous-move features ───────────────────────────────────────── */
        uint8_t mask = 0;

        /* Feature 1: 8-neighborhood of prev */
        if (has_prev && st->prev_neighbor_set[idx])
            mask = 1;

        /* Features 2–5: save atari by capture or extension.
         * Capture (F2/3) takes priority over extension (F4/5). */
        if (n_atari > 0) {
            bool feat2 = false;
            for (int si = 0; si < n_sbc; si++) {
                if (sbc_cells[si] == idx) { feat2 = true; break; }
            }
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

        /* Feature 7: 2-point semeai. Only fires if the atari likely kills. */
        if (n_sem > 0) {
            for (int si = 0; si < n_sem; si++) {
                if (sem_cells[si].cell == idx &&
                    !opponent_can_save(idx, sem_cells[si].egid, g, foe)) {
                    mask |= 64;
                    break;
                }
            }
        }

        /* Feature 6: ko-solve capture */
        for (int ki = 0; ki < n_ko_solve; ki++) {
            if (idx == ko_solve_libs[ki]) { mask |= 32; break; }
        }

        /* Bit 0 piggyback: active for all features 2-7 */
        if (mask & 0x7E) mask |= 1;

        /* Emit prev feature keys */
        for (int b = 0; b < 7; b++)
            if (mask & (1 << b)) st->feat[nf++] = prev_offset + b;

        count++;
    }

    st->feat_start[count] = nf;
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

static float logits_buf[MAX_CAP];

/* Fast approximate exp for softmax sampling.
 * Uses the classic Schraudolph IEEE-754 trick: interpret float bits. */
static inline float fast_expf(float x) {
    /* Clamp to avoid overflow/underflow */
    if (x < -20.0f) return 0.0f;
    if (x > 20.0f) x = 20.0f;
    union { float f; int32_t i; } v;
    v.i = (int32_t)(12102203.0f * x + 1065353216.0f);
    return v.f;
}

int32_t ppat_policy_move(const Game2 *g, PpatState *st, const float *weights) {
    ppat_extract(g, st);
    int n = st->count;
    if (n == 0) return PASS;

    /* Compute logits and find max in one pass */
    float mx = -1e30f;
    for (int i = 0; i < n; i++) {
        float v = 0;
        for (int fi = st->feat_start[i]; fi < st->feat_start[i + 1]; fi++)
            v += weights[st->feat[fi]];
        logits_buf[i] = v;
        if (v > mx) mx = v;
    }

    /* Compute unnormalized weights, sum, and sample */
    float sum = 0;
    for (int i = 0; i < n; i++) {
        float e = fast_expf(logits_buf[i] - mx);
        logits_buf[i] = e;
        sum += e;
    }

    float r = g2_randf() * sum;
    int chosen = n - 1;
    for (int i = 0; i < n; i++) {
        r -= logits_buf[i];
        if (r <= 0) { chosen = i; break; }
    }
    return st->moves[chosen];
}

/* ── Weight file I/O ───────────────────────────────────────────────────────── */

#include <stdio.h>
#include <stdlib.h>

void ppat_save_weights(const char *path, const float *weights, int total,
                       const char *comment) {
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "ppat_save_weights: cannot open %s\n", path); return; }
    fprintf(f, "'use strict';\n");
    if (comment) fprintf(f, "// %s\n", comment);
    fprintf(f, "const _w = { weights: new Float32Array([");
    for (int i = 0; i < total; i++) {
        if (i > 0) fputc(',', f);
        fprintf(f, "%.9g", weights[i]);
    }
    fprintf(f, "]), phases: %d, numPatterns: %d };\n", ppat_phase_count, ppat_num_patterns);
    fprintf(f, "if (typeof module !== 'undefined') module.exports = _w;\n");
    fprintf(f, "else window.PPATWeights = _w;\n");
    fclose(f);
}

float *ppat_load_weights(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "ppat_load_weights: cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(len + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t nread = fread(buf, 1, len, f);
    (void)nread;
    buf[len] = '\0';
    fclose(f);

    const char *pp = strstr(buf, "phases:");
    const char *np = strstr(buf, "numPatterns:");
    if (!pp || !np) {
        fprintf(stderr, "ppat_load_weights: missing phases/numPatterns in %s\n", path);
        free(buf); return NULL;
    }
    int file_phases = atoi(pp + 7);
    int file_np = atoi(np + 12);
    if (file_np != ppat_num_patterns) {
        fprintf(stderr, "ppat_load_weights: numPatterns=%d in file but %d expected\n", file_np, ppat_num_patterns);
        free(buf); return NULL;
    }
    ppat_phase_count = file_phases;
    int total = ppat_total_weights();
    float *weights = calloc(total, sizeof(float));

    const char *start = strchr(buf, '[');
    if (!start) { free(buf); free(weights); return NULL; }
    start++;
    int idx = 0;
    char *end;
    while (idx < total) {
        float v = strtof(start, &end);
        if (end == start) break;
        weights[idx++] = v;
        start = end;
        if (*start == ',') start++;
    }
    free(buf);
    fprintf(stderr, "loaded %d weights from %s (phases=%d)\n", idx, path, file_phases);
    return weights;
}

