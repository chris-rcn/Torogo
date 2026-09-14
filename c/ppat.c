/*
 * ppat.c — 3×3 pattern + previous-move feature library (C port of ppat-lib.js).
 */
#include "ppat.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* ── Canon table ───────────────────────────────────────────────────────────── */

/* Active canonical-ID table, plus the per-cap cache behind it (see ppat.h). */
const int32_t *ppat_canon_id = NULL;

/* ── Twelvecell extension ──────────────────────────────────────────────────── */
/* Canonicalising the four arms under the full D4 is EXACT here rather than the
 * usual lossy shortcut: the feature fires only when the inner ninecell is
 * all-empty, and that inner pattern is fixed by every element of D4, so any
 * transform that canonicalises the arms is a symmetry of the whole twelvecell. */
int ppat_self_atari = 0;
int ppat_file_self_atari = 0;          /* set by ppat_load_weights from the file */
int ppat_twelvecell = 0;
int ppat_file_twelvecell = 0;          /* set by ppat_load_weights from the file */
static int32_t t12_table[PPAT_T12_RAW];
static int32_t t12b_table[PPAT_T12B_RAW];
static bool    t12_built = false;
const int32_t *ppat_t12_canon  = t12_table;
const int32_t *ppat_t12b_canon = t12b_table;

static void ppat_build_t12(void) {
    if (t12_built) return;
    /* Arm order N, E, S, W — the ninecell's D4 restricted to the orthogonals. */
    static const int ROT[4] = {1, 2, 3, 0};   /* 90 degrees: N->E->S->W */
    static const int REF[4] = {0, 3, 2, 1};   /* mirror: E<->W */
    int perms[8][4], cur[4] = {0, 1, 2, 3};
    for (int r = 0; r < 4; r++) {
        for (int i = 0; i < 4; i++) { perms[2*r][i] = cur[i]; perms[2*r+1][i] = cur[REF[i]]; }
        int nxt[4];
        for (int i = 0; i < 4; i++) nxt[i] = cur[ROT[i]];
        for (int i = 0; i < 4; i++) cur[i] = nxt[i];
    }
    int next_id = 0, id_of[PPAT_T12_RAW];
    for (int i = 0; i < PPAT_T12_RAW; i++) id_of[i] = -1;
    for (int raw = 0; raw < PPAT_T12_RAW; raw++) {
        int v[4], r = raw;
        for (int i = 0; i < 4; i++) { v[i] = r % 3; r /= 3; }
        int min_v = raw;
        for (int d = 0; d < 8; d++) {
            int tv[4];
            for (int i = 0; i < 4; i++) tv[perms[d][i]] = v[i];
            int enc = tv[0] + 3*(tv[1] + 3*(tv[2] + 3*tv[3]));
            if (enc < min_v) min_v = enc;
        }
        if (id_of[min_v] < 0) id_of[min_v] = next_id++;
        t12_table[raw] = id_of[min_v];
    }
    if (next_id != PPAT_T12_PATTERNS) {
        fprintf(stderr, "ppat: twelvecell orbits %d != %d\n", next_id, PPAT_T12_PATTERNS);
        exit(1);
    }

    /* Mode 2's JOINT table: positions 0-3 are the diagonals (NE, SE, SW, NW),
     * 4-7 the arms (N, E, S, W), so one D4 element permutes both sets at once.
     * Canonicalising them together is what keeps an enemy arm BESIDE an enemy
     * diagonal distinct from the same two stones opposite each other. */
    {
        static const int ROT8[8] = {1, 2, 3, 0, 5, 6, 7, 4};   /* NE->SE->SW->NW, N->E->S->W */
        static const int REF8[8] = {3, 2, 1, 0, 4, 7, 6, 5};   /* mirror about N-S */
        int perms8[8][8], cur8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        for (int r = 0; r < 4; r++) {
            for (int i = 0; i < 8; i++) { perms8[2*r][i] = cur8[i]; perms8[2*r+1][i] = cur8[REF8[i]]; }
            int nxt[8];
            for (int i = 0; i < 8; i++) nxt[i] = cur8[ROT8[i]];
            for (int i = 0; i < 8; i++) cur8[i] = nxt[i];
        }
        static int32_t id_of8[PPAT_T12B_RAW];
        for (int i = 0; i < PPAT_T12B_RAW; i++) id_of8[i] = -1;
        int next8 = 0;
        for (int raw = 0; raw < PPAT_T12B_RAW; raw++) {
            int v[8], r = raw;
            for (int i = 0; i < 8; i++) { v[i] = r % 3; r /= 3; }
            int min_v = raw;
            for (int d = 0; d < 8; d++) {
                int tv[8];
                for (int i = 0; i < 8; i++) tv[perms8[d][i]] = v[i];
                int enc = 0;
                for (int i = 7; i >= 0; i--) enc = enc * 3 + tv[i];
                if (enc < min_v) min_v = enc;
            }
            if (id_of8[min_v] < 0) id_of8[min_v] = next8++;
            t12b_table[raw] = id_of8[min_v];
        }
        if (next8 != PPAT_T12B_PATTERNS) {
            fprintf(stderr, "ppat: twelvecell2 orbits %d != %d\n", next8, PPAT_T12B_PATTERNS);
            exit(1);
        }
    }
    t12_built = true;
}
static int32_t *canon_by_cap[PPAT_MAX_LIB_CAP + 1];
static int32_t  np_by_cap   [PPAT_MAX_LIB_CAP + 1];
static int32_t  raw_by_cap  [PPAT_MAX_LIB_CAP + 1];
int32_t ppat_num_patterns = 0;
int32_t ppat_lib_cap = 0;      /* 0 = not yet initialised */
int32_t ppat_raw_size = 0;
int     ppat_phase_count = 1;
float   ppat_uniform_below_phase = 0.0f;
int     ppat_load_quiet = 0;

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

/* Pack the 8 cells: orthogonals (0-3) in radix R = 2*cap+1, diagonals (4-7) in
 * radix 3.  R is passed rather than hardcoded so the same packing serves any cap. */
static int encode8(const int *v, int R) {
    return v[0] + R*(v[1] + R*(v[2] + R*(v[3] + R*(v[4] + 3*(v[5] + 3*(v[6] + 3*v[7]))))));
}

void ppat_init(int lib_cap) {
    ppat_build_t12();                          /* cheap, idempotent, cap-independent */
    if (lib_cap < PPAT_MIN_LIB_CAP || lib_cap > PPAT_MAX_LIB_CAP) {
        fprintf(stderr, "ppat_init: lib_cap %d out of range [%d,%d]\n",
                lib_cap, PPAT_MIN_LIB_CAP, PPAT_MAX_LIB_CAP);
        exit(1);
    }
    if (ppat_lib_cap == lib_cap) return;      /* already active */

    if (canon_by_cap[lib_cap]) {              /* built earlier: just swap it in */
        ppat_canon_id     = canon_by_cap[lib_cap];
        ppat_num_patterns = np_by_cap[lib_cap];
        ppat_raw_size     = raw_by_cap[lib_cap];
        ppat_lib_cap      = lib_cap;
        return;
    }

    const int R = 2 * lib_cap + 1;            /* orthogonal radix */
    const int raw_size = R * R * R * R * 81;  /* R^4 * 3^4 */
    int32_t *table = malloc((size_t)raw_size * sizeof(int32_t));
    if (!table) {
        fprintf(stderr, "ppat_init: out of memory for libCap %d (%d entries)\n", lib_cap, raw_size);
        exit(1);
    }

    /* Map: minVariant → assigned dense ID.  Static (not stack): at cap 4 this is
     * 531441 int32 = 2.1 MB, far past any sane stack. */
    static int32_t id_of[PPAT_RAW_SIZE];
    for (int i = 0; i < raw_size; i++) id_of[i] = -1;
    int next_id = 0;

    int v[8], tv[8];
    for (int raw = 0; raw < raw_size; raw++) {
        int r = raw;
        v[0] = r % R; r /= R;
        v[1] = r % R; r /= R;
        v[2] = r % R; r /= R;
        v[3] = r % R; r /= R;
        v[4] = r % 3; r /= 3;
        v[5] = r % 3; r /= 3;
        v[6] = r % 3;
        v[7] = r / 3;

        int min_v = raw;
        for (int di = 0; di < 8; di++) {
            const int *p = D4[di];
            for (int i = 0; i < 8; i++) tv[p[i]] = v[i];
            int enc = encode8(tv, R);
            if (enc < min_v) min_v = enc;
        }

        if (id_of[min_v] == -1) id_of[min_v] = next_id++;
        table[raw] = id_of[min_v];
    }
    canon_by_cap[lib_cap] = table;
    np_by_cap[lib_cap]    = next_id;
    raw_by_cap[lib_cap]   = raw_size;
    ppat_canon_id     = table;
    ppat_num_patterns = next_id;
    ppat_lib_cap      = lib_cap;
    ppat_raw_size     = raw_size;
}

/* ── Internal helpers ──────────────────────────────────────────────────────── */

/* Orthogonal cell value in radix 2*cap+1: 0 empty, 1..cap own with that many
 * liberties (capped), cap+1..2*cap enemy likewise.  At cap 2 this reproduces the
 * historical encoding exactly (own 1/2 = not-atari/atari, enemy 3/4), since a
 * capped count of 1 IS atari.  Note ls[gid[ni]] was already read for the atari
 * test, so raising the cap costs no extra memory traffic. */
static inline int adj_val(int32_t ni, const Game2 *g, int8_t cur) {
    int8_t c = g->cells[ni];
    if (c == EMPTY) return 0;
    const int cap = ppat_lib_cap;
    int lib = g->ls[g->gid[ni]];
    if (lib > cap) lib = cap;
    /* cap 2: lib 1 -> own 2 / enemy 4 (atari), lib 2 -> own 1 / enemy 3.
     * The historical order put not-atari first, so invert the count. */
    int slot = cap + 1 - lib;                 /* 1..cap, ascending in liberties */
    return (c == cur) ? slot : cap + slot;
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

/* Exact self-atari size for the LEGAL candidate idx (colour cur): 0 when the
 * placed group would keep >= 2 liberties, else its merged stone count.  The
 * cheap bound clears most candidates; the exact path ORs the joined chains'
 * liberty bitsets and credits capture-freed points, so snapbacks label
 * correctly as size-1 self-atari. */
static uint32_t sa_lib[MAX_BW];
static int self_atari_size(const Game2 *g, int32_t idx, int b4, int8_t cur) {
    if (not_self_atari_cheap(idx, b4, g, cur)) return 0;
    const int W = g->W;
    int16_t fr[4];  int nfr = 0;
    int16_t capg[4]; int ncap = 0;
    memset(sa_lib, 0, (size_t)W * sizeof(uint32_t));
    for (int d = 0; d < 4; d++) {
        const int32_t ni = g2_nbr[b4 + d];
        const int8_t c = g->cells[ni];
        if (c == EMPTY) { sa_lib[ni >> 5] |= 1u << (ni & 31); continue; }
        const int16_t gid = g->gid[ni];
        if (c == cur) {
            int dup = 0;
            for (int k = 0; k < nfr; k++) dup |= (fr[k] == gid);
            if (!dup) fr[nfr++] = gid;
        } else if (g->ls[gid] == 1) {
            /* adjacent enemy in atari: its lone liberty is idx, so it dies */
            int dup = 0;
            for (int k = 0; k < ncap; k++) dup |= (capg[k] == gid);
            if (!dup) capg[ncap++] = gid;
        }
    }
    int size = 1;
    for (int k = 0; k < nfr; k++) {
        const uint32_t *lw = &g->lw[(int)fr[k] * W];
        for (int w = 0; w < W; w++) sa_lib[w] |= lw[w];
        size += g->ss[fr[k]];
    }
    sa_lib[idx >> 5] &= ~(1u << (idx & 31));
    /* Capture-freed points: a captured stone is a liberty of the merged group
     * iff adjacent to it (the placed stone or a joined chain). */
    for (int k = 0; k < ncap; k++) {
        const uint32_t *sw = &g->sw[(int)capg[k] * W];
        for (int w = 0; w < W; w++) {
            uint32_t bits = sw[w];
            while (bits) {
                const int p = (w << 5) + __builtin_ctz(bits);
                bits &= bits - 1;
                for (int d = 0; d < 4; d++) {
                    const int32_t np = g2_nbr[p * 4 + d];
                    if (np == idx) { sa_lib[p >> 5] |= 1u << (p & 31); break; }
                    if (g->cells[np] == cur) {
                        const int16_t ng = g->gid[np];
                        int adj = 0;
                        for (int k2 = 0; k2 < nfr; k2++) adj |= (fr[k2] == ng);
                        if (adj) { sa_lib[p >> 5] |= 1u << (p & 31); break; }
                    }
                }
            }
        }
    }
    int libs = 0;
    for (int w = 0; w < W && libs < 2; w++) libs += __builtin_popcount(sa_lib[w]);
    return libs >= 2 ? 0 : size;
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
    const int prev_offset = ppat_phase_count * ppat_num_patterns + phase * 7;
    const int t12_offset = ppat_phase_count * (ppat_num_patterns + 7) + phase * ppat_t12_block();
    const int sa_offset  = ppat_phase_count * (ppat_num_patterns + 7 + ppat_t12_block()) + phase * PPAT_SA_N;

    /* The 7 hand-coded previous-move features are disabled — both the
     * emission AND the pre-scan/mask work that feeds it.  A/B at equal time
     * showed their information was not worth their ~28% extraction cost
     * (no-local fine-tune beat the with-local standard in real play,
     * 2026-09).  Weight slots for them still exist in every model; they
     * never fire. */
//    /* Pre-scan: build prevNeighborSet + find atari/2-lib friendly strings.
//     * KNOWN LIMITATION (Features 2–5): We find strings that currently have 1 liberty
//     * adjacent to prev, but don't verify that prev *caused* the atari. The spec says
//     * "new atari" — the string should have had >1 liberty before the opponent's move.
//     * KNOWN LIMITATION (Feature 7): Same issue — we find strings with 2 liberties but
//     * don't verify prev reduced them to 2. */
//    int32_t atari_gids[8];
//    int n_atari = 0;
//    int32_t atari_libs[8];  /* single liberty for each atari group */
//
//    int32_t two_lib_gids[8];
//    int n_two = 0;
//
//    if (has_prev) {
//        int pb4 = prev * 4;
//        for (int d = 0; d < 4; d++) {
//            st->prev_neighbor_set[g2_nbr[pb4 + d]]  = 1;
//            st->prev_neighbor_set[g2_dnbr[pb4 + d]] = 1;
//            /* Only orthogonal neighbors can have had a liberty removed by prev. */
//            int32_t ni = g2_nbr[pb4 + d];
//            if (g->cells[ni] != cur) continue;
//            int32_t gid = g->gid[ni];
//            int32_t ls  = g->ls[gid];
//            if (ls == 1) {
//                bool dup = false;
//                for (int j = 0; j < n_atari; j++) if (atari_gids[j] == gid) { dup = true; break; }
//                if (!dup && n_atari < 8) atari_gids[n_atari++] = gid;
//            } else if (ls == 2) {
//                bool dup = false;
//                for (int j = 0; j < n_two; j++) if (two_lib_gids[j] == gid) { dup = true; break; }
//                if (!dup && n_two < 8) two_lib_gids[n_two++] = gid;
//            }
//        }
//        /* Cache single liberty for each atari group */
//        for (int i = 0; i < n_atari; i++)
//            atari_libs[i] = first_lib(atari_gids[i], g);
//    }
//
//    /* Precompute save-by-capture cells */
//    int32_t sbc_cells[MAX_CAP];
//    int n_sbc = 0;
//    if (n_atari > 0)
//        n_sbc = precompute_save_by_capture(atari_gids, n_atari, g, foe, sbc_cells);
//
//    /* Precompute semeai candidates */
//    SemeaiCandidate sem_cells[MAX_CAP];
//    int n_sem = 0;
//    if (n_two > 0)
//        n_sem = precompute_semeai(two_lib_gids, n_two, g, foe, sem_cells);
//
//    /* Feature 6 pre-scan: find liberty cells that would capture an enemy group
//     * adjacent to our ko stone. */
//    int32_t ko_solve_libs[4];
//    int n_ko_solve = 0;
//    if (my_ko_stone != PASS) {
//        int ks4 = my_ko_stone * 4;
//        for (int d = 0; d < 4; d++) {
//            int32_t ni = g2_nbr[ks4 + d];
//            if (g->cells[ni] != foe) continue;
//            int32_t egid = g->gid[ni];
//            if (g->ls[egid] == 1) {
//                int32_t lib = first_lib(egid, g);
//                if (lib >= 0) ko_solve_libs[n_ko_solve++] = lib;
//            }
//        }
//    }
//
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
        /* Diagonals are read BEFORE the eye check, which needs the hostile
         * count: these are the same four reads the pattern index needs below,
         * so sharing them costs nothing but the handful of eye points that used
         * to skip out first. */
        int vNE = diag_val(g2_dnbr[b4 + 1], g, cur);
        int vSE = diag_val(g2_dnbr[b4 + 3], g, cur);
        int vSW = diag_val(g2_dnbr[b4 + 2], g, cur);
        int vNW = diag_val(g2_dnbr[b4],     g, cur);

        int vN, vS, vW, vE;
        bool eye;
        int empty_nbr_count = 0;   /* mode-2 twelvecell trigger */
        {
            int friend_count = 0, empty_count_e = 0;
            int32_t first_gid = -2, same_group = 0;
            /* Mirrors adj_val() exactly (slot = cap+1-lib, ascending in liberties)
             * but reads gid/ls once per neighbour and folds in the true-eye counts. */
            const int _cap = ppat_lib_cap;
            #define CHECK_AND_ADJ(c, ni, vout) do { \
                if ((c) == EMPTY) { \
                    empty_count_e++; \
                    vout = 0; \
                } else { \
                    int32_t _gid = g->gid[(ni)]; \
                    int _lib = g->ls[_gid]; if (_lib > _cap) _lib = _cap; \
                    int _slot = _cap + 1 - _lib; \
                    if ((c) == cur) { \
                        friend_count++; \
                        if (first_gid == -2) { first_gid = _gid; same_group = 1; } \
                        else if (_gid == first_gid) same_group++; \
                        vout = _slot; \
                    } else { \
                        vout = _cap + _slot; \
                    } \
                } \
            } while(0)
            CHECK_AND_ADJ(c0, ni0, vN);
            CHECK_AND_ADJ(c1, ni1, vS);
            CHECK_AND_ADJ(c2, ni2, vW);
            CHECK_AND_ADJ(c3, ni3, vE);
            #undef CHECK_AND_ADJ
            empty_nbr_count = empty_count_e;
            eye = g2_is_eyelike(friend_count, empty_count_e, same_group,
                                (vNE == 2) + (vSE == 2) + (vSW == 2) + (vNW == 2));
        }
        /* THE playout eye rule lives in g2_is_eyelike; the counts above are
         * handed to it so it need not rescan.  Playouts prune MORE than a root
         * generator may: g2_is_eyelike adds the multi-chain wall with one
         * hostile diagonal, which g2_is_true_eye leaves legal because it cannot
         * prove it is never a move.  This used to be an inlined copy that had
         * drifted, which made the playout fill multi-chain eyes and kill live
         * groups. */
        if (eye) continue;

        const int _R = 2 * ppat_lib_cap + 1;
        int raw = vN + _R*(vE + _R*(vS + _R*(vW + _R*(vNE + 3*(vSE + 3*(vSW + 3*vNW))))));

        st->moves[count] = idx;
        st->feat_start[count] = nf;

        /* Pattern feature */
        st->feat[nf++] = pat_offset + ppat_canon_id[raw];

        /* Twelvecell extension: an all-empty ninecell (raw 0 — every cell codes
         * 0 when empty) gets a second key for the four distance-2 orthogonals. */
        if (ppat_twelvecell == 1 ? raw == 0
                                 : ppat_twelvecell == 2 && empty_nbr_count == 4) {
            const int8_t a0 = g->cells[g2_nbr[ni0 * 4 + 0]];
            const int8_t a1 = g->cells[g2_nbr[ni3 * 4 + 3]];
            const int8_t a2 = g->cells[g2_nbr[ni1 * 4 + 1]];
            const int8_t a3 = g->cells[g2_nbr[ni2 * 4 + 2]];
            const int w0 = a0 == EMPTY ? 0 : a0 == cur ? 1 : 2;
            const int w1 = a1 == EMPTY ? 0 : a1 == cur ? 1 : 2;
            const int w2 = a2 == EMPTY ? 0 : a2 == cur ? 1 : 2;
            const int w3 = a3 == EMPTY ? 0 : a3 == cur ? 1 : 2;
            if (ppat_twelvecell == 1) {
                st->feat[nf++] = t12_offset +
                    ppat_t12_canon[w0 + 3*(w1 + 3*(w2 + 3*w3))];
            } else {
                /* Diagonals first (NE, SE, SW, NW), then the arms — jointly
                 * canonicalised so their relative placement survives. */
                st->feat[nf++] = t12_offset +
                    ppat_t12b_canon[vNE + 3*(vSE + 3*(vSW + 3*(vNW +
                                    3*(w0 + 3*(w1 + 3*(w2 + 3*w3))))))];
            }
        }

        if (ppat_self_atari) {
            const int sa = self_atari_size(g, idx, b4, cur);
            if (sa > 0)
                st->feat[nf++] = sa_offset + (sa < PPAT_SA_N ? sa : PPAT_SA_N) - 1;
        }

        /* Previous-move features disabled — see the note at the pre-scan
         * above. */
//        /* ── Previous-move features ───────────────────────────────────────── */
//        uint8_t mask = 0;
//
//        /* Feature 1: 8-neighborhood of prev */
//        if (has_prev && st->prev_neighbor_set[idx])
//            mask = 1;
//
//        /* Features 2–5: save atari by capture or extension.
//         * Capture (F2/3) takes priority over extension (F4/5). */
//        if (n_atari > 0) {
//            bool feat2 = false;
//            for (int si = 0; si < n_sbc; si++) {
//                if (sbc_cells[si] == idx) { feat2 = true; break; }
//            }
//            bool feat4 = false;
//            if (!feat2) {
//                for (int i = 0; i < n_atari; i++) {
//                    if (atari_libs[i] == idx) { feat4 = true; break; }
//                }
//            }
//            if (feat2 || feat4) {
//                bool sa = false;
//                if (!not_self_atari_cheap(idx, b4, g, cur)) {
//                    Game2 cg;
//                    g2_clone(&cg, g);
//                    g2_play(&cg, idx);
//                    int32_t cid = cg.gid[idx];
//                    sa = (cid != -1 && cg.ls[cid] == 1);
//                }
//                if (feat2) mask |= sa ? 4 : 2;
//                if (feat4) mask |= sa ? 16 : 8;
//            }
//        }
//
//        /* Feature 7: 2-point semeai. Only fires if the atari likely kills. */
//        if (n_sem > 0) {
//            for (int si = 0; si < n_sem; si++) {
//                if (sem_cells[si].cell == idx &&
//                    !opponent_can_save(idx, sem_cells[si].egid, g, foe)) {
//                    mask |= 64;
//                    break;
//                }
//            }
//        }
//
//        /* Feature 6: ko-solve capture */
//        for (int ki = 0; ki < n_ko_solve; ki++) {
//            if (idx == ko_solve_libs[ki]) { mask |= 32; break; }
//        }
//
//        /* Bit 0 piggyback: active for all features 2-7 */
//        if (mask & 0x7E) mask |= 1;
//
//        /* Emit prev feature keys */
//        for (int b = 0; b < 7; b++)
//            if (mask & (1 << b)) st->feat[nf++] = prev_offset + b;
//
        count++;
    }

    st->feat_start[count] = nf;
    st->count = count;

//    /* Clear prevNeighborSet for reuse */
//    if (has_prev) {
//        int pb4 = prev * 4;
//        for (int d = 0; d < 4; d++) {
//            st->prev_neighbor_set[g2_nbr[pb4 + d]]  = 0;
//            st->prev_neighbor_set[g2_dnbr[pb4 + d]] = 0;
//        }
//    }
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

int32_t ppat_policy_move(const Game2 *g, PpatState *st, const float *weights,
                         bool early_pass, float pass_logit, Rng *rng) {
    /* Uniform fast-path: skip feature extraction in the early game where the
     * policy is ≈ uniform.  Threshold is board fullness (cap-empty)/cap in [0,1].
     * g2_random_legal_move only reorders the empty list, so the const-cast is
     * logically safe. */
    if (ppat_uniform_below_phase > 0) {
        float fullness = (float)(g->cap - g->empty_count) / g->cap;
        if (fullness < ppat_uniform_below_phase) return g2_random_legal_move((Game2 *)g, rng);
    }

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

    /* PASS as a candidate, at logit 0, only when the model was trained for it
     * (early_pass, which travels with the weights; ppat-lib.js gates on the same
     * flag, read from the file).
     * The anchor adds no parameter — it IDENTIFIES one that already existed and
     * was unconstrained, since adding a constant to every pattern weight shifts
     * all logits equally and leaves the softmax unchanged.  With it pinned, the
     * absolute level of the board logits means "how good a move must be to be
     * worth playing", and SB fits that like any other weight.  Without it, the
     * policy can only pass by exhausting the move list, so it must spend its
     * probability mass on the board however bad the options are. */
    if (early_pass && pass_logit > mx) mx = pass_logit;

    /* Compute unnormalized weights, sum, and sample */
    float sum = 0;
    for (int i = 0; i < n; i++) {
        float e = fast_expf(logits_buf[i] - mx);
        logits_buf[i] = e;
        sum += e;
    }
    float e_pass = early_pass ? fast_expf(pass_logit - mx) : 0.0f;
    sum += e_pass;

    float r = rng_float(rng) * sum;
    if (early_pass) { r -= e_pass; if (r <= 0) return PASS; }
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
                       bool early_pass, float pass_weight, const char *comment) {
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "ppat_save_weights: cannot open %s\n", path); return; }
    fprintf(f, "'use strict';\n");
    if (comment) fprintf(f, "// %s\n", comment);
    fprintf(f, "const _w = { weights: new Float32Array([");
    for (int i = 0; i < total; i++) {
        if (i > 0) fputc(',', f);
        fprintf(f, "%.9g", weights[i]);
    }
    /* earlyPass travels with the weights: a model trained against the pass
     * anchor is a different policy from one trained without it, and its
     * absolute logit level is only meaningful with the anchor in place. */
    fprintf(f, "]), phases: %d, numPatterns: %d, libCap: %d, earlyPass: %s, passWeight: %.9g, twelvecell: %s, twelvecell2: %s, selfAtari: %s };\n",
            ppat_phase_count, ppat_num_patterns, ppat_lib_cap,
            early_pass ? "true" : "false", pass_weight,
            ppat_twelvecell == 1 ? "true" : "false",
            ppat_twelvecell == 2 ? "true" : "false",
            ppat_self_atari ? "true" : "false");
    fprintf(f, "if (typeof module !== 'undefined') module.exports = _w;\n");
    fprintf(f, "else window.PPATWeights = _w;\n");
    fclose(f);
}

float *ppat_load_weights(const char *path, bool *out_early_pass, float *out_pass_weight) {
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
    /* libCap travels with the model; files predating the field are cap 2.  Build
     * the canon table for the file's cap BEFORE checking numPatterns, so the
     * check compares like with like (and catches a genuine mismatch). */
    const char *lc = strstr(buf, "libCap:");
    int file_cap = lc ? atoi(lc + 7) : PPAT_LEGACY_LIB_CAP;
    ppat_init(file_cap);
    /* earlyPass travels with the model, exactly as in ppat-lib.js: a file that
     * does not declare it was trained without the pass anchor, so its absolute
     * logit level is arbitrary and offering the pass would produce a frequency
     * that is an accident of training.  Reported to the caller rather than
     * stored: two models with different flags may be played in one process. */
    const char *ep = strstr(buf, "earlyPass:");
    if (out_early_pass) *out_early_pass = ep && strncmp(ep + 10, " true", 5) == 0;
    const char *pw = strstr(buf, "passWeight:");
    if (out_pass_weight) *out_pass_weight = pw ? (float)atof(pw + 11) : 0.0f;
    if (file_np != ppat_num_patterns) {
        fprintf(stderr, "ppat_load_weights: numPatterns: %d in file but %d expected (libCap %d)\n",
                file_np, ppat_num_patterns, file_cap);
        free(buf); return NULL;
    }
    ppat_phase_count = file_phases;

    /* Ladder models are no longer supported (ladder features removed). */
    const char *lp = strstr(buf, "ladder:");
    if (lp) {
        const char *v = lp + 7;
        while (*v == ' ' || *v == '\t') v++;
        if (strncmp(v, "true", 4) == 0) {
            fprintf(stderr, "ppat_load_weights: ladder models are no longer supported (%s)\n", path);
            free(buf); return NULL;
        }
    }

    /* Does the FILE carry the twelvecell block?  Recorded for the caller; the
     * run's own setting governs ppat_twelvecell (a file without the block
     * fine-tunes into it, its 21 new weights starting at zero — that is what
     * the appended layout buys). */
    ppat_file_twelvecell = strstr(buf, "twelvecell2: true") != NULL ? 2
                         : strstr(buf, "twelvecell: true")  != NULL ? 1 : 0;
    ppat_file_self_atari = strstr(buf, "selfAtari: true") != NULL;
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
    if (!ppat_load_quiet)
        fprintf(stderr, "loaded %d weights from %s (phases: %d)\n",
                idx, path, file_phases);
    return weights;
}

