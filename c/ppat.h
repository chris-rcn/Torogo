/*
 * ppat.h — 3×3 pattern + previous-move feature library (C port of ppat-lib.js).
 *
 * Patterns encode the 8 neighbors of a candidate move:
 *   4 orthogonal: 5 states (EMPTY, FRIEND, FRIEND_ATARI, FOE, FOE_ATARI)
 *   4 diagonal:   3 states (EMPTY, FRIEND, FOE)
 * Canonicalised under D4 spatial symmetry (8 transforms).
 *
 * Previous-move features (7 bits):
 *   bit 0: contiguous to previous move (8-neighborhood)
 *   bit 1: save by capture (not self-atari)
 *   bit 2: save by capture (self-atari)
 *   bit 3: save by extension (not self-atari)
 *   bit 4: save by extension (self-atari)
 *   bit 5: ko-solve capture
 *   bit 6: 2-point semeai
 */
#ifndef PPAT_H
#define PPAT_H

#include "game2.h"

/* ── Constants ─────────────────────────────────────────────────────────────── */
/* Orthogonal-neighbour liberty cap.  Each orthogonal is encoded in radix
 * 2*cap+1 (0 empty, 1..cap own with that many liberties, cap+1..2*cap enemy);
 * diagonals are always shape-only (radix 3).  cap 2 is the historical encoding
 * (atari vs not) and the SB paper's; files without a `libCap` field are cap 2.
 * cap 1 degenerates to presence-only (every stone caps to 1 liberty, so the
 * orthogonals encode 0/1/2 like the diagonals) — the whole 3x3 becomes pure
 * shape, useful as an ablation isolating what the atari bit is worth.
 * The cap is a property of a trained model and travels in its weights file.
 * Raw space = (2*cap+1)^4 * 3^4: 6561 (cap 1), 50625 (2), 194481 (3), 531441 (4). */
#define PPAT_MIN_LIB_CAP  1
#define PPAT_MAX_LIB_CAP  4
/* Weight files predating the libCap field are the historical encoding, cap 2.
 * NOT PPAT_MIN_LIB_CAP: that floor dropped to 1 when cap-1 support landed, and
 * tying the legacy default to it silently reinterpreted every old file. */
#define PPAT_LEGACY_LIB_CAP  2
#define PPAT_RAW_SIZE  531441          /* 9^4 * 3^4 — the cap-4 (max) raw space */

/* Widened to 32-bit: cap 4 canonicalises to ~71k patterns, past int16's range.
 * A POINTER, not an array: one table per libCap is built on first use and kept,
 * so alternating caps (a match between models built at different caps) costs a
 * pointer swap instead of a multi-megabyte rebuild. */
extern const int32_t *ppat_canon_id;
extern int32_t ppat_lib_cap;           /* active cap (set by ppat_init) */
extern int32_t ppat_raw_size;          /* (2*cap+1)^4 * 81 for the active cap */
extern int32_t ppat_num_patterns;      /* set by ppat_init() */
extern int     ppat_phase_count;       /* number of game phases (default 1 = no phase splitting) */
extern float   ppat_uniform_below_phase; /* default 0 = off; >0: ppat_policy_move plays uniform random while board fullness (cap-empty)/cap < this fraction [0,1] */
extern int     ppat_load_quiet;        /* default 0; when 1, ppat_load_weights suppresses its success message (errors still print) */

/* Max features per candidate: 1 pattern + 7 prev-move */
#define PPAT_MAX_FEAT 8

/* ── Feature state (reusable across calls) ─────────────────────────────────── */
typedef struct {
    int32_t moves[MAX_CAP];                /* move index per candidate */
    int32_t feat[MAX_CAP * PPAT_MAX_FEAT]; /* flat feature key array */
    int16_t feat_start[MAX_CAP + 1];       /* feat_start[i]..feat_start[i+1] = keys for candidate i */
    uint8_t prev_neighbor_set[MAX_CAP];
    int32_t count;
} PpatState;

/* ── Public API ────────────────────────────────────────────────────────────── */

/* Build the canonical-ID table for `lib_cap` (PPAT_MIN_LIB_CAP..PPAT_MAX_LIB_CAP).
 * Idempotent per cap: a repeat call with the same cap is a no-op.  Must be called
 * before any extraction, and again if a loaded model uses a different cap. */
void     ppat_init(int lib_cap);

/* Total weight count: phase_count * (num_patterns + 7) */
static inline int ppat_total_weights(void) {
    return ppat_phase_count * (ppat_num_patterns + 7);
}

/* Extract features for all legal non-true-eye moves into `st`. */
void     ppat_extract     (const Game2 *g, PpatState *st);
int32_t  ppat_policy_move (const Game2 *g, PpatState *st, const float *weights, Rng *rng);

/* ── Weight file I/O (JS-compatible format) ────────────────────────────────── */

/* Save weights to a JS module file.  comment is optional (NULL ok). */
void  ppat_save_weights(const char *path, const float *weights, int total,
                        const char *comment);

/* Load weights from a JS module file.  Sets ppat_phase_count from the file.
 * Returns malloc'd float array of size ppat_total_weights(), or NULL on error
 * (including legacy ladder files, which are no longer supported).
 * Caller must free(). */
float *ppat_load_weights(const char *path);

#endif /* PPAT_H */
