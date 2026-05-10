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
#define PPAT_RAW_SIZE 50625         /* 5^4 * 3^4 */

extern int16_t ppat_canon_id[PPAT_RAW_SIZE];
extern int32_t ppat_num_patterns;   /* set by ppat_init() */
extern int     ppat_phase_count;    /* number of game phases (default 1 = no phase splitting) */

/* Max features per candidate: 1 pattern + 7 prev-move = 8 */
#define PPAT_MAX_FEAT 8

/* ── Feature state (reusable across calls) ─────────────────────────────────── */
typedef struct {
    int32_t moves[MAX_CAP];           /* move index per candidate */
    int32_t feat[MAX_CAP * PPAT_MAX_FEAT]; /* flat feature key array */
    int16_t feat_start[MAX_CAP + 1];  /* feat_start[i]..feat_start[i+1] = keys for candidate i */
    uint8_t prev_neighbor_set[MAX_CAP];
    int32_t count;
} PpatState;

/* ── Public API ────────────────────────────────────────────────────────────── */

void     ppat_init(void);           /* call once at startup (builds canon table) */

/* Total weight count: ppat_phase_count * (ppat_num_patterns + 7) */
static inline int ppat_total_weights(void) {
    return ppat_phase_count * (ppat_num_patterns + 7);
}

void     ppat_extract(const Game2 *g, PpatState *st);
int32_t  ppat_policy_move(const Game2 *g, PpatState *st, const float *weights);

/* ── Weight file I/O (JS-compatible format) ────────────────────────────────── */

/* Save weights to a JS module file.  comment is optional (NULL ok). */
void  ppat_save_weights(const char *path, const float *weights, int total,
                        const char *comment);

/* Load weights from a JS module file.  Sets ppat_phase_count.
 * Returns malloc'd float array of size ppat_total_weights(), or NULL on error.
 * Caller must free(). */
float *ppat_load_weights(const char *path);

#endif /* PPAT_H */
