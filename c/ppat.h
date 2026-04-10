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

extern int32_t ppat_canon_id[PPAT_RAW_SIZE];
extern int32_t ppat_num_patterns;   /* set by ppat_init() */

/* ── Feature state (reusable across calls) ─────────────────────────────────── */
typedef struct {
    int32_t moves[CAP];
    int32_t pat_ids[CAP];
    uint8_t prev_masks[CAP];
    uint8_t prev_neighbor_set[CAP];
    int32_t count;
} PpatState;

/* ── Weights for policy move ───────────────────────────────────────────────── */
typedef struct {
    const float *pat;   /* ppat_num_patterns floats */
    const float *prev;  /* 7 floats */
} PpatWeights;

/* ── Public API ────────────────────────────────────────────────────────────── */

void     ppat_init(void);           /* call once at startup (builds canon table) */
void     ppat_extract(const Game2 *g, PpatState *st);
int32_t  ppat_policy_move(const Game2 *g, PpatState *st, const PpatWeights *w);

#endif /* PPAT_H */
