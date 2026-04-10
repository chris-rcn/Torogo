/*
 * game2.h — Toroidal Go board engine (C port of game2.js).
 *
 * Board size is compile-time configurable via BOARD_SIZE.
 * All group tracking uses bitsets for stones and liberties.
 */
#ifndef GAME2_H
#define GAME2_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

/* ── Board size (compile-time constant) ────────────────────────────────────── */
#ifndef BOARD_SIZE
#define BOARD_SIZE 9
#endif

#define CAP       (BOARD_SIZE * BOARD_SIZE)
#define BW        ((CAP + 31) >> 5)          /* bitset width in uint32 words */
#define MAX_G     (4 * CAP + 4)              /* max group slots */

/* ── Constants ─────────────────────────────────────────────────────────────── */
#define EMPTY  0
#define BLACK  1
#define WHITE -1
#define PASS  -1

/* ── Neighbor tables (toroidal, precomputed once) ──────────────────────────── */
extern int32_t g2_nbr[CAP * 4];   /* orthogonal: [idx*4+0]=N, +1=S, +2=W, +3=E */
extern int32_t g2_dnbr[CAP * 4];  /* diagonal:   [idx*4+0]=NW, +1=NE, +2=SW, +3=SE */

void g2_init_topology(void);      /* call once at startup */

/* ── Game state ────────────────────────────────────────────────────────────── */
typedef struct {
    int8_t   cells[CAP];
    int32_t  gid[CAP];           /* group ID per cell, -1 if empty */
    int32_t  next_gid;

    /* Group arrays (indexed by gid) */
    int8_t   gc[MAX_G];          /* group color */
    int32_t  ss[MAX_G];          /* stone size */
    int32_t  ls[MAX_G];          /* liberty count */
    uint32_t sw[MAX_G * BW];     /* stone bitset */
    uint32_t lw[MAX_G * BW];     /* liberty bitset */

    /* Empty-cell tracking (swap-and-pop) */
    int32_t  empty_cells[CAP];
    int32_t  empty_slot[CAP];
    int32_t  empty_count;

    /* Game state */
    int8_t   current;            /* BLACK or WHITE */
    int32_t  ko;                 /* PASS or board index */
    int32_t  consecutive_passes;
    bool     game_over;
    int32_t  move_count;
    int32_t  last_move;

    /* Capture scratch (filled by play) */
    int32_t  last_captures[CAP];
    int32_t  last_capture_count;
} Game2;

/* ── Public API ────────────────────────────────────────────────────────────── */
void  g2_new(Game2 *g);                        /* init with center stone */
void  g2_new_empty(Game2 *g);                  /* init without center stone */
void  g2_clone(Game2 *dst, const Game2 *src);  /* deep copy */
bool  g2_play(Game2 *g, int32_t idx);          /* returns success */
bool  g2_is_legal(const Game2 *g, int32_t idx);
bool  g2_is_true_eye(const Game2 *g, int32_t idx);
bool  g2_is_capture(const Game2 *g, int32_t idx);
int32_t g2_random_legal_move(Game2 *g);        /* returns idx or PASS */

/* Scoring */
typedef struct { float black; float white; } Score;
Score   g2_estimate_score(const Game2 *g);
int8_t  g2_estimate_winner(const Game2 *g);    /* BLACK or WHITE */

/* ── Inline helpers ────────────────────────────────────────────────────────── */
static inline int g2_popcount(uint32_t x) {
    return __builtin_popcount(x);
}

/* ── Fast RNG (xorshift32, period 2^32-1) ──────────────────────────────────── */
extern uint32_t g2_rng_state;

static inline uint32_t g2_rand(void) {
    uint32_t x = g2_rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return g2_rng_state = x;
}

static inline uint32_t g2_rand_n(uint32_t n) {
    return g2_rand() % n;
}

static inline float g2_randf(void) {
    return (float)(g2_rand() >> 1) / (float)(0x7FFFFFFFu);
}

void g2_seed(uint32_t seed);

static inline float g2_komi(void) {
#if BOARD_SIZE == 5
    return 24.5f;
#elif BOARD_SIZE == 6
    return 35.5f;
#elif BOARD_SIZE == 7
    return 48.5f;
#else
    return 3.5f;
#endif
}

#endif /* GAME2_H */
