/*
 * game2.h — Toroidal Go board engine (C port of game2.js).
 *
 * MAX_BOARD_SIZE sets the maximum supported board size (for array allocation).
 * Actual board size is set at runtime via g2_new(g, N).
 * All group tracking uses bitsets for stones and liberties.
 */
#ifndef GAME2_H
#define GAME2_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

/* ── Maximum board size (for static array allocation) ──────────────────────── */
#ifndef MAX_BOARD_SIZE
#define MAX_BOARD_SIZE 13
#endif

#define MAX_CAP   (MAX_BOARD_SIZE * MAX_BOARD_SIZE)
#define MAX_BW    ((MAX_CAP + 31) >> 5)          /* max bitset width in uint32 words */
#define MAX_G     (MAX_CAP / 2 + 4)                /* max group slots (groups are reused) */

/* ── Constants ─────────────────────────────────────────────────────────────── */
#define EMPTY  0
#define BLACK  1
#define WHITE -1
#define PASS  -1

/* ── Neighbor tables (toroidal, precomputed per size) ──────────────────────── */
extern int32_t g2_nbr[MAX_CAP * 4];
extern int32_t g2_dnbr[MAX_CAP * 4];

void g2_init_topology(int N);      /* call once per board size */

/* ── Game state ────────────────────────────────────────────────────────────── */
typedef struct {
    int      N;                      /* board size (runtime) */
    int      cap;                    /* N * N */
    int      W;                      /* bitset width: (cap + 31) >> 5 */

    int8_t   cells[MAX_CAP];
    int16_t  gid[MAX_CAP];          /* group ID per cell, -1 if empty */
    int16_t  next_gid;

    /* Group-slot free list (reuse captured group IDs) */
    int16_t  free_gids[MAX_G];     /* stack of available gids */
    int16_t  n_free_gids;

    /* Small group arrays (indexed by gid) */
    int8_t   gc[MAX_G];             /* group color */
    int16_t  ss[MAX_G];             /* stone size */
    int16_t  ls[MAX_G];             /* liberty count */

    /* Empty-cell tracking (swap-and-pop) */
    int16_t  empty_cells[MAX_CAP];
    int16_t  empty_slot[MAX_CAP];
    int16_t  empty_count;

    /* Game state */
    int8_t   current;               /* BLACK or WHITE */
    int16_t  ko;                    /* PASS or board index */
    int16_t  ko_stone[3];           /* per-color: cell that created a ko on that color's last turn, or PASS.
                                     * Indexed by color+1: [0]=WHITE(-1+1), [1]=unused, [2]=BLACK(1+1). */
    int16_t  consecutive_passes;
    bool     game_over;
    int16_t  move_count;
    int16_t  last_move;

    /* Capture scratch (filled by play) */
    int16_t  last_captures[MAX_CAP];
    int16_t  last_capture_count;

    /* Large bitset arrays — at the end for efficient partial clone */
    uint32_t sw[MAX_G * MAX_BW];    /* stone bitset */
    uint32_t lw[MAX_G * MAX_BW];    /* liberty bitset */
} Game2;

/* ── Public API ────────────────────────────────────────────────────────────── */
void  g2_new(Game2 *g, int N);                   /* init with center stone */
void  g2_new_empty(Game2 *g, int N);             /* init without center stone */
void  g2_clone(Game2 *dst, const Game2 *src);    /* deep copy */
bool  g2_play(Game2 *g, int32_t idx);            /* returns success */
void  g2_play_unchecked(Game2 *g, int32_t idx);  /* skip legality check (caller must ensure legal) */
bool  g2_is_legal(const Game2 *g, int32_t idx);
bool  g2_is_true_eye(const Game2 *g, int32_t idx);
bool  g2_is_capture(const Game2 *g, int32_t idx);
int32_t g2_random_legal_move(Game2 *g);          /* returns idx or PASS */

/* Scoring */
typedef struct { float black; float white; } Score;
Score   g2_estimate_score(const Game2 *g);
int8_t  g2_estimate_winner(const Game2 *g);      /* BLACK or WHITE */

float g2_komi(int N);

/* ── Internal legality helpers (exposed for inlining in hot paths) ─────────── */
bool g2_is_single_suicide(const Game2 *g, int32_t idx, int8_t color);
bool g2_is_multi_suicide(const Game2 *g, int32_t idx, int8_t color);
bool g2_is_ko(const Game2 *g, int32_t idx, int8_t color);

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

/* Parse an ASCII board diagram into a Game2. Infers size from row count. */
void g2_parse_board(Game2 *g, const char *board, int8_t to_move);

#endif /* GAME2_H */
