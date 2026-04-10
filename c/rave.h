/*
 * rave.h — RAVE-MCTS engine for toroidal Go (C port of ai/rave.js).
 *
 * Tree nodes are pool-allocated for zero malloc during search.
 */
#ifndef RAVE_H
#define RAVE_H

#include "game2.h"
#include <math.h>

/* ── Tuning parameters ─────────────────────────────────────────────────────── */
#ifndef RAVE_EXPLORATION_C
#define RAVE_EXPLORATION_C 0.4f
#endif
#ifndef RAVE_K
#define RAVE_K 800.0f
#endif
#ifndef RAVE_N_EXPAND
#define RAVE_N_EXPAND 2
#endif
#ifndef RAVE_INHERIT
#define RAVE_INHERIT 0.2f
#endif

/* Prior pseudo-counts seeded into wins/visits for new children (50% win rate) */
#define RAVE_PRIOR_WINS   0.001f
#define RAVE_PRIOR_VISITS (2 * RAVE_PRIOR_WINS)

/* Minimum playouts before the "no winning line" resignation triggers */
#define RAVE_RESIGN_MIN_PLAYOUTS 20000

/* Max legal moves per position (CAP non-eyes + 1 pass) */
#define MAX_MOVES (CAP + 1)

/* Nodes per chunk in the pool allocator */
#define RAVE_CHUNK_SIZE 1024

/* ── Tree node ─────────────────────────────────────────────────────────────── */
typedef struct RaveNode RaveNode;

struct RaveNode {
    int32_t   move;             /* move that led to this node */
    RaveNode *parent;
    int32_t   ci;               /* index in parent's children array */
    int8_t    mover;            /* player who made `move` */
    float     total_visits;
    int32_t   selected_child;

    int32_t   legal_moves[MAX_MOVES];
    int32_t   num_moves;
    RaveNode *children[MAX_MOVES];  /* NULL = not yet promoted */

    float     wins[MAX_MOVES];
    float     visits[MAX_MOVES];

    float     rave_wins[CAP];
    float     rave_visits[CAP];
};

/* ── Node pool (linked chunks, never moves existing nodes) ─────────────────── */
typedef struct RaveChunk RaveChunk;
struct RaveChunk {
    RaveNode  nodes[RAVE_CHUNK_SIZE];
    RaveChunk *next;
};

/* ── Search state ──────────────────────────────────────────────────────────── */
typedef struct {
    RaveChunk *head;        /* first chunk (owned) */
    RaveChunk *cur;         /* current chunk being filled */
    int32_t    cur_used;    /* nodes used in current chunk */
    int32_t    total_used;  /* total nodes allocated across all chunks */
    float      played[CAP]; /* playout tracking buffer */
} RaveState;

/* ── Result ────────────────────────────────────────────────────────────────── */
typedef struct {
    int32_t   move;             /* best move (flat index or PASS) */
    float     win_ratio;        /* root win ratio */
    int32_t   playouts;         /* total playouts performed */
} RaveResult;

/* ── Public API ────────────────────────────────────────────────────────────── */

RaveState  *rave_create(void);
void        rave_destroy(RaveState *s);

/* Search.  playout_limit > 0: fixed count.  playout_limit == 0: use time_ms. */
RaveResult  rave_search(RaveState *s, const Game2 *root_game,
                        int playout_limit, int time_ms);

#endif /* RAVE_H */
