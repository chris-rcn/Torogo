/*
 * game3.h — Fully reversible toroidal Go engine (C port of game3.js).
 *
 * Game3 is an alternative engine to Game2 where every state-modifying primitive
 * is recorded onto an operation stack so it can be exactly reversed.  There is
 * no clone(); instead callers `g3_play(...)` and later `g3_undo(...)`.
 *
 * Instances are heap-allocated and sized to the actual N at construction.  No
 * global topology table — each Game3 carries its own neighbor arrays.
 */
#ifndef GAME3_H
#define GAME3_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#include "game2.h"   /* for Game2 (used by g3_from_game2) */

/* ── Constants ─────────────────────────────────────────────────────────────── */
#ifndef G3_EMPTY
#define G3_EMPTY  0
#define G3_BLACK  1
#define G3_WHITE -1
#define G3_PASS  -1
#endif

/* ── Op-stack record types ─────────────────────────────────────────────────── */
typedef enum {
    G3_OP_ADD_STONE = 0,
    G3_OP_REMOVE_STONE,
    G3_OP_ADD_LIBERTY,
    G3_OP_REMOVE_LIBERTY,
    G3_OP_MERGE_GROUPS,
    G3_OP_MOVE,
    G3_OP_PASS,
} G3OpType;

/* Tagged-union op record.  The fields used depend on `type`. */
typedef struct {
    uint8_t type;
    union {
        /* ADD_STONE / REMOVE_STONE */
        struct { int32_t idx; int32_t gid; } stone;
        /* ADD_LIBERTY / REMOVE_LIBERTY */
        struct { int32_t gid; int32_t idx; } liberty;
        /* MERGE_GROUPS — main_libs is a W-word snapshot borrowed from the
         * pool; it is returned to the pool when this op is undone. */
        struct {
            int32_t   main_gid;
            int32_t   other_id;
            uint32_t *main_libs;
        } merge;
        /* MOVE — full snapshot of pre-move scalar state plus indices into
         * the captures stack. */
        struct {
            int32_t move;
            int8_t  color;
            int8_t  previous_current;
            int32_t previous_ko;
            int32_t previous_empty_count;
            int32_t previous_consecutive_passes;
            int32_t previous_last_move;
            int32_t previous_next_gid;
            int32_t ops_start;
            int32_t captures_start;
            int32_t captures_count;
        } move;
        /* PASS */
        struct {
            int8_t  previous_current;
            int32_t previous_consecutive_passes;
        } pass;
    };
} G3Op;

/* Captured-stone record (held in a separate stack so MOVE ops stay POD). */
typedef struct { int32_t idx; int32_t gid; } G3Capture;

/* ── Game3 ─────────────────────────────────────────────────────────────────── */
typedef struct Game3 {
    /* Board geometry */
    int32_t N;
    int32_t cap;          /* N * N */
    int32_t W;            /* bitset width: (cap + 31) >> 5 */
    int32_t max_g;        /* 4 * cap + 4 */

    /* Board state */
    int8_t  *cells;       /* [cap] */
    int32_t *gid;         /* [cap], -1 if empty */
    int32_t  next_gid;

    /* Game state */
    int8_t   current;
    int32_t  ko;
    int32_t  empty_count;
    int32_t  move_count;
    int32_t  last_move;
    bool     game_over;
    int32_t  consecutive_passes;

    /* Group arrays (indexed by gid) */
    uint8_t  *gc;         /* [max_g] group color */
    int32_t  *ss;         /* [max_g] stone count */
    int32_t  *ls;         /* [max_g] liberty count */
    uint32_t *sw;         /* [max_g * W] stone bitset */
    uint32_t *lw;         /* [max_g * W] liberty bitset */

    /* Topology (per-instance) */
    int32_t *nbr;         /* [cap * 4] orthogonal neighbors (N,S,W,E) */
    int32_t *dnbr;        /* [cap * 4] diagonal neighbors  (NW,NE,SW,SE) */

    /* Op stack */
    G3Op    *ops;
    int32_t  op_count;
    int32_t  op_capacity;

    /* Captures stack (parallel to op stack — MOVE ops reference slices) */
    G3Capture *captures;
    int32_t    capture_count;
    int32_t    capture_capacity;

    /* Word-buffer pool: free list of W-word uint32_t arrays */
    uint32_t **pool;
    int32_t    pool_count;
    int32_t    pool_capacity;
} Game3;

/* ── Lifecycle ─────────────────────────────────────────────────────────────── */
Game3 *g3_new(int32_t N);          /* malloc + init; empty board, BLACK to move */
void   g3_free(Game3 *g);
void   g3_reset(Game3 *g);          /* return to constructor state */

/* ── Move primitives ───────────────────────────────────────────────────────── */
bool   g3_play(Game3 *g, int32_t move);   /* returns false if illegal */
bool   g3_undo(Game3 *g);                  /* returns false if op stack empty */

/* ── Legality / eye queries ────────────────────────────────────────────────── */
bool   g3_is_legal(const Game3 *g, int32_t idx);              /* uses g->current */
bool   g3_is_legal_for(const Game3 *g, int32_t idx, int8_t color);
bool   g3_is_valid_move(const Game3 *g, int32_t idx);         /* legal + not true eye */
bool   g3_is_true_eye(const Game3 *g, int32_t idx);

/* Internal helpers (exposed for inlining at call sites that already know color) */
bool   g3_is_single_suicide(const Game3 *g, int32_t idx, int8_t color);
bool   g3_is_multi_suicide (const Game3 *g, int32_t idx, int8_t color);

/* ── Group queries ─────────────────────────────────────────────────────────── */
int32_t g3_group_id_at(const Game3 *g, int32_t idx);
int32_t g3_group_size(const Game3 *g, int32_t gid);
int32_t g3_group_liberty_count(const Game3 *g, int32_t gid);

typedef struct { int32_t count; int32_t lib0; int32_t lib1; } G3GroupLibs2;
G3GroupLibs2 g3_group_libs2(const Game3 *g, int32_t idx);

/* Fills out[] with up to max liberty indices for the group at idx.
 * Returns the total liberty count (which may exceed max). */
int32_t g3_group_libs(const Game3 *g, int32_t idx, int32_t *out, int32_t max);

/* ── Display ───────────────────────────────────────────────────────────────── */
/* Render the board to a freshly allocated string.  Caller frees with
 * g3_free_string().  Stones are rendered as 'X' (BLACK) / 'O' (WHITE) / '.'.
 *   mark_idx   — cell to bracket as `(X)`, or G3_PASS for none
 *                (default in JS = g->last_move; pass G3_PASS to disable)
 *   center_at  — cell to centre the toroidal view on, or G3_PASS for origin
 *   show_axes  — whether to include letter/number axis labels
 */
char  *g3_to_string(const Game3 *g, int32_t mark_idx,
                    int32_t center_at, bool show_axes);
void   g3_free_string(char *s);

/* ── Game2 → Game3 ─────────────────────────────────────────────────────────── */
/* Replay a Game2 position into an existing Game3 by index order.
 * Returns true on success, false if the Game2 position is unreachable by legal
 * play (illegal placement or capture during replay).  Diagnostic is written to
 * stderr on failure. */
bool   g3_from_game2(Game3 *g3, const Game2 *g2);

#endif /* GAME3_H */
