/*
 * ladder2.h — Ladder detection on Game3 using play/undo (C port of ladder2.js).
 *
 * A "ladder" here is a group with 1 or 2 liberties whose fate is decided by a
 * forced sequence of atari moves.  ladder2_can_reach_3libs returns whether the
 * group at idx can be played out to 3+ liberties despite best-attacker play.
 *
 * All public entry points use Game3's play/undo to explore the search tree
 * without cloning.  After each call, the game state is left exactly as it
 * was on entry.
 */
#ifndef LADDER2_H
#define LADDER2_H

#include <stdint.h>
#include <stdbool.h>

#include "game3.h"

#define LADDER2_MAX_LIBS 2
/* A defender's saving moves include captures of adjacent atari'd enemy chains,
 * not just the group's own liberties, so the urgent set can exceed 2.  This
 * bounds the distinct adjacent-enemy captures considered (far above any real
 * count for a 1–2 liberty group). */
#define LADDER2_MAX_CAPTURES 16
#define LADDER2_MAX_URGENT   (LADDER2_MAX_LIBS + LADDER2_MAX_CAPTURES)

/* Result for one group with 1–2 liberties.
 *   libs            — the group's liberties (lib_count entries, 1 or 2)
 *   mover_succeeds  — true if the side to move has at least one winning option:
 *                       defender ⇒ a play that reaches 3+ liberties
 *                       attacker ⇒ a play that prevents 3+ liberties
 *                     Also true for not-urgent groups (passing keeps it safe).
 *   urgent_libs     — for an urgent group, the mover's winning plays: the
 *                     group's liberties and, when defending, captures of
 *                     adjacent atari'd enemy chains.  Empty when not urgent.
 *   valid           — false if the input group had 0 or >2 liberties (matches
 *                     ladder2.js returning null in that case).
 */
typedef struct {
    int32_t libs[LADDER2_MAX_LIBS];
    int32_t lib_count;
    bool    mover_succeeds;
    int32_t urgent_libs[LADDER2_MAX_URGENT];
    int32_t urgent_count;
    bool    valid;
} Ladder2Status;

typedef struct {
    int32_t       gid;
    int8_t        color;
    Ladder2Status status;
} Ladder2Result;

/* Returns true when the group at idx can reach 3+ liberties despite best
 * attacker play.  Uses play/undo on `g`; state is restored on return. */
bool ladder2_can_reach_3libs(Game3 *g, int32_t idx);

/* Examines the group containing the stone at stone_idx (must have 1 or 2
 * liberties).  Sets status.valid = false if the precondition is violated.
 * State is restored on return. */
Ladder2Status ladder2_get_status(Game3 *g, int32_t stone_idx);

/* Runs ladder2_get_status on every group with 1 or 2 liberties whose stone
 * count is >= min_chain_size.  Writes up to `max` entries into `out`.
 * Returns the total number of qualifying groups found (which may exceed
 * `max` if the caller's buffer was too small).  Caller sizing `out` to
 * `g->cap` is always safe (max one group per cell). */
int32_t ladder2_get_all_statuses(Game3 *g, int32_t min_chain_size,
                                  Ladder2Result *out, int32_t max);

#endif /* LADDER2_H */
