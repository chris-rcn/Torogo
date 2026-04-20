#ifndef LADDER_H
#define LADDER_H

#include <stdint.h>
#include <stdbool.h>

/* Ladder status result */
typedef struct {
  bool can_escape;
  int liberty_count;
} LadderStatus;

/* Can reach 3+ liberties despite best attacker play (recursive with undo)
 * Works with game state, using play/undo pattern. */
bool ladder_can_reach_3libs(void *game, int idx, bool use_game3);

/* Get ladder status for a group with 1-2 liberties */
LadderStatus ladder_get_status(void *game, int stone_idx, bool use_game3);

/* Get all ladder statuses in current position */
typedef struct {
  int gid;
  int8_t color;
  LadderStatus status;
} LadderResult;

typedef struct {
  LadderResult *results;
  int count;
  int capacity;
} LadderResults;

LadderResults* ladder_get_all_statuses(void *game, bool use_game3, int min_chain_size);
void ladder_free_results(LadderResults *results);

#endif /* LADDER_H */
