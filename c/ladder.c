#include "ladder.h"
#include "game2.h"
#include <stdlib.h>
#include <string.h>

/* Forward declarations */
static bool ladder_can_reach_3libs_impl(void *game, int idx, bool use_game3);

/* Get group liberty count and first two liberties
 * For game2: g2_get_group_libs2
 * For game3: game3_group_libs2 */
typedef struct {
  int count;
  int lib0;
  int lib1;
} GroupLibs2;

static GroupLibs2 get_group_libs2(void *game, int idx, bool use_game3) {
  GroupLibs2 result = {0, -1, -1};

  if (use_game3) {
    /* game3_group_libs2 */
    extern GroupLibs2 game3_group_libs2(void *game, int idx);
    result = game3_group_libs2(game, idx);
  } else {
    /* game2 version - we need to implement this */
    Game2 *g = (Game2 *)game;
    if (g->gid[idx] == -1) return result;

    int gid = g->gid[idx];
    int lc = g->ls[gid];
    result.count = lc;

    if (lc == 0) return result;

    /* Find first two liberties */
    int found = 0;
    for (int wi = 0; wi < BW && found < 2; wi++) {
      uint32_t w = g->lw[gid * BW + wi];
      while (w && found < 2) {
        int bit = 31 - __builtin_clz(w & -w);
        int lib_idx = wi * 32 + bit;
        if (lib_idx < CAP) {
          if (found == 0) result.lib0 = lib_idx;
          else if (found == 1) result.lib1 = lib_idx;
          found++;
        }
        w &= w - 1;
      }
    }
  }

  return result;
}

static int get_cell(void *game, int idx, bool use_game3) {
  if (use_game3) {
    extern int8_t* game3_get_cells(void *game);
    return game3_get_cells(game)[idx];
  } else {
    Game2 *g = (Game2 *)game;
    return g->cells[idx];
  }
}

static int get_current_player(void *game, bool use_game3) {
  if (use_game3) {
    extern int game3_get_current(void *game);
    return game3_get_current(game);
  } else {
    Game2 *g = (Game2 *)game;
    return g->current;
  }
}

static bool play_move(void *game, int idx, bool use_game3) {
  if (use_game3) {
    extern bool game3_play(void *game, int move);
    return game3_play(game, idx);
  } else {
    Game2 *g = (Game2 *)game;
    return g2_play(g, idx);
  }
}

static void undo_move(void *game, bool use_game3) {
  if (use_game3) {
    extern void game3_undo(void *game);
    game3_undo(game);
  } else {
    /* game2 doesn't support undo, need to use clone-based approach */
    /* This is handled at a higher level */
  }
}

static void clone_game(void *dst, void *src, bool use_game3) {
  if (use_game3) {
    /* game3 doesn't need cloning - uses undo */
  } else {
    g2_clone((Game2 *)dst, (const Game2 *)src);
  }
}

/* Recursive ladder analysis for game3 (uses play/undo) */
static bool ladder_can_reach_3libs_game3(void *game, int idx) {
  GroupLibs2 gl2 = get_group_libs2(game, idx, true);
  int lc = gl2.count;

  if (lc >= 3) return true;
  if (lc == 0) return false;

  int defColor = get_cell(game, idx, true);
  int current = get_current_player(game, true);
  bool defending = (defColor == current);

  int libs[2] = {gl2.lib0, gl2.lib1};
  int libCount = (lc == 1) ? 1 : 2;

  if (defending) {
    /* Defender's turn: needs at least one move that leads to safety */
    for (int i = 0; i < libCount; i++) {
      int lib_idx = libs[i];
      if (!play_move(game, lib_idx, true)) continue;

      bool captured = (get_cell(game, idx, true) == 0);
      bool result = !captured && ladder_can_reach_3libs_game3(game, idx);
      undo_move(game, true);

      if (result) return true;
    }
    return false;
  } else {
    /* Attacker's turn: tries to capture quickly */
    for (int i = 0; i < libCount; i++) {
      int lib_idx = libs[i];
      if (!play_move(game, lib_idx, true)) continue;

      bool captured = (get_cell(game, idx, true) == 0);
      if (captured) {
        undo_move(game, true);
        return false;
      }

      GroupLibs2 afterGl2 = get_group_libs2(game, idx, true);
      int afterLc = afterGl2.count;

      if (afterLc == 0) {
        undo_move(game, true);
        return false;
      }

      if (afterLc == 1) {
        bool result = !ladder_can_reach_3libs_game3(game, idx);
        undo_move(game, true);
        if (result) return false;
      } else {
        undo_move(game, true);
      }
    }
  }

  return true;
}

/* Recursive ladder analysis for game2 (uses cloning) */
static bool ladder_can_reach_3libs_game2(void *game_base, int idx) {
  Game2 *game = (Game2 *)game_base;
  Game2 clone_storage;

  GroupLibs2 gl2 = get_group_libs2(game, idx, false);
  int lc = gl2.count;

  if (lc >= 3) return true;
  if (lc == 0) return false;

  int defColor = get_cell(game, idx, false);
  int current = get_current_player(game, false);
  bool defending = (defColor == current);

  int libs[2] = {gl2.lib0, gl2.lib1};
  int libCount = (lc == 1) ? 1 : 2;

  if (defending) {
    for (int i = 0; i < libCount; i++) {
      int lib_idx = libs[i];
      g2_clone(&clone_storage, game);
      if (!g2_play(&clone_storage, lib_idx)) continue;

      bool captured = (clone_storage.cells[idx] == 0);
      bool result = !captured && ladder_can_reach_3libs_game2(&clone_storage, idx);

      if (result) return true;
    }
    return false;
  } else {
    for (int i = 0; i < libCount; i++) {
      int lib_idx = libs[i];
      g2_clone(&clone_storage, game);
      if (!g2_play(&clone_storage, lib_idx)) continue;

      bool captured = (clone_storage.cells[idx] == 0);
      if (captured) return false;

      GroupLibs2 afterGl2 = get_group_libs2(&clone_storage, idx, false);
      int afterLc = afterGl2.count;

      if (afterLc == 0) return false;

      if (afterLc == 1) {
        bool result = !ladder_can_reach_3libs_game2(&clone_storage, idx);
        if (result) return false;
      }
    }
  }

  return true;
}

bool ladder_can_reach_3libs(void *game, int idx, bool use_game3) {
  if (use_game3) {
    return ladder_can_reach_3libs_game3(game, idx);
  } else {
    return ladder_can_reach_3libs_game2(game, idx);
  }
}

LadderStatus ladder_get_status(void *game, int stone_idx, bool use_game3) {
  LadderStatus result = {0, 0};

  GroupLibs2 gl2 = get_group_libs2(game, stone_idx, use_game3);
  result.liberty_count = gl2.count;

  if (gl2.count < 1 || gl2.count > 2) {
    return result;
  }

  result.can_escape = ladder_can_reach_3libs(game, stone_idx, use_game3);
  return result;
}

LadderResults* ladder_get_all_statuses(void *game, bool use_game3, int min_chain_size) {
  LadderResults *results = (LadderResults *)malloc(sizeof(LadderResults));
  results->capacity = 64;
  results->count = 0;
  results->results = (LadderResult *)malloc(results->capacity * sizeof(LadderResult));

  int cap = CAP;
  int visited_capacity = 64;
  int *visited = (int *)malloc(visited_capacity * sizeof(int));
  int visited_count = 0;

  for (int i = 0; i < cap; i++) {
    int cell = get_cell(game, i, use_game3);
    if (cell == 0) continue;

    int gid = use_game3 ?
      ((extern int game3_group_id_at(void*, int); game3_group_id_at(game, i))) :
      (((Game2 *)game)->gid[i]);

    if (gid == -1) continue;

    /* Check if visited */
    bool found = false;
    for (int j = 0; j < visited_count; j++) {
      if (visited[j] == gid) {
        found = true;
        break;
      }
    }
    if (found) continue;

    /* Add to visited */
    if (visited_count >= visited_capacity) {
      visited_capacity *= 2;
      visited = (int *)realloc(visited, visited_capacity * sizeof(int));
    }
    visited[visited_count++] = gid;

    /* Check group size */
    int group_size = use_game3 ?
      ((extern int game3_group_size(void*, int); game3_group_size(game, gid))) :
      (((Game2 *)game)->ss[gid]);

    if (group_size < min_chain_size) continue;

    /* Check liberties */
    GroupLibs2 gl2 = get_group_libs2(game, i, use_game3);
    if (gl2.count == 0 || gl2.count > 2) continue;

    /* Get ladder status */
    LadderStatus status = ladder_get_status(game, i, use_game3);

    /* Add result */
    if (results->count >= results->capacity) {
      results->capacity *= 2;
      results->results = (LadderResult *)realloc(results->results,
                                                  results->capacity * sizeof(LadderResult));
    }
    results->results[results->count].gid = gid;
    results->results[results->count].color = cell;
    results->results[results->count].status = status;
    results->count++;
  }

  free(visited);
  return results;
}

void ladder_free_results(LadderResults *results) {
  if (!results) return;
  free(results->results);
  free(results);
}
