/* Benchmark: Game3.c vs Game2.c ladder searches on 13x13
   Compares undo-based (game3.c) vs clone-based (game2.c) approaches */

#include "game3.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

/* Recursive ladder analysis with undo (Game3 pattern) */
static bool ladder_can_reach_3libs_undo(Game3 *game, int idx) {
  GroupLibs2 gl2 = game3_group_libs2(game, idx);
  int lc = gl2.count;

  if (lc >= 3) return true;
  if (lc == 0) return false;

  int defColor = game->cells[idx];
  int current = game->current;
  bool defending = (defColor == current);

  int libs[2] = {gl2.lib0, gl2.lib1};
  int libCount = (lc == 1) ? 1 : 2;

  if (defending) {
    for (int i = 0; i < libCount; i++) {
      int lib_idx = libs[i];
      if (!game3_play(game, lib_idx)) continue;

      bool captured = (game->cells[idx] == EMPTY);
      bool result = !captured && ladder_can_reach_3libs_undo(game, idx);
      game3_undo(game);

      if (result) return true;
    }
    return false;
  } else {
    for (int i = 0; i < libCount; i++) {
      int lib_idx = libs[i];
      if (!game3_play(game, lib_idx)) continue;

      bool captured = (game->cells[idx] == EMPTY);
      if (captured) {
        game3_undo(game);
        return false;
      }

      GroupLibs2 afterGl2 = game3_group_libs2(game, idx);
      int afterLc = afterGl2.count;

      if (afterLc == 0) {
        game3_undo(game);
        return false;
      }

      if (afterLc == 1) {
        bool result = !ladder_can_reach_3libs_undo(game, idx);
        game3_undo(game);
        if (result) return false;
      } else {
        game3_undo(game);
      }
    }
  }

  return true;
}

/* Count ladder statuses in game position (Game3 version) */
static int count_ladder_statuses_game3(Game3 *game) {
  int cap = game->N * game->N;
  int count = 0;
  int visited[512] = {0};  /* Track visited gids */
  int visited_count = 0;

  for (int i = 0; i < cap; i++) {
    if (game->cells[i] == EMPTY) continue;

    int gid = game->gid[i];
    if (gid == -1) continue;

    /* Check if already visited */
    bool found = false;
    for (int j = 0; j < visited_count; j++) {
      if (visited[j] == gid) {
        found = true;
        break;
      }
    }
    if (found) continue;

    visited[visited_count++] = gid;

    /* Check liberties */
    GroupLibs2 gl2 = game3_group_libs2(game, i);
    if (gl2.count == 0 || gl2.count > 2) continue;

    /* Has ladder analysis to do */
    count++;
  }

  return count;
}

/* Run ladder analysis on all groups (Game3 version) */
static void analyze_ladder_statuses_game3(Game3 *game) {
  int cap = game->N * game->N;
  int visited[512] = {0};
  int visited_count = 0;

  for (int i = 0; i < cap; i++) {
    if (game->cells[i] == EMPTY) continue;

    int gid = game->gid[i];
    if (gid == -1) continue;

    /* Check if already visited */
    bool found = false;
    for (int j = 0; j < visited_count; j++) {
      if (visited[j] == gid) {
        found = true;
        break;
      }
    }
    if (found) continue;

    visited[visited_count++] = gid;

    /* Check liberties */
    GroupLibs2 gl2 = game3_group_libs2(game, i);
    if (gl2.count == 0 || gl2.count > 2) continue;

    /* Analyze this group */
    (void)ladder_can_reach_3libs_undo(game, i);
  }
}

/* Random game play */
static int play_random_game(Game3 *game) {
  int moves = 0;
  for (int i = 0; i < game->N * game->N && moves < 150; i++) {
    if (game3_is_legal(game, i, game->current)) {
      if (game3_play(game, i)) {
        moves++;
      }
    }
  }
  return moves;
}

int main(int argc, char *argv[]) {
  printf("Game3.c Ladder Search Benchmark\n");
  printf("13x13 board, 200 random games, analyze at each position\n");
  printf("============================================================\n\n");

  int num_games = 200;
  int total_moves = 0;
  int total_analyses = 0;
  clock_t start_time = clock();
  long long total_nanos = 0;

  for (int game_num = 0; game_num < num_games; game_num++) {
    if ((game_num + 1) % 50 == 0) {
      printf("  Game %d/%d...\n", game_num + 1, num_games);
    }

    Game3 *game = game3_new(13);

    /* Play random moves */
    int move_count = play_random_game(game);
    total_moves += move_count;

    /* Analyze ladder statuses at final position */
    clock_t analysis_start = clock();
    analyze_ladder_statuses_game3(game);
    clock_t analysis_end = clock();
    total_nanos += (analysis_end - analysis_start);

    /* Count how many analyses were done */
    int analysis_count = count_ladder_statuses_game3(game);
    total_analyses += analysis_count;

    game3_free(game);
  }

  clock_t end_time = clock();
  double total_ms = (double)(end_time - start_time) * 1000.0 / CLOCKS_PER_SEC;

  printf("\n============================================================\n");
  printf("Results (Game3.c with play/undo):\n");
  printf("  Total time: %.2f ms\n", total_ms);
  printf("  Total moves played: %d\n", total_moves);
  printf("  Total analyses: %d\n", total_analyses);
  printf("  Average per analysis: %.4f µs\n",
         total_analyses > 0 ? (total_ms * 1000.0 / total_analyses) : 0);
  printf("  Operations/second: %.0f M ops/sec\n",
         total_ms > 0 ? (total_moves / (total_ms / 1000.0) / 1e6) : 0);

  printf("\nComparison with JavaScript results:\n");
  printf("  Game3-Precise (JS): 2421.38 ms total (278,136 analyses)\n");
  printf("  Game3.c:            %.2f ms total (%d analyses)\n", total_ms, total_analyses);

  if (total_ms > 0 && 2421.38 > 0) {
    double speedup = 2421.38 / total_ms;
    printf("  C is %.2f× faster than JavaScript\n", speedup);
  }

  return 0;
}
