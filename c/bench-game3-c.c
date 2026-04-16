#include "game3.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Benchmark: Random games with play/undo
void bench_random_games() {
  int num_games = 200;
  int board_size = 13;
  int moves_per_game = 50;

  printf("Benchmarking Game3-C: %d games, %dx%d board, ~%d moves/game\n",
         num_games, board_size, board_size, moves_per_game);

  clock_t start = clock();

  int total_moves = 0;
  for (int game_num = 0; game_num < num_games; game_num++) {
    Game3 *game = game3_new(board_size);

    for (int move = 0; move < board_size * board_size && total_moves < num_games * moves_per_game; move++) {
      if (game3_is_legal(game, move, game->current)) {
        game3_play(game, move);
        total_moves++;
      }
    }

    game3_free(game);
  }

  clock_t end = clock();
  double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

  printf("Results:\n");
  printf("  Total moves: %d\n", total_moves);
  printf("  Time: %.3f seconds\n", elapsed);
  printf("  Moves per second: %.0f\n", total_moves / elapsed);
}

// Benchmark: Play/undo cycles (ladder analysis pattern)
void bench_play_undo_cycles() {
  int num_games = 100;
  int board_size = 13;
  int setup_moves = 20;
  int analysis_depth = 5;

  printf("\nBenchmarking Game3-C play/undo cycles: %d games, depth %d\n",
         num_games, analysis_depth);

  clock_t start = clock();

  int total_operations = 0;

  for (int game_num = 0; game_num < num_games; game_num++) {
    Game3 *game = game3_new(board_size);

    // Set up a game state
    int move_count = 0;
    for (int i = 0; i < board_size * board_size && move_count < setup_moves; i++) {
      if (game3_is_legal(game, i, game->current)) {
        game3_play(game, i);
        move_count++;
      }
    }

    // Analyze positions with play/undo
    for (int analyze = 0; analyze < 10 && analyze < board_size * board_size; analyze++) {
      for (int depth = 0; depth < analysis_depth; depth++) {
        if (game3_is_legal(game, analyze, game->current)) {
          game3_play(game, analyze);
          total_operations++;
        } else {
          break;
        }
      }

      for (int depth = 0; depth < analysis_depth; depth++) {
        game3_undo(game);
        total_operations++;
      }
    }

    game3_free(game);
  }

  clock_t end = clock();
  double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

  printf("Results:\n");
  printf("  Total operations (play + undo): %d\n", total_operations);
  printf("  Time: %.3f seconds\n", elapsed);
  printf("  Operations per second: %.0f\n", total_operations / elapsed);
}

// Benchmark: State verification (comparing board state repeatedly)
void bench_state_verification() {
  int num_games = 100;
  int board_size = 13;

  printf("\nBenchmarking Game3-C state verification: %d games\n", num_games);

  clock_t start = clock();

  int total_checks = 0;

  for (int game_num = 0; game_num < num_games; game_num++) {
    Game3 *game = game3_new(board_size);

    for (int i = 0; i < board_size * board_size && i < 50; i++) {
      if (game3_is_legal(game, i, game->current)) {
        game3_play(game, i);

        // Verify state
        int empty = 0;
        int black = 0;
        int white = 0;
        for (int j = 0; j < board_size * board_size; j++) {
          if (game->cells[j] == EMPTY) empty++;
          else if (game->cells[j] == BLACK) black++;
          else if (game->cells[j] == WHITE) white++;
        }
        total_checks += 3;
      }
    }

    game3_free(game);
  }

  clock_t end = clock();
  double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

  printf("Results:\n");
  printf("  Total state checks: %d\n", total_checks);
  printf("  Time: %.3f seconds\n", elapsed);
  printf("  Checks per second: %.0f\n", total_checks / elapsed);
}

// Benchmark: Group operations
void bench_group_operations() {
  int num_games = 100;
  int board_size = 13;

  printf("\nBenchmarking Game3-C group operations: %d games\n", num_games);

  clock_t start = clock();

  int total_queries = 0;

  for (int game_num = 0; game_num < num_games; game_num++) {
    Game3 *game = game3_new(board_size);

    for (int i = 0; i < board_size * board_size && i < 50; i++) {
      if (game3_is_legal(game, i, game->current)) {
        game3_play(game, i);

        // Query group info for all stones
        for (int j = 0; j < board_size * board_size; j++) {
          if (game->cells[j] != EMPTY) {
            int gid = game3_group_id_at(game, j);
            if (gid != -1) {
              int size = game3_group_size(game, gid);
              int libs = game3_group_liberty_count(game, gid);
              GroupLibs2 gl2 = game3_group_libs2(game, j);
              total_queries += 4;
            }
          }
        }
      }
    }

    game3_free(game);
  }

  clock_t end = clock();
  double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

  printf("Results:\n");
  printf("  Total queries: %d\n", total_queries);
  printf("  Time: %.3f seconds\n", elapsed);
  printf("  Queries per second: %.0f\n", total_queries / elapsed);
}

int main() {
  printf("Game3-C Performance Benchmarks\n");
  printf("============================================================\n\n");

  bench_random_games();
  bench_play_undo_cycles();
  bench_state_verification();
  bench_group_operations();

  printf("\n============================================================\n");
  printf("Benchmarks complete!\n");

  return 0;
}
