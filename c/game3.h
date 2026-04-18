#ifndef GAME3_H
#define GAME3_H

#include <stdint.h>
#include <stdbool.h>

// Constants
#define EMPTY 0
#define BLACK 1
#define WHITE -1
#define PASS -1

// Operation types
typedef enum {
  OP_ADD_STONE,
  OP_REMOVE_STONE,
  OP_ADD_LIBERTY,
  OP_REMOVE_LIBERTY,
  OP_MERGE_GROUPS,
  OP_MOVE,
  OP_PASS,
} OpType;

// Operation structure
typedef struct {
  OpType type;

  // Common fields
  int idx;
  int gid;

  // For MERGE_GROUPS
  int mainGid;
  int otherId;
  int32_t *mainStones;
  int32_t *mainLibs;
  int32_t *otherStones;
  int32_t *otherLibs;
  int mainSize;
  int mainLibCount;
  int otherSize;
  int otherLibCount;
  int W;  // bitset width

  // For MOVE
  int move;
  int color;
  int previousCurrent;
  int previousKo;
  int previousEmptyCount;
  int previousConsecutivePasses;
  int previousLastMove;
  int opsStart;

  // For captured stones (move undo)
  struct {
    int idx;
    int gid;
  } *captured;
  int capturedCount;

  // For PASS
  int previousPassCurrent;
  int previousPassConsecutivePasses;
} Operation;

// Operation stack
typedef struct {
  Operation *ops;
  int count;
  int capacity;
} OpStack;

// Game3 structure
typedef struct {
  // Board state
  int N;
  int boardSize;
  int8_t *cells;
  int current;
  int ko;
  int emptyCount;
  int moveCount;
  int lastMove;
  bool gameOver;
  int consecutivePasses;

  // Group tracking
  int32_t *gid;
  int nextGid;

  // Bitsets
  int W;  // width in 32-bit words
  uint8_t *gc;    // group color
  int32_t *sw;    // stones bitset
  int32_t *ss;    // stones count
  int32_t *lw;    // liberties bitset
  int32_t *ls;    // liberties count

  // Topology
  int32_t *nbr;      // neighbors (4 per cell)
  int32_t *dnbr;     // diagonal neighbors (4 per cell)

  // Operation stack
  OpStack opStack;

  // Temporary arrays
  int *tempIntArr;
  int tempIntArrSize;
} Game3;

// Function declarations
Game3* game3_new(int N);
void game3_free(Game3 *game);

// Core gameplay
bool game3_is_legal(Game3 *game, int idx, int color);
bool game3_is_valid_move(Game3 *game, int idx, int color);
bool game3_play(Game3 *game, int move);
void game3_undo(Game3 *game);

// Group queries
int32_t game3_group_id_at(Game3 *game, int idx);
int game3_group_size(Game3 *game, int gid);
int game3_group_liberty_count(Game3 *game, int gid);
typedef struct {
  int count;
  int lib0;
  int lib1;
} GroupLibs2;
GroupLibs2 game3_group_libs2(Game3 *game, int idx);

// Display
char* game3_to_string(Game3 *game, int markIdx);
void game3_free_string(char *str);

// Eye detection
bool game3_is_true_eye(Game3 *game, int idx);

#endif  // GAME3_H
