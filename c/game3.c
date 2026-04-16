#include "game3.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ────────────────────────────────────────────────────────────────────────────
// Utility functions
// ────────────────────────────────────────────────────────────────────────────

static int pop32(uint32_t x) {
  x = x - ((x >> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
  x = (x + (x >> 4)) & 0x0f0f0f0f;
  return (x * 0x01010101) >> 24;
}

static int clz32(uint32_t x) {
  if (x == 0) return 32;
  int n = 0;
  if (x <= 0x0000FFFF) { n += 16; x <<= 16; }
  if (x <= 0x00FFFFFF) { n += 8; x <<= 8; }
  if (x <= 0x0FFFFFFF) { n += 4; x <<= 4; }
  if (x <= 0x3FFFFFFF) { n += 2; x <<= 2; }
  if (x <= 0x7FFFFFFF) { n += 1; }
  return n;
}

// ────────────────────────────────────────────────────────────────────────────
// Topology initialization
// ────────────────────────────────────────────────────────────────────────────

static void get_topology(int N, int32_t **nbr_out, int32_t **dnbr_out) {
  int cap = N * N;
  int32_t *nbr = (int32_t *)malloc(cap * 4 * sizeof(int32_t));
  int32_t *dnbr = (int32_t *)malloc(cap * 4 * sizeof(int32_t));

  for (int y = 0; y < N; y++) {
    for (int x = 0; x < N; x++) {
      int i = y * N + x;

      // Neighbors (up, down, left, right)
      nbr[i*4+0] = ((y - 1 + N) % N) * N + x;       // up
      nbr[i*4+1] = ((y + 1) % N) * N + x;           // down
      nbr[i*4+2] = y * N + ((x - 1 + N) % N);       // left
      nbr[i*4+3] = y * N + ((x + 1) % N);           // right

      // Diagonal neighbors
      int yu = (y - 1 + N) % N;
      int yd = (y + 1) % N;
      int xl = (x - 1 + N) % N;
      int xr = (x + 1) % N;
      dnbr[i*4+0] = yu * N + xl;      // up-left
      dnbr[i*4+1] = yu * N + xr;      // up-right
      dnbr[i*4+2] = yd * N + xl;      // down-left
      dnbr[i*4+3] = yd * N + xr;      // down-right
    }
  }

  *nbr_out = nbr;
  *dnbr_out = dnbr;
}

// ────────────────────────────────────────────────────────────────────────────
// Operation stack management
// ────────────────────────────────────────────────────────────────────────────

static void opstack_init(OpStack *stack) {
  stack->capacity = 256;
  stack->count = 0;
  stack->ops = (Operation *)malloc(stack->capacity * sizeof(Operation));
  memset(stack->ops, 0, stack->capacity * sizeof(Operation));
}

static void opstack_free(OpStack *stack) {
  for (int i = 0; i < stack->count; i++) {
    Operation *op = &stack->ops[i];
    if (op->mainStones) free(op->mainStones);
    if (op->mainLibs) free(op->mainLibs);
    if (op->otherStones) free(op->otherStones);
    if (op->otherLibs) free(op->otherLibs);
    if (op->captured) free(op->captured);
  }
  free(stack->ops);
}

static void opstack_push(OpStack *stack, Operation op) {
  if (stack->count >= stack->capacity) {
    stack->capacity *= 2;
    stack->ops = (Operation *)realloc(stack->ops, stack->capacity * sizeof(Operation));
  }
  stack->ops[stack->count++] = op;
}

static Operation opstack_pop(OpStack *stack) {
  if (stack->count == 0) {
    Operation empty = {0};
    return empty;
  }
  return stack->ops[--stack->count];
}

static int opstack_len(OpStack *stack) {
  return stack->count;
}

// ────────────────────────────────────────────────────────────────────────────
// Game creation and destruction
// ────────────────────────────────────────────────────────────────────────────

Game3* game3_new(int N) {
  Game3 *game = (Game3 *)malloc(sizeof(Game3));
  game->N = N;
  game->boardSize = N;
  int cap = N * N;

  // Board state
  game->cells = (int8_t *)calloc(cap, sizeof(int8_t));
  game->current = BLACK;
  game->ko = PASS;
  game->emptyCount = cap;
  game->moveCount = 0;
  game->lastMove = PASS;
  game->gameOver = false;
  game->consecutivePasses = 0;

  // Group tracking
  game->gid = (int32_t *)malloc(cap * sizeof(int32_t));
  for (int i = 0; i < cap; i++) game->gid[i] = -1;
  game->nextGid = 0;

  // Bitsets
  game->W = (cap + 31) >> 5;
  int maxGroups = 4 * cap + 4;
  game->gc = (uint8_t *)calloc(maxGroups, sizeof(uint8_t));
  game->sw = (int32_t *)calloc(maxGroups * game->W, sizeof(int32_t));
  game->ss = (int32_t *)calloc(maxGroups, sizeof(int32_t));
  game->lw = (int32_t *)calloc(maxGroups * game->W, sizeof(int32_t));
  game->ls = (int32_t *)calloc(maxGroups, sizeof(int32_t));

  // Topology
  get_topology(N, &game->nbr, &game->dnbr);

  // Operation stack
  opstack_init(&game->opStack);

  // Temporary arrays
  game->tempIntArr = (int *)malloc(16 * sizeof(int));
  game->tempIntArrSize = 0;

  // Initialize center stone
  int center = ((N >> 1) * N) + (N >> 1);
  int centerGid = game->nextGid++;
  game->gc[centerGid] = BLACK;
  game->gid[center] = centerGid;
  game->cells[center] = BLACK;

  // Add center stone to group
  int m = 1 << (center & 31);
  int wi = center >> 5;
  int gb = centerGid * game->W;
  game->sw[gb + wi] |= m;
  game->ss[centerGid]++;
  game->emptyCount--;

  // Add liberties (4 empty neighbors)
  for (int i = 0; i < 4; i++) {
    int ni = game->nbr[center * 4 + i];
    if (game->cells[ni] == EMPTY) {
      int lm = 1 << (ni & 31);
      int lwi = ni >> 5;
      int lb = centerGid * game->W;
      game->lw[lb + lwi] |= lm;
      game->ls[centerGid]++;
    }
  }

  game->current = WHITE;
  game->moveCount = 1;

  return game;
}

void game3_free(Game3 *game) {
  if (!game) return;
  free(game->cells);
  free(game->gid);
  free(game->gc);
  free(game->sw);
  free(game->ss);
  free(game->lw);
  free(game->ls);
  free(game->nbr);
  free(game->dnbr);
  free(game->tempIntArr);
  opstack_free(&game->opStack);
  free(game);
}

// ────────────────────────────────────────────────────────────────────────────
// Raw bitset operations (no operation recording)
// ────────────────────────────────────────────────────────────────────────────

static void add_stone_raw(Game3 *game, int idx, int gid) {
  int m = 1 << (idx & 31);
  int wi = idx >> 5;
  int gb = gid * game->W;
  game->sw[gb + wi] |= m;
  game->ss[gid]++;
}

static void remove_stone_raw(Game3 *game, int idx, int gid) {
  int m = 1 << (idx & 31);
  int wi = idx >> 5;
  int gb = gid * game->W;
  game->sw[gb + wi] &= ~m;
  game->ss[gid]--;
}

static void add_liberty_raw(Game3 *game, int gid, int idx) {
  int m = 1 << (idx & 31);
  int wi = idx >> 5;
  int lb = gid * game->W;
  game->lw[lb + wi] |= m;
  game->ls[gid]++;
}

static void remove_liberty_raw(Game3 *game, int gid, int idx) {
  int m = 1 << (idx & 31);
  int wi = idx >> 5;
  int lb = gid * game->W;
  game->lw[lb + wi] &= ~m;
  game->ls[gid]--;
}

// ────────────────────────────────────────────────────────────────────────────
// Recording operations (with stack)
// ────────────────────────────────────────────────────────────────────────────

static void add_stone(Game3 *game, int idx, int gid, int color) {
  add_stone_raw(game, idx, gid);
  Operation op = {0};
  op.type = OP_ADD_STONE;
  op.idx = idx;
  op.gid = gid;
  opstack_push(&game->opStack, op);
}

static void remove_stone(Game3 *game, int idx, int gid) {
  remove_stone_raw(game, idx, gid);
  Operation op = {0};
  op.type = OP_REMOVE_STONE;
  op.idx = idx;
  op.gid = gid;
  opstack_push(&game->opStack, op);
}

static void add_liberty(Game3 *game, int gid, int idx) {
  int m = 1 << (idx & 31);
  int wi = idx >> 5;
  int lb = gid * game->W;
  if (!(game->lw[lb + wi] & m)) {
    add_liberty_raw(game, gid, idx);
    Operation op = {0};
    op.type = OP_ADD_LIBERTY;
    op.gid = gid;
    op.idx = idx;
    opstack_push(&game->opStack, op);
  }
}

static void remove_liberty(Game3 *game, int gid, int idx) {
  int m = 1 << (idx & 31);
  int wi = idx >> 5;
  int lb = gid * game->W;
  if (game->lw[lb + wi] & m) {
    remove_liberty_raw(game, gid, idx);
    Operation op = {0};
    op.type = OP_REMOVE_LIBERTY;
    op.gid = gid;
    op.idx = idx;
    opstack_push(&game->opStack, op);
  }
}

static int count_liberties(Game3 *game, int gid) {
  int count = 0;
  int gb = gid * game->W;
  for (int wi = 0; wi < game->W; wi++) {
    count += pop32(game->lw[gb + wi]);
  }
  return count;
}

static void merge_groups(Game3 *game, int mainGid, int otherId) {
  int W = game->W;
  int gb = mainGid * W;
  int ob = otherId * W;

  // Snapshot main group
  int32_t *mainStones = (int32_t *)malloc(W * sizeof(int32_t));
  int32_t *mainLibs = (int32_t *)malloc(W * sizeof(int32_t));
  for (int wi = 0; wi < W; wi++) {
    mainStones[wi] = game->sw[gb + wi];
    mainLibs[wi] = game->lw[gb + wi];
  }
  int mainSize = game->ss[mainGid];
  int mainLibCount = game->ls[mainGid];

  // Snapshot other group
  int32_t *otherStones = (int32_t *)malloc(W * sizeof(int32_t));
  int32_t *otherLibs = (int32_t *)malloc(W * sizeof(int32_t));
  for (int wi = 0; wi < W; wi++) {
    otherStones[wi] = game->sw[ob + wi];
    otherLibs[wi] = game->lw[ob + wi];
  }
  int otherSize = game->ss[otherId];
  int otherLibCount = game->ls[otherId];

  // Merge stones
  for (int wi = 0; wi < W; wi++) {
    uint32_t w = game->sw[ob + wi];
    while (w) {
      int bit = 31 - clz32(w & -w);
      game->gid[wi * 32 + bit] = mainGid;
      w &= w - 1;
    }
    game->sw[gb + wi] |= game->sw[ob + wi];
  }
  game->ss[mainGid] += game->ss[otherId];

  // Merge liberties
  for (int wi = 0; wi < W; wi++) {
    game->lw[gb + wi] |= game->lw[ob + wi];
  }
  game->ls[mainGid] = count_liberties(game, mainGid);

  // Push operation
  Operation op = {0};
  op.type = OP_MERGE_GROUPS;
  op.mainGid = mainGid;
  op.otherId = otherId;
  op.mainStones = mainStones;
  op.mainLibs = mainLibs;
  op.mainSize = mainSize;
  op.mainLibCount = mainLibCount;
  op.otherStones = otherStones;
  op.otherLibs = otherLibs;
  op.otherSize = otherSize;
  op.otherLibCount = otherLibCount;
  op.W = W;
  opstack_push(&game->opStack, op);
}

// ────────────────────────────────────────────────────────────────────────────
// Undo operations
// ────────────────────────────────────────────────────────────────────────────

static void undo_operation(Game3 *game, Operation *op) {
  int W = game->W;

  switch (op->type) {
    case OP_ADD_STONE:
      remove_stone_raw(game, op->idx, op->gid);
      break;
    case OP_REMOVE_STONE:
      add_stone_raw(game, op->idx, op->gid);
      break;
    case OP_ADD_LIBERTY:
      remove_liberty_raw(game, op->gid, op->idx);
      break;
    case OP_REMOVE_LIBERTY:
      add_liberty_raw(game, op->gid, op->idx);
      break;
    case OP_MERGE_GROUPS: {
      int gb = op->mainGid * W;
      int ob = op->otherId * W;
      for (int wi = 0; wi < W; wi++) {
        game->sw[gb + wi] = op->mainStones[wi];
        game->lw[gb + wi] = op->mainLibs[wi];
      }
      for (int wi = 0; wi < W; wi++) {
        uint32_t w = op->otherStones[wi];
        while (w) {
          int bit = 31 - clz32(w & -w);
          game->gid[wi * 32 + bit] = op->otherId;
          w &= w - 1;
        }
        game->sw[ob + wi] = op->otherStones[wi];
        game->lw[ob + wi] = op->otherLibs[wi];
      }
      game->ss[op->mainGid] = op->mainSize;
      game->ss[op->otherId] = op->otherSize;
      game->ls[op->mainGid] = op->mainLibCount;
      game->ls[op->otherId] = op->otherLibCount;
      break;
    }
    default:
      break;
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Game logic - legality and suicide detection
// ────────────────────────────────────────────────────────────────────────────

static bool is_single_suicide(Game3 *game, int idx, int color) {
  int32_t *nbr = game->nbr;
  int base = idx * 4;

  for (int i = 0; i < 4; i++) {
    int ni = nbr[base + i];
    int c = game->cells[ni];
    if (c == EMPTY) return false;
    if (c == color) {
      if (game->ls[game->gid[ni]] > 1) return false;
    } else {
      if (game->ls[game->gid[ni]] == 1) return false;
    }
  }
  return true;
}

static bool is_multi_suicide(Game3 *game, int idx, int color) {
  int32_t *nbr = game->nbr;
  int base = idx * 4;
  int W = game->W;
  bool hasFriendly = false;
  int s0 = -1, s1 = -1, s2 = -1, s3 = -1;

  for (int i = 0; i < 4; i++) {
    int ni = nbr[base + i];
    int c = game->cells[ni];
    if (c == EMPTY) return false;

    int gid = game->gid[ni];
    if (gid == s0 || gid == s1 || gid == s2 || gid == s3) continue;

    if      (s0 == -1) s0 = gid;
    else if (s1 == -1) s1 = gid;
    else if (s2 == -1) s2 = gid;
    else                s3 = gid;

    if (c == color) {
      hasFriendly = true;
      if (game->ls[gid] > 1) return false;
      /* Check if this liberty is at idx */
      if (game->ls[gid] == 1 && !((game->lw[gid * W + (idx >> 5)] >> (idx & 31)) & 1)) {
        return false;
      }
    } else {
      /* Enemy group - check if we can capture it */
      if (game->ls[gid] == 1 && ((game->lw[gid * W + (idx >> 5)] >> (idx & 31)) & 1)) {
        return false;
      }
    }
  }

  return hasFriendly;
}

bool game3_is_legal(Game3 *game, int idx, int color) {
  if (game->cells[idx] != EMPTY) return false;
  if (game->ko == idx) return false;
  if (is_single_suicide(game, idx, color)) return false;
  if (is_multi_suicide(game, idx, color)) return false;
  return true;
}

// ────────────────────────────────────────────────────────────────────────────
// Eye detection
// ────────────────────────────────────────────────────────────────────────────

bool game3_is_true_eye(Game3 *game, int idx) {
  int color = game->current;
  int8_t *cells = game->cells;
  int32_t *gidArr = game->gid;
  int32_t *nbr = game->nbr;
  int32_t *dnbr = game->dnbr;
  int base = idx * 4;

  int firstGid = -2, friendCount = 0, emptyCount = 0, sameGroup = 0;
  for (int i = 0; i < 4; i++) {
    int ni = nbr[base + i];
    int c = cells[ni];
    if (c == color) {
      friendCount++;
      int gid = gidArr[ni];
      if (firstGid == -2) {
        firstGid = gid;
        sameGroup = 1;
      } else if (gid == firstGid) {
        sameGroup++;
      }
    } else if (c == EMPTY) {
      emptyCount++;
    }
  }

  if (friendCount == 3 && emptyCount == 1 && sameGroup == 3) return true;
  if (friendCount < 4) return false;
  if (sameGroup == 4) return true;

  int dc = 0;
  for (int i = 0; i < 4; i++) {
    if (cells[dnbr[base + i]] == color) dc++;
  }
  return dc >= 3;
}

bool game3_is_valid_move(Game3 *game, int idx, int color) {
  return game3_is_legal(game, idx, color) && !game3_is_true_eye(game, idx);
}

// ────────────────────────────────────────────────────────────────────────────
// Main play logic
// ────────────────────────────────────────────────────────────────────────────

bool game3_play(Game3 *game, int move) {
  if (move == PASS) {
    Operation op = {0};
    op.type = OP_PASS;
    op.previousPassCurrent = game->current;
    op.previousPassConsecutivePasses = game->consecutivePasses;
    opstack_push(&game->opStack, op);

    game->consecutivePasses++;
    if (game->consecutivePasses >= 2) {
      game->gameOver = true;
    }
    game->current = -game->current;
    game->moveCount++;
    return true;
  }

  if (!game3_is_legal(game, move, game->current)) return false;

  int color = game->current;
  int oppColor = -color;
  int32_t *nbr = game->nbr;
  int base = move * 4;
  int W = game->W;

  int opCountBefore = game->opStack.count;

  int previousKo = game->ko;
  int previousEmptyCount = game->emptyCount;
  int previousConsecutivePasses = game->consecutivePasses;
  int previousLastMove = game->lastMove;

  // Mark cell as occupied
  game->cells[move] = color;
  game->emptyCount--;
  game->lastMove = move;

  // Remove liberties from adjacent opponent groups
  int *oppGroupIds = game->tempIntArr;
  int oppGroupCount = 0;

  for (int i = 0; i < 4; i++) {
    int ni = nbr[base + i];
    int gid = game->gid[ni];
    if (gid != -1) {
      bool found = false;
      for (int j = 0; j < oppGroupCount; j++) {
        if (oppGroupIds[j] == gid) {
          found = true;
          break;
        }
      }
      if (!found) {
        oppGroupIds[oppGroupCount++] = gid;
        remove_liberty(game, gid, move);
      }
    }
  }

  // Find adjacent same-color groups
  int *sameColorGroupIds = oppGroupIds + 4;
  int sameColorCount = 0;
  int *emptyNeighbors = sameColorGroupIds + 4;
  int emptyCount = 0;

  for (int i = 0; i < 4; i++) {
    int ni = nbr[base + i];
    int c = game->cells[ni];
    if (c == color) {
      int gid = game->gid[ni];
      bool found = false;
      for (int j = 0; j < sameColorCount; j++) {
        if (sameColorGroupIds[j] == gid) {
          found = true;
          break;
        }
      }
      if (!found) {
        sameColorGroupIds[sameColorCount++] = gid;
      }
    } else if (c == EMPTY) {
      bool found = false;
      for (int j = 0; j < emptyCount; j++) {
        if (emptyNeighbors[j] == ni) {
          found = true;
          break;
        }
      }
      if (!found) {
        emptyNeighbors[emptyCount++] = ni;
      }
    }
  }

  // Create or merge group
  int mainGid;
  if (sameColorCount == 0) {
    mainGid = game->nextGid++;
    game->gc[mainGid] = color;
    game->gid[move] = mainGid;
    add_stone(game, move, mainGid, color);
  } else {
    mainGid = sameColorGroupIds[0];
    for (int i = 1; i < sameColorCount; i++) {
      if (game->ss[sameColorGroupIds[i]] > game->ss[mainGid]) {
        mainGid = sameColorGroupIds[i];
      }
    }
    game->gid[move] = mainGid;
    add_stone(game, move, mainGid, color);

    for (int i = 0; i < sameColorCount; i++) {
      if (sameColorGroupIds[i] != mainGid) {
        merge_groups(game, mainGid, sameColorGroupIds[i]);
      }
    }
  }

  // Add liberties
  for (int i = 0; i < emptyCount; i++) {
    add_liberty(game, mainGid, emptyNeighbors[i]);
  }

  // Capture opponent groups
  int *captured = (int *)malloc(oppGroupCount * 2 * sizeof(int));
  int capturedCount = 0;

  for (int oi = 0; oi < oppGroupCount; oi++) {
    int oppGid = oppGroupIds[oi];
    if (game->ls[oppGid] == 0) {
      int gb = oppGid * W;
      for (int wi = 0; wi < W; wi++) {
        uint32_t w = game->sw[gb + wi];
        while (w) {
          int bit = 31 - clz32(w & -w);
          int stoneIdx = wi * 32 + bit;
          game->cells[stoneIdx] = EMPTY;
          game->gid[stoneIdx] = -1;
          remove_stone(game, stoneIdx, oppGid);
          captured[capturedCount * 2] = stoneIdx;
          captured[capturedCount * 2 + 1] = oppGid;
          capturedCount++;
          game->emptyCount++;

          int sBase = stoneIdx * 4;
          for (int j = 0; j < 4; j++) {
            int nGid = game->gid[nbr[sBase + j]];
            if (nGid != -1 && nGid != oppGid) {
              add_liberty(game, nGid, stoneIdx);
            }
          }
          w &= w - 1;
        }
      }
    }
  }

  // Update ko rule
  if (capturedCount == 1) {
    game->ko = captured[0];
  } else {
    game->ko = PASS;
  }

  // Push move operation
  Operation op = {0};
  op.type = OP_MOVE;
  op.move = move;
  op.color = color;
  op.previousCurrent = color;
  op.previousKo = previousKo;
  op.previousEmptyCount = previousEmptyCount;
  op.previousConsecutivePasses = previousConsecutivePasses;
  op.previousLastMove = previousLastMove;
  op.opsStart = opCountBefore;
  op.captured = (void *)captured;
  op.capturedCount = capturedCount;
  opstack_push(&game->opStack, op);

  game->consecutivePasses = 0;
  game->current = oppColor;
  game->moveCount++;
  return true;
}

void game3_undo(Game3 *game) {
  if (game->opStack.count == 0) return;

  while (game->opStack.count > 0) {
    Operation op = opstack_pop(&game->opStack);

    if (op.type == OP_PASS) {
      game->current = op.previousPassCurrent;
      game->consecutivePasses = op.previousPassConsecutivePasses;
      if (game->consecutivePasses < 2) {
        game->gameOver = false;
      }
      game->moveCount--;
      return;
    }

    if (op.type == OP_MOVE) {
      while (game->opStack.count > op.opsStart) {
        Operation inner = opstack_pop(&game->opStack);
        undo_operation(game, &inner);
      }

      game->cells[op.move] = EMPTY;
      game->gid[op.move] = -1;

      if (op.captured && op.capturedCount > 0) {
        int *captured = (int *)op.captured;
        for (int i = 0; i < op.capturedCount; i++) {
          int stoneIdx = captured[i * 2];
          int gid = captured[i * 2 + 1];
          game->cells[stoneIdx] = -op.color;
          game->gid[stoneIdx] = gid;
        }
      }

      game->current = op.previousCurrent;
      game->ko = op.previousKo;
      game->emptyCount = op.previousEmptyCount;
      game->consecutivePasses = op.previousConsecutivePasses;
      game->lastMove = op.previousLastMove;
      game->moveCount--;
      return;
    }

    undo_operation(game, &op);
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Group queries
// ────────────────────────────────────────────────────────────────────────────

int32_t game3_group_id_at(Game3 *game, int idx) {
  return game->gid[idx];
}

int game3_group_size(Game3 *game, int gid) {
  return game->ss[gid];
}

int game3_group_liberty_count(Game3 *game, int gid) {
  return game->ls[gid];
}

GroupLibs2 game3_group_libs2(Game3 *game, int idx) {
  GroupLibs2 result = {0, -1, -1};
  int gid = game->gid[idx];
  if (gid == -1) return result;

  int lc = game->ls[gid];
  result.count = lc;
  if (lc == 0) return result;

  int W = game->W;
  int lb = gid * W;
  int cap = game->N * game->N;
  int found = 0;

  for (int wi = 0; wi < W && found < 2; wi++) {
    uint32_t w = game->lw[lb + wi];
    while (w && found < 2) {
      int i = wi * 32 + (31 - clz32(w & -w));
      if (i < cap) {
        if (found == 0) result.lib0 = i;
        else if (found == 1) result.lib1 = i;
        found++;
      }
      w &= w - 1;
    }
  }
  return result;
}

// ────────────────────────────────────────────────────────────────────────────
// Display and scoring
// ────────────────────────────────────────────────────────────────────────────

char* game3_to_string(Game3 *game, int markIdx) {
  int N = game->N;
  int8_t *cells = game->cells;
  int markX = (markIdx != PASS) ? markIdx % N : -1;
  int markY = (markIdx != PASS) ? (markIdx / N) : -1;

  int rows = N;
  int maxLineLen = N * 2 + 10;
  char *result = (char *)malloc((rows * maxLineLen + 1) * sizeof(char));
  int pos = 0;

  for (int y = 0; y < N; y++) {
    int mx = (y == markY) ? markX : -1;
    for (int x = 0; x < N; x++) {
      int c = cells[y * N + x];
      char ch = (c == BLACK) ? '●' : (c == WHITE) ? '○' : '·';
      if (x == 0) {
        if (mx == 0) pos += sprintf(result + pos, "(");
        else pos += sprintf(result + pos, " ");
      }
      if (x > 0) {
        if (x == mx) pos += sprintf(result + pos, "(");
        else if (x - 1 == mx) pos += sprintf(result + pos, ")");
        else pos += sprintf(result + pos, " ");
      }
      pos += sprintf(result + pos, "%c", ch);
    }
    if (mx == N - 1) pos += sprintf(result + pos, ")");
    else pos += sprintf(result + pos, " ");
    pos += sprintf(result + pos, "\n");
  }

  result[pos] = '\0';
  return result;
}

void game3_free_string(char *str) {
  free(str);
}

Score game3_estimate_score(Game3 *game) {
  Score score = {0, 0};
  int N = game->N;
  int cap = N * N;
  int8_t *cells = game->cells;
  int32_t *nbr = game->nbr;

  for (int i = 0; i < cap; i++) {
    int c = cells[i];
    if (c == BLACK) {
      score.black++;
    } else if (c == WHITE) {
      score.white++;
    } else {
      int base = i * 4;
      bool bAdj = false;
      bool wAdj = false;
      for (int k = 0; k < 4; k++) {
        int nc = cells[nbr[base + k]];
        if (nc == BLACK) bAdj = true;
        else if (nc == WHITE) wAdj = true;
      }
      if (bAdj && !wAdj) score.black++;
      else if (wAdj && !bAdj) score.white++;
    }
  }

  float komi = 3.5;
  if (N == 9) komi = 6.5;
  else if (N == 13) komi = 7.5;
  else if (N == 19) komi = 35.5;

  score.white += komi;
  return score;
}

int game3_estimate_winner(Game3 *game) {
  Score score = game3_estimate_score(game);
  return score.black > score.white ? BLACK : WHITE;
}
