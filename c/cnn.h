/*
 * cnn.h — a small residual value network on the toroidal board, forward and
 * backward, in plain C (no dependencies; vectorisable inner loops).
 *
 *   input   VNET_PLANES planes per point, from the side to move's view (vnet_encode)
 *   stem    3x3 conv (planes -> C) + ReLU
 *   blocks  B residual blocks: 3x3 conv + ReLU, 3x3 conv, add the block input, ReLU
 *   head    mean over the board -> dense (C -> H) + ReLU -> dense (H -> 1) -> sigmoid
 *   output  P(side to move wins)
 *
 * Every 3x3 conv wraps around (the board has no edges), so the network is
 * translation-equivariant and, after the mean over the board, its value is
 * exactly translation-invariant.  Reflections and rotations are not built in;
 * the trainer augments with them (vnet_sym_map).
 *
 * Parameters live in one flat float array (vnet_layout gives the offsets), so
 * gradients and optimiser moments are arrays of the same shape.
 */
#ifndef CNN_H
#define CNN_H

#include <stdint.h>
#include "game2.h"
#include "game3.h"

/* Input planes: own stones whose chain has 1 / 2 / 3+ liberties, the same for
 * the opponent, empty, the ko point, the last move, a constant plane that is 1
 * when black is to move (komi makes the value colour-dependent), then ladder2's
 * status of every chain it reads (1-2 liberties, or 3 with the attacker to
 * move): own stones lost even moving first, own stones that only a move now
 * saves, opponent stones dead even if the opponent moves first, opponent
 * stones that a move now captures. */
#define VNET_PLANES 14

typedef struct {
    int C, B, H;                  /* channels, residual blocks, head width */
    int n_params;
    /* offsets into the flat parameter array */
    int stem_w, stem_b;           /* W[9][PLANES][C], b[C] */
    int blk_w1[32], blk_b1[32], blk_w2[32], blk_b2[32];   /* W[9][C][C], b[C] per block */
    int head_w, head_b, out_w, out_b;                     /* W[C][H], b[H], w[H], b[1] */
} VnetLayout;

#define VNET_MAX_BLOCKS 32

VnetLayout vnet_layout(int C, int B, int H);

/* He-normal conv and dense weights, zero biases, and the second conv of every
 * block zero (each block starts as the identity), from `seed`. */
void vnet_init(const VnetLayout *L, float *params, uint64_t seed);

/* Per-thread scratch for one forward + backward pass. */
typedef struct VnetWork VnetWork;
VnetWork *vnet_work_new(const VnetLayout *L, int area);
void      vnet_work_free(VnetWork *w);
float    *vnet_work_input(VnetWork *w);     /* area * VNET_PLANES floats, [point][plane] */

/* nbr9[p*9 + k]: the 3x3 window of point p on the torus (k = 0 centre, 1-4 the
 * orthogonal neighbours, 5-8 the diagonals), from game2's neighbour tables.
 * Call after g2_init_topology(N). */
void vnet_topology(int N, int32_t *nbr9);

/* The compact form a position is stored and encoded from: code[p] & 7 = 0
 * empty, 1-3 own stone whose chain has 1 / 2 / 3+ liberties, 4-6 the same for
 * the opponent ("own" = the side to move); code[p] >> 3 = the chain's ladder
 * status: 0 safe or not read, 1 lost / dead whoever moves, 2 decided by the
 * side to move's next move; ko and last move (or PASS); black to move.
 * g3 is scratch for the ladder reads: g3_new(N), one per thread. */
typedef struct { int16_t ko, last; int8_t black_to_move; } VnetMeta;
void vnet_compact(const Game2 *g, Game3 *g3, int8_t *code, VnetMeta *meta);

/* Planes from the compact form into `x` (area * VNET_PLANES floats), point p
 * written at map[p] (a symmetry permutation, or NULL for the identity). */
void vnet_encode_compact(const int8_t *code, const VnetMeta *meta, int area, const int32_t *map, float *x);

/* Both at once, from a board. */
void vnet_encode(const Game2 *g, Game3 *g3, const int32_t *map, float *x);

/* The 8 reflection/rotation symmetries of the N x N torus as point
 * permutations: map[t*area + p] = image of p under symmetry t (t = 0 identity).
 * Coordinates are permuted directly — a symmetry, not a neighbour lookup. */
void vnet_sym_map(int N, int32_t *map);

/* Forward pass on vnet_work_input(w); returns P(side to move wins) and keeps
 * the activations for vnet_backward. */
float vnet_forward(const VnetLayout *L, const float *params, const int32_t *nbr9, int area, VnetWork *w);

/* Backward pass for the last forward: dy = dLoss/d(pre-sigmoid output) (for
 * cross-entropy against a soft target t, dy = v - t).  ADDS the parameter
 * gradients into grad. */
void vnet_backward(const VnetLayout *L, const float *params, const int32_t *nbr9, int area,
                   VnetWork *w, float dy, float *grad);

/* Binary save / load: "VNET1", C, B, H, planes, n_params, then the floats.
 * vnet_save is atomic (<path>.tmp, then rename).
 * vnet_load fills *L and returns a malloc'd parameter array (NULL on error,
 * after a message). */
int    vnet_save(const char *path, const VnetLayout *L, const float *params);
float *vnet_load(const char *path, VnetLayout *L);

#endif /* CNN_H */
