'use strict';

// patterns.JS — pattern recognition helpers for Go board positions.

// Liberty counts above this threshold are treated as equivalent.
const MAX_LIBS = 2;

// Encode a cell value as a base-3 digit relative to the moving player.
//   null         → 0  (empty)
//   mover colour → 1  (friend)
//   other colour → 2  (enemy)
function cellCode(cell, mover) {
  if (cell === null) return 0;
  return cell === mover ? 1 : 2;
}

// Powers of 3 for weighting the 9 cells (base-3 hash).
const POW3 = [1, 3, 9, 27, 81, 243, 729, 2187, 6561]; // 3^0 … 3^8

// Powers of (MAX_LIBS+1) for weighting the 4 orthogonal liberty counts.
// Orthogonal neighbours occupy flat positions 1, 3, 5, 7 in row-major order:
//
//   0 1 2
//   3 4 5   ← 1,3,5,7 are the 4 orthogonal neighbours of centre (4)
//   6 7 8
//
// We assign each position a fixed weight in base (MAX_LIBS+1):
//   pos 1 → (M+1)^0,  pos 3 → (M+1)^1,  pos 5 → (M+1)^2,  pos 7 → (M+1)^3
const _M1 = MAX_LIBS + 1;
const LIB_WEIGHT = [0, 1, 0, _M1, 0, _M1 * _M1, 0, _M1 * _M1 * _M1, 0];
// (diagonal positions 0,2,6,8 get weight 0; only 1,3,5,7 contribute)

// The 8 symmetries of the square (dihedral group D4) as index permutations
// of the 9-cell row-major 3×3 grid.  SYMMETRY_PERMS[s][i] is the source
// cell index for destination position i under symmetry s.
const SYMMETRY_PERMS = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8], // identity
  [6, 3, 0, 7, 4, 1, 8, 5, 2], // rotate 90° CW
  [8, 7, 6, 5, 4, 3, 2, 1, 0], // rotate 180°
  [2, 5, 8, 1, 4, 7, 0, 3, 6], // rotate 270° CW
  [2, 1, 0, 5, 4, 3, 8, 7, 6], // reflect horizontal
  [6, 7, 8, 3, 4, 5, 0, 1, 2], // reflect vertical
  [0, 3, 6, 1, 4, 7, 2, 5, 8], // reflect main diagonal
  [8, 5, 2, 7, 4, 1, 6, 3, 0], // reflect anti-diagonal
];

// Return the clamped liberty count for the group occupying flat board index
// `idx`, or 0 if the cell is empty.  Uses the board's incremental group
// tracker for an O(1) lookup.
function cellLiberties(board, idx) {
  const gid = board._gid[idx];
  if (gid < 0) return 0;
  const group = board._groups.get(gid);
  return group ? Math.min(group.liberties.size, MAX_LIBS) : 0;
}

// patternHash(game, x, y, mover) → canonical integer in [0, 3^9 · (MAX_LIBS+1)^4)
//
// Returns a rotation- and reflection-invariant hash of the 3×3 neighbourhood
// centred on (x, y).  The hash encodes:
//   • the colour of each cell relative to the mover (empty / friend / enemy)
//   • the liberty count (clamped to MAX_LIBS) of the group at each
//     orthogonal neighbour of the centre
//
// Combined hash = cellHash + 19683 * libHash, where both components are
// permuted consistently under each of the 8 D4 symmetry transforms.
// The minimum combined hash across all transforms is returned.
function patternHash(game, x, y, mover) {
  const board = game.board;
  const grid = board.grid;
  const N = board.size;

  // Collect cell codes and liberty counts for all 9 positions.
  const cells = new Uint8Array(9);
  const libs  = new Uint8Array(9);
  let i = 0;
  for (let dy = -1; dy <= 1; dy++) {
    const ry  = (y + dy + N) % N;
    const row = grid[ry];
    for (let dx = -1; dx <= 1; dx++) {
      const rx = (x + dx + N) % N;
      cells[i] = cellCode(row[rx], mover);
      if (cells[i] !== 0) libs[i] = cellLiberties(board, ry * N + rx);
      i++;
    }
  }

  // Try all 8 symmetry transforms; keep the minimum combined hash.
  let minHash = Infinity;
  for (const perm of SYMMETRY_PERMS) {
    let cellHash = 0;
    let libHash  = 0;
    for (let k = 0; k < 9; k++) {
      const src = perm[k];
      cellHash += cells[src] * POW3[k];
      libHash  += libs[src]  * LIB_WEIGHT[k];
    }
    const combined = cellHash + 19683 * libHash;
    if (combined < minHash) minHash = combined;
  }
  return minHash;
}

if (typeof module !== 'undefined') module.exports = { patternHash, MAX_LIBS };
