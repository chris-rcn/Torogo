'use strict';

// patterns.JS — pattern recognition helpers for Go board positions.

// Encode a cell value as a base-3 digit relative to the moving player.
//   null         → 0  (empty)
//   mover colour → 1  (friend)
//   other colour → 2  (enemy)
function cellCode(cell, mover) {
  if (cell === null) return 0;
  return cell === mover ? 1 : 2;
}

// The 8 symmetries of the square (dihedral group D4) expressed as index
// permutations of the 9-cell row-major 3×3 grid:
//
//   0 1 2
//   3 4 5
//   6 7 8
//
// Each entry lists where cell i moves to under that transform.
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

// Precompute a canonical-form lookup table over all 3^9 = 19683 raw hashes.
// CANONICAL[h] is the minimum hash among all 8 symmetry transforms of h,
// so any two patterns that are rotations/reflections of each other map to
// the same canonical hash.
const CANONICAL = new Uint16Array(19683);
(function buildCanonical() {
  const POW3 = [1, 3, 9, 27, 81, 243, 729, 2187, 6561]; // 3^0 … 3^8
  const cells = new Uint8Array(9);

  for (let h = 0; h < 19683; h++) {
    // Decode h into 9 base-3 digits.
    let tmp = h;
    for (let i = 0; i < 9; i++) { cells[i] = tmp % 3; tmp = (tmp / 3) | 0; }

    // Compute the hash for each of the 8 symmetry transforms, keep minimum.
    let min = h;
    for (let s = 1; s < 8; s++) {
      const perm = SYMMETRY_PERMS[s];
      let th = 0;
      for (let i = 0; i < 9; i++) th += cells[perm[i]] * POW3[i];
      if (th < min) min = th;
    }
    CANONICAL[h] = min;
  }
})();

// patternHash(game, x, y, mover) → canonical integer in [0, 3^9)
//
// Returns the canonical (rotation- and reflection-invariant) hash of the
// 3×3 neighbourhood centred on (x, y), encoded relative to the moving
// player so that the same tactical shape produces the same hash regardless
// of which colour is to move or the board orientation.
// Cells are visited in row-major order (left-to-right, top-to-bottom),
// with toroidal wrapping at board edges matching the rest of the engine.
function patternHash(game, x, y, mover) {
  const grid = game.board.grid;
  const N = game.board.size;
  let hash = 0;
  let weight = 1; // 3^0
  for (let dy = -1; dy <= 1; dy++) {
    const row = grid[(y + dy + N) % N];
    for (let dx = -1; dx <= 1; dx++) {
      const cell = row[(x + dx + N) % N];
      hash += cellCode(cell, mover) * weight;
      weight *= 3;
    }
  }
  return CANONICAL[hash];
}

if (typeof module !== 'undefined') module.exports = { patternHash };
