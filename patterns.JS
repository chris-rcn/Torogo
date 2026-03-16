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

// patternHash(game, x, y, mover) → integer in [0, 3^9)
//
// Returns a hash of the 3×3 neighbourhood centred on (x, y), encoded
// relative to the moving player so that the same tactical shape produces
// the same hash regardless of which colour is to move.
// Cells are visited in row-major order (left-to-right, top-to-bottom),
// with toroidal wrapping at board edges matching the rest of the engine.
//
//   bit weights (base 3):
//   0 1 2
//   3 4 5   ← centre is digit 4 (weight 3^4 = 81)
//   6 7 8
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
  return hash;
}

if (typeof module !== 'undefined') module.exports = { patternHash };
