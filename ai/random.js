'use strict';

/**
 * Random-move policy.
 *
 * Picks a uniformly random legal move, skipping moves that would fill a
 * single-point true eye for the current player.  Falls back to pass when no
 * such move exists.
 *
 * Interface: getMove(game) → { type: 'pass' } | { type: 'place', x, y }
 *   game  - a live Game instance (read-only; do not mutate)
 *
 * Single-point true eye (toroidal board — no corners/edges, every cell has
 * exactly 4 orthogonal and 4 diagonal neighbours):
 *   An empty cell (x, y) is a true eye for `color` when:
 *     1. All 4 orthogonal neighbours are occupied by `color`.
 *     2. At least 3 of the 4 diagonal neighbours are occupied by `color`.
 */

function isTrueEye(board, x, y, color) {
  const N = board.size;
  const ortho = board.getNeighbors(x, y);
  if (!ortho.every(([nx, ny]) => board.get(nx, ny) === color)) return false;
  const diags = [
    [(x + 1) % N,       (y + 1) % N],
    [(x - 1 + N) % N,   (y + 1) % N],
    [(x + 1) % N,       (y - 1 + N) % N],
    [(x - 1 + N) % N,   (y - 1 + N) % N],
  ];
  const friendly = diags.filter(([dx, dy]) => board.get(dx, dy) === color).length;
  return friendly >= 3;
}


module.exports = function getMove(game) {
  if (game.gameOver) return { type: 'pass' };

  const N = game.boardSize;
  const color = game.current;
  const board = game.board;

  // Collect empty cells that are not true eyes for the current player.
  const candidates = [];
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      if (board.get(x, y) !== null) continue;
      if (isTrueEye(board, x, y, color)) continue;
      candidates.push([x, y]);
    }
  }

  // Pick candidates in random order using swap-with-last-and-pop (avoids a
  // full Fisher-Yates shuffle of candidates we may never need to inspect).
  while (candidates.length > 0) {
    const i = Math.floor(Math.random() * candidates.length);
    const [x, y] = candidates[i];
    candidates[i] = candidates[candidates.length - 1];
    candidates.pop();

    // Fast legality check: if any orthogonal neighbour is empty the move
    // cannot be suicide or Ko, so skip the expensive clone.
    if (board.getNeighbors(x, y).some(([nx, ny]) => board.get(nx, ny) === null)) {
      return { type: 'place', x, y };
    }

    // All four neighbours are occupied — verify legality with a clone.
    const clone = game.clone();
    if (clone.placeStone(x, y)) return { type: 'place', x, y };
  }

  return { type: 'pass' };
};
