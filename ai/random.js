'use strict';

/**
 * Random-move policy.
 *
 * Picks a uniformly random legal move, skipping moves that would fill a
 * single-point true eye for the current player.  Falls back to pass when no
 * such move exists.
 *
 * Interface: getMove(game, timeBudgetMs) → { type: 'pass' } | { type: 'place', x, y }
 *   game         - a live Game instance (read-only; do not mutate)
 *   timeBudgetMs - ignored (always fast)
 *
 */

module.exports = function getMove(game, _timeBudgetMs) {
  if (game.gameOver) return { type: 'pass' };

  const N = game.boardSize;
  const color = game.current;
  const board = game.board;

  // Collect empty cells that are not true eyes for the current player.
  const candidates = [];
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      if (board.get(x, y) !== null) continue;
      if (board.isTrueEye(x, y, color)) continue;
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
