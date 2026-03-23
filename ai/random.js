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

  // Collect empty, non-true-eye cells using a single classifyEmpty call each.
  const candidates = [];
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      if (board.get(x, y) !== null) continue;
      const info = board.classifyEmpty(x, y, color);
      if (info.isTrueEye) continue;
      candidates.push([x, y, info.hasEmptyNeighbor]);
    }
  }

  // Pick candidates in random order using swap-with-last-and-pop.
  while (candidates.length > 0) {
    const i = Math.floor(Math.random() * candidates.length);
    const [x, y, hasEmptyNeighbor] = candidates[i];
    candidates[i] = candidates[candidates.length - 1];
    candidates.pop();

    if (hasEmptyNeighbor) return { type: 'place', x, y };
    if (game.isLegal(x, y)) return { type: 'place', x, y };
  }

  return { type: 'pass' };
};
