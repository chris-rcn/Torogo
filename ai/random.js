'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs (require, process, etc.).
// Wrapped in an IIFE to avoid polluting the global namespace.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

/**
 * Random-move policy.
 *
 * Picks a uniformly random legal non-true-eye move.
 * Falls back to pass when no such move exists.
 *
 * Interface: getMove(game, timeBudgetMs) → { type: 'pass' } | { type: 'place', x, y }
 *   game         - a live Game2 instance (read-only; do not mutate)
 *   timeBudgetMs - ignored (always fast)
 */

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
const { PASS } = _isNode ? require('../game2.js') : window.Game2;

function getMove(game, _timeBudgetMs) {
  if (game.gameOver) return { type: 'pass', move: PASS };

  const N   = game.N;
  const cap = N * N;

  // Collect legal non-true-eye moves.
  const candidates = [];
  for (let i = 0; i < cap; i++) {
    if (game.cells[i] !== 0) continue;
    if (game.isTrueEye(i)) continue;
    if (game.isLegal(i)) candidates.push(i);
  }

  // Pick uniformly at random using swap-with-last-and-pop.
  while (candidates.length > 0) {
    const ri = Math.floor(Math.random() * candidates.length);
    const idx = candidates[ri];
    candidates[ri] = candidates[candidates.length - 1];
    candidates.pop();
    return { type: 'place', move: idx, x: idx % N, y: (idx / N) | 0 };
  }

  return { type: 'pass', move: PASS };
}

if (typeof module !== 'undefined') module.exports = { getMove };
else window.getMove = getMove;

})();
