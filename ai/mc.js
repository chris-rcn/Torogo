'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs (require, process, etc.).
// Wrapped in an IIFE to avoid polluting the global namespace.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

/**
 * Monte Carlo policy.
 *
 * For each candidate move, performs random playouts to the end of the game
 * and tracks the win ratio.  After candidate_playouts per move the move with
 * the highest win ratio is returned.
 *
 * Interface: getMove(game, timeBudgetMs) → { type: 'pass' } | { type: 'place', x, y }
 *   game         - a live Game2 instance (read-only; do not mutate)
 *   timeBudgetMs - milliseconds allowed for this decision (required)
 */

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

const performance = (typeof window !== 'undefined') ? window.performance
  : require('perf_hooks').performance;

const { PASS, BLACK, WHITE } = _isNode ? require('../game2.js') : window.Game2;

function playRandom(game2) {
  const N   = game2.N;
  const cap = N * N;
  const cells = game2.cells;
  const nbr   = game2._nbr;

  const empty = [];
  for (let i = 0; i < cap; i++) if (cells[i] === 0) empty.push(i);

  const moveLimit = empty.length + 20;
  let moves = 0;

  while (!game2.gameOver && moves < moveLimit) {
    let placed = false;
    let end = empty.length;

    while (end > 0) {
      const ri  = Math.floor(Math.random() * end);
      const idx = empty[ri];
      empty[ri] = empty[end - 1];
      empty[end - 1] = idx;
      end--;

      if (game2.isTrueEye(idx)) continue;
      if (!game2.isLegal(idx))  continue;

      const base = idx * 4;
      const n0 = cells[nbr[base]], n1 = cells[nbr[base + 1]],
            n2 = cells[nbr[base + 2]], n3 = cells[nbr[base + 3]];
      game2.play(idx);
      empty[end] = empty[empty.length - 1];
      empty.pop();

      if ((n0 && !cells[nbr[base]])     || (n1 && !cells[nbr[base + 1]]) ||
          (n2 && !cells[nbr[base + 2]]) || (n3 && !cells[nbr[base + 3]])) {
        empty.length = 0;
        for (let i = 0; i < cap; i++) if (cells[i] === 0) empty.push(i);
      }

      placed = true;
      moves++;
      break;
    }

    if (!placed) { game2.play(PASS); moves++; }
  }
}

function getMove(game, timeBudgetMs) {
  if (game.gameOver) return { type: 'pass' };

  const game2  = game.cells ? game.clone() : game.toGame2();
  const N      = game2.N;
  const cap    = N * N;
  const player = game2.current;

  const candidates = [];
  for (let i = 0; i < cap; i++) {
    if (game2.cells[i] !== 0)   continue;
    if (game2.isTrueEye(i))     continue;
    if (game2.isLegal(i))       candidates.push(i);
  }
  candidates.push(PASS);

  const stats = candidates.map(() => ({ wins: 0, plays: 0 }));

  const deadline = performance.now() + timeBudgetMs;
  let cidx = 0;
  while (performance.now() < deadline) {
    const clone = game2.clone();
    clone.play(candidates[cidx]);
    playRandom(clone);

    stats[cidx].plays++;
    if (clone.estimateWinner() === player) stats[cidx].wins++;

    cidx = (cidx + 1) % candidates.length;
  }

  let bestIdx = 0, bestRatio = -1;
  for (let i = 0; i < candidates.length; i++) {
    const ratio = stats[i].plays > 0 ? stats[i].wins / stats[i].plays : 0;
    if (ratio > bestRatio) { bestRatio = ratio; bestIdx = i; }
  }

  const best = candidates[bestIdx];
  return best === PASS ? { type: 'pass' }
                       : { type: 'place', x: best % N, y: (best / N) | 0 };
}

if (typeof module !== 'undefined') module.exports = { getMove };
else window.getMove = getMove;

})();
