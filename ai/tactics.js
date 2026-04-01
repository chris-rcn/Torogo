'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs (require, process, etc.).
// Wrapped in an IIFE to avoid polluting the global namespace.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

/**
 * Tactics policy — probabilistic weighted playout.
 *
 * Designed for use inside Monte Carlo simulations: much faster than shore-up.js
 * (no board cloning per move) while being significantly stronger than random.
 *
 * Each legal non-eye move is assigned a weight based on its tactical role:
 *   capture  (takes an opponent group in atari)       weight 50
 *   escape   (extends an own group in atari)          weight 30
 *   shore-up (extends an own 2-liberty group)         weight  8
 *   threaten (puts an opponent 2-liberty group into atari) weight 5
 *   other                                             weight  1
 *
 * A single move can satisfy multiple criteria; the highest weight wins.
 * Weights are computed by a one-pass scan of the immediate neighbor groups —
 * no cloning, no recursion, O(empty cells) per call.
 *
 * Interface: getMove(game, timeBudgetMs) → { type: 'pass' } | { type: 'place', x, y }
 *   game         - a live Game2 instance (read-only; do not mutate)
 *   timeBudgetMs - ignored (always fast)
 */

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
const { PASS, BLACK, WHITE } = _isNode ? require('../game2.js') : window.Game2;

const W_CAPTURE  = 50;
const W_ESCAPE   = 30;
const W_SHOREUP  = 8;
const W_THREATEN = 5;
const W_OTHER    = 1;

function getMove(game, _timeBudgetMs) {
  if (game.gameOver) return { type: 'pass', move: PASS };

  const game2  = game.cells ? game : game.toGame2();
  const N      = game2.N;
  const cap    = N * N;
  const color  = game2.current;
  const opp    = color === BLACK ? WHITE : BLACK;
  const cells  = game2.cells;
  const nbr    = game2._nbr;
  const ls     = game2._ls;
  const gidArr = game2._gid;

  // One pass: collect candidates and compute weights without cloning.
  const candidates = [];
  const weights    = [];
  let   totalWeight = 0;

  for (let i = 0; i < cap; i++) {
    if (cells[i] !== 0)      continue;
    if (game2.isTrueEye(i))  continue;
    if (!game2.isLegal(i))   continue;

    // Inspect the four immediate neighbors to classify the move.
    let w    = W_OTHER;
    const base = i * 4;

    for (let d = 0; d < 4; d++) {
      const ni = nbr[base + d];
      const c  = cells[ni];
      const g  = gidArr[ni];
      if (g === -1) continue;   // empty or off-board sentinel

      if (c === opp) {
        if (ls[g] === 1) { w = W_CAPTURE;  break; }  // capture — highest, short-circuit
        if (ls[g] === 2 && w < W_THREATEN) w = W_THREATEN;
      } else if (c === color) {
        if (ls[g] === 1 && w < W_ESCAPE)  w = W_ESCAPE;
        if (ls[g] === 2 && w < W_SHOREUP) w = W_SHOREUP;
      }
    }

    candidates.push(i);
    weights.push(w);
    totalWeight += w;
  }

  if (candidates.length === 0) return { type: 'pass', move: PASS };

  // Weighted random selection.
  let r = Math.random() * totalWeight;
  for (let i = 0; i < candidates.length; i++) {
    r -= weights[i];
    if (r <= 0) {
      const idx = candidates[i];
      return { type: 'place', move: idx, x: idx % N, y: (idx / N) | 0 };
    }
  }

  // Fallback (floating-point rounding edge case).
  const idx = candidates[candidates.length - 1];
  return { type: 'place', move: idx, x: idx % N, y: (idx / N) | 0 };
}

if (typeof module !== 'undefined') module.exports = { getMove };
else window.getMove = getMove;

})();
