'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs (require, process, etc.).
// Wrapped in an IIFE to avoid polluting the global namespace.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

/**
 * Tactics policy — probabilistic weighted playout.
 *
 * Each legal non-eye move accumulates weight based on its tactical role(s):
 *   capture  (takes an opponent group in atari)       
 *   escape   (extends an own group in atari)          
 *   shore-up (extends an own 2-liberty group)         
 *   threaten (puts an opponent 2-liberty group into atari) 
 *   other                                            
 *
 * Weights are computed by a one-pass scan of the immediate neighbor groups.
 */

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
const { PASS, BLACK, WHITE } = _isNode ? require('../game2.js') : window.Game2;

const W_CAPTURE  = 1;
const W_ESCAPE   = 1;
const W_SHOREUP  = 1;
const W_THREATEN = 1;

const W_OTHER    = 1;

function getMove(game, _timeBudgetMs) {
  if (game.gameOver) return { move: PASS };

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
  const adjChains = new Set();

  for (let i = 0; i < cap; i++) {
    if (!game2.isLegal(i) || game2.isTrueEye(i))  continue;

    // Inspect the four immediate neighbors to classify the move.
    let w    = 0;
    const base = i * 4;

    adjChains.clear();
    for (let d = 0; d < 4; d++) {
      const ni = nbr[base + d];
      const c  = cells[ni];
      const g  = gidArr[ni];
      if (c !== 0 && !adjChains.has(g)) {
        adjChains.add(g);
        if (c === opp) {
          if (ls[g] === 1) w += W_CAPTURE;
          else if (ls[g] === 2) w += W_THREATEN;
        } else {
          if (ls[g] === 1)  w += W_ESCAPE;
          else if (ls[g] === 2) w += W_SHOREUP;
        }
      }
    }

    if (w === 0) w = W_OTHER;

    candidates.push(i);
    weights.push(w);
    totalWeight += w;
  }

  if (candidates.length === 0) return { move: PASS };

  // Weighted random selection.
  let r = Math.random() * totalWeight;
  for (let i = 0; i < candidates.length; i++) {
    r -= weights[i];
    if (r <= 0) {
      const move = candidates[i];
      return { move };
    }
  }

  // Fallback (floating-point rounding edge case).
  const move = candidates[Math.floor(Math.random() * candidates.length)];
  return { move };
}

if (typeof module !== 'undefined') module.exports = { getMove };
else window.getMove = getMove;

})();
