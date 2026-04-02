'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

/**
 * Generic alpha-beta search.
 *
 * evaluate(game) → number ∈ [0,1]: P(BLACK wins) from the static evaluator.
 * BLACK maximises, WHITE minimises.
 *
 * Exports: { ab, search }
 *   ab(game, depth, alpha, beta, evaluate, dither) → value
 *   search(game, depth, evaluate, dither)          → best move index
 */

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
const { BLACK, PASS } = _isNode ? require('./game2.js') : window.Game2;

// Recursive alpha-beta evaluator. Returns V ∈ [0,1] = P(BLACK wins).
// At depth 0 returns the static evaluation; terminal nodes return 0 or 1.
function ab(game, depth, alpha, beta, evaluate, dither) {
  if (game.gameOver) return game.calcWinner() === BLACK ? 1 : 0;
  if (depth <= 0) return evaluate(game) + Math.random() * dither;

  const cap     = game.N * game.N;
  const isBlack = game.current === BLACK;
  let   v       = isBlack ? -Infinity : Infinity;
  let   cutoff  = false;

  for (let i = 0; i < cap && !cutoff; i++) {
    if (game.cells[i] !== 0) continue;
    if (game.isTrueEye(i))   continue;
    if (!game.isLegal(i))    continue;
    const g = game.clone();
    g.play(i);
    const s = ab(g, depth - 1, alpha, beta, evaluate, dither);
    if (isBlack) { if (s > v) v = s; if (v > alpha) alpha = v; if (alpha >= beta) cutoff = true; }
    else         { if (s < v) v = s; if (v < beta)  beta  = v; if (beta <= alpha) cutoff = true; }
  }

  // PASS: always as fallback if no legal move was found; also consider proactively late in game.
  if (!cutoff && (v === (isBlack ? -Infinity : Infinity) || game.emptyCount < cap / 2)) {
    const g = game.clone();
    g.play(PASS);
    const s = ab(g, depth - 1, alpha, beta, evaluate, dither);
    if (isBlack) { if (s > v) v = s; }
    else         { if (s < v) v = s; }
  }

  return v;
}

// Root search: returns the best move index.
// depth=1 is equivalent to a 1-ply greedy policy.
// dither adds uniform noise to each leaf evaluation.
function search(game, depth, evaluate, dither = 0) {
  const cap     = game.N * game.N;
  const isBlack = game.current === BLACK;
  let bestIdx   = PASS;
  let bestScore = isBlack ? -Infinity : Infinity;
  let alpha = -Infinity, beta = Infinity;

  for (let i = 0; i < cap; i++) {
    if (game.cells[i] !== 0) continue;
    if (game.isTrueEye(i))   continue;
    if (!game.isLegal(i))    continue;
    const g = game.clone();
    g.play(i);
    const s = ab(g, depth - 1, alpha, beta, evaluate, dither);
    if (isBlack ? s > bestScore : s < bestScore) {
      bestScore = s; bestIdx = i;
      if (isBlack) alpha = Math.max(alpha, s);
      else         beta  = Math.min(beta,  s);
    }
  }

  if (game.emptyCount < cap / 2) {
    const g = game.clone();
    g.play(PASS);
    const s = ab(g, depth - 1, alpha, beta, evaluate, dither);
    if (isBlack ? s > bestScore : s < bestScore) bestIdx = PASS;
  }

  return bestIdx;
}

const ABSearch = { ab, search };

if (typeof module !== 'undefined') module.exports = ABSearch;
else window.ABSearch = ABSearch;

})();
