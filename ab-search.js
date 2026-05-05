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
// proactivePass=false disables the "consider PASS once half the board is filled"
// branch — PASS is then only considered when no legal move exists or the
// opponent just passed.
function ab(game, depth, alpha, beta, evaluate, dither, proactivePass = true) {
  if (game.gameOver) return game.estimateWinner() === BLACK ? 1 : 0;
  if (depth <= 0) return evaluate(game) + Math.random() * dither;

  const cap     = game.N * game.N;
  const isBlack = game.current === BLACK;
  let   v       = isBlack ? -Infinity : Infinity;
  let   cutoff  = false;

  for (let i = 0; i < cap && !cutoff; i++) {
    if (!game.isLegal(i) || game.isTrueEye(i)) continue;
    const g = game.clone();
    g.play(i);
    const s = ab(g, depth - 1, alpha, beta, evaluate, dither, proactivePass);
    if (isBlack) { if (s > v) v = s; if (v > alpha) alpha = v; if (alpha >= beta) cutoff = true; }
    else         { if (s < v) v = s; if (v < beta)  beta  = v; if (beta <= alpha) cutoff = true; }
  }

  // PASS: always as fallback if no legal move was found; consider proactively
  // late in the game only when proactivePass is enabled.
  const considerPass = v === (isBlack ? -Infinity : Infinity)
                    || game.consecutivePasses > 0
                    || (proactivePass && game.emptyCount < cap / 2);
  if (!cutoff && considerPass) {
    const g = game.clone();
    g.play(PASS);
    const s = ab(g, depth - 1, alpha, beta, evaluate, dither, proactivePass);
    if (isBlack) { if (s > v) v = s; }
    else         { if (s < v) v = s; }
  }

  return v;
}

// Root search: returns the best move index.
// depth=1 is equivalent to a 1-ply greedy policy.
// dither adds uniform noise to each leaf evaluation.
// proactivePass=false: only return PASS as the move when no legal move exists
// or the opponent just passed (recommended for evaluators that systematically
// over-rate the do-nothing position).
function search(game, depth, evaluate, dither = 0, proactivePass = true) {
  const cap     = game.N * game.N;
  const isBlack = game.current === BLACK;
  let bestIdx   = PASS;
  let bestScore = isBlack ? -Infinity : Infinity;
  let alpha = -Infinity, beta = Infinity;

  for (let i = 0; i < cap; i++) {
    if (!game.isLegal(i) || game.isTrueEye(i)) continue;
    const g = game.clone();
    g.play(i);
    const s = ab(g, depth - 1, alpha, beta, evaluate, dither, proactivePass);
    if (isBlack ? s > bestScore : s < bestScore) {
      bestScore = s; bestIdx = i;
      if (isBlack) alpha = Math.max(alpha, s);
      else         beta  = Math.min(beta,  s);
    }
  }

  const considerPass = bestIdx === PASS
                    || game.consecutivePasses > 0
                    || (proactivePass && game.emptyCount < cap / 2);
  if (considerPass) {
    const g = game.clone();
    g.play(PASS);
    const s = ab(g, depth - 1, alpha, beta, evaluate, dither, proactivePass);
    if (isBlack ? s > bestScore : s < bestScore) bestIdx = PASS;
  }

  return bestIdx;
}

const ABSearch = { ab, search };

if (typeof module !== 'undefined') module.exports = ABSearch;
else window.ABSearch = ABSearch;

})();
