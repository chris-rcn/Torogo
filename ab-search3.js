'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

/**
 * Generic alpha-beta search over Game3.
 *
 * Identical interface to ab-search.js, but operates on a Game3 instance.
 * Game3 supports play/undo, so the search reuses a single game object and
 * never clones — substantial speedup at depth ≥ 2.
 *
 * evaluate(game3) → number ∈ [0,1]: P(BLACK wins) from the static evaluator.
 * BLACK maximises, WHITE minimises.  The same `game3` instance is reused
 * across the entire search; evaluate must not mutate it (or, if it does,
 * must restore state before returning).
 *
 * Terminal (gameOver) positions short-circuit to `evaluate(game)` immediately
 * regardless of depth — Game3 has no built-in winner estimator, so the caller
 * is responsible for returning a sane V for terminal states (a +/− area count
 * + komi, or just calling estimateWinner on a Game2 mirror, etc.).
 *
 * Optional `opts` (passed to both `ab` and `search`):
 *   opts.getCandidates(game3) → array/typed-array of board indices to expand,
 *       in the order they should be visited.  PASS is always handled
 *       separately and should NOT appear in the returned list.  When omitted,
 *       defaults to "every legal non-true-eye move in board-index order".
 *   opts.rng — object with a `random()` method returning a value in [0, 1).
 *       Used for the dither term at leaves.  Defaults to the global `Math`.
 *
 * `opts` is forwarded to recursive `ab` calls, so the hooks apply at every
 * depth.
 *
 * Exports: { ab, search }
 *   ab(game3, depth, alpha, beta, evaluate, dither, opts?) → value
 *   search(game3, depth, evaluate, dither, opts?)          → best move index
 */

const Util = (typeof require === 'function') ? require('./util.js') : window.Util;
const { BLACK, PASS } = Util.load('./game3.js', 'Game3');

// Recursive alpha-beta evaluator. Returns V ∈ [0,1] = P(BLACK wins).
// Plays + undoes moves on the supplied game3 instance — no cloning.
function ab(game, depth, alpha, beta, evaluate, dither, opts) {
  if (game.gameOver) return evaluate(game);
  if (depth <= 0) {
    const r = opts && opts.rng ? opts.rng : Math;
    return evaluate(game) + r.random() * dither;
  }

  const cap     = game.N * game.N;
  const isBlack = game.current === BLACK;
  let   v       = isBlack ? -Infinity : Infinity;
  let   cutoff  = false;

  if (opts && opts.getCandidates) {
    const candidates = opts.getCandidates(game);
    for (let j = 0; j < candidates.length && !cutoff; j++) {
      const i = candidates[j];
      game.play(i);
      const s = ab(game, depth - 1, alpha, beta, evaluate, dither, opts);
      game.undo();
      if (isBlack) { if (s > v) v = s; if (v > alpha) alpha = v; if (alpha >= beta) cutoff = true; }
      else         { if (s < v) v = s; if (v < beta)  beta  = v; if (beta <= alpha) cutoff = true; }
    }
  } else {
    // Fast path — no allocation, no hook indirection.
    for (let i = 0; i < cap && !cutoff; i++) {
      if (!game.isLegal(i) || game.isTrueEye(i)) continue;
      game.play(i);
      const s = ab(game, depth - 1, alpha, beta, evaluate, dither, opts);
      game.undo();
      if (isBlack) { if (s > v) v = s; if (v > alpha) alpha = v; if (alpha >= beta) cutoff = true; }
      else         { if (s < v) v = s; if (v < beta)  beta  = v; if (beta <= alpha) cutoff = true; }
    }
  }

  // PASS: always as fallback if no legal move was found; also consider proactively late in game.
  if (!cutoff && (v === (isBlack ? -Infinity : Infinity) || game.consecutivePasses > 0 || game.emptyCount < cap / 2)) {
    game.play(PASS);
    const s = ab(game, depth - 1, alpha, beta, evaluate, dither, opts);
    game.undo();
    if (isBlack) { if (s > v) v = s; }
    else         { if (s < v) v = s; }
  }

  return v;
}

// Root search: returns the best move index.
// depth=1 is equivalent to a 1-ply greedy policy.
// dither adds uniform noise to each leaf evaluation.
function search(game, depth, evaluate, dither = 0, opts) {
  const cap     = game.N * game.N;
  const isBlack = game.current === BLACK;
  let bestIdx   = PASS;
  let bestScore = isBlack ? -Infinity : Infinity;
  let alpha = -Infinity, beta = Infinity;

  if (opts && opts.getCandidates) {
    const candidates = opts.getCandidates(game);
    for (let j = 0; j < candidates.length; j++) {
      const i = candidates[j];
      game.play(i);
      const s = ab(game, depth - 1, alpha, beta, evaluate, dither, opts);
      game.undo();
      if (isBlack ? s > bestScore : s < bestScore) {
        bestScore = s; bestIdx = i;
        if (isBlack) alpha = Math.max(alpha, s);
        else         beta  = Math.min(beta,  s);
      }
    }
  } else {
    for (let i = 0; i < cap; i++) {
      if (!game.isLegal(i) || game.isTrueEye(i)) continue;
      game.play(i);
      const s = ab(game, depth - 1, alpha, beta, evaluate, dither, opts);
      game.undo();
      if (isBlack ? s > bestScore : s < bestScore) {
        bestScore = s; bestIdx = i;
        if (isBlack) alpha = Math.max(alpha, s);
        else         beta  = Math.min(beta,  s);
      }
    }
  }

  if (game.consecutivePasses > 0 || game.emptyCount < cap / 2) {
    game.play(PASS);
    const s = ab(game, depth - 1, alpha, beta, evaluate, dither, opts);
    game.undo();
    if (isBlack ? s > bestScore : s < bestScore) bestIdx = PASS;
  }

  return bestIdx;
}

const ABSearch3 = { ab, search };

if (typeof module !== 'undefined') module.exports = ABSearch3;
else window.ABSearch3 = ABSearch3;

})();
