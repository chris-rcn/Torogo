'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.
// Wrapped in an IIFE to avoid polluting the global namespace.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

/**
 * Pattern-weight policy agent with alpha-beta search.
 *
 * Value function: V(s) = σ(Σ polarity_i · w[key_i]) = P(BLACK wins)
 * Move selection: full-width alpha-beta, BLACK maximises V, WHITE minimises V.
 *
 * Weights and specs are loaded from a JS file specified by the PAT_DATA
 * environment variable (Node) or by calling loadWeights() directly.
 */

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

const { BLACK, PASS } = _isNode ? require('../game2.js') : window.Game2;
const { extractFeatures, evaluateFeatures, loadWeights } = _isNode ? require('../vpatterns.js') : window.VPatterns;
const Util = _isNode ? require('../util.js') : window.Util;

const MIN_LIBS    = Util.envInt('MIN_LIBS',    1);
const MAX_LIBS    = Util.envInt('MAX_LIBS',    1);
const DEPTH       = Util.envInt('SEARCH_DEPTH', 1);
const DITHER       = Util.envFloat('DITHER', 0.002);

// ── Agent state ───────────────────────────────────────────────────────────────

const defaultSpecs = [];
for (let maxLibs = MIN_LIBS; maxLibs <= MAX_LIBS; maxLibs++)
  for (const size of [1, 2, 3])
    defaultSpecs.push({ size, maxLibs });

let model = { weights: new Map(), specs: defaultSpecs };

// ── Alpha-beta search ─────────────────────────────────────────────────────────

// Recursive alpha-beta evaluator. Returns V ∈ [0,1] = P(BLACK wins).
// At depth 0 returns the static evaluation; terminal nodes return 0 or 1.
function ab(game, depth, alpha, beta, m, dither) {
  if (game.gameOver) return game.calcWinner() === BLACK ? 1 : 0;
  if (depth <= 0) {
    const v = evaluateFeatures(extractFeatures(game, m.specs), m.weights);
    return v + Math.random() * dither;
  }

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
    const s = ab(g, depth - 1, alpha, beta, m, dither);
    if (isBlack) { if (s > v) v = s; if (v > alpha) alpha = v; if (alpha >= beta) cutoff = true; }
    else         { if (s < v) v = s; if (v < beta)  beta  = v; if (beta <= alpha) cutoff = true; }
  }

  // PASS: always as fallback if no legal move was found; also consider proactively late in game.
  if (!cutoff && (v === (isBlack ? -Infinity : Infinity) || game.emptyCount < cap / 2)) {
    const g = game.clone();
    g.play(PASS);
    const s = ab(g, depth - 1, alpha, beta, m, dither);
    if (isBlack) { if (s > v) v = s; }
    else         { if (s < v) v = s; }
  }

  return v;
}

// Root search: returns the best move index.
// depth=1 is equivalent to the original 1-ply greedy policy.
// dither adds uniform noise to each leaf evaluation.
// m: { weights: Map, specs: [...] }
function search(game, m, depth = 1, dither = 0) {
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
    const s = ab(g, depth - 1, alpha, beta, m, dither);
    if (isBlack ? s > bestScore : s < bestScore) {
      bestScore = s; bestIdx = i;
      if (isBlack) alpha = Math.max(alpha, s);
      else         beta  = Math.min(beta,  s);
    }
  }

  if (game.emptyCount < cap / 2) {
    const g = game.clone();
    g.play(PASS);
    const s = ab(g, depth - 1, alpha, beta, m, dither);
    if (isBlack ? s > bestScore : s < bestScore) bestIdx = PASS;
  }

  return bestIdx;
}

function getMove(game) {
  return { move: search(game, model, DEPTH, DITHER) };
}

// ── Persistence ───────────────────────────────────────────────────────────────

// Auto-load weights and specs if PAT_DATA env var is set.
if (_isNode && process.env.PAT_DATA) {
  model = loadWeights(process.env.PAT_DATA);
}

// ── Exports ───────────────────────────────────────────────────────────────────

const PatternAgent = { getMove, search };

if (typeof module !== 'undefined') module.exports = PatternAgent;
else window.PatternAgent = PatternAgent;

})();
