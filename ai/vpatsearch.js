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
 * Weights and specs are loaded from a JS file specified by the VPAT_DATA
 * environment variable (Node) or by calling loadWeights() directly.  A
 * health-coded model (size:H<N> specs) additionally needs HEALTH_DATA.
 */

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

const { extractFeatures, evaluateFeatures, deltaZ, loadWeights, prepareSpecs, makeWeights } = _isNode ? require('../vpatterns.js') : window.VPatterns;
const { search: abSearch } = _isNode ? require('../ab-search.js') : window.ABSearch;
const Util = _isNode ? require('../util.js') : window.Util;
const { BLACK, PASS } = _isNode ? require('../game2.js') : window.game;

const MIN_LIBS = Util.envInt  ('MIN_LIBS',     1);
const MAX_LIBS = Util.envInt  ('MAX_LIBS',     1);
const SEARCH_DEPTH    = Util.envInt  ('SEARCH_DEPTH', 1);
const DITHER   = Util.envFloat('DITHER',       0.002);
const VPAT_DATA = Util.envStr ('VPAT_DATA',    'out/ref13.js');
// Health-coded specs (size:H<N>) need a frozen health model; vpatterns no
// longer reads the environment itself, so the caller supplies it.
const HEALTH_DATA = Util.envStr ('HEALTH_DATA', '');

// ── Agent state ───────────────────────────────────────────────────────────────

const defaultSpecs = [];
for (let maxLibs = MIN_LIBS; maxLibs <= MAX_LIBS; maxLibs++)
  for (const size of [1, 2, 3])
    defaultSpecs.push({ size, maxLibs });

let model = { weights: makeWeights(), specs: defaultSpecs, preparedSpecs: prepareSpecs(defaultSpecs) };

// ── Search ────────────────────────────────────────────────────────────────────

function search(game, m, depth = 1, dither = 0) {
  if (depth === 1) return search1(game, m, dither);
  const evaluate = g => evaluateFeatures(extractFeatures(g, m.preparedSpecs, false, undefined, true), m.weights);
  return abSearch(game, depth, evaluate, dither);
}

// Depth-1 fast path: one base extraction, then speculative-incremental
// deltaZ per candidate (V = sigma(zBase + dz)).  Captures fall back to full
// extraction on a SEPARATE prepSpecs, so the base planes deltaZ reads stay
// valid for the remaining candidates.  Ladder-coded (size:L), chain-
// attribute (C) and phase-binned (pN) specs have no incremental contract,
// so EVERY candidate takes the fallback path there — supported, several
// times slower per move.  Semantics mirror ab() at depth 1:
// board-index order, strict-improvement argmax, PASS considered last under
// the same conditions, terminal PASS scored exactly.
function search1(game, m, dither) {
  const prep = m.preparedSpecs;
  const incremental = !(prep.hasLadder || prep.hasPhasedPatterns || prep.hasHealth || prep.hasTurn);
  const f = extractFeatures(game, prep, false, undefined, true);
  evaluateFeatures(f, m.weights);
  const zBase = f.z;
  const cap = game.N * game.N;
  const isBlack = game.current === BLACK;
  let v = isBlack ? -Infinity : Infinity, best = PASS, found = false;
  for (let i = 0; i < cap; i++) {
    if (!game.isLegal(i) || game.isTrueEye(i)) continue;
    found = true;
    const d = incremental ? deltaZ(game, prep, m.weights, i) : NaN;
    let s;
    if (d === d) {
      s = 1 / (1 + Math.exp(-(zBase + d))) + (dither > 0 ? Math.random() * dither : 0);
    } else {
      // reuse the health model the caller already resolved onto m.preparedSpecs
      const fb = m._fbPrep || (m._fbPrep = prepareSpecs(m.specs,
        { health: m.preparedSpecs && m.preparedSpecs.healthModel }));
      const g = game.clone();
      g.play(i);
      s = evaluateFeatures(extractFeatures(g, fb, false, undefined, true), m.weights) + (dither > 0 ? Math.random() * dither : 0);
    }
    if (isBlack ? s > v : s < v) { v = s; best = i; }
  }
  if (!found || game.consecutivePasses > 0 || game.emptyCount < cap / 2) {
    const g = game.clone();
    g.play(PASS);
    const s = g.gameOver ? (g.estimateWinner() === BLACK ? 1 : 0)
                         : 1 / (1 + Math.exp(-zBase)) + (dither > 0 ? Math.random() * dither : 0);
    if (isBlack ? s > v : s < v) best = PASS;
  }
  return best;
}

function getMove(game) {
  return { move: search(game, model, SEARCH_DEPTH, DITHER) };
}

// Position value oracle: the static evaluation itself — V(s) = P(BLACK wins).
// Each call is an independent position with no Game3 in hand, so a ladder spec
// must rebuild one — inherent here (game3RebuildOk), not a missed reuse.
function valueB(game) {
  const g = game.cells ? game : game.toGame2();
  if (g.gameOver) return g.calcWinner() === BLACK ? 1 : 0;
  return evaluateFeatures(extractFeatures(g, model.preparedSpecs, false, undefined, true, undefined, true), model.weights);
}

// ── Persistence ───────────────────────────────────────────────────────────────

// Auto-load weights and specs if VPAT_DATA env var is set.
if (_isNode && VPAT_DATA) {
  model = loadWeights(VPAT_DATA, HEALTH_DATA);
}

// ── Exports ───────────────────────────────────────────────────────────────────

const PatternAgent = { getMove, search, valueB };

if (typeof module !== 'undefined') module.exports = PatternAgent;
else window.PatternAgent = PatternAgent;

})();
