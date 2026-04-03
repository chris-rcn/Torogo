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

const { extractFeatures, evaluateFeatures, loadWeights, prepareSpecs } = _isNode ? require('../vpatterns.js') : window.VPatterns;
const { search: abSearch } = _isNode ? require('../ab-search.js') : window.ABSearch;
const Util = _isNode ? require('../util.js') : window.Util;

const MIN_LIBS = Util.envInt  ('MIN_LIBS',     1);
const MAX_LIBS = Util.envInt  ('MAX_LIBS',     1);
const DEPTH    = Util.envInt  ('SEARCH_DEPTH', 1);
const DITHER   = Util.envFloat('DITHER',       0.002);
const PAT_DATA = Util.envStr  ('PAT_DATA',     'out/vpatterns-S1L3_S2L3_S3L1.js');

// ── Agent state ───────────────────────────────────────────────────────────────

const defaultSpecs = [];
for (let maxLibs = MIN_LIBS; maxLibs <= MAX_LIBS; maxLibs++)
  for (const size of [1, 2, 3])
    defaultSpecs.push({ size, maxLibs });

let model = { weights: new Map(), specs: defaultSpecs, preparedSpecs: prepareSpecs(defaultSpecs) };

// ── Search ────────────────────────────────────────────────────────────────────

function search(game, m, depth = 1, dither = 0) {
  const evaluate = g => evaluateFeatures(extractFeatures(g, m.preparedSpecs), m.weights);
  return abSearch(game, depth, evaluate, dither);
}

function getMove(game) {
  return { move: search(game, model, DEPTH, DITHER) };
}

// ── Persistence ───────────────────────────────────────────────────────────────

// Auto-load weights and specs if PAT_DATA env var is set.
if (_isNode && PAT_DATA) {
  model = loadWeights(PAT_DATA);
}

// ── Exports ───────────────────────────────────────────────────────────────────

const PatternAgent = { getMove, search };

if (typeof module !== 'undefined') module.exports = PatternAgent;
else window.PatternAgent = PatternAgent;

})();
