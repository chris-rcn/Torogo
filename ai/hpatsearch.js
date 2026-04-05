'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.
// Wrapped in an IIFE to avoid polluting the global namespace.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

/**
 * Hierarchical pattern-weight policy agent with alpha-beta search.
 *
 * Value function: V(s) = σ(Σ polarity_i · w[key_i]) = P(BLACK wins)
 * Move selection: full-width alpha-beta, BLACK maximises V, WHITE minimises V.
 *
 * Weights are loaded from a JS file specified by the DATA_FILE environment
 * variable (Node) or by calling loadModel() directly.
 */

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

const { createModel, extractFeatures, evaluateFeatures } = _isNode ? require('../hpatterns.js') : window.HPatterns;
const { search: abSearch } = _isNode ? require('../ab-search.js') : window.ABSearch;
const Util = _isNode ? require('../util.js') : window.Util;

const DEPTH     = Util.envInt  ('HP_SEARCH_DEPTH', 1);
const DITHER    = Util.envFloat('HP_DITHER',       0.002);
const DATA_FILE = Util.envStr  ('HP_DATA',    '');

// ── Agent state ───────────────────────────────────────────────────────────────

let model = createModel({}, Infinity);

// ── Search ────────────────────────────────────────────────────────────────────

function search(game, m, depth = 1, dither = 0) {
  const evaluate = g => evaluateFeatures(extractFeatures(g, m), m.weights);
  return abSearch(game, depth, evaluate, dither);
}

function getMove(game) {
  return { move: search(game, model, DEPTH, DITHER) };
}

// ── Persistence ───────────────────────────────────────────────────────────────

function loadModel(filePath) {
  const raw = _isNode ? require(require('path').resolve(filePath)) : window[filePath];
  const m = createModel(raw.maxStones, raw.maxSize === Infinity ? Infinity : raw.maxSize);
  m.weights = new Map(raw.weights);
  return m;
}

// Auto-load weights if DATA_FILE env var is set.
if (_isNode && DATA_FILE) {
  model = loadModel(DATA_FILE);
}

// ── Exports ───────────────────────────────────────────────────────────────────

const HPatAgent = { getMove, search, loadModel };

if (typeof module !== 'undefined') module.exports = HPatAgent;
else window.HPatAgent = HPatAgent;

})();
