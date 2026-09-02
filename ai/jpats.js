'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.
// Wrapped in an IIFE to avoid polluting the global namespace.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

/**
 * jpats value-function agent: gated hierarchical patterns + alpha-beta search.
 *
 * Value: V(s) = sigma(sum polarity_i * w[key_i]) = P(BLACK wins).  BLACK
 * maximises, WHITE minimises.
 *
 * Weights come from a train-jpats.js checkpoint named by JP_DATA (Node) or
 * window.jpatsModel (browser).  The checkpoint carries the geometry the model
 * was trained with — minPSize and maxPSize — and both are restored, because the
 * feature set is only meaningful under the same gating that produced it.
 *
 * Extraction never passes collectPending, so the agent cannot grow or mutate
 * the weight table: playing is read-only, admission belongs to training.
 */

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

const { createModel, extractFeatures, evaluateFeatures, weightsMap } = _isNode ? require('../jpats-lib.js') : window.JPats;
const { search: abSearch } = _isNode ? require('../ab-search.js') : window.ABSearch;
const Util = _isNode ? require('../util.js') : window.Util;

const JP_SEARCH_DEPTH = Util.envInt  ('JP_SEARCH_DEPTH', 1);
const JP_DITHER       = Util.envFloat('JP_DITHER',       0.002);
const JP_DATA         = Util.envStr  ('JP_DATA',         '');

// ── Agent state ───────────────────────────────────────────────────────────────

let model = createModel();

// ── Search ────────────────────────────────────────────────────────────────────

function search(game, m, depth = 1, dither = 0) {
  const evaluate = g => evaluateFeatures(extractFeatures(g, m), m.weights);
  return abSearch(game, depth, evaluate, dither);
}

function getMove(game) {
  return { move: search(game, model, JP_SEARCH_DEPTH, JP_DITHER) };
}

// P(BLACK wins) for the position, for callers that want the value directly.
function valueB(game) {
  return evaluateFeatures(extractFeatures(game, model), model.weights);
}

// ── Persistence ───────────────────────────────────────────────────────────────

function loadModel(filePath) {
  const raw = _isNode ? require(require('path').resolve(filePath)) : window[filePath];
  const m = createModel({
    minPSize: raw.minPSize !== undefined ? raw.minPSize : 3,
    maxPSize: raw.maxPSize === undefined || raw.maxPSize === null ? Infinity : raw.maxPSize,
  });
  m.weights = weightsMap(raw);
  return m;
}

if (_isNode && JP_DATA) model = loadModel(JP_DATA);
else if (!_isNode && typeof window !== 'undefined' && window.jpatsModel) {
  model = createModel({
    minPSize: window.jpatsModel.minPSize !== undefined ? window.jpatsModel.minPSize : 3,
    maxPSize: window.jpatsModel.maxPSize == null ? Infinity : window.jpatsModel.maxPSize,
  });
  model.weights = weightsMap(window.jpatsModel);
}

// ── Exports ───────────────────────────────────────────────────────────────────

const JPatAgent = { getMove, valueB, search, loadModel };

if (typeof module !== 'undefined') module.exports = JPatAgent;
else window.JPatAgent = JPatAgent;

})();
