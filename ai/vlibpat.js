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
 * Weights and specs are loaded from a JS file specified by the VLIBPAT_DATA
 * environment variable (Node) or by calling loadWeights() directly.
 */

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

const { extractFeatures, evaluateFeatures, loadWeights, prepareSpecs } = _isNode ? require('../vlibpat.js') : window.VlibPat;
const { game3FromGame2 } = _isNode ? require('../game3.js') : window.Game3;
const { search: abSearch } = _isNode ? require('../ab-search3.js') : window.ABSearch3;
const Util = _isNode ? require('../util.js') : window.Util;

const DEPTH    = Util.envInt  ('SEARCH_DEPTH', 1);
const DITHER   = Util.envFloat('DITHER',       0.002);
const PAT_DATA = Util.envStr  ('VLIBPAT_DATA', '');

// ── Agent state ───────────────────────────────────────────────────────────────

const defaultSpecs = [{ size: 1 }, { size: 2 }, { size: 3 }];

let model = { weights: new Map(), specs: defaultSpecs, preparedSpecs: prepareSpecs(defaultSpecs) };

// ── Search ────────────────────────────────────────────────────────────────────

function search(game, m, depth = 1, dither = 0) {
  const game3 = game3FromGame2(game);
  // extractFeatures auto-detects Game3 and skips the per-call rebuild.
  const evaluate = g => evaluateFeatures(extractFeatures(g, m.preparedSpecs), m.weights);
  return abSearch(game3, depth, evaluate, dither);
}

function getMove(game) {
  return { move: search(game, model, DEPTH, DITHER) };
}

// ── Persistence ───────────────────────────────────────────────────────────────

// Auto-load weights and specs if VLIBPAT_DATA env var is set.
if (_isNode && PAT_DATA) {
  model = loadWeights(PAT_DATA);
}

// ── Exports ───────────────────────────────────────────────────────────────────

const VlibpatAgent = { getMove, search };

if (typeof module !== 'undefined') module.exports = VlibpatAgent;
else window.VlibpatAgent = VlibpatAgent;

})();
