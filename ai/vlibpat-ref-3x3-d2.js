'use strict';

// Fixed-config reference agent: same weights as vlibpat-ref-3x3 but with
// search depth 2 instead of 1.  3-component spec (size-2 ladder + size-3
// ladder + size-3 NL), 655k training games, ~207k weights.
//
// All parameters are hardcoded.  This script reads no environment variables.

const path = require('path');
const { extractFeatures, evaluateFeatures, loadWeights } = require('../vlibpat.js');
const { game3FromGame2 } = require('../game3.js');
const { search: abSearch } = require('../ab-search3.js');

// ── Hardcoded configuration ──────────────────────────────────────────────────

const WEIGHTS_PATH = path.join(__dirname, '..', 'ref', 'vlibpat-9-2L3L3NL-onpol70-9.js');
const DEPTH        = 2;
const DITHER       = 0.002;

// Load weights eagerly at module init.
const model = loadWeights(WEIGHTS_PATH);

// ── Search ───────────────────────────────────────────────────────────────────

function search(game, m, depth, dither) {
  const game3 = game3FromGame2(game);
  // extractFeatures auto-detects Game3 and skips the per-call rebuild.
  const evaluate = g => evaluateFeatures(extractFeatures(g, m.preparedSpecs), m.weights);
  return abSearch(game3, depth, evaluate, dither);
}

function getMove(game) {
  return { move: search(game, model, DEPTH, DITHER) };
}

module.exports = { getMove, search };
