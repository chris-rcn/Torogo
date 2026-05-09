'use strict';

// Fixed-config reference agent: vlibpat 3-component spec (size-2 ladder +
// size-3 ladder + size-3 NL), trained at LR=0.3, EMA α=0.9, 70% on-policy +
// 5% random + 25% softmax-npat external, through 655k games total.  Weights
// file out/vlibpat-9-2L3L3NL-onpol70-9.js, ~207k weights.
// Verified ~78% / 1000 games vs vlibpat-ref-2x2.
//
// All parameters are hardcoded.  This script reads no environment variables
// — use it when a stable, reproducible reference policy is needed (e.g. as
// a fixed opponent in evaluation).

const path = require('path');
const { extractFeatures, evaluateFeatures, loadWeights } = require('../vlibpat.js');
const { search: abSearch } = require('../ab-search.js');

// ── Hardcoded configuration ──────────────────────────────────────────────────

const WEIGHTS_PATH = path.join(__dirname, '..', 'out', 'vlibpat-9-2L3L3NL-onpol70-9.js');
const DEPTH        = 1;
const DITHER       = 0.002;

// Load weights eagerly at module init.
const model = loadWeights(WEIGHTS_PATH);

// ── Search ───────────────────────────────────────────────────────────────────

function search(game, m, depth, dither) {
  const evaluate = g => evaluateFeatures(extractFeatures(g, m.preparedSpecs), m.weights);
  return abSearch(game, depth, evaluate, dither);
}

function getMove(game) {
  return { move: search(game, model, DEPTH, DITHER) };
}

module.exports = { getMove, search };
