'use strict';

// Fixed-config reference agent: vlibpat 3-component spec (size-2 ladder +
// size-3 ladder + size-3 NL), trained at LR=0.3, EMA α=0.9, 70% on-policy +
// 5% random + 25% softmax-npat external, through 655k games total.  Weights
// Verified ~78% / 1000 games vs vlibpat-ref-2x2.
//
// All parameters are hardcoded.  This script reads no environment variables
// — use it when a stable, reproducible reference policy is needed (e.g. as
// a fixed opponent in evaluation).

const path = require('path');
const { extractFeatures, evaluateFeatures, loadWeights } = require('../vlibpat.js');
const { game3FromGame2 } = require('../game3.js');
const { search: abSearch } = require('../ab-search3.js');

// ── Hardcoded configuration ──────────────────────────────────────────────────

const WEIGHTS_PATH = path.join(__dirname, '..', 'ref', 'vlibpat-4074.js');
const DEPTH        = 1;
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

// Raw vlibpat value of a Game2 position: P(BLACK wins) in [0,1] (leaf eval, no
// search).  For use as an SB value oracle.
function valueB(game, options) {   // options unused (static leaf eval); kept for API uniformity
  return evaluateFeatures(extractFeatures(game3FromGame2(game), model.preparedSpecs), model.weights);
}

module.exports = { getMove, search, valueB };
