'use strict';

// Fixed-config reference agent: vlibpat trained at LR=0.05, EMA α=0.92,
// 90% on-policy + 10% random, resumed from the 154k off-policy size-2 ladder
// checkpoint and trained through 234k games total.  Weights file
// ref/vlibpat-9-2-onpol-ema-12.js, spec [{size:2}] (size-2 ladder only).
// Verified 68.5% / 200 games vs rave-500 and 62.5% / 200 games vs npat.
//
// All parameters are hardcoded.  This script reads no environment variables
// — use it when a stable, reproducible reference policy is needed (e.g. as
// --ext for new training runs, or as a fixed opponent in evaluation).

const path = require('path');
const { extractFeatures, evaluateFeatures, loadWeights } = require('../vlibpat.js');
const { game3FromGame2 } = require('../game3.js');
const { search: abSearch } = require('../ab-search3.js');

// ── Hardcoded configuration ──────────────────────────────────────────────────

const WEIGHTS_PATH = path.join(__dirname, '..', 'ref', 'vlibpat-9-2-onpol-ema-12.js');
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

module.exports = { getMove, search };
