'use strict';

// Softmax-sampling npat policy with hardcoded weights file.
//
// Same weights as ai/npat.js (out/npat-9-QD-pat4-6.js) but uses softmax
// sampling over the logits (NPat.policyMove) instead of greedy argmax.
// Provides a stochastic external move source for training runs (--ext).
//
// All parameters are hardcoded.  This script reads no environment variables.

const path = require('path');
const NPat = require('../npat-lib.js');
const { PASS } = require('../game2.js');
const { game3FromGame2 } = require('../game3.js');

// ── Hardcoded configuration ──────────────────────────────────────────────────

const WEIGHTS_PATH = path.join(__dirname, '..', 'out', 'npat-9-QD-pat4-6.js');

// ── Load weights ─────────────────────────────────────────────────────────────

const raw = require(path.resolve(WEIGHTS_PATH));
if (raw.tactStoneLimit !== undefined && raw.tactStoneLimit !== NPat.TACT_STONE_LIMIT) {
  throw new Error(
    `ref-npat-softmax: TACT_STONE_LIMIT mismatch — file ${path.basename(WEIGHTS_PATH)} ` +
    `was trained at ${raw.tactStoneLimit}, runtime is ${NPat.TACT_STONE_LIMIT}.`
  );
}

let has33c = false, hasE = false;
for (const [k] of raw.weights) {
  if (typeof k === 'string') continue;
  if      (k >= NPat.SHAPE33C_RAW_BASE && k < NPat.TYPE_E_RAW_BASE) has33c = true;
  else if (k >= NPat.TYPE_E_RAW_BASE)                                hasE   = true;
}
const weights = NPat.createWeights({
  initialCapacity: Math.max(1024, raw.weights.size | 0),
  use33c: has33c, useE: hasE,
});
for (const [k, v] of raw.weights) {
  const idx = NPat.internWeight(weights, k);
  weights.vals[idx] = v;
}

const stateByN = new Map();

function getMove(game) {
  if (game.gameOver) return { move: PASS };
  let state = stateByN.get(game.N);
  if (!state) { state = NPat.createState(game.N); stateByN.set(game.N, state); }
  const game3 = game3FromGame2(game);
  const { move } = NPat.policyMove(game, state, weights, Math, game3);
  return { move };
}

console.error(`ref-npat-softmax: loaded ${weights.size} weights from ${path.basename(WEIGHTS_PATH)} ` +
  `(3x3c=${has33c} E=${hasE}) [softmax sampling]`);

module.exports = { getMove };
