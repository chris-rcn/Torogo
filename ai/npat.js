'use strict';

// Greedy npat policy.  Loads a weights checkpoint (path from $NPAT_WEIGHTS or
// the default below) and picks the move with the highest logit each call.
// Only invoked from Node — selfplay loads its policies via require().

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
if (!_isNode) return; // npat policy is Node-only (loads a weights file at startup)

const path = require('path');
const NPat = require('../npat-lib.js');
const Util = require('../util.js');
const { PASS } = require('../game2.js');
const { Game3, game3FromGame2 } = require('../game3.js');

// Softmax temperature for move selection — 0 = argmax, 1 = standard softmax,
// in-between values give sharper (< 1) or flatter (> 1) sampling.  Default 0
// reproduces the previous greedy behavior.
const NPAT_TEMP = Util.envFloat('NPAT_TEMP', 0);

const weightsPath = process.env.NPAT_WEIGHTS
  || path.join(__dirname, '..', 'npat-data.js');
const raw = require(path.resolve(weightsPath));
if (raw.tactStoneLimit !== undefined && raw.tactStoneLimit !== NPat.TACT_STONE_LIMIT) {
  throw new Error(
    `npat: TACT_STONE_LIMIT mismatch — file ${path.basename(weightsPath)} ` +
    `was trained at ${raw.tactStoneLimit}, runtime is ${NPat.TACT_STONE_LIMIT}. ` +
    `Set NPAT_STONE_LIMIT=${raw.tactStoneLimit} before launching.`
  );
}

// Infer feature flags from the raw key ranges in the file.
let has33c = false, hasP12 = false;
for (const [k] of raw.weights) {
  if (typeof k === 'string') continue; // ignore any orphan string keys
  if      (k >= NPat.SHAPE33C_RAW_BASE && k < NPat.P12_RAW_BASE) has33c = true;
  else if (k >= NPat.P12_RAW_BASE)                                hasP12 = true;
}
const weights = NPat.createWeights({
  initialCapacity: Math.max(1024, raw.weights.size | 0),
  use33c: has33c, useP12: hasP12,
});
for (const [k, v] of raw.weights) {
  const idx = NPat.internWeight(weights, k);
  weights.vals[idx] = v;
}

// One state per board size we encounter, lazily built.
const stateByN = new Map();

function getMove(game) {
  if (game.gameOver) return { move: PASS };
  let state = stateByN.get(game.N);
  if (!state) { state = NPat.createState(game.N); stateByN.set(game.N, state); }
  // Rebuild Game3 from current Game2 each call (selfplay doesn't expose move
  // history, and Game3 is cheap relative to npat extraction).
  const game3 = game3FromGame2(game);
  const move = NPat.policyMove(game, state, weights, Math, game3, NPAT_TEMP).move;
  return { move };
}

console.error(`npat: loaded ${weights.size} weights from ${path.basename(weightsPath)} ` +
  `(3x3c=${has33c} p12=${hasP12})`);

module.exports = { getMove };

})();
