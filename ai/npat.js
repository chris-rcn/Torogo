'use strict';

// Greedy npat policy.  Loads a weights checkpoint (path from $NPAT_WEIGHTS or
// the default below) and picks the move with the highest logit each call.
// Only invoked from Node — selfplay loads its policies via require().

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
if (!_isNode) return; // npat policy is Node-only (loads a weights file at startup)

const path = require('path');
const NPat = require('../npat-lib.js');
const { PASS } = require('../game2.js');
const { Game3, game3FromGame2 } = require('../game3.js');

const weightsPath = process.env.NPAT_WEIGHTS
  || path.join(__dirname, '..', 'out', 'npat-9-QD-pat4-6.js');
const raw = require(path.resolve(weightsPath));
if (raw.tactStoneLimit !== undefined && raw.tactStoneLimit !== NPat.TACT_STONE_LIMIT) {
  throw new Error(
    `npat: TACT_STONE_LIMIT mismatch — file ${path.basename(weightsPath)} ` +
    `was trained at ${raw.tactStoneLimit}, runtime is ${NPat.TACT_STONE_LIMIT}. ` +
    `Set NPAT_STONE_LIMIT=${raw.tactStoneLimit} before launching.`
  );
}

// Infer feature flags from the raw key types/ranges in the file.
let has33c = false, hasE = false;
for (const [k] of raw.weights) {
  if (typeof k === 'string') continue; // ignore any orphan string keys
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

// One state per board size we encounter, lazily built.
const stateByN = new Map();

function getMove(game) {
  if (game.gameOver) return { move: PASS };
  let state = stateByN.get(game.N);
  if (!state) { state = NPat.createState(game.N); stateByN.set(game.N, state); }
  // Rebuild Game3 from current Game2 each call (selfplay doesn't expose move
  // history, and Game3 is cheap relative to npat extraction).
  const game3 = game3FromGame2(game);
  const move = NPat.greedyMove(game, state, weights, game3);
  return { move };
}

console.error(`npat: loaded ${weights.size} weights from ${path.basename(weightsPath)} ` +
  `(3x3c=${has33c} E=${hasE})`);

module.exports = { getMove };

})();
