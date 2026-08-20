'use strict';

// Softmax-sampling featurepol policy with hardcoded weights file.
//
// Same weights as ai/ref-featurepol.js (ref/featurepol-6082.js) but samples
// from the softmax over the logits (temperature 1) instead of greedy argmax,
// so its games vary — a stochastic reference for the CGOS ladder, mirroring
// ai/ref-npat-softmax.js.
//
// All parameters are hardcoded.  This script reads no environment variables.

const path = require('path');
const FeaturePol = require('../featurepol-lib.js');
const { PASS } = require('../game2.js');
const { game3FromGame2 } = require('../game3.js');

const WEIGHTS = path.join(__dirname, '..', 'ref', 'featurepol-6082.js');

const { weights, modelName } = FeaturePol.loadModel({ name: 'ref-featurepol-softmax', path: WEIGHTS });
const stateByN = new Map();

function getMove(game) {
  if (game.gameOver) return { move: PASS };
  let state = stateByN.get(game.N);
  if (!state) { state = FeaturePol.createState(game.N, weights.spec); stateByN.set(game.N, state); }
  const game3 = weights.spec.needsLadder ? game3FromGame2(game) : undefined;
  const { move } = FeaturePol.policyMove(game, state, weights, Math, game3);
  return { move };
}

console.error(`ref-featurepol-softmax: loaded ${weights.size} weights from ${modelName} ` +
  `spec='${weights.spec.str}' [softmax sampling]`);

module.exports = { getMove };
