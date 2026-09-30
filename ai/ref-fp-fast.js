'use strict';

// Softmax-sampling featurepol policy with hardcoded weights file and
// temperature: ref/ref-fp-fast.js (stone8AdjLib3) sampled at temperature 1,
// so its games vary — a stochastic reference, mirroring
// ai/ref-featurepol-softmax.js.
//
// Frozen once fielded: any strength-affecting change gets a new name.
// All parameters are hardcoded.  This script reads no environment variables.

const path = require('path');
const FeaturePol = require('../featurepol-lib.js');
const { PASS } = require('../game2.js');
const { game3FromGame2 } = require('../game3.js');

const WEIGHTS     = path.join(__dirname, '..', 'ref', 'ref-fp-fast.js');
const TEMPERATURE = 1;

const { weights, modelName } = FeaturePol.loadModel({ name: 'ref-fp-fast', path: WEIGHTS });
const stateByN = new Map();

function getMove(game) {
  if (game.gameOver) return { move: PASS };
  let state = stateByN.get(game.N);
  if (!state) { state = FeaturePol.createState(game.N, weights.spec); stateByN.set(game.N, state); }
  const game3 = weights.spec.needsLadder ? game3FromGame2(game) : undefined;
  const { move } = FeaturePol.policyMove(game, state, weights, Math, game3, TEMPERATURE);
  return { move };
}

console.error(`ref-fp-fast: loaded ${weights.size} weights from ${modelName} ` +
  `spec='${weights.spec.str}' [softmax sampling, temperature ${TEMPERATURE}]`);

module.exports = { getMove };
