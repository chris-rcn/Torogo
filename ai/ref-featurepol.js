'use strict';

// featurepol policy agent.  Loads a featurepol checkpoint (path from
// Node-only (loads a weights file at startup).

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
if (!_isNode) return;

const path = require('path');
const FeaturePol = require('../featurepol-lib.js');
const { PASS } = require('../game2.js');
const { game3FromGame2 } = require('../game3.js');

const TEMP = 0;
const WEIGHTS = "ref/featurepol-6082.js";

const { weights, modelName } = FeaturePol.loadModel({ name: 'featurepol', path: WEIGHTS });
const _stateByN = new Map();

function getMove(game, _budgetMs, opts) {
  if (game.gameOver) return { move: PASS };
  let state = _stateByN.get(game.N);
  if (!state) { state = FeaturePol.createState(game.N, weights.spec); _stateByN.set(game.N, state); }
  const game3 = weights.spec.needsLadder ? game3FromGame2(game) : undefined;
  const rng = opts && opts.rng;
  const move = FeaturePol.policyMove(game, state, weights, rng, game3, TEMP).move;
  return { move };
}

console.error(`featurepol: loaded ${weights.size} weights from ${modelName}  spec='${weights.spec.str}'  temp=${TEMP}`);

module.exports = { getMove };

})();
