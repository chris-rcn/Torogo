'use strict';

// featurepol-b: a SECOND featurepol policy agent, identical to featurepol.js but reading its
// checkpoint from FPOL_DATA_B (and temperature from FPOL_TEMP_B).  Exists solely so two
// distinct featurepol checkpoints can be evaluated head-to-head in one selfplay run, which the
// single shared FPOL_DATA env of featurepol.js cannot express.

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
if (!_isNode) return;

const path = require('path');
const FeaturePol = require('../featurepol-lib.js');
const { PASS } = require('../game2.js');
const { game3FromGame2 } = require('../game3.js');

const TEMP = process.env.FPOL_TEMP_B !== undefined ? parseFloat(process.env.FPOL_TEMP_B) : 0;
const WEIGHTS = process.env.FPOL_DATA_B || path.join(__dirname, '..', 'featurepol-data.js');

const { weights, modelName } = FeaturePol.loadModel({ name: 'featurepol-b', path: WEIGHTS });
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

console.error(`featurepol-b: loaded ${weights.size} weights from ${modelName}  spec='${weights.spec.str}'  temp=${TEMP}`);

module.exports = { getMove };

})();
