'use strict';

// ref-fp2: frozen featurepol softmax reference agent.
//
// A self-contained frozen COPY of the featurepol policy agent with hardcoded
// config — it reads NO environment variables:
//   weights:      ref/ref-fp2-data.js  (spec includes vpat9; its value model is
//                 embedded in the file, so no external vpat model is needed)
//   temperature:  1  (samples from the softmax over the logits — a stochastic
//                 reference, like ref-featurepol-softmax)
//   rank shortlist: 0  (whole-board ranking — the vpat9 rank feature scores
//                 every candidate, matching featurepol's and the trainer's default)
//
// Node-only (loads a weights file at startup).

const path = require('path');
const FeaturePol = require('../featurepol-lib.js');
const { PASS } = require('../game2.js');
const { game3FromGame2 } = require('../game3.js');

const WEIGHTS   = path.join(__dirname, '..', 'ref', 'ref-fp2-data.js');
const TEMP      = 1;
const RANK_TOPN = 3;

FeaturePol.setRankTopN(RANK_TOPN);
const { weights, modelName } = FeaturePol.loadModel({ name: 'ref-fp2', path: WEIGHTS });
const stateByN = new Map();

function getMove(game, _budgetMs, opts) {
  if (game.gameOver) return { move: PASS };
  let state = stateByN.get(game.N);
  if (!state) { state = FeaturePol.createState(game.N, weights.spec); stateByN.set(game.N, state); }
  const game3 = weights.spec.needsLadder ? game3FromGame2(game) : undefined;
  const rng = opts && opts.rng;
  const { move } = FeaturePol.policyMove(game, state, weights, rng, game3, TEMP);
  return { move };
}

console.error(`ref-fp2: loaded ${weights.size} weights from ${modelName} ` +
  `spec='${weights.spec.str}' temp=${TEMP} rank-topn=${RANK_TOPN}` +
  `${weights.vpatModel ? ' vpat=embedded' : ''} [softmax sampling]`);

module.exports = { getMove };
