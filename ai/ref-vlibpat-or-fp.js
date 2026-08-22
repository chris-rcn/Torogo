'use strict';

// Fixed-config reference agent: each move, a fair coin picks the decider —
// vlibpat (depth-1 value argmax, dither-free) or featurepol (softmax
// sampling, temperature 1).  Variance comes from the per-move policy
// mixture plus the fp side's sampling; the playstyle switches mid-game.
//
// All parameters are hardcoded.  This script reads no environment variables.

const path = require('path');
const { PASS } = require('../game2.js');
const { game3FromGame2 } = require('../game3.js');
const { extractFeatures: vExtract, evaluateFeatures: vEval, loadWeights: vLoadWeights } = require('../vlibpat.js');
const { search: abSearch } = require('../ab-search3.js');
const FeaturePol = require('../featurepol-lib.js');

const vModel = vLoadWeights(path.join(__dirname, '..', 'ref', 'vlibpat-4074.js'));
const { weights: fpWeights, modelName } = FeaturePol.loadModel({ name: 'ref-vlibpat-or-fp',
  path: path.join(__dirname, '..', 'ref', 'featurepol-6082.js') });

console.error(`ref-vlibpat-or-fp: vlibpat=${vModel.weights.size}w featurepol=${fpWeights.size}w  [50/50 per-move: vlib argmax / fp softmax]`);

let fpState = null;

function vlibMove(game) {
  const game3 = game3FromGame2(game);
  const evaluate = g => vEval(vExtract(g, vModel.preparedSpecs), vModel.weights);
  return abSearch(game3, 1, evaluate, 0);
}

function fpMove(game) {
  const N = game.N;
  if (!fpState || fpState.moves.length < N * N) fpState = FeaturePol.createState(N, fpWeights.spec);
  const game3 = fpWeights.spec.needsLadder ? game3FromGame2(game) : undefined;
  return FeaturePol.policyMove(game, fpState, fpWeights, Math, game3).move;
}

function getMove(game) {
  if (game.gameOver) return { move: PASS };
  return { move: Math.random() < 0.5 ? fpMove(game) : vlibMove(game) };
}

module.exports = { getMove };
