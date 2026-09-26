'use strict';

// ref-fp-heavy: frozen featurepol reference agent with a heavy vpat<n> ranking.
//
// A self-contained frozen COPY of the featurepol policy agent with hardcoded
// config — it reads NO environment variables:
//   weights:      ref/ref-fp-heavy-data.js  (frozen copy of
//                 out/featurepol-7c799j4s.js; spec includes vpat9, its value
//                 model embedded, so no external vpat model is needed)
//   temperature:  ramps DOWN with phase (board fullness), 1.0 at the empty board
//                 to 0.5 at the full board: diverse, exploratory openings and a
//                 sharp, near-greedy endgame.  Samples from the softmax over the
//                 logits — a stochastic reference, like ref-fp2.
//   rank shortlist: 14  (the vpat9 rank feature scores the best 14 candidates by
//                 the other spaces — a heavier pass than ref-fp2's 3)
//
// Node-only (loads a weights file at startup).

const path = require('path');
const FeaturePol = require('../featurepol-lib.js');
const { PASS } = require('../game2.js');
const { game3FromGame2 } = require('../game3.js');

const WEIGHTS    = path.join(__dirname, '..', 'ref', 'ref-fp-heavy-data.js');
// Softmax temperature ramps DOWN linearly with phase (board fullness):
// TEMP_START at the empty board, TEMP_END at the full board.
const TEMP_START = 1.8;   // phase 0
const TEMP_END   = 0.1;   // phase 1
const RANK_TOPN  = 14;

const { weights, modelName } = FeaturePol.loadModel({ name: 'ref-fp-heavy', path: WEIGHTS });
weights.rankTopN = RANK_TOPN;
const stateByN = new Map();

function getMove(game, _budgetMs, opts) {
  if (game.gameOver) return { move: PASS };
  let state = stateByN.get(game.N);
  if (!state) { state = FeaturePol.createState(game.N, weights.spec); stateByN.set(game.N, state); }
  const game3 = weights.spec.needsLadder ? game3FromGame2(game) : undefined;
  const rng = opts && opts.rng;
  const phase = 1 - game.emptyCount / (game.N * game.N);
  const temp = TEMP_START + (TEMP_END - TEMP_START) * phase;
  const { move } = FeaturePol.policyMove(game, state, weights, rng, game3, temp);
  return { move };
}

console.error(`ref-fp-heavy: loaded ${weights.size} weights from ${modelName} ` +
  `spec='${weights.spec.str}' temp=${TEMP_START}->${TEMP_END} by phase rank-topn=${RANK_TOPN}` +
  `${weights.vpatModel ? ' vpat=embedded' : ''} [softmax sampling]`);

module.exports = { getMove };
