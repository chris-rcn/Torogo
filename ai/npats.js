'use strict';

// Greedy policy agent: plays the policy softmax argmax each move.  The
// counterpart of ai/npat.js for the new policy system — use it to measure a
// trained policy model directly (e.g. head-to-head against npat) without a
// search on top.  Model selected by $NPATS_WEIGHTS (default: npats-data.js,
// whose empty generation-0 form plays uniformly random legal moves).
// Only invoked from Node — selfplay loads its policies via require().

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
if (!_isNode) return; // policy agent is Node-only (loads a weights file at startup)

const NPats = require('../npats-lib.js');
const { PASS } = require('../game2.js');

const { weights, cfg, modelName } = NPats.loadModel({
  name: 'npats',
  path: process.env.NPATS_WEIGHTS,
});
const model = { weights, cfg };
console.error(`npats: loaded ${weights.size} weights from ${modelName} [argmax]`);

const stateByN = new Map();

function getMove(game) {
  if (game.gameOver) return { move: PASS };
  let state = stateByN.get(game.N);
  if (!state) { state = NPats.createState(game.N); stateByN.set(game.N, state); }
  const n = NPats.computeProbs(game, state, undefined, model, 0);  // T=0 → one-hot argmax
  if (n === 0) return { move: PASS };
  let best = 0;
  for (let i = 1; i < n; i++) if (state.probs[i] > state.probs[best]) best = i;
  return { move: state.moves[best] };
}

module.exports = { getMove };

})();
