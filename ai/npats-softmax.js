'use strict';

// Softmax-sampling policy agent: samples each move from the policy softmax —
// the stochastic counterpart of ai/npats.js, as ref-npat-softmax is to
// npat.  Gives naturally diverse selfplay comparisons (no --rand-moves
// needed) and measures the whole distribution rather than just the argmax.
// Model selected by $NPATS_WEIGHTS (default: npats-data.js); sampling
// temperature by $NPATS_TEMP (default 1 = standard softmax, < 1 sharper,
// 0 = argmax).
// Only invoked from Node — selfplay loads its policies via require().

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
if (!_isNode) return; // policy agent is Node-only (loads a weights file at startup)

const NPats = require('../npats-lib.js');
const Util   = require('../util.js');
const { PASS } = require('../game2.js');

const NPATS_TEMP = Util.envFloat('NPATS_TEMP', 1);

const { weights, cfg, modelName } = NPats.loadModel({
  name: 'npats-softmax',
  path: process.env.NPATS_WEIGHTS,
});
const model = { weights, cfg };
console.error(`npats-softmax: loaded ${weights.size} weights from ${modelName} [softmax T=${NPATS_TEMP}]`);

const stateByN = new Map();

function getMove(game) {
  if (game.gameOver) return { move: PASS };
  let state = stateByN.get(game.N);
  if (!state) { state = NPats.createState(game.N); stateByN.set(game.N, state); }
  const n = NPats.computeProbs(game, state, undefined, model, NPATS_TEMP);
  if (n === 0) return { move: PASS };
  let r = Math.random(), chosen = n - 1;
  for (let i = 0; i < n; i++) {
    r -= state.probs[i];
    if (r <= 0) { chosen = i; break; }
  }
  return { move: state.moves[chosen] };
}

module.exports = { getMove };

})();
