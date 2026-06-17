'use strict';

// Softmax-sampling npat policy with hardcoded weights file.
//
// Same weights as ai/npat.js (npat-data.js) but uses softmax
// sampling over the logits (NPat.policyMove) instead of greedy argmax.
// Provides a stochastic external move source for training runs (--ext).
//
// All parameters are hardcoded.  This script reads no environment variables.

const NPat = require('../npat-lib.js');
const { PASS } = require('../game2.js');
const { game3FromGame2 } = require('../game3.js');

// ── Load weights (canonical npat-data.js) ────────────────────────────────────

const { weights, modelName } = NPat.loadModel({ name: 'ref-npat-softmax' });

const stateByN = new Map();

function getMove(game) {
  if (game.gameOver) return { move: PASS };
  let state = stateByN.get(game.N);
  if (!state) { state = NPat.createState(game.N); stateByN.set(game.N, state); }
  const game3 = game3FromGame2(game);
  const { move } = NPat.policyMove(game, state, weights, Math, game3);
  return { move };
}

console.error(`ref-npat-softmax: loaded ${weights.size} weights from ${modelName} ` +
  `(3x3c=${weights.cfg.use33c} p12=${weights.cfg.useP12}) [softmax sampling]`);

module.exports = { getMove };
