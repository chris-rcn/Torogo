'use strict';

// Greedy npat policy.  Loads a weights checkpoint (path from $NPAT_WEIGHTS or
// the default below) and picks the move with the highest logit each call.
// Only invoked from Node — selfplay loads its policies via require().

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
if (!_isNode) return; // npat policy is Node-only (loads a weights file at startup)

const NPat = require('../npat-lib.js');
const Util = require('../util.js');
const { PASS } = require('../game2.js');
const { Game3, game3FromGame2 } = require('../game3.js');

// Softmax temperature for move selection — 0 = argmax, 1 = standard softmax,
// in-between values give sharper (< 1) or flatter (> 1) sampling.  Default 0
// reproduces the previous greedy behavior.
const NPAT_TEMP = 0;

const { weights, modelName } = NPat.loadModel({ name: 'ref-npat', path: 'ref/npat-p8.js' });

// One state per board size we encounter, lazily built.
const stateByN = new Map();

function getMove(game) {
  if (game.gameOver) return { move: PASS };
  let state = stateByN.get(game.N);
  if (!state) { state = NPat.createState(game.N); stateByN.set(game.N, state); }
  // Rebuild Game3 from current Game2 each call (selfplay doesn't expose move
  // history, and Game3 is cheap relative to npat extraction).
  const game3 = game3FromGame2(game);
  const move = NPat.policyMove(game, state, weights, Math, game3, NPAT_TEMP).move;
  return { move };
}

console.error(`ref-npat: loaded ${weights.size} weights from ${modelName} (${NPat.cfgFlags(weights.cfg)})`);

module.exports = { getMove };

})();
