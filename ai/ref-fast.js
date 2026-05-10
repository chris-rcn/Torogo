'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs (require, process, etc.).
// Loaded as a plain <script> tag; ladder-pat-lib.js must be loaded first.

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

const { PASS } = _isNode ? require('../game2.js') : window.Game2;
const Util = _isNode ? require('../util.js') : window.Util;
const { createLadderPat, softmax } = _isNode ? require('../ladder-pat-lib.js') : window.LadderPat;

// Load weights.
let _weights = null;
let _ladderPat = null;

function _loadWeights() {
  if (_weights !== null) return;
  const path = 'out/ref-fast-weights.js';
  if (!path) return;
  if (_isNode) {
    const resolve = require('path').resolve;
    _weights = require(resolve(path));
  } else if (window.ladderPatWeights) {
    _weights = window.ladderPatWeights;
  }
}
_loadWeights();

function _weightFn(key) {
  const w = _weights.get(key);
  return w !== undefined ? w : 0;
}

function getMove(game) {
  if (game.gameOver) return { type: 'pass', move: PASS, info: 'game already over' };

  const game2 = game.cells ? game.clone() : game.toGame2();
  const N = game2.N;

  if (!_weights) {
    return { type: 'pass', move: PASS, info: 'no weights loaded' };
  }

  if (!_ladderPat) _ladderPat = createLadderPat(_weights);

  const candidates = _ladderPat.getFeatures(game2);
  if (candidates.length === 0) return { type: 'pass', move: PASS };

  const { vals, sum } = softmax(candidates, _weightFn);

  // Sample from the policy distribution.
  let r = Math.random() * sum, chosen = candidates.length - 1;
  for (let i = 0; i < candidates.length; i++) {
    r -= vals[i];
    if (r <= 0) { chosen = i; break; }
  }

  const m = candidates[chosen].move;
  return m === PASS ? { type: 'pass', move: PASS }
                    : { type: 'place', move: m, x: m % N, y: (m / N) | 0 };
}

if (typeof module !== 'undefined') module.exports = { getMove };
else window.getMove = getMove;

})();
