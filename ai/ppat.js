'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs (require, process, etc.).
// Wrapped in an IIFE to avoid polluting the global namespace.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

const performance = (typeof window !== 'undefined') ? window.performance
  : require('perf_hooks').performance;

const { PASS, BLACK, WHITE } = _isNode ? require('../game2.js') : window.Game2;
const Util = _isNode ? require('../util.js') : window.Util;
const { createState, ppatMove, loadWeights } = _isNode ? require('../ppat-lib.js') : window.PPatterns;

const _weightsArr = _isNode
  ? loadWeights(Util.envStr('PPAT_WEIGHTS', ''))
  : loadWeights((typeof window !== 'undefined' && window.PPATWeights) || null);

let _ppatState = null;
function _ensureState(N) {
  if (_ppatState === null || _ppatState.moves.length < N * N)
    _ppatState = createState(N);
}

function getMove(game, timeBudgetMs) {
  if (game.gameOver) return { type: 'pass', move: PASS, info: 'game already over' };

  const game2 = game.cells ? game.clone() : game.toGame2();
  const N = game2.N;
  _ensureState(N);
  const m = ppatMove(game2, _ppatState, _weightsArr);
  const result = m === PASS ? { type: 'pass', move: PASS }
                            : { type: 'place', move: m, x: m % N, y: (m / N) | 0 };
  return result;
}

if (typeof module !== 'undefined') module.exports = { getMove };
else window.getMove = getMove;

})();
