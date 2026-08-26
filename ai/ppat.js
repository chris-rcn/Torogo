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

// ── Factory ──────────────────────────────────────────────────────────────────
// create(cfg) → { getMove }.  cfg is a Util.makeCfg reader (slot-aware env:
// P1_/P2_ prefixes in selfplay), so two ppat models can be compared head to
// head in one process:
//   P1_PPAT_DATA=a.js P2_PPAT_DATA=b.js node selfplay.js --p1 ppat --p2 ppat
// All per-instance state lives in this closure.
function create(cfg) {
  cfg = cfg || Util.makeCfg();

  // PPAT_DATA defaults to the root ppat-data.js (as puct-ppat/puct-ppat-fp/
  // cascade/rave-ppat do); window.PPATWeights in the browser.  This agent IS
  // the ppat policy, so failing to load weights would silently turn it into a
  // uniform-random player — a hard error is the honest outcome.
  const _ppatPath = _isNode
    ? cfg.str('PPAT_DATA', require('path').join(__dirname, '..', 'ppat-data.js'))
    : null;
  const _model = _isNode
    ? loadWeights(_ppatPath)
    : loadWeights((typeof window !== 'undefined' && window.PPATWeights) || null);
  if (!_model) throw new Error(`ppat: cannot load ppat weights from ${_isNode ? _ppatPath : 'window.PPATWeights'}`);

  // Play uniform-random moves while board fullness < this fraction [0,1] (0 = off).
  _model.uniformBelowPhase = cfg.float('PPAT_UNIFORM_BELOW_PHASE', 0);

  console.log(`ppat[${cfg.slot != null ? cfg.slot : '-'}]: ${_model.weights.length} weights ` +
              `(${_model.phaseCount} phase(s), libCap ${_model.libCap}) from ` +
              `${_isNode ? require('path').basename(_ppatPath) : 'window.PPATWeights'} [policy sampling, no search]`);

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
    const m = ppatMove(game2, _ppatState, _model);
    const result = m === PASS ? { type: 'pass', move: PASS }
                              : { type: 'place', move: m, x: m % N, y: (m / N) | 0 };
    return result;
  }

  return { getMove };
}

// Lazy default instance for direct-require callers.
let _default = null;
function _def() { return _default || (_default = create(Util.makeCfg())); }

if (typeof module !== 'undefined') module.exports = { create, getMove: (g, b) => _def().getMove(g, b) };
else window.getMove = (g, b) => _def().getMove(g, b);

})();
