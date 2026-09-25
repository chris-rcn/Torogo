'use strict';

// featurepol policy agent.  create(cfg) factory (slot-scoped config) at the top,
// so per-slot P<n>_FPOL_* overrides work under selfplay — each slot builds its
// own instance from its own FPOL_DATA.  Node-only (loads a weights file).

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
if (!_isNode) return;

const path = require('path');
const Util = require('../util.js');
const FeaturePol = require('../featurepol-lib.js');
const { PASS } = require('../game2.js');
const { game3FromGame2 } = require('../game3.js');

function create(cfg) {
  cfg = cfg || Util.makeCfg();

  const FPOL_TEMP = cfg.float('FPOL_TEMP', 0);
  const FPOL_DATA = cfg.str('FPOL_DATA', path.join(__dirname, '..', 'featurepol-cbk7wa32.js'));
  // Rank-feature shortlist (the deployment analogue of the trainer's
  // --eval-rank-topn, and sharing its default of 3): rank the vpat<n> feature
  // over only the best N moves by the other feature spaces.  0 = off (no
  // ranking pass); N < 0 = rank every candidate (whole board).  No-op for specs
  // without vpat<n>.  Set per-slot on the loaded weights, so two slots hold
  // their own widths.
  const FPOL_RANK_TOPN = cfg.int('FPOL_RANK_TOPN', 3);

  const { weights, modelName } = FeaturePol.loadModel({ name: 'featurepol', path: FPOL_DATA });
  weights.rankTopN = FPOL_RANK_TOPN;
  const _stateByN = new Map();

  function getMove(game, _budgetMs, opts) {
    if (game.gameOver) return { move: PASS };
    let state = _stateByN.get(game.N);
    if (!state) { state = FeaturePol.createState(game.N, weights.spec); _stateByN.set(game.N, state); }
    const game3 = weights.spec.needsLadder ? game3FromGame2(game) : undefined;
    const rng = opts && opts.rng;
    const move = FeaturePol.policyMove(game, state, weights, rng, game3, FPOL_TEMP).move;
    return { move };
  }

  console.error(`featurepol[${cfg.slot != null ? cfg.slot : '-'}]: loaded ${weights.size} weights from ` +
                `${modelName}  spec='${weights.spec.str}'  temp=${FPOL_TEMP}`);
  return { getMove };
}

let _default = null;
function _def() { return _default || (_default = create(Util.makeCfg())); }

module.exports = { create, getMove: (g, b, o) => _def().getMove(g, b, o) };

})();
