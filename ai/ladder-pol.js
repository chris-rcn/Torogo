'use strict';

// ladder-pol — alpha-beta over the moves ladder2.js calls urgent.
//
// At every node the candidates are the side to move's URGENT moves: the union
// of getLadderStatus's urgentLibs over every chain ladder2 reads (chains with
// 1-2 liberties, and the opponent's 3-liberty chains) — the moves that save a
// chain of its own that would otherwise be captured in a ladder, or capture an
// opponent chain that would otherwise escape.  With no urgent move, the
// candidates are the moves featurepol ranks highest (ref-fp-heavy's model and
// ranking): QUIET_MOVES_ROOT of them at the root, QUIET_MOVES below it.  Leaves are
// valued by a vpat model; a finished game by its winner.
//
// Config:
//   AB_DEPTH    search depth in plies                                   (default 2)
//   QUIET_MOVES_ROOT  candidates at the root when no move is urgent there:
//               featurepol's top this many                              (default 2)
//   QUIET_MOVES candidates at any other node with no urgent move          (default 1)
//   FPOL_DATA   featurepol model for the quiet moves (default
//               ref/ref-fp-heavy-data.js, ref-fp-heavy's; its vpat<n> ranking
//               term over the top FP_RANK_TOPN (14) candidates, as ref-fp-heavy)
//   VPAT_DATA   leaf evaluator (default ref/ref-ab-fp-vpat-data.js, the
//               ladder-aware model ref-ab2-fp4-vpat searches with)

(function () {

const Util = (typeof require === 'function') ? require('../util.js') : window.Util;
const { PASS, BLACK } = Util.load('./game2.js', 'Game2');
const { game3FromGame2 } = Util.load('./game3.js', 'Game3');
const Ladder2    = Util.load('./ladder2.js', 'Ladder2');
const FeaturePol = Util.load('./featurepol-lib.js', 'FeaturePol');
const VPat       = Util.load('./vpatterns.js', 'VPatterns');
const { makeRng } = Util.load('./xorshift.js', 'XorShift');

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
const FP_RANK_TOPN = 14;   // ref-fp-heavy's vpat<n> ranking width

function create(cfg) {
  cfg = cfg || Util.makeCfg();
  const AB_DEPTH = cfg.int('AB_DEPTH', 2);
  if (!(AB_DEPTH >= 1)) throw new Error(`ladder-pol: AB_DEPTH must be at least 1, got ${AB_DEPTH}`);
  const QUIET_MOVES_ROOT = cfg.int('QUIET_MOVES_ROOT', 2);
  const QUIET_MOVES = cfg.int('QUIET_MOVES', 1);
  if (!(QUIET_MOVES_ROOT >= 1)) throw new Error(`ladder-pol: QUIET_MOVES_ROOT must be at least 1, got ${QUIET_MOVES_ROOT}`);
  if (!(QUIET_MOVES >= 1)) throw new Error(`ladder-pol: QUIET_MOVES must be at least 1, got ${QUIET_MOVES}`);

  const fpPath = _isNode ? cfg.str('FPOL_DATA', require('path').join(__dirname, '..', 'ref', 'ref-fp-heavy-data.js')) : undefined;
  const { weights: fpWeights, modelName } = FeaturePol.loadModel({ name: 'ladder-pol', path: fpPath });
  if (fpWeights.spec.rankSpaces && fpWeights.spec.rankSpaces.length > 0) fpWeights.rankTopN = FP_RANK_TOPN;
  const vpatPath = _isNode ? cfg.str('VPAT_DATA', require('path').join(__dirname, '..', 'ref', 'ref-ab-fp-vpat-data.js')) : null;
  const vpatModel = _isNode ? VPat.loadWeights(vpatPath) : VPat.modelFromRaw(window.vpatternsModel);
  console.error(`ladder-pol[${cfg.slot != null ? cfg.slot : '-'}]: depth ${AB_DEPTH}, quiet moves: featurepol top ${QUIET_MOVES_ROOT} at the root, ${QUIET_MOVES} below ` +
    `(${modelName}${fpWeights.rankTopN > 0 ? `, rank-topn ${fpWeights.rankTopN}` : ''}), leaves ${_isNode ? require('path').basename(vpatPath) : 'window.vpatternsModel'} ` +
    `(${VPat.specString(vpatModel.specs)})`);

  let fpState = null, fpScores = null;

  // The side to move's urgent moves (deduped, in read order), else
  // featurepol's top `quiet`; `.urgent` says which.
  function candidates(g, g3, quiet) {
    const urgent = [];
    for (const { status } of Ladder2.getAllLadderStatuses(g3)) {
      if (!status) continue;
      for (const m of status.urgentLibs) if (!urgent.includes(m)) urgent.push(m);
    }
    const c = urgent.length > 0 ? urgent : fpTop(g, g3, quiet);
    c.urgent = urgent.length > 0;
    return c;
  }

  function fpTop(g, g3, quiet) {
    const N = g.N;
    if (!fpState || fpState.moves.length < N * N) {
      fpState = FeaturePol.createState(N, fpWeights.spec);
      fpScores = new Float64Array(N * N + 1);
    }
    FeaturePol.extractFeatures(g, fpState, fpWeights, g3);
    const n = FeaturePol.scoreAll(fpState, fpWeights, fpScores);
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => fpScores[b] - fpScores[a]);
    return order.slice(0, quiet).map(i => fpState.moves[i]);
  }

  // P(BLACK wins) of a leaf.
  function evaluate(g, g3) {
    if (g.gameOver) return g.estimateWinner() === BLACK ? 1 : 0;
    return VPat.evaluate(g, vpatModel, g3);
  }

  // Alpha-beta in P(BLACK wins); g3 is kept in lockstep with g (play before
  // recursing, undo after) for the ladder reads, featurepol and the evaluator.
  function ab(g, depth, alpha, beta, g3) {
    if (g.gameOver || depth === 0) return evaluate(g, g3);
    const cand = candidates(g, g3, QUIET_MOVES);
    if (cand.length === 0) return evaluate(g, g3);
    const maxing = g.current === BLACK;
    let best = maxing ? -Infinity : Infinity;
    for (const mv of cand) {
      const c = g.clone();
      if (!c.play(mv) || !g3.play(mv)) throw new Error(`ladder-pol: candidate ${mv} did not play`);
      const v = ab(c, depth - 1, alpha, beta, g3);
      g3.undo();
      if (maxing) { if (v > best) best = v; if (best > alpha) alpha = best; }
      else        { if (v < best) best = v; if (best < beta)  beta  = best; }
      if (beta <= alpha) break;
    }
    return best;
  }

  function getMove(game, _budgetMs, options = {}) {
    if (game.gameOver) return { move: PASS };
    if (game.consecutivePasses > 0 && game.calcWinner() === game.current) return { move: PASS, info: 'end the game; ahead' };
    const rng = options.rng || makeRng();
    const g3 = game3FromGame2(game);
    const cand = candidates(game, g3, QUIET_MOVES_ROOT);
    if (cand.length === 0) return { move: PASS };
    const mover = game.current;
    let best = cand[0], bestV = -Infinity;
    for (const mv of cand) {
      const c = game.clone();
      if (!c.play(mv) || !g3.play(mv)) throw new Error(`ladder-pol: candidate ${mv} did not play`);
      const v = ab(c, AB_DEPTH - 1, -Infinity, Infinity, g3);
      g3.undo();
      const mvV = (mover === BLACK ? v : 1 - v) + rng.random() * 1e-9;
      if (mvV > bestV) { bestV = mvV; best = mv; }
    }
    return { move: best, info: `${cand.urgent ? 'urgent' : 'quiet'} ${cand.length} v=${bestV.toFixed(3)}` };
  }

  return { getMove };
}

let _default = null;
function _def() { return _default || (_default = create(Util.makeCfg())); }

if (typeof module !== 'undefined') {
  module.exports = { create, getMove: (g, b, o) => _def().getMove(g, b, o) };
} else {
  window.create  = create;
  window.getMove = (g, b, o) => _def().getMove(g, b, o);
}

})();
