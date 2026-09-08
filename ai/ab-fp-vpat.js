'use strict';

// ab-fp-vpat: EXPERIMENTAL — shallow alpha-beta with featurepol move
// ordering/pruning at every node and a vpatterns pattern evaluator at the
// leaves.  Probes whether search depth can substitute for the evaluator's
// missing tactics at low time scales (the mc-ppat vote frontier's rival).
//
// Every node: featurepol ranks the legal moves, the top AB_TOP_K are
// searched best-first (ordering doubles as pruning — alpha-beta over a
// K-wide tree).  Leaves: the vpat evaluator's P(BLACK wins); terminals:
// estimateWinner.  Game2 clones per child — at K=3, depth 3 that is ~40
// clones per decision, negligible next to the fp extractions.
//
// Config (cfg reader, slot-aware):
//   AB_DEPTH    plies of lookahead (1 = score own moves' results) (default 2)
//   AB_TOP_K    fp candidates searched per node                   (default 3)
//   FPOL_DATA   featurepol weights          (default featurepol-cbk7wa32.js)
//   VPAT_DATA   leaf evaluator              (default out/vpat-pe-55dq1yxv-best.js)
//   DITHER      uniform noise on root values                    (default 0.001)
//   SOFTMAX_RATIO  fraction of MOVES taken from fp softmax sampling
//               (temperature 1) instead of the search — the corpus-
//               generation diversity knob (the epsilon/on-policy recipe;
//               fp-softmax is the incumbent diversity engine).  Set ~0.2
//               for corpus generation (default 0 = full strength)

const path = require('path');
const Util = require('../util.js');
const { PASS, BLACK } = require('../game2.js');
const { game3FromGame2 } = require('../game3.js');
const { makeRng } = require('../xorshift.js');
const FeaturePol = require('../featurepol-lib.js');
const VPat = require('../vpatterns.js');

function create(cfg) {
  cfg = cfg || Util.makeCfg();

  const AB_DEPTH = Math.max(1, cfg.int('AB_DEPTH', 2));
  const AB_TOP_K = Math.max(1, cfg.int('AB_TOP_K', 3));
  const DITHER   = cfg.float('DITHER', 0.001);
  const SOFTMAX_RATIO = cfg.float('SOFTMAX_RATIO', 0);

  const fpWeights = FeaturePol.loadModel({ name: 'ab-fp-vpat',
    path: cfg.str('FPOL_DATA', path.join(__dirname, '..', 'featurepol-cbk7wa32.js')) }).weights;
  const vpatModel = VPat.loadWeights(cfg.str('VPAT_DATA',
    path.join(__dirname, '..', 'out', 'vpat-pe-55dq1yxv-best.js')));

  console.log(`ab-fp-vpat[${cfg.slot != null ? cfg.slot : '-'}]: depth=${AB_DEPTH} top-K=${AB_TOP_K}` +
              (SOFTMAX_RATIO > 0 ? ` softmax-ratio=${SOFTMAX_RATIO}` : '') + `  ` +
              `fp=${fpWeights.map.size}w  vpat=${vpatModel.weights.size}w (${vpatModel.specs.map(s => s.size + ':' + (s.maxLibs || 'L')).join(',')})`);

  const rng = makeRng();
  let fpState = null, fpScores = null;

  // fp's top-K moves of g, best-first.  The shared fpState is overwritten on
  // every call, so the indices are copied out before any recursion.
  function fpTopK(g, out) {
    const N = g.N;
    if (!fpState || fpState.moves.length < N * N) {
      fpState  = FeaturePol.createState(N, fpWeights.spec);
      fpScores = new Float64Array(N * N + 1);
    }
    const game3 = fpWeights.spec.needsLadder ? game3FromGame2(g) : undefined;
    FeaturePol.extractFeatures(g, fpState, fpWeights, game3);
    const n = FeaturePol.scoreAll(fpState, fpWeights, fpScores);
    if (n === 0) return 0;
    const k = Math.min(AB_TOP_K, n);
    const order = new Array(n);
    for (let i = 0; i < n; i++) order[i] = i;
    order.sort((a, b) => fpScores[b] - fpScores[a]);
    for (let j = 0; j < k; j++) out[j] = fpState.moves[order[j]];
    return k;
  }

  function evaluate(g) {
    if (g.gameOver) return g.estimateWinner() === BLACK ? 1 : 0;
    return VPat.evaluate(g, vpatModel);
  }

  // Alpha-beta over fp's top-K, BLACK maximises.
  function ab(g, depth, alpha, beta) {
    if (g.gameOver || depth === 0) return evaluate(g);
    const cand = new Int32Array(AB_TOP_K);
    const k = fpTopK(g, cand);
    if (k === 0) return evaluate(g);
    const maxing = g.current === BLACK;
    let best = maxing ? -Infinity : Infinity;
    for (let j = 0; j < k; j++) {
      const c = g.clone();
      c.play(cand[j]);
      const v = ab(c, depth - 1, alpha, beta);
      if (maxing) { if (v > best) best = v; if (best > alpha) alpha = best; }
      else        { if (v < best) best = v; if (best < beta)  beta  = best; }
      if (beta <= alpha) break;
    }
    return best;
  }

  function getMove(game, _budgetMs, options = {}) {
    if (game.gameOver) return { move: PASS };
    const r = options.rng || rng;
    if (SOFTMAX_RATIO > 0 && r.random() < SOFTMAX_RATIO) {
      const N = game.N;
      if (!fpState || fpState.moves.length < N * N) {
        fpState  = FeaturePol.createState(N, fpWeights.spec);
        fpScores = new Float64Array(N * N + 1);
      }
      const game3 = fpWeights.spec.needsLadder ? game3FromGame2(game) : undefined;
      const m = FeaturePol.policyMove(game, fpState, fpWeights, r, game3, 1).move;
      return { move: m, info: 'fp-softmax (mix)' };
    }
    const cand = new Int32Array(AB_TOP_K);
    const k = fpTopK(game, cand);
    if (k === 0) return { move: PASS };
    const mover = game.current;
    let best = cand[0], bestV = -Infinity;
    for (let j = 0; j < k; j++) {
      const c = game.clone();
      c.play(cand[j]);
      const v = ab(c, AB_DEPTH - 1, -Infinity, Infinity);
      const mv = (mover === BLACK ? v : 1 - v) + (DITHER > 0 ? r.random() * DITHER : 0);
      if (mv > bestV) { bestV = mv; best = cand[j]; }
    }
    return { move: best, info: `ab=${bestV.toFixed(3)} d${AB_DEPTH}k${AB_TOP_K}` };
  }

  return { getMove };
}

let _default = null;
function _def() { return _default || (_default = create(Util.makeCfg())); }
module.exports = { create, getMove: (g, b, o) => _def().getMove(g, b, o) };
