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
//   VPAT_DATA   leaf evaluator              (default ref/ref-ab-fp-vpat-data.js)
//   DITHER      uniform noise on root values                    (default 0.005)
//               Both DITHER and AB_TEMP scale by (1 - phase): full strength
//               of the knob at an empty board, zero at phase 1 — diversity
//               lives in the opening, determinism in the endgame.
//   AB_TEMP     root value-softmax: sample the move among the K searched
//               candidates with probability ∝ exp(v/AB_TEMP) over their
//               minimax values (mover-relative, win-prob units).  Deviations
//               concentrate where the search itself calls the moves equal,
//               so strength cost stays tiny while openings branch.
//               (default 0.002; 0 = argmax).  Gentler than FP_SOFTMAX_MOVES, which
//               plays a weaker policy's move outright
//   FP_SOFTMAX_MOVES  while the board holds FEWER than this many stones,
//               play an fp softmax move (temperature 1) instead of
//               searching — opening diversity confined to the first
//               stones, full search strength after (default 3; 0 = off)

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
  const DITHER   = cfg.float('DITHER', 0.005);
  const AB_TEMP  = cfg.float('AB_TEMP', 0.002);
  const FP_SOFTMAX_MOVES = cfg.int('FP_SOFTMAX_MOVES', 3);

  const fpWeights = FeaturePol.loadModel({ name: 'ab-fp-vpat',
    path: cfg.str('FPOL_DATA', path.join(__dirname, '..', 'featurepol-cbk7wa32.js')) }).weights;
  const vpatModel = VPat.loadWeights(cfg.str('VPAT_DATA',
    path.join(__dirname, '..', 'ref', 'ref-ab-fp-vpat-data.js')));

  console.log(`ab-fp-vpat[${cfg.slot != null ? cfg.slot : '-'}]: depth=${AB_DEPTH} top-K=${AB_TOP_K}` +
              (AB_TEMP > 0 ? ` temp=${AB_TEMP}` : '') +
              (FP_SOFTMAX_MOVES > 0 ? ` fp-softmax<${FP_SOFTMAX_MOVES}st` : '') + `  ` +
              `fp=${fpWeights.map.size}w  vpats=${Util.fmt4i(vpatModel.weights.size).trim()} (${vpatModel.specs.map(s => s.size + ':' + (s.maxLibs || 'L')).join(',')})`);

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
    const phaseScale = 1 - game.phase();
    const dither = DITHER * phaseScale;
    const temp   = AB_TEMP * phaseScale;
    if (FP_SOFTMAX_MOVES > 0 && (game.N * game.N - game.emptyCount) < FP_SOFTMAX_MOVES) {
      const N = game.N;
      if (!fpState || fpState.moves.length < N * N) {
        fpState  = FeaturePol.createState(N, fpWeights.spec);
        fpScores = new Float64Array(N * N + 1);
      }
      const game3 = fpWeights.spec.needsLadder ? game3FromGame2(game) : undefined;
      const m = FeaturePol.policyMove(game, fpState, fpWeights, r, game3, 1).move;
      return { move: m, info: 'fp-softmax (opening)' };
    }
    const cand = new Int32Array(AB_TOP_K);
    const k = fpTopK(game, cand);
    if (k === 0) return { move: PASS };
    const mover = game.current;
    const vals = new Float64Array(k);
    let best = cand[0], bestV = -Infinity;
    for (let j = 0; j < k; j++) {
      const c = game.clone();
      c.play(cand[j]);
      const v = ab(c, AB_DEPTH - 1, -Infinity, Infinity);
      const mv = (mover === BLACK ? v : 1 - v) + (dither > 0 ? r.random() * dither : 0);
      vals[j] = mv;
      if (mv > bestV) { bestV = mv; best = cand[j]; }
    }
    if (temp > 0 && k > 1) {
      // Root value-softmax over the searched candidates' minimax values.
      let sum = 0;
      const w = new Float64Array(k);
      for (let j = 0; j < k; j++) { w[j] = Math.exp((vals[j] - bestV) / temp); sum += w[j]; }
      let u = r.random() * sum;
      for (let j = 0; j < k; j++) { u -= w[j]; if (u <= 0) return { move: cand[j], info: `ab~=${vals[j].toFixed(3)} d${AB_DEPTH}k${AB_TOP_K}t${AB_TEMP}` }; }
    }
    return { move: best, info: `ab=${bestV.toFixed(3)} d${AB_DEPTH}k${AB_TOP_K}` };
  }

  return { getMove };
}

let _default = null;
function _def() { return _default || (_default = create(Util.makeCfg())); }
module.exports = { create, getMove: (g, b, o) => _def().getMove(g, b, o) };
