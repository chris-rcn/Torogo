'use strict';

// ab-fp-vpat: shallow alpha-beta with featurepol move
// ordering/pruning at every node and a vpatterns pattern evaluator at the
// leaves.  Probes whether search depth can substitute for the evaluator's
// missing tactics at low time scales (the mc-ppat vote frontier's rival).
//
// Every node: featurepol ranks the legal moves, the top AB_WIDTH are
// searched best-first (ordering doubles as pruning — alpha-beta over a
// K-wide tree).  Leaves: the vpat evaluator's P(BLACK wins); terminals:
// estimateWinner.  Game2 clones per child — at K=3, depth 3 that is ~40
// clones per decision, negligible next to the fp extractions.
//
// Config (cfg reader, slot-aware):
//   AB_DEPTH    plies of lookahead (1 = score own moves' results) (default 2)
//   AB_WIDTH    fp candidates searched at the ROOT                (default 4)
//   AB_WIDTH_SHRINK  subtracted from the width per ply of depth  (default 0)
//               Root searches AB_WIDTH, the next ply AB_WIDTH - SHRINK, and so
//               on, floored at 1 — a width-1 ply extends the principal line
//               rather than branching.
//               Deep nodes are the many, and fp's ordering is most trustworthy
//               about its top two, so narrowing with depth buys plies cheaply.
//   FPOL_DATA   featurepol weights          (default featurepol-cbk7wa32.js)
//   FPOL_RANK_TOPN  for a model with a vpat<n> ranking term (fp-heavy), rank
//               over its top N candidates, as ref-fp-heavy does; 0 = off; no
//               effect on a model without one                       (default 14)
//   VPAT_DATA   leaf evaluator              (default ref/ref-ab-fp-vpat-data.js)
//   DITHER      uniform noise on root values                    (default 0.001)
//               Both DITHER and AB_TEMP scale by (1 - phase): full strength
//               of the knob at an empty board, zero at phase 1 — diversity
//               lives in the opening, determinism in the endgame.
//   AB_TEMP     root value-softmax: sample the move among the K searched
//               candidates with probability ∝ exp(v/AB_TEMP) over their
//               minimax values (mover-relative, win-prob units).  Deviations
//               concentrate where the search itself calls the moves equal,
//               so strength cost stays tiny while openings branch.
//               (default 0.003; 0 = argmax).  Gentler than FP_SOFTMAX_RATIO, which
//               plays a weaker policy's move outright
//   FP_SOFTMAX_RATIO  per-move probability of playing an fp softmax move
//               (temperature 1) instead of searching, scaled by (1 - phase):
//               FP_SOFTMAX_RATIO on an empty board, tapering to 0 at the
//               endgame (default 0.4; 0 = off)
//   ROOT_SYMMETRY  prune symmetry-equivalent root candidates (default 1; 0=off).
//               A symmetric root (common in the opening) has orbits of equal-
//               value moves, so searching one representative per orbit is exact
//               and concentrates the width on distinct moves.

const path = require('path');
const Util = require('../util.js');
const { PASS, BLACK } = require('../game2.js');
const { game3FromGame2 } = require('../game3.js');
const { makeRng } = require('../xorshift.js');
const FeaturePol = require('../featurepol-lib.js');
const VPat = require('../vpatterns.js');
const Symmetry = require('../symmetry.js');

function create(cfg) {
  cfg = cfg || Util.makeCfg();

  const AB_DEPTH = Math.max(1, cfg.int('AB_DEPTH', 2));
  const AB_WIDTH = Math.max(1, cfg.int('AB_WIDTH', 4));
  const AB_WIDTH_SHRINK = Math.max(0, cfg.int('AB_WIDTH_SHRINK', 0));
  // ply 0 is the root; a node reached with `depth` remaining sits at
  // AB_DEPTH - depth.  Floored at 1: below the root a width-1 node chooses
  // nothing, but it still extends the principal line one ply instead of
  // stopping, which beats the static eval it replaces.  Only the ROOT is
  // degenerate at width 1, and the root uses AB_WIDTH directly.
  function widthAt(ply) {
    const w = AB_WIDTH - ply * AB_WIDTH_SHRINK;
    return w < 1 ? 1 : w;
  }
  const DITHER   = cfg.float('DITHER', 0.001);
  const AB_TEMP  = cfg.float('AB_TEMP', 0.003);
  const FP_SOFTMAX_RATIO = cfg.float('FP_SOFTMAX_RATIO', 0.4);
  const ROOT_SYMMETRY = cfg.int('ROOT_SYMMETRY', 1) !== 0;

  const fpWeights = FeaturePol.loadModel({ name: 'ab-fp-vpat',
    path: cfg.str('FPOL_DATA', path.join(__dirname, '..', 'featurepol-cbk7wa32.js')) }).weights;
  if (fpWeights.spec.rankSpaces && fpWeights.spec.rankSpaces.length > 0) fpWeights.rankTopN = cfg.int('FPOL_RANK_TOPN', 14);
  const vpatModel = VPat.loadWeights(cfg.str('VPAT_DATA',
    path.join(__dirname, '..', 'ref', 'ref-ab-fp-vpat-data.js')));

  console.log(`ab-fp-vpat[${cfg.slot != null ? cfg.slot : '-'}]: depth=${AB_DEPTH} width=${AB_WIDTH}` +
              (AB_WIDTH_SHRINK > 0 ? ` shrink=${AB_WIDTH_SHRINK} (${[...Array(AB_DEPTH).keys()].map(widthAt).join('/')})` : '') +
              (AB_TEMP > 0 ? ` temp=${AB_TEMP}` : '') +
              (FP_SOFTMAX_RATIO > 0 ? ` fp-softmax-ratio=${FP_SOFTMAX_RATIO}` : '') +
              (ROOT_SYMMETRY ? '' : ` root-symmetry=off`) + `  ` +
              `fp=${fpWeights.map.size}w${fpWeights.rankTopN > 0 ? ` rank-topn=${fpWeights.rankTopN}` : ''}  vpats=${Util.fmt4i(vpatModel.weights.size).trim()} (${VPat.specString(vpatModel.specs)})`);

  const rng = makeRng();
  let fpState = null, fpScores = null;

  // fp's top-K moves of g, best-first.  The shared fpState is overwritten on
  // every call, so the indices are copied out before any recursion.
  function fpTopK(g, out, width, game3) {
    const N = g.N;
    if (!fpState || fpState.moves.length < N * N) {
      fpState  = FeaturePol.createState(N, fpWeights.spec);
      fpScores = new Float64Array(N * N + 1);
    }
    FeaturePol.extractFeatures(g, fpState, fpWeights, game3);
    const n = FeaturePol.scoreAll(fpState, fpWeights, fpScores);
    if (n === 0) return 0;
    const k = Math.min(width, n);
    const order = new Array(n);
    for (let i = 0; i < n; i++) order[i] = i;
    order.sort((a, b) => fpScores[b] - fpScores[a]);
    for (let j = 0; j < k; j++) out[j] = fpState.moves[order[j]];
    return k;
  }

  function evaluate(g, game3) {
    if (g.gameOver) return g.estimateWinner() === BLACK ? 1 : 0;
    return VPat.evaluate(g, vpatModel, game3);
  }

  // Alpha-beta over fp's top-K, BLACK maximises.  `game3` is maintained in
  // lockstep with `g` (play before recursing, undo after) so featurepol
  // extraction and the leaf vpat eval reuse one Game3 instead of rebuilding per
  // node.  Behaviour-identical to rebuilding.
  function ab(g, depth, alpha, beta, game3) {
    if (g.gameOver || depth === 0) return evaluate(g, game3);
    const cand = new Int32Array(AB_WIDTH);
    const k = fpTopK(g, cand, widthAt(AB_DEPTH - depth), game3);
    if (k === 0) return evaluate(g, game3);
    const maxing = g.current === BLACK;
    let best = maxing ? -Infinity : Infinity;
    for (let j = 0; j < k; j++) {
      const c = g.clone();
      c.play(cand[j]);
      if (game3) game3.play(cand[j]);
      const v = ab(c, depth - 1, alpha, beta, game3);
      if (game3) game3.undo();
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
    // fp softmax move (temperature 1) instead of searching, with probability
    // FP_SOFTMAX_RATIO tapered by (1 - phase) so it's likeliest on an empty board
    // and reaches zero at the endgame.  The draw is short-circuited when the knob
    // is off, so existing configs consume no extra rng.
    if (FP_SOFTMAX_RATIO > 0 && r.random() < FP_SOFTMAX_RATIO * phaseScale) {
      const N = game.N;
      if (!fpState || fpState.moves.length < N * N) {
        fpState  = FeaturePol.createState(N, fpWeights.spec);
        fpScores = new Float64Array(N * N + 1);
      }
      const game3 = fpWeights.spec.needsLadder ? game3FromGame2(game) : undefined;
      const m = FeaturePol.policyMove(game, fpState, fpWeights, r, game3, 1).move;
      return { move: m, info: 'fp-softmax (ratio)' };
    }
    const cand = new Int32Array(AB_WIDTH);
    // One Game3 for the whole search, advanced in lockstep with the recursion
    // and reused for featurepol extraction + the leaf vpat eval instead of a
    // rebuild per node.  Reused-vs-rebuilt gives identical features (speed only).
    const needG3 = fpWeights.spec.needsLadder || VPat.needsGame3(vpatModel.preparedSpecs);
    const g3 = needG3 ? game3FromGame2(game) : undefined;
    let k = fpTopK(game, cand, AB_WIDTH, g3);
    if (k === 0) return { move: PASS };
    // Root symmetry pruning: a symmetric root has orbits of equal-value moves,
    // so search one representative per orbit and skip the duplicates.
    let roots = cand.subarray(0, k);
    if (ROOT_SYMMETRY) {
      const sym = Symmetry.of(game);
      if (sym.hasSymmetry()) { roots = sym.distinctMoves(Array.from(roots)); k = roots.length; }
    }
    const mover = game.current;
    const vals = new Float64Array(k);
    let best = roots[0], bestV = -Infinity;
    for (let j = 0; j < k; j++) {
      const c = game.clone();
      c.play(roots[j]);
      if (g3) g3.play(roots[j]);
      const v = ab(c, AB_DEPTH - 1, -Infinity, Infinity, g3);
      if (g3) g3.undo();
      const mv = (mover === BLACK ? v : 1 - v) + (dither > 0 ? r.random() * dither : 0);
      vals[j] = mv;
      if (mv > bestV) { bestV = mv; best = roots[j]; }
    }
    if (temp > 0 && k > 1) {
      // Root value-softmax over the searched candidates' minimax values.
      let sum = 0;
      const w = new Float64Array(k);
      for (let j = 0; j < k; j++) { w[j] = Math.exp((vals[j] - bestV) / temp); sum += w[j]; }
      let u = r.random() * sum;
      for (let j = 0; j < k; j++) { u -= w[j]; if (u <= 0) return { move: roots[j], info: `ab~=${vals[j].toFixed(3)} d${AB_DEPTH}k${AB_WIDTH}t${AB_TEMP}` }; }
    }
    return { move: best, info: `ab=${bestV.toFixed(3)} d${AB_DEPTH}k${AB_WIDTH}` };
  }

  return { getMove };
}

let _default = null;
function _def() { return _default || (_default = create(Util.makeCfg())); }
module.exports = { create, getMove: (g, b, o) => _def().getMove(g, b, o) };
