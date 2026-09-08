'use strict';

// Fixed-config reference agent: depth-2 alpha-beta over featurepol's top-5
// moves at every node, TD-trained ladder-aware vpatterns evaluator at the
// leaves, with the settled stochasticity profile — fp-softmax for the first
// 3 stones, then phase-scaled root value-softmax (temp 0.002) and dither
// (0.005), deterministic toward phase 1.  Self-contained copy of
// ai/ab-fp-vpat.js, frozen 2026-09-08 as a ladder rung — frozen names must
// never track live files.  All parameters are hardcoded; this script reads
// no environment variables.
//
// Provenance: 78.2%/500 vs ref-featurepol-softmax at 3.5ms/mv with exactly
// this profile; game diversity 94% unique prefixes at k=2, 100% by k=6,
// 0.00% duplicate sampled positions over phase [0,1] (200 bare games).
// Evaluator weights: ref/ref-ab-fp-vpat-data.js (frozen copy of the
// 2h1a4wlh TD snapshot) — IMMUTABLE.

const path = require('path');
const { PASS, BLACK } = require('../game2.js');
const { game3FromGame2 } = require('../game3.js');
const { makeRng } = require('../xorshift.js');
const FeaturePol = require('../featurepol-lib.js');
const VPat = require('../vpatterns.js');

// ── Hardcoded configuration ──────────────────────────────────────────────────

const AB_DEPTH = 2;
const AB_TOP_K = 5;
const DITHER   = 0.005;    // phase-scaled: effective dither = DITHER * (1 - phase)
const AB_TEMP  = 0.002;    // phase-scaled root value-softmax temperature
const FP_SOFTMAX_MOVES = 3;   // fp softmax while the board holds fewer stones

const FP_PATH   = path.join(__dirname, '..', 'featurepol-cbk7wa32.js');
const VPAT_PATH = path.join(__dirname, '..', 'ref', 'ref-ab-fp-vpat-data.js');

// ── Load weights ─────────────────────────────────────────────────────────────

const fpWeights = FeaturePol.loadModel({ name: 'ref-ab2-fp5-vpat', path: FP_PATH }).weights;
const vpatModel = VPat.loadWeights(VPAT_PATH);

console.error(`ref-ab2-fp5-vpat: depth=${AB_DEPTH} top-K=${AB_TOP_K} temp=${AB_TEMP} dither=${DITHER} ` +
              `fp-softmax<${FP_SOFTMAX_MOVES}st  fp=${fpWeights.map.size}w  vpats=${vpatModel.weights.size} ` +
              `(${vpatModel.specs.map(s => s.size + ':' + (s.maxLibs || 'L')).join(',')})`);

const rng = makeRng();
let fpState = null, fpScores = null;

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
    let sum = 0;
    const w = new Float64Array(k);
    for (let j = 0; j < k; j++) { w[j] = Math.exp((vals[j] - bestV) / temp); sum += w[j]; }
    let u = r.random() * sum;
    for (let j = 0; j < k; j++) { u -= w[j]; if (u <= 0) return { move: cand[j], info: `ab~=${vals[j].toFixed(3)}` }; }
  }
  return { move: best, info: `ab=${bestV.toFixed(3)}` };
}

module.exports = { getMove };
