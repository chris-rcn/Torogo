'use strict';

// Combines npat's policy value and vlibpat's depth-1 position value into ONE
// per-move score, then plays the argmax — no tree search, no hard candidate
// filtering (contrast ref-search-topk, which uses npat as a top-K filter and
// vlibpat as the sole judge).
//
//   score(m) = vm(m) + MERGE_C * ln P_npat(m)
//
// where vm(m) is vlibpat's P(win for the mover) of the position after m, and
// P_npat(m) is npat's softmax probability for m.  The ln-prior softly penalises
// npat-implausible moves (rather than hard-excluding them), and vlibpat refines
// the ranking among plausible ones.  MERGE_C trades the two off: MERGE_C=0 is pure
// vlibpat-depth-1 argmax over all moves; large MERGE_C approaches npat's argmax.
//
// Env (for tuning; defaults are the chosen config):
//   MERGE_C     blend coefficient (default 0.08)
//   MERGE_TOPK  cap candidates to npat's top-K by prior; 0 = all legal moves (default 0)

const path = require('path');
const { PASS, BLACK } = require('../game2.js');
const { game3FromGame2 } = require('../game3.js');
const { extractFeatures: vExtract, evaluateFeatures: vEval, loadWeights: vLoadWeights } = require('../vlibpat.js');
const NPat = require('../npat-lib.js');

// Findings from sweeping vs ref-search-topk (size 13, 4 rand moves, 1000-game
// confirms): NOTHING reliably beats it.  Best results reach PARITY (~0.50):
//   - value-merge: mode=logit (logit(vm)+MERGE_C*npatLogit) MERGE_C=0.08          → ~0.50
//   - mode=earlystop (vlibpat decider, npat-ordered, adaptive deepen)  → ~0.50
// Other value-merge formulas (add/mult/convex/pow) cap in the 0.40s.  The
// bottleneck is vlibpat's RANKING accuracy: re-selecting/combining the two static
// values can't exceed filter-then-judge; only real lookahead would.  Default is
// earlystop (ties at ref-search-topk-like speed, ~0.9ms/move).
const MERGE_C      = process.env.MERGE_C    !== undefined ? parseFloat(process.env.MERGE_C)     : 0.08;
const MERGE_TOPK   = process.env.MERGE_TOPK !== undefined ? parseInt(process.env.MERGE_TOPK, 10) : 0;   // 0 = all
const MERGE_MODE   = process.env.MERGE_MODE || 'earlystop';   // combine values: add | mult | logit | convex | pow | earlystop
const MERGE_EXP    = process.env.MERGE_EXP !== undefined ? parseFloat(process.env.MERGE_EXP) : 2;   // pow mode: vm + MERGE_C * p^MERGE_EXP

// mode=earlystop: vlibpat is the sole decider, candidates examined in npat order;
// stop once examined >= ES_KMIN and the next candidate's npat prob falls below a
// threshold that RISES with the best vlibpat value found so far (ES_TAU +
// ES_LAMBDA*vBest) — i.e. dig deeper only while the best move looks weak.  KMAX caps.
const ES_TAU    = process.env.ES_TAU    !== undefined ? parseFloat(process.env.ES_TAU)    : 0.02;
const ES_LAMBDA = process.env.ES_LAMBDA !== undefined ? parseFloat(process.env.ES_LAMBDA) : 0.05;
const ES_KMIN   = process.env.ES_KMIN   !== undefined ? parseInt(process.env.ES_KMIN, 10) : 6;
const ES_KMAX   = process.env.ES_KMAX   !== undefined ? parseInt(process.env.ES_KMAX, 10) : 20;
const dither = 0.001;
const LN_FLOOR = 1e-12;

const vpatPath = path.join(__dirname, '..', 'ref', 'vlibpat-4074.js');
const npatPath = path.join(__dirname, '..', 'npat-data.js');

const vModel = vLoadWeights(vpatPath);
const { weights: npatWeights } = NPat.loadModel({ name: 'npat-vlibpat-merge', path: npatPath });
const _stateByN = new Map();

console.error(`npat-vlibpat-merge: vlibpat=${vModel.weights.size}w npat=${npatWeights.size}w  mode=${MERGE_MODE}  MERGE_C=${MERGE_C}  topK=${MERGE_TOPK || 'all'}`);

function _vEvaluate(g3) {
  const f = vExtract(g3, vModel.preparedSpecs);
  vEval(f, vModel.weights);
  return f.val;   // P(BLACK wins)
}

function getMove(game) {
  if (game.gameOver) return { move: PASS };
  const N = game.N;
  let state = _stateByN.get(N);
  if (!state) { state = NPat.createState(N); _stateByN.set(N, state); }
  const game3 = game3FromGame2(game);
  NPat.policyMove(game, state, npatWeights, Math, game3);   // populates moves/logits/probs/count
  const n = state.count;
  if (n === 0) return { move: PASS };

  if (MERGE_MODE === 'earlystop') {
    // vlibpat is the decider; walk npat's ranking, stop adaptively.
    const ord = Array.from({ length: n }, (_, i) => i).sort((a, b) => state.probs[b] - state.probs[a]);
    const isBlack = game.current === BLACK;
    let vBest = -Infinity, bestMove = PASS, examined = 0;
    for (let j = 0; j < n; j++) {
      const i = ord[j];
      if (examined >= ES_KMIN && state.probs[i] < ES_TAU + ES_LAMBDA * vBest) break;
      game3.play(state.moves[i]);
      const val = _vEvaluate(game3);
      game3.undo();
      const vm = isBlack ? val : 1 - val;
      if (vm > vBest) { vBest = vm; bestMove = state.moves[i]; }
      if (++examined >= ES_KMAX) break;
    }
    return { move: bestMove };
  }

  // Candidate set: all legal moves, or npat's top-K by prior.
  let order = null, k = n;
  if (MERGE_TOPK > 0 && MERGE_TOPK < n) {
    order = Array.from({ length: n }, (_, i) => i).sort((a, b) => state.probs[b] - state.probs[a]);
    k = MERGE_TOPK;
  }

  const isBlack = game.current === BLACK;
  const probs = state.probs, logits = state.logits, moves = state.moves;

  // Gather per-candidate vlibpat (mover win prob) and npat (prob, logit).
  const cm = new Array(k), cvm = new Float64Array(k), cp = new Float64Array(k), clg = new Float64Array(k);
  for (let j = 0; j < k; j++) {
    const i = order ? order[j] : j;
    cm[j] = moves[i];
    game3.play(moves[i]);
    const val = _vEvaluate(game3);          // P(BLACK wins) after the move
    game3.undo();
    cvm[j] = isBlack ? val : 1 - val;       // mover's win probability
    cp[j]  = probs[i];
    clg[j] = logits[i];
  }

  // Combine the two values into one score per the chosen formula.
  const score = new Float64Array(k);
  if (MERGE_MODE === 'convex') {
    // Per-position min-max normalise both to [0,1]; MERGE_C is the npat weight in [0,1].
    let vMin = Infinity, vMax = -Infinity, pMin = Infinity, pMax = -Infinity;
    for (let j = 0; j < k; j++) { if (cvm[j] < vMin) vMin = cvm[j]; if (cvm[j] > vMax) vMax = cvm[j]; if (cp[j] < pMin) pMin = cp[j]; if (cp[j] > pMax) pMax = cp[j]; }
    const vR = (vMax - vMin) || 1, pR = (pMax - pMin) || 1;
    for (let j = 0; j < k; j++) score[j] = (1 - MERGE_C) * ((cvm[j] - vMin) / vR) + MERGE_C * ((cp[j] - pMin) / pR);
  } else {
    for (let j = 0; j < k; j++) {
      const vm = cvm[j];
      if (MERGE_MODE === 'mult')       score[j] = Math.log(vm > LN_FLOOR ? vm : LN_FLOOR) + MERGE_C * Math.log(cp[j] > LN_FLOOR ? cp[j] : LN_FLOOR);
      else if (MERGE_MODE === 'logit') { const vc = vm < 1e-6 ? 1e-6 : vm > 1 - 1e-6 ? 1 - 1e-6 : vm; score[j] = Math.log(vc / (1 - vc)) + MERGE_C * clg[j]; }
      else if (MERGE_MODE === 'pow')   score[j] = vm + MERGE_C * Math.pow(cp[j], MERGE_EXP);   // polynomial reward on raw npat prob
      else /* add */             score[j] = vm + MERGE_C * Math.log(cp[j] > LN_FLOOR ? cp[j] : LN_FLOOR);
    }
  }

  let bestMove = PASS, bestScore = -Infinity;
  for (let j = 0; j < k; j++) { const s = score[j] + Math.random() * dither; if (s > bestScore) { bestScore = s; bestMove = cm[j]; } }
  return { move: bestMove };
}

module.exports = { getMove };
