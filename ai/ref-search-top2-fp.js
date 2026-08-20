'use strict';

// Fixed-config reference agent: featurepol narrows the candidate set to its
// top-2 moves, vlibpat picks the best of those by depth-1 value.  The
// featurepol-proposal sibling of ai/ref-search-topk.js (which proposes with
// npat).
//
// All parameters are hardcoded.  This script reads no environment variables —
// use it when a stable, reproducible reference policy is needed.

const path = require('path');
const { PASS } = require('../game2.js');
const { game3FromGame2 } = require('../game3.js');
const { extractFeatures: vExtract, evaluateFeatures: vEval, loadWeights: vLoadWeights } = require('../vlibpat.js');
const { search: abSearch } = require('../ab-search3.js');
const FeaturePol = require('../featurepol-lib.js');

// ── Hardcoded configuration ──────────────────────────────────────────────────

const TOP_K  = 2;
const DITHER = 0.001;

const VPAT_PATH = path.join(__dirname, '..', 'ref', 'vlibpat-4074.js');
const FP_PATH   = path.join(__dirname, '..', 'ref', 'featurepol-6082.js');

// ── Load weights ─────────────────────────────────────────────────────────────

const vModel = vLoadWeights(VPAT_PATH);

const { weights: fpWeights, modelName } = FeaturePol.loadModel({ name: 'ref-search-top2-fp', path: FP_PATH });

console.error(`ref-search-top2-fp: vlibpat=${vModel.weights.size}w featurepol=${fpWeights.size}w (${modelName})  top-K=${TOP_K}`);

// ── Move selection ───────────────────────────────────────────────────────────
//
// One featurepol state (+ score buffer) per board size; reused across calls
// (extractFeatures repopulates moves/features on every invocation).

const _stateByN = new Map();

function _vEvaluate(g3) {
  const f = vExtract(g3, vModel.preparedSpecs);
  vEval(f, vModel.weights);
  return f.val;
}

function _topKCandidates(game, game3) {
  const N = game.N;
  let st = _stateByN.get(N);
  if (!st) {
    st = { state: FeaturePol.createState(N, fpWeights.spec), scores: new Float64Array(N * N + 1) };
    _stateByN.set(N, st);
  }
  const { state, scores } = st;
  FeaturePol.extractFeatures(game, state, fpWeights, game3);
  const n = FeaturePol.scoreAll(state, fpWeights, scores);
  if (n === 0) return [];
  const k = Math.min(TOP_K, n);
  const order = new Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  if (k < n) order.sort((a, b) => scores[b] - scores[a]);
  const candidates = new Array(k);
  for (let j = 0; j < k; j++) candidates[j] = state.moves[order[j]];
  return candidates;
}

function getMove(game) {
  if (game.gameOver) return { move: PASS };
  const game3 = game3FromGame2(game);
  const candidates = _topKCandidates(game, game3);
  if (candidates.length === 0) return { move: PASS };
  const move = abSearch(game3, 1, _vEvaluate, DITHER, {
    getCandidates: () => candidates,
  });
  return { move };
}

module.exports = { getMove };
