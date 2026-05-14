'use strict';

// Hybrid reference agent: npat narrows the candidate set to its top-K moves,
// vlibpat picks the best of those by depth-1 value.
//
// Weights are the canonical references shared with ai/vlibpat-ref-3x3.js and
// ai/npat.js / ai/ref-npat-softmax.js (loaded eagerly at module init).
//
// Env config:
//   NPAT_FILTER_TOPK — number of top-npat candidates to evaluate (default 2)

const path = require('path');
const { PASS } = require('../game2.js');
const { game3FromGame2 } = require('../game3.js');
const { extractFeatures: vExtract, evaluateFeatures: vEval, loadWeights: vLoadWeights } = require('../vlibpat.js');
const { search: abSearch } = require('../ab-search3.js');
const NPat = require('../npat-lib.js');

// ── Configuration ────────────────────────────────────────────────────────────

const TOP_K  = Math.max(1, parseInt(process.env.NPAT_FILTER_TOPK || '2', 10));
const DITHER = 0.002;

const VPAT_PATH = path.join(__dirname, '..', 'ref', 'vlibpat-9-2L3L3NL-onpol70-9.js');
const NPAT_PATH = path.join(__dirname, '..', 'npat-data.js');

// ── Load weights ─────────────────────────────────────────────────────────────

const vModel = vLoadWeights(VPAT_PATH);

const npatRaw = require(path.resolve(NPAT_PATH));
let has33c = false, hasP12 = false;
for (const [k] of npatRaw.weights) {
  if (typeof k === 'string') continue;
  if      (k >= NPat.SHAPE33C_RAW_BASE && k < NPat.P12_RAW_BASE) has33c = true;
  else if (k >= NPat.P12_RAW_BASE)                                hasP12 = true;
}
const npatWeights = NPat.createWeights({
  initialCapacity: Math.max(1024, npatRaw.weights.size | 0),
  use33c: has33c, useP12: hasP12,
});
for (const [k, v] of npatRaw.weights) {
  const idx = NPat.internWeight(npatWeights, k);
  npatWeights.vals[idx] = v;
}

console.error(`vlibpat-npat-topk: vlibpat=${vModel.weights.size}w npat=${npatWeights.size}w  top-K=${TOP_K}  (3x3c=${has33c} p12=${hasP12})`);

// ── Move selection ───────────────────────────────────────────────────────────
//
// One npat state per board size; reused across calls (extractFeatures
// repopulates moves/patIds on every invocation).

const _stateByN = new Map();

// vlibpat depth-1 leaf evaluator.  extractFeatures detects the Game3 and uses
// it directly — no per-candidate rebuild.
function _vEvaluate(g3) {
  const f = vExtract(g3, vModel.preparedSpecs);
  vEval(f, vModel.weights);
  return f.val;
}

// Pre-compute the top-K npat candidate list once at the root.  npat-lib's
// extractFeatures needs a Game2 (uses Game2-only fields like _emptyCells), so
// we pass the Game2 here and let the lockstep Game3 carry the position into
// the αβ candidate scan.
function _topKCandidates(game, game3) {
  const N = game.N;
  let state = _stateByN.get(N);
  if (!state) { state = NPat.createState(N); _stateByN.set(N, state); }
  NPat.policyMove(game, state, npatWeights, Math, game3);
  const n = state.count;
  if (n === 0) return [];
  const k = Math.min(TOP_K, n);
  const probs = state.probs;
  const order = new Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  if (k < n) order.sort((a, b) => probs[b] - probs[a]);
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
