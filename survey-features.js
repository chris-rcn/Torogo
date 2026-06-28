'use strict';

// survey-features.js — rank a featurepol model's spec spaces by how much each
// one actually influences move SELECTION, to surface the least useful term.
//
// Plays softmax self-play with the full loaded model.  At every position, for
// each spec space it computes that space's per-move partial logit, then ablates
// the space (subtracts its partial logit from every candidate) and measures how
// much the policy moves.  Periodically prints the spaces sorted by contribution
// (least useful first); a final table prints when the games are exhausted.
//
// The full move logit is the SUM of the per-space partial logits (each space
// owns a disjoint set of hash keys), so ablation = "what if this space's weights
// were all zero".  A space only changes the softmax through the SPREAD of its
// partial logit across candidates — a constant offset cancels — so a near-zero
// spread (and near-zero KL / flip rate) marks a term that pulls no weight.
//
// Metrics per space, averaged over positions (>1 candidate):
//   KL      mean KL(P_full || P_ablated)   — primary; policy shift from removal
//   flip%   % of positions where the GREEDY (argmax) move changes when removed
//   spread  mean std-dev of the space's partial logit across candidates
//   |dev|   mean abs deviation of that partial logit from its per-position mean
//   keys    distinct keys this space emitted over the run
//
// Usage:
//   node survey-features.js --data <featurepol.js> [--games N]   (no --games: run forever)
//
//   Reports fire on an EXPONENTIAL cadence: the threshold of scored positions
//   doubles after each report, plus one final table at the end.

const path = require('path');
const FP = require('./featurepol-lib.js');
const { Game2, PASS } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');
const { makeRng } = require('./xorshift.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['data', 'games']);
if (opts.help || !opts.data) {
  console.log('Usage: node survey-features.js --data <featurepol.js> [--games N]');
  process.exit(opts.help ? 0 : 1);
}
const N        = 13;
const MAXGAMES = opts.games !== undefined ? parseInt(opts.games, 10) : Infinity;   // default: run forever
const SEED     = 1;
const MTEMP    = 1;   // softmax temperature (must be > 0 for KL)

const { weights: model } = FP.loadModel({ name: 'survey-features', path: opts.data });
const rng        = makeRng(SEED);
const modelState = FP.createState(N, model.spec);
const needLadder = model.spec.needsLadder;

// One probe per spec space: a single-space spec whose keys we look up in the full
// model's weight table (model.map: hash -> dense idx, model.vals[idx] = weight).
const spaces = model.spec.spaces.map(sp => {
  const spec = FP.parseSpec(sp.str);
  return {
    str:        sp.str,
    spec,
    state:      FP.createState(N, spec),
    weights:    FP.createWeights({ spec }),
    needLadder: spec.needsLadder,
    subval:     [],   // local dense idx -> learned model weight (0 if unlearned)
    synced:     0,    // local weight-table size already mirrored into subval
    sumKL: 0, flips: 0, sumSpread: 0, sumAbs: 0,
  };
});

let positions = 0;   // scored positions (candidates > 1)
let games = 0;

// Mirror any newly-interned probe keys' learned weights from the full model.
function syncSubval(st) {
  if (st.weights.size === st.synced) return;
  for (const [hash, d] of st.weights.map) {
    if (d >= st.synced) {
      const mi = model.map.get(hash);
      st.subval[d] = mi !== undefined ? model.vals[mi] : 0;
    }
  }
  st.synced = st.weights.size;
}

// softmax of arr[0..n) at temperature MTEMP, written into out.
function softmaxInto(arr, n, out) {
  let m = -Infinity;
  for (let i = 0; i < n; i++) if (arr[i] > m) m = arr[i];
  let sum = 0;
  const invT = 1 / MTEMP;
  for (let i = 0; i < n; i++) { const e = Math.exp((arr[i] - m) * invT); out[i] = e; sum += e; }
  const inv = 1 / sum;
  for (let i = 0; i < n; i++) out[i] *= inv;
}

const partial = new Float64Array(N * N);   // per-move partial logit for one space
const ablLog  = new Float64Array(N * N);   // ablated logits
const ablP    = new Float64Array(N * N);   // ablated softmax

function scorePosition(game, game3) {
  // Full model: logits + P_full for this position.
  FP.extractFeatures(game, modelState, model, needLadder ? game3 : undefined);
  const n = modelState.count;
  if (n === 0) return -1;                      // no candidates -> end game
  if (n === 1) return modelState.moves[0];     // single move: play it, nothing to rank
  FP.computeSoftmax(modelState, model, MTEMP);
  const L = modelState.logits, Pfull = modelState.probs;
  let gFull = 0;
  for (let i = 1; i < n; i++) if (L[i] > L[gFull]) gFull = i;

  for (const st of spaces) {
    FP.extractFeatures(game, st.state, st.weights, st.needLadder ? game3 : undefined);
    syncSubval(st);
    const ss = st.state;
    // partial logit per move = sum of this space's key weights
    let mean = 0;
    for (let i = 0; i < n; i++) {
      let v = 0;
      for (let k = ss.keyOff[i]; k < ss.keyOff[i + 1]; k++) v += st.subval[ss.keys[k]];
      partial[i] = v; mean += v;
    }
    mean /= n;
    // spread (std) and mean abs deviation of the partial logit across candidates
    let var_ = 0, absdev = 0;
    for (let i = 0; i < n; i++) { const d = partial[i] - mean; var_ += d * d; absdev += d < 0 ? -d : d; }
    st.sumSpread += Math.sqrt(var_ / n);
    st.sumAbs    += absdev / n;
    // ablate this space: L' = L - partial, then KL(P_full || P_ablated)
    let gAbl = 0;
    for (let i = 0; i < n; i++) { ablLog[i] = L[i] - partial[i]; if (ablLog[i] > ablLog[gAbl]) gAbl = i; }
    softmaxInto(ablLog, n, ablP);
    let kl = 0;
    for (let i = 0; i < n; i++) {
      const p = Pfull[i];
      if (p > 0) kl += p * Math.log(p / (ablP[i] > 1e-300 ? ablP[i] : 1e-300));
    }
    st.sumKL += kl;
    if (gAbl !== gFull) st.flips++;
  }
  positions++;
  // sample the actual move from the full policy to advance the game
  let r = rng.random(), chosen = n - 1;
  for (let i = 0; i < n; i++) { r -= Pfull[i]; if (r <= 0) { chosen = i; break; } }
  return modelState.moves[chosen];
}

function report(tag) {
  const rows = spaces.map(st => ({
    str: st.str,
    KL: st.sumKL / positions,
    flip: 100 * st.flips / positions,
    spread: st.sumSpread / positions,
    absdev: st.sumAbs / positions,
    keys: st.weights.size,
  })).sort((a, b) => a.KL - b.KL);   // least useful first
  const w = Math.max(8, ...rows.map(r => r.str.length));
  console.log(`\n=== ${tag}: ${games} games, ${positions} scored positions (least useful first) ===`);
  console.log(`${'space'.padEnd(w)}  ${'KL'.padStart(10)}  ${'flip%'.padStart(7)}  ${'spread'.padStart(8)}  ${'|dev|'.padStart(8)}  ${'keys'.padStart(8)}`);
  for (const r of rows) {
    console.log(`${r.str.padEnd(w)}  ${r.KL.toFixed(6).padStart(10)}  ${r.flip.toFixed(2).padStart(7)}  ${r.spread.toFixed(4).padStart(8)}  ${r.absdev.toFixed(4).padStart(8)}  ${String(r.keys).padStart(8)}`);
  }
  // spec ordered by DECREASING importance (most useful term first), ready to paste as --spec
  console.log(`spec: ${rows.map(r => r.str).reverse().join(',')}`);
}

console.error(`survey-features: model ${model.size}w, ${spaces.length} spaces from ${path.basename(opts.data)}  size=${N} seed=${SEED}`);

let nextReport = 1000;
for (let g = 0; g < MAXGAMES; g++) {
  const game  = new Game2(N);
  const game3 = game3FromGame2(game);
  games = g + 1;
  let mv = 0;
  while (!game.gameOver && mv < N * N * 4) {
    const res = scorePosition(game, game3);
    if (res < 0) break;                        // no candidates -> end this game
    game.play(res);
    game3.play(res);
    mv++;
    if (positions >= nextReport) { report('progress'); nextReport *= 2; }
  }
}
report('final');
