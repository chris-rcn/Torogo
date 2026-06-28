'use strict';

// train-featurepol-points.js — train a featurepol model to PREDICT per-move point
// differentials (territory gained over passing), from offline samples produced by
// gen-point-diffs.js.  Pure regression: for each candidate move the model's raw score
// (FeaturePol.scoreAll) is fit to the playout-estimated diff by squared error, using
// the generic FeaturePol.applyScoreGradient step (grad_i = target_i − score_i).  No
// softmax, no self-play — the labels are precomputed.
//
// A trained model is an ordinary featurepol checkpoint; an agent plays it greedily
// (argmax predicted point gain = territory-greedy policy).
//
// Usage:
//   node train-featurepol-points.js --data FILE --spec '<spec>' [options]
//   --data FILE        point-diffs file from gen-point-diffs.js  (REQUIRED)
//   --spec S           feature spec  (required unless --load supplies it)
//   --lr F             learning rate; with --epochs it runs in two phases — constant for the
//                      first half of epochs (build-up), then a linear ramp-down to ~0 over the
//                      second half (quiescence/denoise)
//   --weight-decay F   decoupled L2 shrink per update (default 0; for nested/overlapping specs
//                      it acts as a coarse-to-fine backoff, shrinking under-evidenced fine keys)
//   --val F            fraction of samples held out for validation RMS (default 0.05)
//   --epochs N         stop after N passes over the training set (default: run until killed)
//   --load PATH        resume from saved weights
//   --save PATH        where to save (default out/featurepol-points-<rand>.js)
//   --seed N           shuffle/split seed (default 1)
//
// Columns: elapsed, smpl (samples seen), epoch, lr, nWts, avgW, tRMS (train residual RMS),
//          vRMS (held-out residual RMS).

const fs   = require('fs');
const path = require('path');
const FeaturePol = require('./featurepol-lib.js');
const { Game2, parseMove } = require('./game2.js');
const { makeRng } = require('./xorshift.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help', 'center'],
  ['data', 'spec', 'lr', 'weight-decay', 'val', 'epochs', 'load', 'save', 'seed', 'size']);
if (opts.help || !opts.data || (!opts.spec && !opts.load)) {
  console.log("Usage: node train-featurepol-points.js --data FILE --spec '<spec>' [--lr 0.001] [--weight-decay 0] [--val 0.05] [--epochs N] [--load PATH] [--save PATH] [--seed N]");
  process.exit(opts.help ? 0 : 1);
}
const DATA         = opts.data;
const LR0          = parseFloat(opts.lr || '0.1');           // initial learning rate
let   LR           = LR0;                                       // current; linearly decays to ~0 across --epochs (if set)
// Default 0: a single non-redundant spec needs none (regression to fixed targets settles on
// its own).  But for NESTED/overlapping specs (e.g. stones8,stones12) a small decay turns the
// additive sum into a coarse-to-fine backoff: it shrinks under-evidenced fine-feature keys
// toward 0, so the model falls back to the coarse term where the fine one lacks data.
const WEIGHT_DECAY = opts['weight-decay'] !== undefined ? parseFloat(opts['weight-decay']) : 0;
const VAL_FRAC     = opts.val !== undefined ? parseFloat(opts.val) : 0.1;
const EPOCHS       = opts.epochs !== undefined ? parseInt(opts.epochs, 10) : Infinity;   // default: until killed
const SEED         = parseInt(opts.seed || '1', 10);
const CENTER       = opts.center || false;                     // fit within-position contrasts (subtract per-position mean residual)
const LOAD_PATH    = opts.load || null;
const SAVE_PATH    = opts.save || `out/featurepol-points-${Math.random().toString(36).slice(2, 10)}.js`;

// ── Load the point-diffs file ───────────────────────────────────────────────────

const rawLines = fs.readFileSync(DATA, 'utf8').split('\n');
let SIZE = parseInt(opts.size || '9', 10);
for (const l of rawLines) {                                   // header: "# ... size=N ..."
  const m = /^#.*\bsize=(\d+)/.exec(l);
  if (m) { SIZE = parseInt(m[1], 10); break; }
}

// Parse "<coord,...> ; <coord>=<diff> ..." → { prefix, lm (label moves), lv (label diffs) }.
const samples = [];
for (const line of rawLines) {
  if (!line || line[0] === '#') continue;
  const semi = line.indexOf(';');
  if (semi < 0) continue;
  const prefix = line.slice(0, semi).split(',').filter(Boolean).map(c => parseMove(c, SIZE));
  const toks = line.slice(semi + 1).trim().split(/\s+/).filter(Boolean);
  const lm = new Int32Array(toks.length), lv = new Float32Array(toks.length);
  for (let i = 0; i < toks.length; i++) {
    const eq = toks[i].lastIndexOf('=');
    lm[i] = parseMove(toks[i].slice(0, eq), SIZE);
    lv[i] = parseFloat(toks[i].slice(eq + 1));
  }
  if (lm.length) samples.push({ prefix: Int32Array.from(prefix), lm, lv });
}
if (samples.length === 0) { console.error(`Error: no samples parsed from ${DATA}`); process.exit(1); }

const rng = makeRng(SEED);
function shuffle(arr) { for (let i = arr.length - 1; i > 0; i--) { const j = (rng.random() * (i + 1)) | 0; const t = arr[i]; arr[i] = arr[j]; arr[j] = t; } }

shuffle(samples);
const nVal = Math.min(samples.length - 1, Math.floor(samples.length * VAL_FRAC));
const valSet = samples.slice(0, nVal);
const trainSet = samples.slice(nVal);

// ── Model (fresh or resumed) ────────────────────────────────────────────────────

const loaded = (LOAD_PATH && fs.existsSync(LOAD_PATH))
  ? FeaturePol.loadModel({ name: 'featurepol-points', path: LOAD_PATH })
  : null;
if (LOAD_PATH && !loaded) console.warn(`Warning: --load file not found: ${LOAD_PATH}`);

const SPEC = opts.spec || (loaded && loaded.spec.str);
if (!SPEC) { console.error('Error: --spec is required unless --load points to an existing model.'); process.exit(1); }

let weights;
try { weights = FeaturePol.createWeights({ spec: SPEC }); }
catch (e) { console.error(`Error: ${e.message}`); process.exit(1); }
let totalUpdates = 0;
const wStats = { absSum: 0, count: 0 };

if (loaded) {
  totalUpdates = loaded.totalUpdates;
  const cliSpaces   = new Set(weights.spec.spaces.map(s => s.str));
  const savedSpaces = new Set(loaded.weights.spec.spaces.map(s => s.str));
  const kept    = [...cliSpaces].filter(s => savedSpaces.has(s));
  const added   = [...cliSpaces].filter(s => !savedSpaces.has(s));
  const removed = [...savedSpaces].filter(s => !cliSpaces.has(s));
  for (const [hash, srcIdx] of loaded.weights.map) weights.vals[FeaturePol.internKey(weights, hash)] = loaded.weights.vals[srcIdx];
  console.log(`Resumed from ${LOAD_PATH}: spaces kept ${kept.length}, added ${added.length}${added.length ? ` (${added.join(', ')})` : ''}, removed ${removed.length}${removed.length ? ` (${removed.join(', ')})` : ''}`);
}

const state = FeaturePol.createState(SIZE, weights.spec);
const cap = SIZE * SIZE;
const preds = new Float64Array(cap), grad = new Float64Array(cap);
let misaligned = 0;

function saveWeights() {
  fs.mkdirSync(path.dirname(SAVE_PATH), { recursive: true });
  fs.writeFileSync(SAVE_PATH, FeaturePol.serialize(weights, { spec: weights.spec.str, ema: 0, totalUpdates }));
}

// Replay a sample's position, predict per-move scores, accumulate residual stats into
// `acc`, and (if doUpdate) take the regression step.  Targets are matched to the model's
// candidate moves by order (same replay → same _emptyCells order); if that ever diverges
// it falls back to a per-sample map and counts it.
function runSample(s, acc, doUpdate) {
  const game = new Game2(SIZE);
  for (let i = 0; i < s.prefix.length; i++) game.play(s.prefix[i]);
  FeaturePol.extractFeatures(game, state, weights);
  const n = state.count;
  FeaturePol.scoreAll(state, weights, preds);

  let aligned = n === s.lm.length;
  if (aligned) for (let i = 0; i < n; i++) if (state.moves[i] !== s.lm[i]) { aligned = false; break; }
  let lookup = null;
  if (!aligned) { misaligned++; lookup = new Map(); for (let i = 0; i < s.lm.length; i++) lookup.set(s.lm[i], s.lv[i]); }

  // --center: subtract the mean residual over this position's matched candidates, so we fit
  // only the WITHIN-position contrasts (the part argmax actually uses) and not the large
  // common tempo baseline.  Gradient of ½Σ((pᵢ−p̄)−(tᵢ−t̄))² wrt pᵢ is exactly (residᵢ − resid̄),
  // so centering the raw residual by its mean is all it takes.  rbar = 0 ⇒ plain regression.
  let rbar = 0;
  if (CENTER) {
    let rsum = 0, rcnt = 0;
    for (let i = 0; i < n; i++) {
      const tgt = aligned ? s.lv[i] : lookup.get(state.moves[i]);
      if (tgt === undefined) continue;
      rsum += preds[i] - tgt; rcnt++;
    }
    rbar = rcnt ? rsum / rcnt : 0;
  }
  for (let i = 0; i < n; i++) {
    const tgt = aligned ? s.lv[i] : lookup.get(state.moves[i]);
    if (tgt === undefined) { grad[i] = 0; continue; }
    const resid = (preds[i] - tgt) - rbar;     // centered residual (rbar=0 ⇒ plain squared error)
    grad[i] = -resid;                          // ascent on −½·resid² → minimise (centered) squared error
    acc.sumSq += resid * resid; acc.count++;
  }
  if (doUpdate) { FeaturePol.applyScoreGradient(state, grad, weights, LR, WEIGHT_DECAY, wStats); totalUpdates++; }
}

function valRMS() {
  if (valSet.length === 0) return 0;
  const acc = { sumSq: 0, count: 0 };
  for (const s of valSet) runSample(s, acc, false);
  return acc.count ? Math.sqrt(acc.sumSq / acc.count) : 0;
}

// ── Train ───────────────────────────────────────────────────────────────────────

console.log(`spec: ${weights.spec.str}  spaces: ${weights.nSpaces}  needsLadder: ${weights.spec.needsLadder}`);
console.log(`data: ${DATA}  size: ${SIZE}  samples: ${samples.length} (train ${trainSet.length}, val ${valSet.length})`);
console.log(`lr: ${LR0}  weight-decay: ${WEIGHT_DECAY}  epochs: ${EPOCHS === Infinity ? 'unlimited' : EPOCHS}${EPOCHS === Infinity ? '' : '  (lr: constant first half, then quiesce)'}  objective: ${CENTER ? 'within-position contrast (centered)' : 'absolute regression'}`);
console.log(`save: ${SAVE_PATH}${loaded ? `  (resumed from ${LOAD_PATH})` : ''}`);
console.log();
const COLS = ['elapsed', 'smpl', 'epoch', 'lr', 'nWts', 'avgW', 'tRMS', 'vRMS'];
const COLW = [7, 6, 5, 6, 6, 7, 7, 7];
const printRow = cells => console.log(cells.map((c, i) => String(c).padStart(COLW[i])).join('  '));
printRow(COLS);

const t0 = Date.now();
let nextPrintAt = t0 + 1000;
let smpl = 0, epoch = 0;
const trainAcc = { sumSq: 0, count: 0 };

function report() {
  let wNz = 0; for (let i = 0; i < weights.size; i++) if (weights.vals[i] !== 0) wNz++;
  const avgW = wStats.count > 0 ? wStats.absSum / wStats.count : 0;
  printRow([
    Util.fmtMs(Date.now() - t0), Util.fmt4i(smpl), epoch, Util.fmtRatio4(LR), Util.fmt4i(wNz),
    avgW.toFixed(4),
    (trainAcc.count ? Math.sqrt(trainAcc.sumSq / trainAcc.count) : 0).toFixed(3),
    valRMS().toFixed(3),
  ]);
  saveWeights();
  trainAcc.sumSq = 0; trainAcc.count = 0;
  if (misaligned) { console.warn(`  note: ${misaligned} samples had candidate order mismatch (fell back to map lookup)`); misaligned = 0; }
}

while (epoch < EPOCHS) {
  shuffle(trainSet);
  epoch++;
  // Two-phase schedule (when --epochs is set): hold LR constant for the first half of
  // epochs (build-up — lets avgW reach its operating value), then linearly ramp it down
  // toward ~0 over the second half (quiescence — settles the weights / reduces noise).
  if (EPOCHS !== Infinity) {
    const half = Math.floor(EPOCHS / 2);
    LR = epoch <= half ? LR0 : LR0 * (EPOCHS - epoch + 1) / (EPOCHS - half);
  }
  for (const s of trainSet) {
    runSample(s, trainAcc, true);
    smpl++;
    if (Date.now() >= nextPrintAt) { report(); nextPrintAt = t0 + Math.round((Date.now() - t0) * 1.4); }
  }
}
if (trainAcc.count) report();                 // final row + save for the last (partial) interval
console.log(`done: ${epoch} epochs, ${smpl} samples -> ${SAVE_PATH}`);
