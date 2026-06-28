'use strict';

// train-featurepol-mean.js — CLOSED-FORM trainer for SINGLE-TERM featurepol point models.
//
// For a single-space spec every candidate move has exactly one feature key, so the score
// (= that key's weight) and the squared-error fit DECOUPLE across keys: the least-squares
// optimum is simply, per pattern, the mean of the point differentials of every move that
// shows that pattern.  So we compute it directly in ONE pass — sum the targets per key,
// divide by the count — with no learning rate, no epochs, no SGD, and no sensitivity.  It is
// the exact optimum the point SGD trainer can only approach, and it serves as a fast, exact
// reference point.
//
// Multi-term specs do NOT decouple (a move's score sums several keys, so the keys are coupled
// through the moves they share) — there the per-key average is not the joint optimum, so this
// trainer rejects them; use train-featurepol-points.js for those.
//
// Usage:
//   node train-featurepol-mean.js --data FILE --spec '<single-term spec>' [--save PATH] [--size N]

const fs   = require('fs');
const path = require('path');
const FeaturePol = require('./featurepol-lib.js');
const { Game2, parseMove } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help', 'by-position'], ['data', 'spec', 'save', 'size', 'shrink']);
if (opts.help || !opts.data || !opts.spec) {
  console.log("Usage: node train-featurepol-mean.js --data FILE --spec '<single-term spec>' [--by-position] [--shrink L] [--save PATH] [--size N]");
  process.exit(opts.help ? 0 : 1);
}
// Shrinkage toward 0: weight[k] = sum[k] / (cnt[k] + L).  L acts as a pseudo-count of zero-
// valued evidence, so it pulls LOW-count (rare, noisily-estimated) patterns hardest toward
// neutral while barely touching well-evidenced common ones.  L=0 ⇒ exact mean.
const SHRINK = opts.shrink !== undefined ? parseFloat(opts.shrink) : 0;
const DATA        = opts.data;
const SPEC        = opts.spec;
// Weighting of a pattern's mean.  Default (per-occurrence): every candidate counts equally —
// the literal "sum the samples / count".  --by-position: average a pattern's targets WITHIN
// each position first, then average those per-position means equally, so a high-branching
// position can't over-count a shared pattern (matches the SGD per-key-mean fixed point).
const BY_POSITION = opts['by-position'] || false;
const SAVE_PATH   = opts.save || `out/featurepol-mean-${SPEC.replace(/[^a-zA-Z0-9]/g, '')}${BY_POSITION ? '-bypos' : ''}.js`;

let weights;
try { weights = FeaturePol.createWeights({ spec: SPEC }); }
catch (e) { console.error(`Error: ${e.message}`); process.exit(1); }
if (weights.nSpaces !== 1) {
  console.error(`Error: closed-form mean supports SINGLE-term specs only (got ${weights.nSpaces} spaces in '${weights.spec.str}').`);
  console.error('Multi-term specs couple their keys and need the joint solve — use train-featurepol-points.js.');
  process.exit(1);
}

// ── Load the point-diffs file (same format as train-featurepol-points.js) ────────

const rawLines = fs.readFileSync(DATA, 'utf8').split('\n');
let SIZE = parseInt(opts.size || '9', 10);
for (const l of rawLines) { const m = /^#.*\bsize=(\d+)/.exec(l); if (m) { SIZE = parseInt(m[1], 10); break; } }

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

const state    = FeaturePol.createState(SIZE, weights.spec);
const needLadder = weights.spec.needsLadder;

// Per-key accumulators (indexed by dense weight idx; grow as keys are interned).  The final
// weight is sum[k]/cnt[k] in both modes — only WHAT accumulates differs: per-occurrence adds
// each target with cnt++; per-position adds each position's within-position mean with cnt++.
let sum = new Float64Array(weights.vals.length), cnt = new Int32Array(weights.vals.length);
let locSum = new Float64Array(weights.vals.length), locCnt = new Int32Array(weights.vals.length);
const touched = new Int32Array(SIZE * SIZE);   // distinct keys seen in one position (≤ candidates)
function ensure(n) {
  if (n <= sum.length) return;
  let cap = sum.length; while (cap < n) cap *= 2;
  const ns = new Float64Array(cap); ns.set(sum); sum = ns;
  const nc = new Int32Array(cap); nc.set(cnt); cnt = nc;
  const ls = new Float64Array(cap); ls.set(locSum); locSum = ls;
  const lc = new Int32Array(cap); lc.set(locCnt); locCnt = lc;
}

console.log(`spec: ${weights.spec.str}  (single term)  weighting: ${BY_POSITION ? 'per-position' : 'per-occurrence'}  shrink: ${SHRINK}`);
console.log(`data: ${DATA}  size: ${SIZE}  samples: ${samples.length}`);
console.log(`save: ${SAVE_PATH}`);
console.log();

const t0 = Date.now();
let positions = 0, candidates = 0, misaligned = 0, nextReport = 1000;

for (const s of samples) {
  const game = new Game2(SIZE);
  for (let i = 0; i < s.prefix.length; i++) game.play(s.prefix[i]);
  const game3 = needLadder ? game3FromGame2(game) : undefined;
  FeaturePol.extractFeatures(game, state, weights, game3);
  const n = state.count;

  let aligned = n === s.lm.length;
  if (aligned) for (let i = 0; i < n; i++) if (state.moves[i] !== s.lm[i]) { aligned = false; break; }
  let lookup = null;
  if (!aligned) { misaligned++; lookup = new Map(); for (let i = 0; i < s.lm.length; i++) lookup.set(s.lm[i], s.lv[i]); }

  ensure(weights.size);
  if (!BY_POSITION) {
    for (let i = 0; i < n; i++) {                 // per-occurrence: every candidate votes
      const tgt = aligned ? s.lv[i] : lookup.get(state.moves[i]);
      if (tgt === undefined) continue;
      for (let k = state.keyOff[i], e = state.keyOff[i + 1]; k < e; k++) {
        const idx = state.keys[k];
        sum[idx] += tgt; cnt[idx]++; candidates++;
      }
    }
  } else {                                         // per-position: one vote per position
    let tc = 0;
    for (let i = 0; i < n; i++) {                 // accumulate within-position per-key sums
      const tgt = aligned ? s.lv[i] : lookup.get(state.moves[i]);
      if (tgt === undefined) continue;
      for (let k = state.keyOff[i], e = state.keyOff[i + 1]; k < e; k++) {
        const idx = state.keys[k];
        if (locCnt[idx] === 0) touched[tc++] = idx;
        locSum[idx] += tgt; locCnt[idx]++; candidates++;
      }
    }
    for (let t = 0; t < tc; t++) {                 // flush each key's within-position mean as one vote
      const idx = touched[t];
      sum[idx] += locSum[idx] / locCnt[idx]; cnt[idx]++;
      locSum[idx] = 0; locCnt[idx] = 0;
    }
  }
  positions++;
  if (positions >= nextReport) {
    console.log(`  ${Util.fmtMs(Date.now() - t0).padStart(7)}  positions: ${Util.fmt4i(positions)}  candidates: ${Util.fmt4i(candidates)}  patterns: ${Util.fmt4i(weights.size)}`);
    nextReport = Math.ceil(nextReport * 2);
  }
}

// Closed-form (optionally shrunk): weight[k] = sum[k] / (cnt[k] + SHRINK).
let absSum = 0, learned = 0;
for (let idx = 0; idx < weights.size; idx++) {
  if (cnt[idx] > 0) { weights.vals[idx] = sum[idx] / (cnt[idx] + SHRINK); absSum += Math.abs(weights.vals[idx]); learned++; }
}

fs.mkdirSync(path.dirname(SAVE_PATH), { recursive: true });
fs.writeFileSync(SAVE_PATH, FeaturePol.serialize(weights, { spec: weights.spec.str, ema: 0, totalUpdates: candidates }));

console.log();
console.log(`patterns: ${learned}  avgW: ${(learned ? absSum / learned : 0).toFixed(4)}  candidates: ${candidates}`);
if (misaligned) console.log(`note: ${misaligned} positions had candidate-order mismatch (used map lookup)`);
console.log(`done: ${positions} positions -> ${SAVE_PATH}`);
