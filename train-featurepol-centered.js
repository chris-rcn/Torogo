'use strict';

// train-featurepol-centered.js — exact CLOSED-FORM centered (within-position contrast) fit for
// SINGLE-TERM specs.  Minimises  Σ_pos Σ_i ((w[k_i]−w̄_pos) − (t_i−t̄_pos))²  — the within /
// fixed-effects least squares — so the per-pattern values fit the WITHIN-position contrasts
// (each position's mean removed), which is the ranking-relevant signal.  Deterministic, no
// epochs, no learning rate (the epoch sensitivity of centered SGD was an SGD artifact, not a
// property of the centered objective).
//
// Centering couples the keys that co-occur in a position, so this does NOT decouple into a
// per-key average.  We form the normal equations A w = b and solve directly:
//   A = diag(totalCount) − Σ_pos (1/n_pos) c_pos c_posᵀ        (c_pos[k] = # candidates in pos with key k)
//   b = totalTargetSum    − Σ_pos t̄_pos · c_pos
// A is singular (the centered loss is invariant to a global shift of all weights), so a small
// ridge is added to the diagonal before solving — the global shift is irrelevant to argmax.
//
// Single-term only (multi-term needs the joint solve across spaces too).

const fs   = require('fs');
const path = require('path');
const FeaturePol = require('./featurepol-lib.js');
const { Game2, parseMove } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['data', 'spec', 'save', 'size', 'ridge']);
if (opts.help || !opts.data || !opts.spec) {
  console.log("Usage: node train-featurepol-centered.js --data FILE --spec '<single-term spec>' [--ridge L] [--save PATH] [--size N]");
  process.exit(opts.help ? 0 : 1);
}
const DATA      = opts.data;
const SPEC      = opts.spec;
const RIDGE     = opts.ridge !== undefined ? parseFloat(opts.ridge) : 1e-2;   // diagonal regulariser (pins the global-shift null space)
const SAVE_PATH = opts.save || `out/featurepol-centered-${SPEC.replace(/[^a-zA-Z0-9]/g, '')}.js`;

let weights;
try { weights = FeaturePol.createWeights({ spec: SPEC }); }
catch (e) { console.error(`Error: ${e.message}`); process.exit(1); }
if (weights.nSpaces !== 1) {
  console.error(`Error: centered closed form supports SINGLE-term specs only (got ${weights.nSpaces} spaces).`);
  process.exit(1);
}

// ── Load point-diffs (same format as train-featurepol-points.js) ─────────────────

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

const state = FeaturePol.createState(SIZE, weights.spec);
const needLadder = weights.spec.needsLadder;

// Per-position helper: replay, extract, return matched (keyIdx, target) for each candidate.
function forEachMatched(s, fn) {
  const game = new Game2(SIZE);
  for (let i = 0; i < s.prefix.length; i++) game.play(s.prefix[i]);
  const game3 = needLadder ? game3FromGame2(game) : undefined;
  FeaturePol.extractFeatures(game, state, weights, game3);
  const n = state.count;
  let aligned = n === s.lm.length;
  if (aligned) for (let i = 0; i < n; i++) if (state.moves[i] !== s.lm[i]) { aligned = false; break; }
  let lookup = null;
  if (!aligned) { lookup = new Map(); for (let i = 0; i < s.lm.length; i++) lookup.set(s.lm[i], s.lv[i]); }
  for (let i = 0; i < n; i++) {
    const tgt = aligned ? s.lv[i] : lookup.get(state.moves[i]);
    if (tgt === undefined) continue;
    for (let k = state.keyOff[i], e = state.keyOff[i + 1]; k < e; k++) fn(state.keys[k], tgt);
  }
}

console.log(`spec: ${weights.spec.str}  (single term, centered closed form)  ridge: ${RIDGE}`);
console.log(`data: ${DATA}  size: ${SIZE}  samples: ${samples.length}`);
console.log(`save: ${SAVE_PATH}`);

// ── Pass 1: intern keys, accumulate totals (diagonal of A and first part of b) ────
const t0 = Date.now();
let cnt = [], tsum = [];   // grow dynamically (dense idx)
for (const s of samples) forEachMatched(s, (idx, tgt) => {
  cnt[idx] = (cnt[idx] || 0) + 1;
  tsum[idx] = (tsum[idx] || 0) + tgt;
});
const P = weights.size;
console.log(`pass 1: ${P} patterns interned  (${Util.fmtMs(Date.now() - t0)})`);

// ── Pass 2: accumulate the within-position outer products into A and the b correction ──
const A = new Float64Array(P * P), b = new Float64Array(P);
for (let k = 0; k < P; k++) { A[k * P + k] = cnt[k] || 0; b[k] = tsum[k] || 0; }

const locCnt = new Float64Array(P), touched = new Int32Array(SIZE * SIZE);
let locTsum = 0, locN = 0;
for (const s of samples) {
  let tc = 0; locTsum = 0; locN = 0;
  forEachMatched(s, (idx, tgt) => {
    if (locCnt[idx] === 0) touched[tc++] = idx;
    locCnt[idx] += 1; locTsum += tgt; locN += 1;
  });
  if (locN > 0) {
    const tbar = locTsum / locN, invN = 1 / locN;
    for (let a = 0; a < tc; a++) {
      const ka = touched[a], ca = locCnt[ka];
      b[ka] -= tbar * ca;
      const base = ka * P;
      for (let bb = 0; bb < tc; bb++) { const kb = touched[bb]; A[base + kb] -= invN * ca * locCnt[kb]; }
    }
  }
  for (let a = 0; a < tc; a++) locCnt[touched[a]] = 0;   // reset scratch
}
console.log(`pass 2: normal equations built  (${Util.fmtMs(Date.now() - t0)})`);

// ── Ridge + solve A w = b (Gaussian elimination, partial pivoting) ────────────────
for (let k = 0; k < P; k++) A[k * P + k] += RIDGE;
for (let col = 0; col < P; col++) {
  let piv = col, mx = Math.abs(A[col * P + col]);
  for (let r = col + 1; r < P; r++) { const v = Math.abs(A[r * P + col]); if (v > mx) { mx = v; piv = r; } }
  if (piv !== col) {
    for (let c = 0; c < P; c++) { const t = A[col * P + c]; A[col * P + c] = A[piv * P + c]; A[piv * P + c] = t; }
    const tb = b[col]; b[col] = b[piv]; b[piv] = tb;
  }
  const d = A[col * P + col];
  for (let r = col + 1; r < P; r++) {
    const f = A[r * P + col] / d;
    if (f === 0) continue;
    const br = r * P, bc = col * P;
    for (let c = col; c < P; c++) A[br + c] -= f * A[bc + c];
    b[r] -= f * b[col];
  }
}
const w = new Float64Array(P);
for (let r = P - 1; r >= 0; r--) {
  let sV = b[r];
  for (let c = r + 1; c < P; c++) sV -= A[r * P + c] * w[c];
  w[r] = sV / A[r * P + r];
}

let absSum = 0;
for (let k = 0; k < P; k++) { weights.vals[k] = w[k]; absSum += Math.abs(w[k]); }

fs.mkdirSync(path.dirname(SAVE_PATH), { recursive: true });
fs.writeFileSync(SAVE_PATH, FeaturePol.serialize(weights, { spec: weights.spec.str, ema: 0, totalUpdates: samples.length }));
console.log(`patterns: ${P}  avgW: ${(P ? absSum / P : 0).toFixed(4)}  solved in ${Util.fmtMs(Date.now() - t0)}`);
console.log(`done -> ${SAVE_PATH}`);
