'use strict';

// train-featurepol-lsq.js — exact CLOSED-FORM ordinary least squares for ANY featurepol spec,
// including MULTI-TERM.  A move's score is the SUM of its keys (one per space), so for more than
// one space the keys COUPLE and the per-pattern mean is no longer the optimum — we form the full
// normal equations and solve:
//   A = XᵀX   (A[k,l] = # candidates whose key-set contains both k and l)
//   b = Xᵀy   (b[k]   = Σ targets over candidates containing k)
//   solve A w = b
// For a single term A is diagonal and this reduces to the per-occurrence mean.  For multi-term
// each space is a complete partition (every candidate has exactly one key per space), so the
// "+1 on all of space S's keys" indicators are collinear across spaces — A is singular along the
// inter-space shift (which leaves every score, and every argmax, unchanged).  A ridge on the
// diagonal pins it.  Deterministic, no epochs, no learning rate.

const fs   = require('fs');
const path = require('path');
const FeaturePol = require('./featurepol-lib.js');
const { Game2, parseMove } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['data', 'spec', 'save', 'size', 'ridge']);
if (opts.help || !opts.data || !opts.spec) {
  console.log("Usage: node train-featurepol-lsq.js --data FILE --spec '<spec>' [--ridge L] [--save PATH] [--size N]");
  process.exit(opts.help ? 0 : 1);
}
const DATA      = opts.data;
const SPEC      = opts.spec;
const RIDGE     = opts.ridge !== undefined ? parseFloat(opts.ridge) : 1e-2;
const SAVE_PATH = opts.save || `out/featurepol-lsq-${SPEC.replace(/[^a-zA-Z0-9]/g, '')}.js`;

let weights;
try { weights = FeaturePol.createWeights({ spec: SPEC }); }
catch (e) { console.error(`Error: ${e.message}`); process.exit(1); }

// ── Load point-diffs ─────────────────────────────────────────────────────────────
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

// Replay + extract, invoke fn(keyArray, off, end, tgt) per matched candidate.
function forEachCandidate(s, fn) {
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
    fn(state.keys, state.keyOff[i], state.keyOff[i + 1], tgt);
  }
}

console.log(`spec: ${weights.spec.str}  (${weights.nSpaces} space${weights.nSpaces > 1 ? 's' : ''}, OLS closed form)  ridge: ${RIDGE}`);
console.log(`data: ${DATA}  size: ${SIZE}  samples: ${samples.length}`);
console.log(`save: ${SAVE_PATH}`);

const t0 = Date.now();
// Pass 1: intern all keys so we know P before sizing the dense matrix.
let cand = 0;
for (const s of samples) forEachCandidate(s, () => { cand++; });
const P = weights.size;
console.log(`pass 1: ${P} keys interned, ${cand} candidates  (${Util.fmtMs(Date.now() - t0)})`);

// Pass 2: accumulate A = XᵀX and b = Xᵀy (each candidate adds an outer product of its key set).
const A = new Float64Array(P * P), b = new Float64Array(P);
for (const s of samples) forEachCandidate(s, (keys, off, end, tgt) => {
  for (let a = off; a < end; a++) {
    const ka = keys[a];
    b[ka] += tgt;
    const base = ka * P;
    for (let bb = off; bb < end; bb++) A[base + keys[bb]] += 1;
  }
});
console.log(`pass 2: normal equations built  (${Util.fmtMs(Date.now() - t0)})`);

// Ridge + Gaussian elimination with partial pivoting.
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
fs.writeFileSync(SAVE_PATH, FeaturePol.serialize(weights, { spec: weights.spec.str, ema: 0, totalUpdates: cand }));
console.log(`keys: ${P}  avgW: ${(P ? absSum / P : 0).toFixed(4)}  total: ${Util.fmtMs(Date.now() - t0)}`);
console.log(`done -> ${SAVE_PATH}`);
