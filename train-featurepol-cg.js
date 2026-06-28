'use strict';

// train-featurepol-cg.js — exact least-squares fit via CGLS (conjugate gradient on the least-
// squares problem), MATRIX-FREE.  Solves the same OLS as train-featurepol-lsq.js (minimise
// ||Xw − y||²) but never forms the P×P normal-equations matrix: each iteration does only the
// sparse products  q = X·p  and  s = Xᵀ·r  (two passes over the cached sparse data).  Cost and
// memory scale with nnz (≈ candidates × spaces), essentially independent of the pattern count P,
// so large-P specs (e.g. stones12, ~48k patterns) that blow up the O(P³) dense solve are fine.
//
// Deterministic — no learning rate, no epochs, no seed.  CGLS converges to the minimum-norm
// least-squares solution, so the rank-deficient inter-space null (each space is a partition) is
// handled automatically without a ridge.  Works for any spec, single or multi term.

const fs   = require('fs');
const path = require('path');
const FeaturePol = require('./featurepol-lib.js');
const { Game2, parseMove } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help', 'center'], ['data', 'spec', 'save', 'size', 'iters', 'tol', 'ridge', 'point-weight', 'weight']);
if (opts.help || !opts.data || !opts.spec) {
  console.log("Usage: node train-featurepol-cg.js --data FILE[,FILE...] --spec '<spec>' [--point-weight 1] [--weight sqrt|n|log|none] [--center] [--ridge L] [--iters 500] [--tol 1e-6] [--save PATH] [--size N]");
  process.exit(opts.help ? 0 : 1);
}
const DATA      = opts.data;
const SPEC      = opts.spec;
// Blended target (point/win delta data):  pw·pointDelta + (1−pw)·winDelta.  pw=1 → point, pw=0 → win.
const POINT_WEIGHT = opts['point-weight'] !== undefined ? parseFloat(opts['point-weight']) : 0;   // default: pure win
if (!(POINT_WEIGHT >= 0 && POINT_WEIGHT <= 1)) { console.error(`Error: --point-weight must be in [0,1] (got '${opts['point-weight']}')`); process.exit(1); }
// Per-row weighted least squares (distill data only): w_i from the visit count n_i.
const WEIGHT    = opts.weight || 'sqrt';   // 'n' → w=n (inverse-variance), 'sqrt' → w=√n, 'log' → w=log(1+n), 'none' → w=1
if (!['sqrt', 'n', 'log', 'none'].includes(WEIGHT)) { console.error(`Error: --weight must be sqrt|n|log|none (got '${WEIGHT}')`); process.exit(1); }
const CENTER    = opts.center || false;   // fit within-position contrasts: minimise ||M(Xw−y)||², M = per-position mean-removal
const RIDGE     = opts.ridge !== undefined ? parseFloat(opts.ridge) : 0;   // Tikhonov: minimise ||Xw−y||² + RIDGE·||w||²
const MAXIT     = opts.iters !== undefined ? parseInt(opts.iters, 10) : 500;
const TOL       = opts.tol !== undefined ? parseFloat(opts.tol) : 1e-6;   // stop when ||grad||² / initial < TOL
const SAVE_PATH = opts.save || `out/featurepol-cg-${SPEC.replace(/[^a-zA-Z0-9]/g, '')}.js`;

let weights;
try { weights = FeaturePol.createWeights({ spec: SPEC }); }
catch (e) { console.error(`Error: ${e.message}`); process.exit(1); }

// ── Load training data (NDJSON from gen-deltas.js / convert-to-ndjson.js; ≥1 files) ──
// Record: {size, pos(csv moves), moves(csv candidates), p[](point means), pp, w[]?, pw?}.
// Per-candidate target:  --target point → p[i]−pp ;  --target win → w[i]−pw  (needs w/pw).
// Files may mix board sizes (features are local patterns → board-size-agnostic); each sample
// carries its own size, and a per-size feature state is used at extract time.
const FILES = DATA.split(',').map(s => s.trim()).filter(Boolean);
// Stream a file line-by-line in fixed chunks: the data files can exceed Node's ~512 MB single-
// string limit (readFileSync('utf8') would throw).  Records are ASCII JSON, so decoding chunk
// boundaries as UTF-8 is safe.  fn(line) per non-empty line; fn may return true to stop early.
function forEachLine(file, fn) {
  const fd = fs.openSync(file, 'r');
  const buf = Buffer.allocUnsafe(1 << 20);   // 1 MB chunks
  let leftover = '', bytes;
  try {
    while ((bytes = fs.readSync(fd, buf, 0, buf.length, null)) > 0) {
      const chunk = leftover + buf.toString('utf8', 0, bytes);
      let start = 0, nl;
      while ((nl = chunk.indexOf('\n', start)) !== -1) {
        if (nl > start && fn(chunk.slice(start, nl)) === true) return;
        start = nl + 1;
      }
      leftover = chunk.slice(start);
    }
    if (leftover) fn(leftover);
  } finally { fs.closeSync(fd); }
}

// Pick schema from the first record: distill data has q/n (search targets); else it's point/win deltas.
let _r0 = null;
for (const f of FILES) {
  if (!fs.existsSync(f)) { console.error(`Error: data file not found: ${f}`); process.exit(1); }
  forEachLine(f, l => { _r0 = JSON.parse(l); return true; });
  if (_r0) break;
}
if (!_r0) { console.error(`Error: no records in ${DATA}`); process.exit(1); }
const DISTILL = _r0.q !== undefined;
// Fail loudly on a flag that does nothing for the detected schema, rather than silently ignoring it.
if (!DISTILL && opts.weight !== undefined) {
  console.error("Error: --weight only applies to distill data (q/n); point/win data uses --point-weight."); process.exit(1);
}
const wfn = WEIGHT === 'none' ? (() => 1) : WEIGHT === 'n' ? (x => x) : WEIGHT === 'log' ? (x => Math.log1p(x)) : (x => Math.sqrt(x));   // per-row WLS weight w_i

const samples = [];
const sizeCounts = new Map();
let sP = 0, sP2 = 0, sW = 0, sW2 = 0, nDelta = 0;   // running stats for std-normalising deltas (point/win)
for (const file of FILES) {
  if (!fs.existsSync(file)) { console.error(`Error: data file not found: ${file}`); process.exit(1); }
  let n0 = samples.length;
  forEachLine(file, (line) => {
    if (!line) return;
    const rec = JSON.parse(line);
    const sz = rec.size || (opts.size !== undefined ? parseInt(opts.size, 10) : 9);
    const prefix = rec.pos ? rec.pos.split(',').filter(Boolean).map(c => parseMove(c, sz)) : [];
    const mvStr = rec.moves.split(',').filter(Boolean);
    const n = mvStr.length;
    if (!n) return;
    const lm = new Int32Array(n);
    for (let i = 0; i < n; i++) lm[i] = parseMove(mvStr[i], sz);

    if (DISTILL) {
      // Q-distillation.  Per-row WLS weight from the visit count n[i] (always).  Target is the
      // win delta q[i]−qp by default; with --point-weight>0 it blends in the point delta p[i]−pp,
      // std-normalised (computed after the load, mirroring the point/win mode below).
      const qp = rec.qp ?? 0.5;
      const lw = new Float32Array(n);
      for (let i = 0; i < n; i++) lw[i] = wfn(rec.n[i]);
      if (POINT_WEIGHT > 0) {
        if (rec.p === undefined || rec.pp === undefined) {
          console.error(`Error: --point-weight ${POINT_WEIGHT} needs point data, but ${file} has distill records without p/pp`); process.exit(1);
        }
        const pp = rec.pp ?? 0;
        const pd = new Float32Array(n);                                 // point delta p−pp
        const wd = POINT_WEIGHT < 1 ? new Float32Array(n) : null;       // win delta q−qp (skip at pw=1)
        for (let i = 0; i < n; i++) {
          const dp = rec.p[i] - pp; pd[i] = dp; sP += dp; sP2 += dp * dp;
          if (wd) { const dw = rec.q[i] - qp; wd[i] = dw; sW += dw; sW2 += dw * dw; }
        }
        nDelta += n;
        samples.push({ prefix: Int32Array.from(prefix), lm, pd, wd, lw, size: sz });
      } else {
        const lv = new Float32Array(n);
        for (let i = 0; i < n; i++) lv[i] = rec.q[i] - qp;
        samples.push({ prefix: Int32Array.from(prefix), lm, lv, lw, size: sz });
      }
    } else {
      if (POINT_WEIGHT < 1 && (rec.w === undefined || rec.pw === undefined)) {
        console.error(`Error: --point-weight ${POINT_WEIGHT} needs win data, but ${file} has records without w/pw`); process.exit(1);
      }
      if (POINT_WEIGHT > 0 && rec.p === undefined) {
        console.error(`Error: --point-weight ${POINT_WEIGHT} needs point data, but ${file} has records without p`); process.exit(1);
      }
      const pp = rec.pp ?? 0, pw = rec.pw ?? 0;
      const pd = POINT_WEIGHT > 0 ? new Float32Array(n) : null;
      const wd = POINT_WEIGHT < 1 ? new Float32Array(n) : null;
      for (let i = 0; i < n; i++) {
        if (pd) { const d = rec.p[i] - pp; pd[i] = d; sP += d; sP2 += d * d; }
        if (wd) { const d = rec.w[i] - pw; wd[i] = d; sW += d; sW2 += d * d; }
      }
      nDelta += n;
      samples.push({ prefix: Int32Array.from(prefix), lm, pd, wd, size: sz });
    }
    sizeCounts.set(sz, (sizeCounts.get(sz) || 0) + 1);
  });
  if (FILES.length > 1) console.log(`  loaded ${samples.length - n0} positions from ${file}`);
}
if (samples.length === 0) { console.error(`Error: no samples parsed from ${DATA}`); process.exit(1); }

if (DISTILL && POINT_WEIGHT === 0) {
  console.log(`distill: target q−qp  weight ${WEIGHT}`);
} else {
  // Std-normalise each delta to unit scale, then blend: lv = pw·(pointΔ/σ_point) + (1−pw)·(winΔ/σ_win).
  // point/win mode: winΔ = w−pw.  distill (point-weight>0): winΔ = q−qp, and rows keep their WLS weight (lw).
  const sdP = POINT_WEIGHT > 0 ? (Math.sqrt(Math.max(0, sP2 / nDelta - (sP / nDelta) ** 2)) || 1) : 1;
  const sdW = POINT_WEIGHT < 1 ? (Math.sqrt(Math.max(0, sW2 / nDelta - (sW / nDelta) ** 2)) || 1) : 1;
  const cp = POINT_WEIGHT / sdP, cw = (1 - POINT_WEIGHT) / sdW;
  for (const s of samples) {
    const m = s.lm.length, lv = new Float32Array(m);
    for (let i = 0; i < m; i++) { let t = 0; if (s.pd) t += cp * s.pd[i]; if (s.wd) t += cw * s.wd[i]; lv[i] = t; }
    s.lv = lv; s.pd = null; s.wd = null;
  }
  const lbl = DISTILL ? 'distill blend' : 'normalize';
  console.log(`${lbl}: σ_point ${sdP.toFixed(3)}  σ_win ${sdW.toFixed(4)}  point-weight ${POINT_WEIGHT}${DISTILL ? `  weight ${WEIGHT}` : ''}`);
}

const needLadder = weights.spec.needsLadder;
const _stateBySize = new Map();
function stateFor(sz) {
  let st = _stateBySize.get(sz);
  if (!st) { st = FeaturePol.createState(sz, weights.spec); _stateBySize.set(sz, st); }
  return st;
}

function forEachCandidate(s, fn) {
  const game = new Game2(s.size);
  for (let i = 0; i < s.prefix.length; i++) game.play(s.prefix[i]);
  const game3 = needLadder ? game3FromGame2(game) : undefined;
  const state = stateFor(s.size);
  FeaturePol.extractFeatures(game, state, weights, game3);
  const n = state.count;
  let aligned = n === s.lm.length;
  if (aligned) for (let i = 0; i < n; i++) if (state.moves[i] !== s.lm[i]) { aligned = false; break; }
  let idxOf = null;
  if (!aligned) { idxOf = new Map(); for (let i = 0; i < s.lm.length; i++) idxOf.set(s.lm[i], i); }
  for (let i = 0; i < n; i++) {
    let j;
    if (aligned) j = i; else { j = idxOf.get(state.moves[i]); if (j === undefined) continue; }
    fn(state.keys, state.keyOff[i], state.keyOff[i + 1], s.lv[j], s.lw ? s.lw[j] : 1);
  }
}

console.log(`spec: ${weights.spec.str}  (${weights.nSpaces} space${weights.nSpaces > 1 ? 's' : ''}, CGLS)  objective: ${CENTER ? 'within-position contrast (centered)' : 'absolute regression'}`);
console.log(`data: ${DATA}  sizes: ${[...sizeCounts.entries()].map(([s, c]) => `${s}:${c}`).join(' ')}  samples: ${samples.length}  ${DISTILL ? `weight: ${WEIGHT}` : `point-weight: ${POINT_WEIGHT}`}`);
console.log(`save: ${SAVE_PATH}`);

// ── Pass 1: intern keys, count rows (candidates) and non-zeros ────────────────────
const t0 = Date.now();
let nCand = 0, nnz = 0;
const cntArr = [];
for (const s of samples) forEachCandidate(s, (keys, off, end) => {
  nCand++; nnz += end - off;
  for (let k = off; k < end; k++) cntArr[keys[k]] = (cntArr[keys[k]] || 0) + 1;
});
const P = weights.size;
// Jacobi (diagonal) preconditioner: scale each key's column by D[k] = 1/√count[k] so the Gram
// diagonal becomes ~1.  Counts span ~6 orders of magnitude (common vs rare patterns), giving a
// condition number ~1e6 that stalls plain CG on the rare-pattern directions; this normalises it.
const D = new Float64Array(P);
for (let k = 0; k < P; k++) D[k] = cntArr[k] > 0 ? 1 / Math.sqrt(cntArr[k]) : 0;
console.log(`pass 1: ${P} keys, ${nCand} rows, ${nnz} non-zeros  (${Util.fmtMs(Date.now() - t0)})`);

// ── Pass 2: build CSR (rowPtr, col), targets y, row scales (WLS), and groups (--center) ──
// Weighted least squares min Σ wᵢ(Xw−y)ᵢ² is plain LS on the row-scaled system (√wᵢ·rowᵢ, √wᵢ·yᵢ);
// rs[i]=√wᵢ is applied in the matrix products and the target.  rs=1 (uniform) for non-distill data.
const rowPtr = new Int32Array(nCand + 1), col = new Int32Array(nnz), yv = new Float64Array(nCand);
const rs = new Float64Array(nCand);   // per-row scale = √(WLS weight)
const groupPtr = CENTER ? new Int32Array(samples.length + 1) : null;   // contiguous row range per position
let ci = 0, ri = 0, gi = 0;
for (const s of samples) {
  const rStart = ri;
  forEachCandidate(s, (keys, off, end, tgt, weight) => {
    const sc = Math.sqrt(weight);
    rs[ri] = sc; yv[ri] = sc * tgt;
    for (let k = off; k < end; k++) col[ci++] = keys[k];
    rowPtr[++ri] = ci;
  });
  if (CENTER && ri > rStart) groupPtr[gi++] = rStart;   // record only non-empty positions
}
const nGroups = gi;
if (CENTER) groupPtr[nGroups] = nCand;                  // sentinel end (groups are contiguous)
console.log(`pass 2: CSR built${CENTER ? `, ${nGroups} position-groups` : ''}  (${Util.fmtMs(Date.now() - t0)})`);

// Subtract each position's mean over its candidate rows (the projection M); idempotent.
function centerByGroup(v) {
  for (let g = 0; g < nGroups; g++) {
    const a = groupPtr[g], b = groupPtr[g + 1];
    let s = 0; for (let i = a; i < b; i++) s += v[i];
    const m = s / (b - a);
    for (let i = a; i < b; i++) v[i] -= m;
  }
}
if (CENTER) centerByGroup(yv);   // y ← M y; residual then stays centered, so only the forward op needs M

// ── CGLS ──────────────────────────────────────────────────────────────────────────
// We solve in the SCALED space (variable w' with X̃ = X·diag(D)); the actual weight is D∘w'.
const w = new Float64Array(P), p = new Float64Array(P), sVec = new Float64Array(P), dp = new Float64Array(P);
const r = new Float64Array(nCand), q = new Float64Array(nCand);

function Xtimes(vec, out) {            // out = diag(rs)·X · vec  (row-scaled for WLS)
  for (let i = 0; i < nCand; i++) { let acc = 0; for (let j = rowPtr[i], e = rowPtr[i + 1]; j < e; j++) acc += vec[col[j]]; out[i] = rs[i] * acc; }
}
function Xtrans(vec, out) {            // out = Xᵀ·diag(rs) · vec  (keys)
  out.fill(0);
  for (let i = 0; i < nCand; i++) { const v = rs[i] * vec[i]; for (let j = rowPtr[i], e = rowPtr[i + 1]; j < e; j++) out[col[j]] += v; }
}

// Ridge is standard Tikhonov λ||w||² in the ORIGINAL weight space; since w = D∘w', in the scaled
// space that is λ·Σ Dₖ²·w'ₖ², so the per-key ridge coefficient is RIDGE·Dₖ² (= RIDGE/countₖ),
// which shrinks rare (small-count) patterns hardest — the ill-conditioned, noise-prone ones.
const rg = new Float64Array(P); for (let k = 0; k < P; k++) rg[k] = RIDGE * D[k] * D[k];
r.set(yv);                            // r = y − X̃·0 = y
Xtrans(r, sVec);                      // s = D∘(Xᵀr) − λDₖ²w'  (gradient in scaled space)
for (let k = 0; k < P; k++) sVec[k] = D[k] * sVec[k] - rg[k] * w[k];
p.set(sVec);
let gamma = 0; for (let k = 0; k < P; k++) gamma += sVec[k] * sVec[k];
const gamma0 = gamma;
let iter = 0;
for (; iter < MAXIT; iter++) {
  for (let k = 0; k < P; k++) dp[k] = D[k] * p[k];
  Xtimes(dp, q);                      // q = X(D∘p) = X̃p
  if (CENTER) centerByGroup(q);       // q ← M·X̃p  (within-position contrast); keeps r centered so Xᵀ needs no M
  let qq = 0; for (let i = 0; i < nCand; i++) qq += q[i] * q[i];   // ||X̃p||² + ΣλDₖ²pₖ²
  if (RIDGE) for (let k = 0; k < P; k++) qq += rg[k] * p[k] * p[k];
  if (qq === 0) break;
  const alpha = gamma / qq;
  for (let k = 0; k < P; k++) w[k] += alpha * p[k];
  for (let i = 0; i < nCand; i++) r[i] -= alpha * q[i];
  Xtrans(r, sVec);
  for (let k = 0; k < P; k++) sVec[k] = D[k] * sVec[k] - rg[k] * w[k];
  let gnew = 0; for (let k = 0; k < P; k++) gnew += sVec[k] * sVec[k];
  if ((iter & 15) === 0 || gnew <= TOL * gamma0) {
    let rr = 0; for (let i = 0; i < nCand; i++) rr += r[i] * r[i];
    console.log(`  iter ${String(iter).padStart(4)}  rms ${Math.sqrt(rr / nCand).toFixed(4)}  grad² ${(gnew / gamma0).toExponential(2)}  (${Util.fmtMs(Date.now() - t0)})`);
  }
  if (gnew <= TOL * gamma0) { iter++; break; }
  const beta = gnew / gamma;
  for (let k = 0; k < P; k++) p[k] = sVec[k] + beta * p[k];
  gamma = gnew;
}

let absSum = 0;
for (let k = 0; k < P; k++) { weights.vals[k] = D[k] * w[k]; absSum += Math.abs(weights.vals[k]); }   // unscale w' → w
fs.mkdirSync(path.dirname(SAVE_PATH), { recursive: true });
fs.writeFileSync(SAVE_PATH, FeaturePol.serialize(weights, { spec: weights.spec.str, ema: 0, totalUpdates: nCand }));
let rr = 0; for (let i = 0; i < nCand; i++) rr += r[i] * r[i];
console.log(`converged in ${iter} iters  rms ${Math.sqrt(rr / nCand).toFixed(4)}  avgW ${(P ? absSum / P : 0).toFixed(4)}  total ${Util.fmtMs(Date.now() - t0)}`);
console.log(`done -> ${SAVE_PATH}`);
