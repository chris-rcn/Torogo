#!/usr/bin/env node
'use strict';

// tune-raveheavy.js — optimize 3×3 pattern weights using move-quality evaluation.
//
// CMA-ES minimizes the rmsErr of raveheavy move choices against pre-computed
// KataGo win-rate rankings (produced by createmovedetails.js).
//
// Usage:
//   node tune-raveheavy.js --file <details> --budget <ms> [options]
//
//   --file        move-details file produced by createmovedetails.js (required)
//   --budget      time budget per move in ms (required)
//   --sigma       initial CMA-ES step size (default 0.5)
//   --init-mean   initial mean for all weights (default 0; try -2 to start devalued)
//   --reg-weight  L2 regularization coefficient (default 0)
//   --discovery   random games played to seed the pattern vocabulary (default 500)
//   --out         output JS file for learned weights (default auto-named)

const fs = require('fs');
const { Game2, PASS, BLACK } = require('./game2.js');
const { patternHash2 }       = require('./pattern9.js');
const { makeGetMove }        = require('./ai/raveheavy.js');
const { loadPositions, evalPositions } = require('./evalmovedetails.js');

// ── Args ──────────────────────────────────────────────────────────────────────

const argv = process.argv.slice(2);
const get  = (flag, def) => { const i = argv.indexOf(flag); return i !== -1 ? argv[i+1] : def; };

const file        = get('--file', null);
const budgetMs    = parseInt(get('--budget', '1'),   10);
const batchArg    = parseInt(get('--batch',            '1'),   10);
const displayArg  = parseInt(get('--display-positions', '1'),   10);
const sigma0      = parseFloat(get('--sigma',        '0.5'));
const initMean    = parseFloat(get('--init-mean',    '0'));
const regWeight   = parseFloat(get('--reg-weight',   '0'));
const discoveryN  = parseInt(get('--discovery',      '500'), 10);
const outFile     = get('--out', `tuned-weights-${Math.floor(Math.random() * 1000000)}.js`);

if (!file || budgetMs <= 0) {
  process.stderr.write('Usage: node tune-raveheavy.js --file <details> --budget <ms> [options]\n');
  process.exit(1);
}

const positions  = loadPositions(file);
const boardN     = positions[0].boardSize;
const batchN   = batchArg   > 0 ? Math.min(batchArg,   positions.length) : positions.length;
const displayN = displayArg > 0 ? Math.min(displayArg, positions.length) : positions.length;
const sampleBatch = () => Array.from({ length: displayN }, () => positions[Math.floor(Math.random() * positions.length)]);

// ── Standard normal sampler (Box-Muller) ─────────────────────────────────────

let _spare = 0, _hasSpare = false;
function randn() {
  if (_hasSpare) { _hasSpare = false; return _spare; }
  const u = 1 - Math.random(), v = Math.random();
  const mag = Math.sqrt(-2 * Math.log(u));
  _spare = mag * Math.cos(2 * Math.PI * v);
  _hasSpare = true;
  return mag * Math.sin(2 * Math.PI * v);
}

// ── Separable CMA-ES ──────────────────────────────────────────────────────────

function initCMA(n) {
  const lambda = 4 + Math.floor(3 * Math.log(n));
  const mu     = Math.floor(lambda / 2);

  const rawW = new Float64Array(mu);
  for (let i = 0; i < mu; i++) rawW[i] = Math.log(mu + 0.5) - Math.log(i + 1);
  const sumW = rawW.reduce((a, b) => a + b, 0);
  const w = rawW.map(x => x / sumW);

  const mueff = 1 / w.reduce((s, x) => s + x * x, 0);

  const cs   = (mueff + 2) / (n + mueff + 5);
  const ds   = 1 + cs + 2 * Math.max(0, Math.sqrt((mueff - 1) / (n + 1)) - 1);
  const cc   = (4 + mueff / n) / (n + 4 + 2 * mueff / n);
  const c1   = 2 / ((n + 1.3) ** 2 + mueff);
  const cmu  = Math.min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff));
  const chiN = Math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n));

  return {
    n, lambda, mu, w, mueff, cs, ds, cc, c1, cmu, chiN,
    m:     new Float64Array(n).fill(initMean),
    sigma: sigma0,
    D:     new Float64Array(n).fill(1),
    pc:    new Float64Array(n),
    ps:    new Float64Array(n),
    gen:   0,
  };
}

function sampleCMA(state) {
  const { n, lambda, m, sigma, D } = state;
  const xs = new Float64Array(n * lambda);
  for (let k = 0; k < lambda; k++) {
    const base = k * n;
    for (let j = 0; j < n; j++) xs[base + j] = m[j] + sigma * D[j] * randn();
  }
  return xs;
}

function stepCMA(state, xs, fs) {
  const { n, lambda, mu, w, mueff, cs, ds, cc, c1, cmu, chiN, m, D, pc, ps } = state;

  const order = Array.from({ length: lambda }, (_, i) => i).sort((a, b) => fs[a] - fs[b]);

  const m_old = m.slice();
  const delta = new Float64Array(n);
  for (let i = 0; i < mu; i++) {
    const base = order[i] * n;
    for (let j = 0; j < n; j++) delta[j] += w[i] * (xs[base + j] - m[j]);
  }
  for (let j = 0; j < n; j++) m[j] += delta[j];

  const psCoeff = Math.sqrt(cs * (2 - cs) * mueff);
  let psNorm2 = 0;
  for (let j = 0; j < n; j++) {
    ps[j] = (1 - cs) * ps[j] + psCoeff * (delta[j] / (state.sigma * D[j]));
    psNorm2 += ps[j] * ps[j];
  }
  const psNorm = Math.sqrt(psNorm2);

  state.sigma *= Math.exp((cs / ds) * (psNorm / chiN - 1));
  state.sigma  = Math.min(state.sigma, 1e6);

  const hsig = psNorm / (Math.sqrt(1 - (1 - cs) ** (2 * (state.gen + 1))) * chiN) < 1.4 + 2 / (n + 1) ? 1 : 0;

  const pcCoeff = hsig * Math.sqrt(cc * (2 - cc) * mueff);
  for (let j = 0; j < n; j++) {
    pc[j] = (1 - cc) * pc[j] + pcCoeff * (delta[j] / state.sigma);
  }

  for (let j = 0; j < n; j++) {
    let d2 = (1 - c1 - cmu) * D[j] * D[j];
    d2 += c1 * (pc[j] * pc[j] + (1 - hsig) * cc * (2 - cc) * D[j] * D[j]);
    for (let i = 0; i < mu; i++) {
      const step = (xs[order[i] * n + j] - m_old[j]) / state.sigma;
      d2 += cmu * w[i] * step * step;
    }
    D[j] = Math.sqrt(Math.max(1e-20, d2));
  }

  state.gen++;
}

// ── Pattern discovery ─────────────────────────────────────────────────────────
// Play random games and collect every 3×3 hash that could appear as a local
// response candidate (i.e. the 3×3 window of the move just played).

function discoverPatterns(N, patternIndex) {
  const game2 = new Game2(N);
  while (!game2.gameOver) {
    const lm = game2.lastMove;
    if (lm >= 0) {
      const lx = lm % N, ly = (lm / N) | 0;
      const current = game2.current;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = lx + dx, ny = ly + dy;
          if (nx < 0 || nx >= N || ny < 0 || ny >= N) continue;
          const idx = ny * N + nx;
          if (game2.cells[idx] !== 0) continue;
          if (!game2.isTrueEye(idx) && game2.isLegal(idx)) {
            const h = patternHash2(game2, idx, current);
            if (!patternIndex.has(h)) patternIndex.set(h, patternIndex.size);
          }
        }
      }
    }
    game2.play(game2.randomLegalMove());
  }
}

// Hash 0 (tenuki/non-local) is fixed at weight 0; only discovered patterns are optimized.
const patternIndex = new Map();
for (let r = 0; r < discoveryN; r++) discoverPatterns(boardN, patternIndex);

const n = patternIndex.size;
const patternHashes = new Array(n);
for (const [h, i] of patternIndex) patternHashes[i] = h;

process.stderr.write(`Found ${n} unique patterns (after ${discoveryN} discovery games).\n`);

if (n === 0) {
  process.stderr.write('No patterns found — nothing to optimize.\n');
  process.exit(1);
}

// ── Fitness evaluation ────────────────────────────────────────────────────────

function makeAgent(weights) {
  const table = new Map();
  for (let i = 0; i < n; i++) table.set(patternHashes[i], weights[i]);
  return makeGetMove(table);
}

// Returns rmsErr + regularization (CMA-ES minimizes).
function evalFitness(weights) {
  const batch = batchN === positions.length
    ? positions
    : Array.from({ length: batchN }, () => positions[Math.floor(Math.random() * positions.length)]);
  const { rmsErr } = evalPositions(makeAgent(weights), batch, budgetMs);
  let l2 = 0;
  for (let j = 0; j < n; j++) l2 += (weights[j] - initMean) ** 2;
  return rmsErr + regWeight * l2 / n;
}

// ── Save weights ──────────────────────────────────────────────────────────────

function saveWeights() {
  const body = patternHashes.map((h, i) => `[${h},${cma.m[i]}]`).join(',');
  fs.writeFileSync(outFile,
    `'use strict';\n` +
    `// Auto-generated by tune-raveheavy.js — do not edit by hand.\n` +
    `const patternTable = new Map([${body}]);\n` +
    `if (typeof module !== 'undefined') module.exports = patternTable;\n` +
    `else window.patternTable = patternTable;\n`
  );
}

// ── CMA-ES optimization loop ──────────────────────────────────────────────────

const cma = initCMA(n);

process.stderr.write(
  `CMA-ES: n=${n}  lambda=${cma.lambda}  mu=${cma.mu}  ` +
  `positions=${positions.length}  batch=${batchN}  display=${displayN}  budget=${budgetMs}ms  reg-weight=${regWeight}  out=${outFile}\n`
);
process.stderr.write('─'.repeat(62) + '\n');

let iter = 0;
let nextReport = 1;
const startTime = Date.now();
let rmsErrSum = 0, reportCount = 0;

// eslint-disable-next-line no-constant-condition
while (true) {
  iter++;

  const xs = sampleCMA(cma);
  const fitnesses = new Float64Array(cma.lambda);
  for (let k = 0; k < cma.lambda; k++) {
    fitnesses[k] = evalFitness(xs.subarray(k * n, (k + 1) * n));
  }

  stepCMA(cma, xs, fitnesses);

  if (iter === nextReport) {
    const { rmsErr } = evalPositions(makeAgent(cma.m), sampleBatch(), budgetMs);
    rmsErrSum += rmsErr; reportCount++;
    const avgRmsErr = rmsErrSum / reportCount;
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    let maxW = 0, sumW = 0;
    for (let j = 0; j < n; j++) { const a = Math.abs(cma.m[j]); if (a > maxW) maxW = a; sumW += cma.m[j]; }
    const avgW = sumW / n;
    process.stderr.write(
      `iter ${String(iter).padStart(6)}` +
      `  rmsErr=${rmsErr.toFixed(4).padStart(8)}` +
      `  avg=${avgRmsErr.toFixed(4).padStart(8)}` +
      `  sigma=${cma.sigma.toFixed(4).padStart(8)}` +
      `  maxW=${maxW.toFixed(3).padStart(7)}` +
      `  avgW=${avgW.toFixed(3).padStart(7)}` +
      `  elapsed=${String(elapsed + 's').padStart(9)}\n`
    );
    saveWeights();
    nextReport = Math.ceil(nextReport * 1.5);
  }
}
