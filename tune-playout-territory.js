#!/usr/bin/env node
'use strict';

// tune-playout.js — optimize 3×3 pattern weights for K=2 tournament playout
// selection using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
//
// Reads scored positions from --file (output of score-positions.js).  Each
// position already has a ground-truth score from many uniform random playouts.
// CMA-ES minimizes biasWeight*bias² + variance, where bias² is the squared
// error between the mean K-tournament playout estimate and the ground-truth
// score, and variance is the playout variance across the same rollouts.
//
// K=2 tournament: on each playout move, 2 random legal non-eye candidates are
// sampled.  The one whose 3×3 pattern has the higher learned weight wins with
// probability proportional to softmax: p(c1) = 1/(1+exp(w2-w1)).  Unknown
// patterns get weight 0 (neutral).
//
// Uses separable CMA-ES (diagonal covariance) which is O(n) memory and
// appropriate for ~independent parameters like per-pattern weights.
//
// Usage:
//   node tune-playout.js --file scored.tmp --playouts <n> [options]
//
//   --file               scored.tmp produced by score-positions.js (required)
//   --playouts           rollouts per position (fitness and checkpoint eval; required)
//   --bias-weight        multiplier on bias² in fitness function (default 1)
//   --variance-weight    multiplier on variance in fitness function (default 0)
//   --reg-weight         L2 regularization coefficient; penalizes large weights (default 0)
//   --batch              positions per generation; 0 = all (default 0)
//   --display-positions  positions sampled for checkpoint display; 0 = all (default 0)
//   --sigma              initial CMA-ES step size (default 0.5)
//   --out                output JS file for learned weights (default tuned-weights.js)

const fs = require('fs');
const { Game2, PASS, BLACK } = require('./game2.js');
const { patternHash2 }       = require('./pattern9.js');
const { selectPatternMove }  = require('./ai/raveheavy.js');

// ── Args ──────────────────────────────────────────────────────────────────────

const argv = process.argv.slice(2);
const get  = (flag, def) => { const i = argv.indexOf(flag); return i !== -1 ? argv[i+1] : def; };

const file            = get('--file', null);
const playouts        = parseInt(get('--playouts', '0'), 10);
const biasWeight      = parseFloat(get('--bias-weight',     '1'));
const varianceWeight  = parseFloat(get('--variance-weight', '0'));
const regWeight       = parseFloat(get('--reg-weight',   '0'));
const batchArg        = parseInt(get('--batch',              '0'), 10);
const displayArg      = parseInt(get('--display-positions', '300'), 10);
const sigma0          = parseFloat(get('--sigma',        '0.5'));
const discoveryPlayouts = 1000;
const outFile         = get('--out', `tuned-weights-${Math.floor(Math.random() * 1000000)}.js`);

if (!file || playouts <= 0) {
  process.stderr.write('Usage: node tune-playout.js --file <scored.tmp> --playouts <n> [options]\n');
  process.exit(1);
}

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
//
// State:
//   m      — mean weight vector (n)
//   sigma  — global step size (scalar)
//   D      — diagonal standard deviations: C = diag(D²) (n)
//   pc     — evolution path for covariance (n)
//   ps     — evolution path for sigma (n)
//   gen    — generation counter
//
// Sampling: x = m + sigma * D * z,  z ~ N(0,I)

function initCMA(n) {
  const lambda = 4 + Math.floor(3 * Math.log(n));
  const mu     = Math.floor(lambda / 2);

  // Log-linear recombination weights, normalized to sum 1.
  const rawW = new Float64Array(mu);
  for (let i = 0; i < mu; i++) rawW[i] = Math.log(mu + 0.5) - Math.log(i + 1);
  const sumW = rawW.reduce((a, b) => a + b, 0);
  const w = rawW.map(x => x / sumW);

  const mueff = 1 / w.reduce((s, x) => s + x * x, 0);

  // Strategy parameters from Hansen's CMA-ES tutorial.
  const cs   = (mueff + 2) / (n + mueff + 5);
  const ds   = 1 + cs + 2 * Math.max(0, Math.sqrt((mueff - 1) / (n + 1)) - 1);
  const cc   = (4 + mueff / n) / (n + 4 + 2 * mueff / n);
  const c1   = 2 / ((n + 1.3) ** 2 + mueff);
  const cmu  = Math.min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff));
  const chiN = Math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n));

  return {
    n, lambda, mu, w, mueff, cs, ds, cc, c1, cmu, chiN,
    m:     new Float64Array(n),       // mean
    sigma: sigma0,
    D:     new Float64Array(n).fill(1), // diagonal std devs
    pc:    new Float64Array(n),       // covariance evolution path
    ps:    new Float64Array(n),       // sigma evolution path
    gen:   0,
  };
}

// Sample lambda candidates.  Returns flat Float64Array of shape [lambda × n].
function sampleCMA(state) {
  const { n, lambda, m, sigma, D } = state;
  const xs = new Float64Array(n * lambda);
  for (let k = 0; k < lambda; k++) {
    const base = k * n;
    for (let j = 0; j < n; j++) {
      xs[base + j] = m[j] + sigma * D[j] * randn();
    }
  }
  return xs;
}

// Update state given samples xs and their fitnesses fs (lower = better).
function stepCMA(state, xs, fs) {
  const { n, lambda, mu, w, mueff, cs, ds, cc, c1, cmu, chiN, m, D, pc, ps } = state;

  // Sort candidates by fitness ascending.
  const order = Array.from({ length: lambda }, (_, i) => i).sort((a, b) => fs[a] - fs[b]);

  // Weighted mean step: delta = sum_i w[i] * (x_{i:λ} - m)
  const m_old = m.slice();
  const delta = new Float64Array(n);
  for (let i = 0; i < mu; i++) {
    const base = order[i] * n;
    for (let j = 0; j < n; j++) delta[j] += w[i] * (xs[base + j] - m[j]);
  }
  for (let j = 0; j < n; j++) m[j] += delta[j];

  // ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * delta/(sigma*D)
  // (uses normalized step z = delta/(sigma*D) to track progress in isotropic space)
  const psCoeff = Math.sqrt(cs * (2 - cs) * mueff);
  let psNorm2 = 0;
  for (let j = 0; j < n; j++) {
    ps[j] = (1 - cs) * ps[j] + psCoeff * (delta[j] / (state.sigma * D[j]));
    psNorm2 += ps[j] * ps[j];
  }
  const psNorm = Math.sqrt(psNorm2);

  // sigma update.
  state.sigma *= Math.exp((cs / ds) * (psNorm / chiN - 1));
  state.sigma  = Math.min(state.sigma, 1e6);   // safety cap

  // hsig: 1 if ps hasn't grown too large (normal progress).
  // When 0, prevents pc from shrinking C during stagnation.
  const hsig = psNorm / (Math.sqrt(1 - (1 - cs) ** (2 * (state.gen + 1))) * chiN) < 1.4 + 2 / (n + 1) ? 1 : 0;

  // pc = (1-cc)*pc + hsig * sqrt(cc*(2-cc)*mueff) * delta/sigma
  const pcCoeff = hsig * Math.sqrt(cc * (2 - cc) * mueff);
  for (let j = 0; j < n; j++) {
    pc[j] = (1 - cc) * pc[j] + pcCoeff * (delta[j] / state.sigma);
  }

  // D² update (element-wise separable covariance).
  // = (1-c1-cmu)*D² + c1*(pc² + stagnation correction) + cmu*rank-μ term
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

// ── Move encoding ─────────────────────────────────────────────────────────────

function decodeMove(s, N) {
  if (s === '..') return PASS;
  return (s.charCodeAt(1) - 97) * N + (s.charCodeAt(0) - 97);
}

// ── K=2 pattern playout ───────────────────────────────────────────────────────

// Run one playout from game2 (mutated in place) using local response patterns.
// patternWeights[patternIndex.get(hash)] is the logit weight for that pattern.
// Returns black's territory score (komi excluded).
function runPatternPlayout(game2, patternWeights, patternIndex) {
  const N   = game2.N;
  const cap = N * N;
  const cells = game2.cells;

  const empty = [];
  for (let i = 0; i < cap; i++) if (cells[i] === 0) empty.push(i);

  const moveLimit = empty.length + 20;
  let moves = 0;

  const weightOf = h => { const i = patternIndex.get(h); return i !== undefined ? patternWeights[i] : 0; };

  while (!game2.gameOver && moves < moveLimit) {
    const chosen = selectPatternMove(game2, weightOf, patternHash2);

    if (chosen === PASS) { game2.play(PASS); moves++; continue; }

    const { capturedCount } = game2.playInfo(chosen);
    const pos = empty.indexOf(chosen);
    if (pos !== -1) { empty[pos] = empty[empty.length - 1]; empty.pop(); }
    if (capturedCount) {
      empty.length = 0;
      for (let i = 0; i < cap; i++) if (cells[i] === 0) empty.push(i);
    }
    moves++;
  }

  return game2.calcScore().black;
}

// ── Fitness evaluation ────────────────────────────────────────────────────────

// Fitness = biasWeight * bias² + varianceWeight * variance + regWeight * mean(w²), averaged over the batch.
function evalFitness(patternWeights, patternIndex, batch) {
  let total = 0;
  for (const { game2, baseline } of batch) {
    let sum = 0, sumSq = 0;
    for (let p = 0; p < playouts; p++) {
      const s = runPatternPlayout(game2.clone(), patternWeights, patternIndex);
      sum   += s;
      sumSq += s * s;
    }
    const mean     = sum / playouts;
    const variance = sumSq / playouts - mean * mean;
    total += biasWeight * (mean - baseline) ** 2 + varianceWeight * variance;
  }
  let l2 = 0;
  for (let j = 0; j < patternWeights.length; j++) l2 += patternWeights[j] ** 2;
  return total / batch.length + regWeight * l2 / patternWeights.length;
}

// Evaluate a weight vector against a batch of positions.
// Returns { bias2, variance, fitness } each averaged over the batch.
function fullEval(patternWeights, patternIndex, batch) {
  let totalBias2 = 0, totalVariance = 0;
  for (const { game2, baseline } of batch) {
    let sum = 0, sumSq = 0;
    for (let p = 0; p < playouts; p++) {
      const s = runPatternPlayout(game2.clone(), patternWeights, patternIndex);
      sum   += s;
      sumSq += s * s;
    }
    const mean     = sum / playouts;
    const variance = sumSq / playouts - mean * mean;
    totalBias2    += (mean - baseline) ** 2;
    totalVariance += variance;
  }
  const bias2    = totalBias2    / batch.length;
  const variance = totalVariance / batch.length;
  let l2 = 0;
  for (let j = 0; j < patternWeights.length; j++) l2 += patternWeights[j] ** 2;
  const l2term = regWeight * l2 / patternWeights.length;
  return { bias2, variance, fitness: biasWeight * bias2 + varianceWeight * variance + l2term };
}

// ── Load training positions ───────────────────────────────────────────────────

const rawLines = fs.readFileSync(file, 'utf8').trim().split('\n');
const positions = [];

for (const line of rawLines) {
  if (!line.trim()) continue;
  const sp = line.indexOf(' ');
  const moveList = line.slice(0, sp);
  const rest     = line.slice(sp + 1).trim().split(/\s+/);
  const baseline = parseFloat(rest[0]);

  const fields = moveList.split(',');
  const N = parseInt(fields[0], 10);
  // fields[1] is the auto-placed center stone (constructor handles it); skip it.
  const moves = fields.slice(2).map(s => decodeMove(s.trim(), N));

  const g = new Game2(N);
  for (const idx of moves) {
    if (g.gameOver) break;
    g.play(idx);
  }
  if (g.gameOver) continue;

  positions.push({ game2: g, baseline });
}

process.stderr.write(`Loaded ${positions.length} training positions.\n`);

// ── Pattern discovery via full games from a fresh board ───────────────────────
// Running K=2 rollouts from mid-game positions only finds patterns reachable
// from those positions.  Starting from a fresh board covers the full game arc
// and discovers patterns that would otherwise only get weight 0 at eval time.

function discoverPatterns(N, patternIndex) {
  const game2 = new Game2(N);
  const cap   = N * N;
  const cells = game2.cells;

  const empty = [];
  for (let i = 0; i < cap; i++) if (cells[i] === 0) empty.push(i);

  const moveLimit = empty.length + 20;
  let moves = 0;

  while (!game2.gameOver && moves < moveLimit) {
    const current = game2.current;

    // Hash every legal non-eye cell on the board at this position.
    let chosen = -1;
    let end = empty.length;
    while (end > 0) {
      const ri  = Math.floor(Math.random() * end);
      const idx = empty[ri];
      empty[ri] = empty[end - 1];
      empty[end - 1] = idx;
      end--;

      if (game2.isTrueEye(idx) || !game2.isLegal(idx)) continue;

      const h = patternHash2(game2, idx, current);
      if (!patternIndex.has(h)) patternIndex.set(h, patternIndex.size);

      if (chosen === -1) chosen = idx;  // play the first legal move found
    }

    if (chosen === -1) { game2.play(PASS); moves++; continue; }

    const { capturedCount } = game2.playInfo(chosen);
    const pos = empty.indexOf(chosen);
    if (pos !== -1) { empty[pos] = empty[empty.length - 1]; empty.pop(); }
    if (capturedCount) {
      empty.length = 0;
      for (let i = 0; i < cap; i++) if (cells[i] === 0) empty.push(i);
    }
    moves++;
  }
}

const patternIndex = new Map();   // hash (uint32) → compact index

const boardN = positions[0].game2.N;
for (let r = 0; r < discoveryPlayouts; r++) discoverPatterns(boardN, patternIndex);

const n = patternIndex.size;
const patternHashes = new Array(n);
for (const [h, i] of patternIndex) patternHashes[i] = h;

process.stderr.write(`Found ${n} unique patterns (after ${discoveryPlayouts} discovery games).\n`);

if (n === 0) {
  process.stderr.write('No patterns found — nothing to optimize.\n');
  process.exit(1);
}

// ── CMA-ES optimization ───────────────────────────────────────────────────────

const cma    = initCMA(n);
const batchN   = batchArg   > 0 ? Math.min(batchArg,   positions.length) : positions.length;
const displayN = displayArg > 0 ? Math.min(displayArg, positions.length) : positions.length;

// Fixed display batch — sampled once so checkpoint values are comparable across iterations.
const displayBatch = Array.from({ length: displayN }, () => positions[Math.floor(Math.random() * positions.length)]);

process.stderr.write(
  `CMA-ES: n=${n}  lambda=${cma.lambda}  mu=${cma.mu}  ` +
  `playouts=${playouts}  bias-weight=${biasWeight}  variance-weight=${varianceWeight}  reg-weight=${regWeight}  batch=${batchN}  display=${displayN}  out=${outFile}\n`
);
process.stderr.write('─'.repeat(62) + '\n');

function saveWeights() {
  const body = patternHashes.map((h, i) => `[${h},${cma.m[i]}]`).join(',');
  fs.writeFileSync(outFile,
    `'use strict';\n` +
    `// Auto-generated by tune-playout.js — do not edit by hand.\n` +
    `const patternTable = new Map([${body}]);\n` +
    `if (typeof module !== 'undefined') module.exports = patternTable;\n` +
    `else window.patternTable = patternTable;\n`
  );
}

let iter = 0;
let nextReport = 1;  // next iteration to report and save (doubles each time)
const startTime = Date.now();

// eslint-disable-next-line no-constant-condition
while (true) {
  iter++;

  // Random batch of positions for this generation.
  const batch = batchN === positions.length
    ? positions
    : Array.from({ length: batchN }, () => positions[Math.floor(Math.random() * positions.length)]);

  // Sample and evaluate lambda candidates.
  const xs = sampleCMA(cma);
  const fitnesses = new Float64Array(cma.lambda);
  for (let k = 0; k < cma.lambda; k++) {
    fitnesses[k] = evalFitness(xs.subarray(k * n, (k + 1) * n), patternIndex, batch);
  }

  // CMA-ES update.
  stepCMA(cma, xs, fitnesses);

  if (iter === nextReport) {
    const { bias2, variance, fitness } = fullEval(cma.m, patternIndex, displayBatch);
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    let maxW = 0;
    for (let j = 0; j < n; j++) { const a = Math.abs(cma.m[j]); if (a > maxW) maxW = a; }
    process.stderr.write(
      `iter ${String(iter).padStart(6)}` +
      `  bias²=${bias2.toFixed(3).padStart(10)}` +
      `  var=${variance.toFixed(3).padStart(10)}` +
      `  fitness=${fitness.toFixed(3).padStart(10)}` +
      `  sigma=${cma.sigma.toFixed(4).padStart(8)}` +
      `  maxW=${maxW.toFixed(3).padStart(7)}` +
      `  elapsed=${String(elapsed + 's').padStart(9)}\n`
    );
    saveWeights();
    nextReport = Math.ceil(nextReport * 1.5);
  }
}
