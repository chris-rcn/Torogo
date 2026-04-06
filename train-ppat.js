#!/usr/bin/env node
'use strict';

// train-ppat.js — Simulation Balancing (Huang, Coulom, Lin 2010, Algorithm 1).
//
// Input: newline-delimited JSON produced by createmovedetails.js.
// Each line: { boardSize, history, candidates: [{m, kwr}] }
//   kwr = win-ratio × 1000; best candidate's kwr/1000 is used as V*(s).
//
// For each training position s₁:
//   V ← average z over M rollouts from s₁     (outcome estimate)
//   g ← average of z·Σ_t ψ(sₜ,aₜ) over N rollouts   (policy gradient)
//   θ ← θ + α·(V*(s₁) − V)·g
//
// ψ(s,a) = φ(s,a) − Σ_b πθ(s,b)φ(s,b)   (score function / log-policy gradient)
// z ∈ {−1, +1}  (loss/win for the player to move at s₁)
// V and g use separate rollouts for independent estimates.
//
// Runs indefinitely; prints a summary line at exponentially spaced intervals
// (factor 1.5).
//
// Usage:
//   node train-ppat.js --file <datafile> [--alpha <f>] [--M <n>] [--N <n>]
//                      [--iteration-limit <n>] [--position-limit <n>]

const fs   = require('fs');
const { performance } = require('perf_hooks');
const { Game2, BLACK, WHITE, PASS, parseMove } = require('./game2.js');
const { createState, extractFeatures, NUM_PATTERNS } = require('./ppat.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help']);
if (opts.help || !opts.file) {
  console.error('Usage: node train-ppat.js --file <datafile> [--alpha <f>] [--M <n>] [--N <n>]');
  console.error('                          [--iteration-limit <n>] [--position-limit <n>]');
  process.exit(1);
}

const dataFile      = opts.file;
const alpha         = opts.alpha            !== undefined ? parseFloat(opts.alpha)                    : 10;
const M             = opts.M                !== undefined ? parseInt(opts.M, 10)                      : 50;
const N             = opts.N                !== undefined ? parseInt(opts.N, 10)                      : M;
const iterLimit     = opts['iteration-limit'] !== undefined ? parseInt(opts['iteration-limit'], 10)   : Infinity;
const positionLimit = opts['position-limit']  !== undefined ? parseInt(opts['position-limit'], 10)    : Infinity;

// ── Parameter vector ──────────────────────────────────────────────────────────
// theta[0..NUM_PATTERNS-1]  : pattern weights
// theta[NUM_PATTERNS..+6]   : prev-move feature weights (bits 0–6)

const TOTAL = NUM_PATTERNS + 7;
const theta  = new Float32Array(TOTAL);

// ── Scratch buffers ───────────────────────────────────────────────────────────

const rolloutFeatSt  = createState(19);
const rolloutGradBuf = new Float32Array(TOTAL);  // per-rollout gradient accumulator
const gBuf           = new Float32Array(TOTAL);  // g across N rollouts

let rolloutLogits = new Float32Array(512);
let rolloutProbs  = new Float32Array(512);

// ── Helpers ───────────────────────────────────────────────────────────────────

function logitOf(featState, i) {
  let v = theta[featState.patIds[i]];
  let m = featState.prevMasks[i];
  for (let b = 0; m; b++, m >>= 1) if (m & 1) v += theta[NUM_PATTERNS + b];
  return v;
}

function softmax(src, dst, n) {
  let max = src[0];
  for (let i = 1; i < n; i++) if (src[i] > max) max = src[i];
  let sum = 0;
  for (let i = 0; i < n; i++) { dst[i] = Math.exp(src[i] - max); sum += dst[i]; }
  const inv = 1 / sum;
  for (let i = 0; i < n; i++) dst[i] *= inv;
}

// ── Rollout ───────────────────────────────────────────────────────────────────
//
// Simulates one complete game from `game` under πθ.
// If `gradAcc` is non-null, accumulates Σ_t ψ(sₜ,aₜ) into it (in-place).
// Returns z ∈ {−1, +1} from `player`'s perspective.

function rollout(game, player, gradAcc) {
  const sim = game.clone();

  while (!sim.gameOver) {
    extractFeatures(sim, rolloutFeatSt);
    const n = rolloutFeatSt.count;

    if (n === 0) { sim.play(PASS); continue; }

    if (rolloutLogits.length < n) {
      rolloutLogits = new Float32Array(n * 2);
      rolloutProbs  = new Float32Array(n * 2);
    }

    for (let i = 0; i < n; i++) rolloutLogits[i] = logitOf(rolloutFeatSt, i);
    softmax(rolloutLogits, rolloutProbs, n);

    // Sample action.
    let r = Math.random(), chosen = n - 1;
    for (let i = 0; i < n; i++) { r -= rolloutProbs[i]; if (r <= 0) { chosen = i; break; } }

    // ψ(s,a) = φ(s,a) − Σ_b π(b|s)φ(s,b)
    if (gradAcc !== null) {
      for (let i = 0; i < n; i++) {
        const p = rolloutProbs[i];
        gradAcc[rolloutFeatSt.patIds[i]] -= p;
        let m = rolloutFeatSt.prevMasks[i];
        for (let b = 0; m; b++, m >>= 1) if (m & 1) gradAcc[NUM_PATTERNS + b] -= p;
      }
      gradAcc[rolloutFeatSt.patIds[chosen]] += 1;
      let m = rolloutFeatSt.prevMasks[chosen];
      for (let b = 0; m; b++, m >>= 1) if (m & 1) gradAcc[NUM_PATTERNS + b] += 1;
    }

    sim.play(rolloutFeatSt.moves[chosen]);
  }

  const winner = sim.estimateWinner();
  return winner === player ? 1 : -1;
}

// ── Core update (Algorithm 1) ─────────────────────────────────────────────────

function updateTheta(game, vStar) {
  const player = game.current;

  // ── V: M rollouts, no gradient tracking ─────────────────────────────────────
  let V = 0;
  for (let i = 0; i < M; i++) V += rollout(game, player, null);
  V /= M;

  // ── g: N rollouts with gradient tracking ─────────────────────────────────────
  gBuf.fill(0);
  for (let j = 0; j < N; j++) {
    rolloutGradBuf.fill(0);
    const z = rollout(game, player, rolloutGradBuf);
    const scale = z / N;
    for (let k = 0; k < TOTAL; k++) gBuf[k] += scale * rolloutGradBuf[k];
  }

  // ── θ ← θ + α(V* − V)·g ─────────────────────────────────────────────────────
  const bias = vStar - V;
  const ascale = alpha * bias;
  for (let k = 0; k < TOTAL; k++) theta[k] += ascale * gBuf[k];
  return bias;
}

// ── Load data ─────────────────────────────────────────────────────────────────

const lines      = fs.readFileSync(dataFile, 'utf8').split('\n').filter(l => l.trim()).slice(0, positionLimit);
const weightsFile = dataFile.replace(/(\.[^.]+)?$/, '-weights.js');

// ── Main loop ─────────────────────────────────────────────────────────────────

const startTime = performance.now();
let totalPositions = 0;
let iterations     = 0;
let nextPrint      = 1;
let passDeltaSum   = 0;
let passDeltaCount = 0;

console.log(`${'iters'.padStart(6)}  ${'positions'.padStart(9)}  ${'|ΔV|'.padStart(6)}  ${'elapsed'.padStart(8)}`);

function printStats() {
  const elapsed   = ((performance.now() - startTime) / 1000).toFixed(1) + 's';
  const meanDelta = passDeltaCount > 0 ? (passDeltaSum / passDeltaCount).toFixed(3) : '  n/a';
  passDeltaSum   = 0;
  passDeltaCount = 0;
  const featWeights = Array.from(theta.subarray(NUM_PATTERNS, NUM_PATTERNS + 7))
    .map(w => w.toFixed(3)).join('  ');
  console.log(
    `${String(iterations).padStart(6)}  ` +
    `${String(totalPositions).padStart(9)}  ` +
    `${String(meanDelta).padStart(6)}  ` +
    `${elapsed.padStart(8)}  ` +
    `[${featWeights}]`
  );

  const patArr  = JSON.stringify(Array.from(theta.subarray(0, NUM_PATTERNS)));
  const prevArr = JSON.stringify(Array.from(theta.subarray(NUM_PATTERNS, NUM_PATTERNS + 7)));
  const js = `'use strict';
// Generated by train-ppat.js — iterations: ${iterations}, positions: ${totalPositions}, elapsed: ${elapsed}
const _w = { pat: new Float32Array(${patArr}), prev: new Float32Array(${prevArr}) };
if (typeof module !== 'undefined') module.exports = _w;
else window.PPATWeights = _w;
`;
  fs.writeFileSync(weightsFile, js);
}

while (true) {
  for (const line of lines) {
    const { boardSize: N, history, candidates } = JSON.parse(line);

    const game = new Game2(N);
    let ok = true;
    for (const c of history) {
      if (!game.play(parseMove(c, N))) { ok = false; break; }
    }
    if (!ok || game.gameOver) continue;

    // V*(s) = value of the best candidate (pre-sorted descending by kwr).
    const best = candidates.find(c => c.kwr !== null);
    if (!best) continue;
    const vStar = best.kwr / 1000;

    const delta = updateTheta(game, vStar);
    passDeltaSum   += Math.abs(delta);
    passDeltaCount++;
    totalPositions++;
  }

  iterations++;
  if (iterations >= iterLimit) { printStats(); break; }
  if (iterations >= nextPrint) {
    printStats();
    nextPrint = Math.ceil(nextPrint * 1.5);
  }
}
