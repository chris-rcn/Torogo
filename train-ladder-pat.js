#!/usr/bin/env node
'use strict';

// train-ladder-pol.js — Learn playout policy weights via REINFORCE self-play.
//
// Combined features: ppat 3×3 patterns + pattern1 ladder features.
// Training: play full self-play games, update weights using policy gradient.
//
// Usage:
//   node train-ladder-pol.js [--lr <f>] [--games <n>] [--size <n>] [--out <path>]

const fs = require('fs');
const { performance } = require('perf_hooks');
const { Game2, BLACK, WHITE, PASS } = require('./game2.js');
const { getAllLadderStatuses } = require('./ladder2.js');
const { makeIntMap } = require('./int-map.js');
const { createLadderPat } = require('./ladder-pat-lib.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help']);
if (opts.help) {
  console.error('Usage: node train-ladder-pol.js [--lr <f>] [--games <n>] [--size <n>] [--out <path>]');
  process.exit(1);
}

const lr           = opts.getFloat('lr', 0.01);
const gameLimit    = opts.getInt('games', 0);  // 0 = infinite
const boardSize    = opts.getInt('size', 9);
const maxAdjLibs       = opts.getInt('max-adj-libs', 3);
const minLadderStones  = opts.getInt('min-ladder-stones', 2);
const maxLadderStones  = opts.getInt('max-ladder-stones', 8);
const loadFile        = opts.load || '';
const refAgent        = opts.ref || 'ai/ref-fast.js';
const evalGames       = opts.getInt('eval-games', 200);
const outFile         = opts.out || `out/ladder-pat-${Date.now().toString(36)}.js`;

const Ref = require('./' + refAgent);

// ── Weight table (hash → weight index) ──────────────────────────────────────

const keyToIdx = makeIntMap();
let weights = [];

function resolveKey(key) {
  let wi = keyToIdx.get(key);
  if (wi < 0) {
    wi = weights.length;
    keyToIdx.set(key, wi);
    weights.push(0);
  }
  return wi;
}

// Resolve candidates from hash keys to weight indices (in-place replacement).
function resolveCandidates(candidates) {
  for (const c of candidates) {
    const keys = c.keys;
    for (let i = 0; i < keys.length; i++) keys[i] = resolveKey(keys[i]);
  }
  return candidates;
}

function weightFn(key) { return weights[resolveKey(key)]; }

// ── Load existing weights ───────────────────────────────────────────────────

if (loadFile) {
  const resolve = require('path').resolve;
  const loaded = require(resolve(loadFile));
  loaded.forEach((v, k) => {
    const wi = resolveKey(k);
    weights[wi] = v;
  });
  console.log(`loaded ${weights.length} weights from ${loadFile}`);
}

// ── Feature extraction ──────────────────────────────────────────────────────

const ladderPat = createLadderPat({ maxAdjLibs, maxLadderStones, minLadderStones });

// ── Softmax + sampling (training uses weight indices) ───────────────────────

function softmaxSampleByIdx(candidates) {
  const n = candidates.length;
  const logits = new Float64Array(n);
  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    let v = 0;
    for (const wi of candidates[i].keys) v += weights[wi];
    logits[i] = v;
    if (v > max) max = v;
  }
  let sum = 0;
  const probs = new Float64Array(n);
  for (let i = 0; i < n; i++) { probs[i] = Math.exp(logits[i] - max); sum += probs[i]; }
  const inv = 1 / sum;
  for (let i = 0; i < n; i++) probs[i] *= inv;

  let r = Math.random(), chosen = n - 1;
  for (let i = 0; i < n; i++) { r -= probs[i]; if (r <= 0) { chosen = i; break; } }

  return { chosen, probs };
}

// ── Play one game, collect trajectory ───────────────────────────────────────

function playGame() {
  const game = new Game2(boardSize);
  const trajectory = []; // { candidates, chosen, probs, player }

  while (!game.gameOver) {
    const candidates = resolveCandidates(ladderPat.getFeatures(game));
    if (candidates.length === 0) {
      game.play(PASS);
      continue;
    }

    const { chosen, probs } = softmaxSampleByIdx(candidates);
    trajectory.push({ candidates, chosen, probs, player: game.current });
    game.play(candidates[chosen].move);
  }

  const winner = game.estimateWinner();
  return { trajectory, winner, moveCount: game.moveCount };
}

// ── Compute gradient for one game ───────────────────────────────────────────

function accumulateGradient(trajectory, winner, grad) {
  for (const { candidates, chosen, probs, player } of trajectory) {
    const z = (winner === player) ? 1 : -1;
    const n = candidates.length;

    // ψ(s,a) = φ(s,a) - Σ_b π(b|s)φ(s,b)
    // grad += z · ψ(s,a)

    // Subtract expected features (weighted by probability)
    for (let i = 0; i < n; i++) {
      const p = probs[i];
      for (const wi of candidates[i].keys) {
        if (grad[wi] === undefined) grad[wi] = 0;
        grad[wi] -= z * p;
      }
    }

    // Add chosen action's features
    for (const wi of candidates[chosen].keys) {
      if (grad[wi] === undefined) grad[wi] = 0;
      grad[wi] += z;
    }
  }
}

// ── Eval: play policy vs reference agent ─────────────────────────────────────

function evalVsRef(nGames) {
  let wins = 0;
  const opp = { ku: 0, su: 0, kw: 0, sw: 0 };
  const prob = { ku: 0, su: 0, kw: 0, sw: 0 };

  const cap = boardSize * boardSize;
  const ku = new Int32Array(cap), su = new Int32Array(cap);
  const kw = new Int32Array(cap), sw = new Int32Array(cap);
  for (let g = 0; g < nGames; g++) {
    const policyColor = (g % 2 === 0) ? BLACK : WHITE;
    const game = new Game2(boardSize);
    while (!game.gameOver) {
      if (game.current === policyColor) {
        const collectLadderStats = maxLadderStones > 0 || Math.random() < 0.1;
        if (collectLadderStats) {
          ku.fill(0); su.fill(0); kw.fill(0); sw.fill(0);
          const statuses = getAllLadderStatuses(game, 1);
          for (const { gid, color, status } of statuses) {
            const cs = game.groupSize(gid);
            const isOpp = color !== game.current;
            if (status.moverSucceeds) {
              const arr = isOpp ? ku : su;
              for (const lib of status.urgentLibs) arr[lib] += cs;
            } else {
              const arr = isOpp ? kw : sw;
              for (const lib of status.libs) arr[lib] += cs;
            }
          }
        }

        const candidates = resolveCandidates(ladderPat.getFeatures(game));
        if (candidates.length === 0) { game.play(PASS); continue; }
        const { chosen, probs } = softmaxSampleByIdx(candidates);

        if (collectLadderStats) {
          // Accumulate probability mass per ladder type
          let hasKu = false, hasSu = false, hasKw = false, hasSw = false;
          for (let i = 0; i < candidates.length; i++) {
            const m = candidates[i].move;
            if (ku[m] > 0) { prob.ku += probs[i]; hasKu = true; }
            if (su[m] > 0) { prob.su += probs[i]; hasSu = true; }
            if (kw[m] > 0) { prob.kw += probs[i]; hasKw = true; }
            if (sw[m] > 0) { prob.sw += probs[i]; hasSw = true; }
          }
          if (hasKu) opp.ku++;
          if (hasSu) opp.su++;
          if (hasKw) opp.kw++;
          if (hasSw) opp.sw++;
        }

        game.play(candidates[chosen].move);
      } else {
        const result = Ref.getMove(game, 0);
        game.play(result.move);
      }
    }
    if (game.estimateWinner() === policyColor) wins++;
  }
  return { winRate: wins / nGames, opp, prob };
}

// ── Save weights ────────────────────────────────────────────────────────────

function saveWeights() {
  const entries = [];
  keyToIdx.forEach((key, wi) => {
    if (weights[wi] !== 0) entries.push(`[${key},${weights[wi]}]`);
  });
  fs.writeFileSync(outFile,
    `'use strict';\n` +
    `// Auto-generated by train-ladder-pol.js — do not edit by hand.\n` +
    `// maxAdjLibs: ${maxAdjLibs}, minLadderStones: ${minLadderStones}, maxLadderStones: ${maxLadderStones}\n` +
    `const ladderPatWeights = new Map([${entries.join(',')}]);\n` +
    `ladderPatWeights.maxAdjLibs = ${maxAdjLibs};\n` +
    `ladderPatWeights.minLadderStones = ${minLadderStones};\n` +
    `ladderPatWeights.maxLadderStones = ${maxLadderStones};\n` +
    `if (typeof module !== 'undefined') module.exports = ladderPatWeights;\n` +
    `else window.ladderPatWeights = ladderPatWeights;\n`
  );
}

// ── Training loop ───────────────────────────────────────────────────────────

const startTimeMs = performance.now();
let totalGames = 0;
let totalMoves = 0;
let totalGameMs = 0;
let wins = [0, 0]; // [black wins, white wins]
let nextPrintMs = performance.now() + 1000;

console.log(`size: ${boardSize}  lr: ${lr}  maxAdjLibs: ${maxAdjLibs}  minLadderStones: ${minLadderStones}  maxLadderStones: ${maxLadderStones}  ref: ${refAgent}  out: ${outFile}`);
function fmtRatio(s, o) { return o > 0 ? (s / o).toFixed(2) : '  - '; }

console.log(`${'games'.padStart(7)}  ${'weights'.padStart(7)}  ${'avg|w|'.padStart(7)}  ${'avgLen'.padStart(6)}  ${'gameMs'.padStart(6)}  ${'vsRef'.padStart(6)}  ${'ku'.padStart(5)}  ${'su'.padStart(5)}  ${'kw'.padStart(5)}  ${'sw'.padStart(5)}  ${'elapsed'.padStart(8)}`);

function printStats() {
  const { winRate, opp, prob } = evalVsRef(evalGames);
  const vsRand = (100 * winRate).toFixed(0) + '%';
  const elapsed = ((performance.now() - startTimeMs) / 1000).toFixed(1) + 's';
  const avgLen = totalGames > 0 ? (totalMoves / totalGames).toFixed(0) : '0';
  const gameMs = totalGames > 0 ? (totalGameMs / totalGames).toFixed(0) : '0';
  let absSum = 0;
  for (let i = 0; i < weights.length; i++) absSum += Math.abs(weights[i]);
  const avgMag = weights.length > 0 ? (absSum / weights.length).toFixed(4) : '0';
  console.log(
    `${String(totalGames).padStart(7)}  ` +
    `${String(weights.length).padStart(7)}  ` +
    `${avgMag.padStart(7)}  ` +
    `${avgLen.padStart(6)}  ` +
    `${gameMs.padStart(6)}  ` +
    `${vsRand.padStart(6)}  ` +
    `${fmtRatio(prob.ku, opp.ku).padStart(5)}  ` +
    `${fmtRatio(prob.su, opp.su).padStart(5)}  ` +
    `${fmtRatio(prob.kw, opp.kw).padStart(5)}  ` +
    `${fmtRatio(prob.sw, opp.sw).padStart(5)}  ` +
    `${elapsed.padStart(8)}`
  );
  nextPrintMs = (performance.now() - startTimeMs) / 1000 * 1.5 + (performance.now() - startTimeMs) / 1000;
  nextPrintMs = nextPrintMs * 500 + performance.now(); // convert back to ms offset
}

while (gameLimit === 0 || totalGames < gameLimit) {
  const t0 = performance.now();
  const grad = {};
  const { trajectory, winner, moveCount } = playGame();
  accumulateGradient(trajectory, winner, grad);
  totalMoves += moveCount;
  totalGames++;
  if (winner === BLACK) wins[0]++;
  else wins[1]++;

  // Apply gradient
  for (const key in grad) {
    const wi = parseInt(key, 10);
    weights[wi] += lr * grad[key];
  }
  totalGameMs += performance.now() - t0;

  const elapsedMs = performance.now() - startTimeMs;
  if (elapsedMs > nextPrintMs) {
    printStats();
    saveWeights();
    nextPrintMs = performance.now() + 0.5 * (performance.now() - startTimeMs);
  }
}

printStats();
saveWeights();
