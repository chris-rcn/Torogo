#!/usr/bin/env node
'use strict';

// train-patterns.js — learn pattern weights via logistic TD(2).
//
// Value function (absolute, P(BLACK wins)):
//   V(s) = σ( Σ  polarity_i · w[key_i] )
//
// Update rule — logistic TD, 2-step lookahead (same player to move):
//   Δw_k = (LR / n_features) · (target − V) · V·(1−V) · polarity_k
//   target = V(s_{t+2})   for t+2 < game length  [bootstrap]
//          = 1 / 0.5 / 0  for terminal            [BLACK wins / draw / WHITE wins]
//
//   Targets are absolute (P(BLACK wins)), independent of current player.
//   This keeps terminal updates consistent when both final PASS states
//   share the same board position — they receive the same target and
//   reinforce each other rather than cancelling.
//
// Move selection — full-width single-ply search:
//   Policy player: BLACK maximises V(s'), WHITE minimises V(s').
//   Reference opponent: uniform-random legal non-eye move.
//
// Features: pattern1 + pattern2 + pattern3 (maxLibs = 1), all cells.
//
// Usage:
//   node train-patterns.js [--size 9] [--games 2000] [--save weights.json]
//                          [--load weights.json]  [--log 200]

const { Game2, BLACK, WHITE, PASS } = require('./game2.js');
const { pattern1, pattern2, pattern3 } = require('./patterns.js');
const fs = require('fs');

// ── Hyperparameters ───────────────────────────────────────────────────────────

const LR       = 0.3;   // learning rate
const MAX_LIBS = 1;     // liberty cap (1 = presence/absence only, no liberty count)

// ── Weight table ──────────────────────────────────────────────────────────────

const weights = new Map();  // pattern key (int32) → weight (float)

function getW(key) {
  const w = weights.get(key);
  return w !== undefined ? w : 0;
}

function addW(key, delta) {
  weights.set(key, getW(key) + delta);
}

// ── Feature extraction ────────────────────────────────────────────────────────

// Returns [{key, polarity}] for every non-empty pattern on the board.
// Each cell anchors pattern1 (1×1), pattern2 (2×2 TL), pattern3 (3×3 TL).
function extractFeatures(game) {
  const cap = game.N * game.N;
  const out = [];
  for (let i = 0; i < cap; i++) {
    let f;
    f = pattern1(game, MAX_LIBS, i); if (f !== null) out.push(f);
    f = pattern2(game, MAX_LIBS, i); if (f !== null) out.push(f);
    f = pattern3(game, MAX_LIBS, i); if (f !== null) out.push(f);
  }
  return out;
}

// ── Value function ────────────────────────────────────────────────────────────

// V(s) = σ(Σ polarity_i · w[key_i])  = P(BLACK wins)
function valueOf(features) {
  let z = 0;
  for (const f of features) z += f.polarity * getW(f.key);
  return 1 / (1 + Math.exp(-z));
}

// Absolute terminal outcome: 1=BLACK wins, 0.5=draw, 0=WHITE wins.
function absoluteOutcome(game) {
  const w = game.calcWinner();
  return w === BLACK ? 1 : w === WHITE ? 0 : 0.5;
}

// ── Logistic TD update ────────────────────────────────────────────────────────

// Δw_k = (LR / n) · (target − V) · V(1−V) · polarity_k
function tdUpdate(features, v, target) {
  const n = features.length;
  if (n === 0) return;
  const step = (LR / n) * (target - v) * v * (1 - v);
  for (const f of features) addW(f.key, step * f.polarity);
}

// ── Move selection ────────────────────────────────────────────────────────────

// Full-width single-ply search using absolute V = P(BLACK wins).
//   BLACK to move: maximise V(s') over all resulting states.
//   WHITE to move: minimise V(s') over all resulting states.
function pickMovePolicy(game) {
  const cap     = game.N * game.N;
  const isBlack = game.current === BLACK;

  // Evaluate a candidate move and return the resulting absolute value.
  function scoreMove(idx) {
    const g = game.clone();
    g.play(idx);
    return g.gameOver ? absoluteOutcome(g) : valueOf(extractFeatures(g));
  }

  let bestIdx   = PASS;
  let bestScore = isBlack ? -Infinity : Infinity;

  function isBetter(s) { return isBlack ? s > bestScore : s < bestScore; }

  for (let i = 0; i < cap; i++) {
    if (game.cells[i] !== 0) continue;
    if (game.isTrueEye(i))   continue;
    if (!game.isLegal(i))    continue;
    const s = scoreMove(i);
    if (isBetter(s)) { bestScore = s; bestIdx = i; }
  }

  // Consider PASS: override bestIdx only when PASS is strictly better.
  if (isBetter(scoreMove(PASS))) bestIdx = PASS;

  return bestIdx;
}

// Uniform-random reference opponent (non-eye legal moves or PASS).
function pickMoveRandom(game) {
  const m = game.randomLegalMove();
  return m !== undefined ? m : PASS;
}

// ── Play one game ─────────────────────────────────────────────────────────────

// policyIsBlack: true  → BLACK uses policy, WHITE uses random.
//                false → WHITE uses policy, BLACK uses random.
// Returns trajectory [{features, v}] and the winner.
function playGame(N, policyIsBlack) {
  const game     = new Game2(N, false);   // empty board, BLACK moves first
  const maxMoves = N * N * 4;            // safety cap
  const traj     = [];                   // [{features, v}]

  while (!game.gameOver && traj.length < maxMoves) {
    const features = extractFeatures(game);
    const v        = valueOf(features);
    traj.push({ features, v });

    const usePolicy = (game.current === BLACK) === policyIsBlack;
    game.play(usePolicy ? pickMovePolicy(game) : pickMoveRandom(game));
  }

  return { traj, winner: game.calcWinner() };
}

// ── Train on one game ─────────────────────────────────────────────────────────

// Apply 2-step logistic TD updates.  All targets are absolute (P(BLACK wins)).
// V values in traj[] were computed with weights frozen at game start (batch TD).
function trainGame(N, policyIsBlack) {
  const { traj, winner } = playGame(N, policyIsBlack);
  const T = traj.length;

  // Absolute terminal outcome.
  const outcome = winner === BLACK ? 1 : winner === WHITE ? 0 : 0.5;

  for (let t = 0; t < T; t++) {
    const { features, v } = traj[t];

    // Bootstrap 2 steps ahead (same-player parity preserved); fall back to
    // actual outcome when fewer than 2 steps remain.
    const target = (t + 2 < T) ? traj[t + 2].v : outcome;

    tdUpdate(features, v, target);
  }

  return winner;
}

// ── Persistence ───────────────────────────────────────────────────────────────

function saveWeights(path) {
  const obj = {};
  for (const [k, v] of weights) obj[String(k)] = v;
  fs.writeFileSync(path, JSON.stringify(obj));
}

function loadWeights(path) {
  const raw = JSON.parse(fs.readFileSync(path, 'utf8'));
  for (const [k, v] of Object.entries(raw)) weights.set(Number(k), v);
}

// ── CLI ───────────────────────────────────────────────────────────────────────

function parseArgs() {
  const opts = {};
  for (let i = 2; i < process.argv.length; i++) {
    const a = process.argv[i];
    if (a.startsWith('--')) {
      const eq = a.indexOf('=');
      if (eq === -1) { opts[a.slice(2)] = process.argv[++i]; }
      else           { opts[a.slice(2, eq)] = a.slice(eq + 1); }
    }
  }
  return opts;
}

const opts      = parseArgs();
const SIZE      = parseInt(opts.size  || '9',    10);
const N_GAMES   = parseInt(opts.games || '2000', 10);
const SAVE_PATH = opts.save || 'pattern-weights.json';
const LOAD_PATH = opts.load || null;
const LOG_EVERY = parseInt(opts.log   || '200',  10);

if (LOAD_PATH) {
  if (fs.existsSync(LOAD_PATH)) {
    loadWeights(LOAD_PATH);
    console.log(`Loaded ${weights.size} weights from ${LOAD_PATH}`);
  } else {
    console.warn(`Warning: --load file not found: ${LOAD_PATH}`);
  }
}

process.on('SIGINT', () => {
  saveWeights(SAVE_PATH);
  console.log(`\nInterrupted — saved ${weights.size} weights to ${SAVE_PATH}`);
  process.exit(0);
});

console.log(`Training: size=${SIZE}  games=${N_GAMES}  lr=${LR}  maxLibs=${MAX_LIBS}`);
console.log(`Opponents: policy (single-ply) vs random, alternating colour each game`);
console.log(`Save → ${SAVE_PATH}${LOAD_PATH ? `  (resumed from ${LOAD_PATH})` : ''}`);
console.log();

let blackWins = 0, whiteWins = 0, draws = 0;
const t0 = Date.now();

for (let g = 1; g <= N_GAMES; g++) {
  // Alternate which colour is the policy player each game.
  const policyIsBlack = (g % 2 === 1);
  const winner = trainGame(SIZE, policyIsBlack);

  if (winner === BLACK)      blackWins++;
  else if (winner === WHITE) whiteWins++;
  else                       draws++;

  if (g % LOG_EVERY === 0 || g === N_GAMES) {
    const total = blackWins + whiteWins + draws;
    const bPct  = total ? (100 * blackWins / total).toFixed(1) : '—';
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    console.log(
      `game ${String(g).padStart(6)} / ${N_GAMES}` +
      `  weights=${weights.size}` +
      `  B%=${bPct.padStart(5)}` +
      `  elapsed=${elapsed}s`
    );
    blackWins = whiteWins = draws = 0;
  }
}

saveWeights(SAVE_PATH);
console.log(`\nDone. Saved ${weights.size} pattern weights → ${SAVE_PATH}`);
