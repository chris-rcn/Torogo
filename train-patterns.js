#!/usr/bin/env node
'use strict';

// train-patterns.js — learn pattern weights via logistic TD(2) self-play.
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
//   Both PASS states at game end share the same board and receive the same
//   target, so they reinforce each other rather than cancelling.
//
// Training: pure self-play — both colours use the pattern policy.
//   BLACK: maximise V(s')   WHITE: minimise V(s')  (full-width single-ply)
//
// Evaluation: play eval games against a configurable reference agent to
//   measure how much the policy has improved.  Eval games do not update weights.
//
// Features: pattern1 + pattern2 + pattern3 (maxLibs = 1), all cells.
//
// Status is printed at an exponentially increasing interval (× 1.5 each time).
//
// Usage:
//   node train-patterns.js [--size 9] [--save weights.json]
//                          [--load weights.json] [--eval random]
//                          [--eval-games 40]
//
// Runs indefinitely (Ctrl-C to stop).  Weights are saved at every print.

const path = require('path');
const { Game2, BLACK, WHITE, PASS } = require('./game2.js');
const { pattern1, pattern2, pattern3 } = require('./patterns.js');
const fs = require('fs');

// ── Hyperparameters ───────────────────────────────────────────────────────────

const LR       = 0.3;   // learning rate
const MAX_LIBS = 1;     // liberty cap (1 = stone presence only, no liberty count)

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

// ── Policy move selection (full-width single-ply) ─────────────────────────────

// Uses absolute V = P(BLACK wins):
//   BLACK to move: maximise V(s')   WHITE to move: minimise V(s')
function pickMovePolicy(game) {
  const cap     = game.N * game.N;
  const isBlack = game.current === BLACK;

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

  if (isBetter(scoreMove(PASS))) bestIdx = PASS;

  return bestIdx;
}

// ── Self-play training game ───────────────────────────────────────────────────

// Both colours use the policy.  Apply 2-step logistic TD after the game.
// All targets are absolute (P(BLACK wins)).
function trainGame(N) {
  const game     = new Game2(N, false);
  const maxMoves = N * N * 4;
  const traj     = [];   // [{features, v}]

  while (!game.gameOver && traj.length < maxMoves) {
    const features = extractFeatures(game);
    const v        = valueOf(features);
    traj.push({ features, v });
    game.play(pickMovePolicy(game));
  }

  const T       = traj.length;
  const outcome = absoluteOutcome(game);

  for (let t = 0; t < T; t++) {
    const { features, v } = traj[t];
    const target = (t + 2 < T) ? traj[t + 2].v : outcome;
    tdUpdate(features, v, target);
  }

  return game.calcWinner();
}

// ── Evaluation against a reference agent ─────────────────────────────────────

// Play nGames of policy vs agent, alternating colours.
// Returns { policyWins, agentWins, draws, winRate }.
function evalVsAgent(N, agentGetMove, nGames) {
  let policyWins = 0, agentWins = 0, draws = 0;

  for (let g = 0; g < nGames; g++) {
    const policyIsBlack = (g % 2 === 0);
    const game     = new Game2(N, false);
    const maxMoves = N * N * 4;
    let   moves    = 0;

    while (!game.gameOver && moves++ < maxMoves) {
      let idx;
      if ((game.current === BLACK) === policyIsBlack) {
        idx = pickMovePolicy(game);
      } else {
        const mv = agentGetMove(game, 0);
        idx = mv.move !== undefined ? mv.move : PASS;
      }
      game.play(idx);
    }

    const winner = game.calcWinner();
    if (winner === null) {
      draws++;
    } else if ((winner === BLACK) === policyIsBlack) {
      policyWins++;
    } else {
      agentWins++;
    }
  }

  return { policyWins, agentWins, draws, winRate: policyWins / nGames };
}

// ── Persistence ───────────────────────────────────────────────────────────────

function saveWeights(savePath) {
  const obj = {};
  for (const [k, v] of weights) obj[String(k)] = v;
  fs.writeFileSync(savePath, JSON.stringify(obj));
}

function loadWeights(loadPath) {
  const raw = JSON.parse(fs.readFileSync(loadPath, 'utf8'));
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

const opts        = parseArgs();
const SIZE        = parseInt(opts.size          || '9',      10);
const SAVE_PATH   = opts.save                   || 'pattern-weights.json';
const LOAD_PATH   = opts.load                   || null;
const EVAL_AGENT  = opts.eval                   || 'random';
const EVAL_GAMES  = parseInt(opts['eval-games'] || '40',     10);

// Load eval agent from ai/ folder.
const { getMove: evalGetMove } =
  require(path.join(__dirname, 'ai', EVAL_AGENT + '.js'));

if (LOAD_PATH) {
  if (fs.existsSync(LOAD_PATH)) {
    loadWeights(LOAD_PATH);
    console.log(`Loaded ${weights.size} weights from ${LOAD_PATH}`);
  } else {
    console.warn(`Warning: --load file not found: ${LOAD_PATH}`);
  }
}


console.log(`Training: size=${SIZE}  lr=${LR}  maxLibs=${MAX_LIBS}  (runs forever, Ctrl-C to stop)`);
console.log(`Self-play training, eval vs ${EVAL_AGENT} every (×1.5) games`);
console.log(`Save → ${SAVE_PATH}${LOAD_PATH ? `  (resumed from ${LOAD_PATH})` : ''}`);
console.log();

// Print header.
console.log(
  `${'game'.padStart(7)}  ${'weights'.padStart(8)}` +
  `  ${'policy%'.padStart(7)}  ${'elapsed'.padStart(8)}`
);

const t0 = Date.now();
let nextPrint = 1;
let g = 0;

// Runs indefinitely; Ctrl-C (SIGINT default) kills the process immediately.
while (true) {
  g++;
  trainGame(SIZE);

  if (g >= nextPrint) {
    const { winRate } = evalVsAgent(SIZE, evalGetMove, EVAL_GAMES);
    const pct     = (100 * winRate).toFixed(1);
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1) + 's';
    console.log(
      `${String(g).padStart(7)}  ${String(weights.size).padStart(8)}` +
      `  ${(pct + '%').padStart(7)}  ${elapsed.padStart(8)}`
    );
    saveWeights(SAVE_PATH);
    nextPrint = Math.max(nextPrint + 1, Math.round(nextPrint * 1.5));
  }
}
