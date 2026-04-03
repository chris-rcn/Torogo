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
//   Move selection uses absolute V = P(BLACK wins).
//   BLACK maximises V(s'), WHITE minimises V(s')  (full-width single-ply)
//
// Evaluation: play eval games against a configurable reference agent to
//   measure how much the policy has improved.  Eval games do not update weights.
//
// Features: pattern1 + pattern2 + pattern3 (maxLibs = 1), all cells.
//
// Status is printed at an exponentially increasing interval (× 1.5 each time).
//
// Runs indefinitely (Ctrl-C to stop).  Weights are saved at every print.

const path = require('path');
const { Game2, BLACK, PASS } = require('./game2.js');
const { evaluateFeatures, extractFeatures, loadWeights, saveWeights } = require('./vpatterns.js');
const { search } = require('./ai/vpatsearch.js');
const { loadPositions, evalPositionsSample } = require('./evalmovedetails.js');
const { evalValueAccuracy } = require('./eval-value-accuracy.js');
const Util = require('./util.js');
const fs = require('fs');
const Playout = require('./ai/playout.js');

// ── Arguments ─────────────────────────────────────────────────────────────────

const opts       = Util.parseArgs(process.argv.slice(2));
const TRAIN_SIZE = parseInt(opts['train-size']  || opts.size || '9',  10);
const EVAL_SIZE  = parseInt(opts['eval-size']   || opts.size || '9',  10);
const SAVE_PATH  = opts.save  || `out/vpatterns-${Math.random().toString(36).slice(2, 10)}.js`;
const LOAD_PATH  = opts.load  || null;
const EVAL_AGENT = opts.eval  || 'random';
const EPSILON    = parseFloat(opts.epsilon      || '0.1');
const POSITIONS_FILE  = opts['positions-file']   || null;
const POSITIONS_N     = parseInt(opts['positions-n'] || '0', 10);
const ACCURACY_FILE   = opts['accuracy-file']    || null;
const ACCURACY_GAMES  = parseInt(opts['accuracy-games'] || '100', 10);
const LR         = parseFloat(opts['lr']     || '0.3');
const BUDGET     = parseFloat(opts['budget']    || '1');

// ── Features ───────────────────────────

let specs = [
  { size: 1, maxLibs: 1 },
  { size: 2, maxLibs: 1 },
  { size: 3, maxLibs: 1 },
];

// ── Weight table ──────────────────────────────────────────────────────────────

let weights = new Map();  // pattern key (int32) → weight (float)

function addWeight(key, delta) {
  const w = weights.get(key);
  weights.set(key, (w !== undefined ? w : 0) + delta);
}


// ── Training helpers ──────────────────────────────────────────────────────────

// Absolute terminal outcome: 1=BLACK wins, 0=WHITE wins.
function absoluteOutcome(game) {
  return game.calcWinner() === BLACK ? 1 : 0;
}

// Δw_k = (LR / n) · (target − V) · polarity_k
function tdUpdate(features, v, target) {
  const n = features.length;
  if (n === 0) return;
  const step = (LR / n) * (target - v);
  for (const f of features) addWeight(f.key, step * f.polarity);
}

// ── Self-play training ────────────────────────────────────────────────────────

// Both colours use the policy.  Apply 2-step logistic TD inline during play.
// All targets are absolute (P(BLACK wins)).
function trainGame(N) {
  const game     = new Game2(N, false);
  const maxMoves = N * N * 4;
  const tStartMs = Date.now();

  let prev2 = null, prev1 = null;  // feature sets from 2 and 1 steps ago
  let moves = 0;

  while (!game.gameOver && moves < maxMoves) {
    const features = extractFeatures(game, specs);
    const v        = evaluateFeatures(features, weights);

    if (prev2 !== null) {
      const v2 = evaluateFeatures(prev2, weights);
      tdUpdate(prev2, v2, v);
    }

    prev2 = prev1;
    prev1 = features;
    let move;
    if (Math.random() < EPSILON) {
      move = game.randomLegalMove();
      //move = Playout.getMove(game).move;
    } else {
      move = search(game, { weights, specs: specs }, 1);
    }
    game.play(move);
    moves++;
  }

  const elapsedMs = Date.now() - tStartMs;
  const outcome   = absoluteOutcome(game);

  for (const features of [prev2, prev1]) {
    if (features === null) continue;
    const v = evaluateFeatures(features, weights);
    tdUpdate(features, v, outcome);
  }

  return { winner: game.calcWinner(), elapsedMs, moves };
}

// ── Evaluation against a reference agent ─────────────────────────────────────

// Play nGames of policy vs agent, alternating colours.
// Returns { results } where each element is 1 (policy win), 0 (agent win), or 0.5 (draw).
function evalVsReference(N, refGetMove, nGames, budget) {
  const results = [];

  for (let g = 0; g < nGames; g++) {
    const policyIsBlack = (g % 2 === 0);
    const game     = new Game2(N, false);
    const maxMoves = N * N * 4;
    let   moves    = 0;

    while (!game.gameOver && moves++ < maxMoves) {
      let idx;
      if ((game.current === BLACK) === policyIsBlack) {
        idx = search(game, { weights, specs: specs });
      } else {
        const mv = refGetMove(game, budget);
        idx = mv.move !== undefined ? mv.move : PASS;
      }
      game.play(idx);
    }

    const winner = game.calcWinner();
    if (winner === null) {
      results.push(0.5);
    } else if ((winner === BLACK) === policyIsBlack) {
      results.push(1);
    } else {
      results.push(0);
    }
  }

  return { results };
}

// ── CLI ───────────────────────────────────────────────────────────────────────

// Load eval agent from ai/ folder.
const { getMove: evalGetMove } =
  require(path.join(__dirname, 'ai', EVAL_AGENT + '.js'));

// Load positions for move-quality eval (optional).
let evalPositionsPool = null;
if (POSITIONS_FILE) {
  evalPositionsPool = loadPositions(POSITIONS_FILE);
  console.log(`Loaded ${evalPositionsPool.length} positions from ${POSITIONS_FILE}  batch=${POSITIONS_N || 'all'}`);
}

if (LOAD_PATH) {
  if (fs.existsSync(LOAD_PATH)) {
    ({ weights, specs } = loadWeights(LOAD_PATH));
    console.log(`Loaded ${weights.size} weights from ${LOAD_PATH}`);
  } else {
    console.warn(`Warning: --load file not found: ${LOAD_PATH}`);
  }
}


console.log(`LR=${LR}  epsilon=${EPSILON}  train-size=${TRAIN_SIZE}  eval-size=${EVAL_SIZE}  ref=${EVAL_AGENT}`);
console.log(`Out: ${SAVE_PATH}${LOAD_PATH ? `  (resumed from ${LOAD_PATH})` : ''}${evalPositionsPool ? `  positions: ${evalPositionsPool.length} batch=${POSITIONS_N || 'all'}` : ''}`);
console.log(`Specs: ${JSON.stringify(specs)}`);
console.log();

// Print header.
console.log(
  `${'game'.padStart(7)}  ${'elapsedS'.padStart(8)}  ${'weights'.padStart(8)}` +
  `  ${'win%'.padStart(6)}(${'n'.padStart(3)})  ${'winAvg%'.padStart(7)}` +
  `  ${'avglen'.padStart(6)}` +
  (ACCURACY_FILE ? `  ${'vacc%'.padStart(6)}` : '') +
  (evalPositionsPool ? `  ${'rmsErr'.padStart(7)}  ${'rmsAvg'.padStart(7)}` : '') +
  `  ${'avg|w|'.padStart(7)}  ${'rms(w)'.padStart(7)}  ${'max|w|'.padStart(7)}` +
  `  ${'tTrain'.padStart(7)}  ${'tTest'.padStart(6)}  ${'turnMs'.padStart(6)}`
);

const t0 = Date.now();
let nextPrintAt = t0 + 1000;
let g = 0;
let totalMoves = 0;
let intervalGames = 0;
let intervalMoves = 0;
let moveElapsedMs = 0;
let intervalTrainMs = 0;
let refBudgetMs = BUDGET;
const evalHistory = [];   // per-interval game results (1/0.5/0)
const rmsHistory  = [];   // per-interval rmsErr values

while (true) {
  g++;
  const { moves, elapsedMs } = trainGame(TRAIN_SIZE);
  totalMoves += moves;
  intervalGames++;
  intervalMoves += moves;
  moveElapsedMs += elapsedMs;
  intervalTrainMs += elapsedMs;
  const timePerMoveMs = moveElapsedMs / totalMoves;

  if (Date.now() >= nextPrintAt) {
    const tTestStart = Date.now();
    const resultsBatch = [];
    while (true) {
      const { results } = evalVsReference(EVAL_SIZE, evalGetMove, 2, refBudgetMs);
      for (const r of results) resultsBatch.push(r);
      const tTestMs   = Date.now() - tTestStart;
      if (tTestMs > 0.3 * intervalTrainMs) break;
      if (resultsBatch.length >= 500) break;
    }
    for (const r of resultsBatch) evalHistory.push(r);

    // Latest interval win rate.
    const latestWR  = resultsBatch.reduce((s, r) => s + r, 0) / resultsBatch.length;
    const latestPct = (100 * latestWR).toFixed(1);

    // Rolling average over the most recent half of all recorded games.
    const half    = Math.max(1, Math.floor(evalHistory.length / 2));
    const avgWR   = evalHistory.slice(-half).reduce((s, r) => s + r, 0) / half;
    const avgPct  = (100 * avgWR).toFixed(1);

    const avgLen  = (intervalMoves / intervalGames).toFixed(1);
    intervalGames = 0;
    intervalMoves = 0;
    let vaccStr = '';
    if (ACCURACY_FILE) {
      const { accuracy } = evalValueAccuracy(ACCURACY_FILE, { weights, specs }, { nGames: ACCURACY_GAMES });
      vaccStr = `  ${(100 * accuracy).toFixed(1).padStart(5)}%`;
    }
    let rmsStr = '';
    if (evalPositionsPool) {
      const { rmsErr } = evalPositionsSample(game => ({ move: search(game, { weights, specs: specs }) }), evalPositionsPool, POSITIONS_N || evalPositionsPool.length, 0);
      rmsHistory.push(rmsErr);
      const rmsHalf   = Math.max(1, Math.floor(rmsHistory.length / 2));
      const rmsAvg    = rmsHistory.slice(-rmsHalf).reduce((s, r) => s + r, 0) / rmsHalf;
      rmsStr = `  ${rmsErr.toFixed(4).padStart(7)}  ${rmsAvg.toFixed(4).padStart(7)}`;
    }
    let wAbsSum = 0, wAbsMax = 0, wSqSum = 0;
    for (const w of weights.values()) {
      const a = Math.abs(w);
      wAbsSum += a;
      wSqSum  += w * w;
      if (a > wAbsMax) wAbsMax = a;
    }
    const wAvg   = weights.size > 0 ? wAbsSum / weights.size : 0;
    const wRms = weights.size > 0 ? Math.sqrt(wSqSum / weights.size) : 0;
    const wStr = `  ${wAvg.toFixed(3).padStart(7)}  ${wRms.toFixed(3).padStart(7)}  ${wAbsMax.toFixed(3).padStart(7)}`;

    const tTestMs   = Date.now() - tTestStart;
    const tTrainStr = (intervalTrainMs / 1000).toFixed(1) + 's';
    const tTestStr  = (tTestMs / 1000).toFixed(1) + 's';
    intervalTrainMs = 0;
    const nextMs = Date.now() - t0;
    const elapsedS = ((Date.now() - t0) / 1000).toFixed(0);
    console.log(
      `${String(g).padStart(7)}  ${elapsedS.padStart(8)}  ${String(weights.size).padStart(8)}` +
      `  ${(latestPct + '%').padStart(6)}(${String(resultsBatch.length).padStart(3)})  ${(avgPct + '%').padStart(7)}` +
      `  ${avgLen.padStart(6)}${vaccStr}${rmsStr}${wStr}  ${tTrainStr.padStart(7)}  ${tTestStr.padStart(6)}  ${timePerMoveMs.toFixed(1).padStart(6)}`
    );
    saveWeights(SAVE_PATH, { weights, specs: specs });
    nextPrintAt = t0 + Math.round(nextMs * 1.4);
  }
}
