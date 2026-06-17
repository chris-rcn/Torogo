#!/usr/bin/env node
'use strict';

// train-patterns.js — learn pattern weights via TD(λ) self-play.
//
// Value function (absolute, P(BLACK wins)):
//   V(s) = σ( Σ  polarity_i · w[key_i] )
//
// Update rule — λ-return TD applied at episode end, 2-ply lookahead
// (bootstraps from the next position where the same player moves):
//   G_t^λ      = (1−λ)·V(s_{t+2}) + λ·G_{t+2}^λ      (recursive form)
//   G_{M-1}^λ  = G_{M-2}^λ = outcome ∈ {1, 0.5, 0}   (terminal, M = #moves)
//   Δw_k       = (LR / n_features) · (G_t^λ − V_t) · polarity_k
//
//   Targets are absolute (P(BLACK wins)), independent of current player.
//   The two parity classes (even-t and odd-t) form independent chains.
//   λ = 0  → pure 2-step TD (target = V(s_{t+2}))
//   λ = 1  → pure Monte Carlo (target = outcome)
//   0<λ<1  → exponentially-weighted bootstrap trading bias for variance
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
// Status is printed at an exponentially increasing interval (× 1.4 each time),
// capped so the gap between prints never exceeds 6 hours.
//
// Runs indefinitely (Ctrl-C to stop).  Weights are saved at every print.

const path = require('path');
const { Game2, BLACK, PASS } = require('./game2.js');
const { evaluateFeatures, extractFeatures, prepareSpecs, loadWeights, saveWeights } = require('./vpatterns.js');
const { search } = require('./ai/vpatsearch.js');
const { loadPositions, evalPositionsSample } = require('./evalmovedetails.js');
const { evalValueAccuracy } = require('./eval-value-accuracy.js');
const Util = require('./util.js');
const fs = require('fs');

// ── Arguments ─────────────────────────────────────────────────────────────────

const opts       = Util.parseArgs(process.argv.slice(2));
const TRAIN_SIZE = parseInt(opts['train-size']  || opts.size || '9',  10);
const EVAL_SIZE  = parseInt(opts['eval-size']   || opts.size || '13', 10);
const SAVE_PATH  = opts.save  || `out/vpatterns-${Math.random().toString(36).slice(2, 10)}.js`;
const LOAD_PATH  = opts.load  || null;
const EVAL_AGENT = opts.eval  || '';     // empty disables in-training reference test games
const EXT_AGENT  = opts.ext   || '';     // off-policy move source: (1-epsilon) fraction of moves come from this agent
const LIMIT_GAMES = opts.limit !== undefined ? parseInt(opts.limit, 10) : 0;
const EPSILON    = parseFloat(opts.epsilon      || '0.1');
const ON_POLICY  = parseFloat(opts['on-policy'] || '0');   // share of non-random moves from own search1ply (vs --ext)
const POSITIONS_FILE  = opts['positions-file']   || null;
const POSITIONS_N     = parseInt(opts['positions-n'] || '0', 10);
const ACCURACY_FILE   = opts['accuracy-file']    || null;
const ACCURACY_GAMES  = parseInt(opts['accuracy-games'] || '100', 10);
const LR         = parseFloat(opts['lr']       || '0.3');
const BUDGET     = parseFloat(opts['budget']   || '1');

// ── Features ───────────────────────────

let specs = [
  { size: 1, maxLibs: 6 },
  { size: 2, maxLibs: 6 },
  { size: 3, maxLibs: 6 },
];
let prepSpecs = prepareSpecs(specs);

// ── Weight table ──────────────────────────────────────────────────────────────

let weights  = new Map();  // pattern key (int32) → weight (float)
let wAbsSum = 0, wUpdateCount = 0;  // running |weight| sum/count over every feature update this run

// ── Training helpers ──────────────────────────────────────────────────────────

// Absolute terminal outcome: 1=BLACK wins, 0=WHITE wins.
function absoluteOutcome(game) {
//  return game.estimateWinner() === BLACK ? 1 : 0;
  return game.calcWinner() === BLACK ? 1 : 0;
}

// Plain SGD:
//   g_k  = (target − V) / n · polarity_k
//   Δw_k = lr · g_k
function tdUpdate(features, target, lr) {
  const n = features.count;
  if (n === 0) return;
  const step = lr * (target - features.val) / n;
  const { keys, pols } = features;
  for (let i = 0; i < n; i++) {
    const k = keys[i];
    const w = (weights.get(k) ?? 0) + pols[i] * step;
    weights.set(k, w);
    wAbsSum += Math.abs(w);
    wUpdateCount++;
  }
}

// ── Self-play training ────────────────────────────────────────────────────────

// Custom 1-ply search which bypasses non-capture moves.
function search1ply(game) {
  const area = game.N * game.N;
  const isBlack = game.current === BLACK;
  let bestMove = PASS;
  let bestScore = isBlack ? -Infinity : Infinity;
  for (let coord = 0; coord < area; coord++) {
    if (!game.isLegal(coord) || game.isTrueEye(coord)) continue;
//    const g = game.clone();
//    g.play(coord);
//    const features = extractFeatures(g, specs);
    const features = extractFeatures(game, prepSpecs, true, coord);
    evaluateFeatures(features, weights);
    if (isBlack === (features.val > bestScore)) { 
      bestScore = features.val;
      bestMove = coord;
    }
  }
  if (bestMove === PASS) {
    return PASS;
  }
  if (game.consecutivePasses > 0 || game.emptyCount < area/2) {
    const passFeatures = extractFeatures(game, prepSpecs);
    evaluateFeatures(passFeatures, weights);
    if (isBlack === (passFeatures.val > bestScore)) {
      bestScore = passFeatures.val;
      bestMove = PASS;
    }
  }
  return bestMove;
}

// Both colours use the policy.  Per-position features and values are collected
// during play; at episode end the λ-return target is computed by a single
// backward pass and applied to each position.
function trainGame(N) {
  const game     = new Game2(N, false);
  const maxMoves = N * N * 4;
  const tStartMs = Date.now();

  let moves = 0;
  const featsArr = [];
  const vals = [];

  while (!game.gameOver && moves < maxMoves) {
    const features = extractFeatures(game, prepSpecs);
    evaluateFeatures(features, weights);
    featsArr.push(features);
    vals.push(features.val);

    let move;
    if (Math.random() < EPSILON) {
      move = game.randomLegalMove();
    } else if (extGetMove && Math.random() > ON_POLICY) {
      move = extGetMove(game).move;
    } else {
      move = search1ply(game);
    }
    game.play(move);
    moves++;
  }

  const elapsedMs = Date.now() - tStartMs;
  const outcome   = absoluteOutcome(game);

  // TD(0) backward pass with 2-ply lookahead.  Each parity class (even-t and
  // odd-t) is its own chain — same-player moves are 2 apart.  The target for
  // step t is the value of the next same-parity step (V_{t+2}); base cases
  // G_{M-1} = G_{M-2} = outcome.
  let G0 = outcome, G1 = outcome;
  for (let t = featsArr.length - 1; t >= 0; t--) {
    if ((t & 1) === 0) {
      tdUpdate(featsArr[t], G0, LR);
      G0 = vals[t];
    } else {
      tdUpdate(featsArr[t], G1, LR);
      G1 = vals[t];
    }
  }

  let correct = 0;
  for (const v of vals) {
    if ((v >= 0.5) === (outcome === 1)) correct++;
  }

  return { winner: game.estimateWinner(), elapsedMs, moves, correct, nVals: vals.length };
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
        idx = search(game, { weights, specs, preparedSpecs: prepSpecs });
      } else {
        const mv = refGetMove(game, budget);
        idx = mv.move !== undefined ? mv.move : PASS;
      }
      if (!game.play(idx)) {
        console.log("Illegal move!");
      }
    }

    const winner = game.calcWinner();
    if ((winner === BLACK) === policyIsBlack) {
      results.push(1);
    } else {
      results.push(0);
    }
  }

  return { results };
}

// ── CLI ───────────────────────────────────────────────────────────────────────

// Load eval agent from ai/ folder (only when --eval was supplied).
const evalGetMove = EVAL_AGENT
  ? require(path.join(__dirname, 'ai', EVAL_AGENT + '.js')).getMove
  : null;

// Load off-policy move-source agent (only when --ext was supplied).
const extGetMove = EXT_AGENT
  ? require(path.join(__dirname, 'ai', EXT_AGENT + '.js')).getMove
  : null;

// Load positions for move-quality eval (optional).
let evalPositionsPool = null;
if (POSITIONS_FILE) {
  evalPositionsPool = loadPositions(POSITIONS_FILE);
  console.log(`Loaded ${evalPositionsPool.length} positions from ${POSITIONS_FILE}  batch=${POSITIONS_N || 'all'}`);
}

if (LOAD_PATH) {
  if (fs.existsSync(LOAD_PATH)) {
    ({ weights, specs, preparedSpecs: prepSpecs } = loadWeights(LOAD_PATH));
    console.log(`Loaded ${weights.size} weights from ${LOAD_PATH}`);
  } else {
    console.warn(`Warning: --load file not found: ${LOAD_PATH}`);
  }
}


console.log(`LR=${LR}  epsilon=${EPSILON}  on-policy=${ON_POLICY}  train-size=${TRAIN_SIZE}  eval-size=${EVAL_SIZE}  ref=${EVAL_AGENT || '(none)'}  ext=${EXT_AGENT || '(none)'}`);
console.log(`Out: ${SAVE_PATH}${LOAD_PATH ? `  (resumed from ${LOAD_PATH})` : ''}${evalPositionsPool ? `  positions: ${evalPositionsPool.length} batch=${POSITIONS_N || 'all'}` : ''}`);
console.log(`Specs: ${JSON.stringify(specs)}`);
console.log();

// Print header.
console.log([
  'game'.padStart(4),
  'tElp'.padStart(5),
  'tGm '.padStart(5),
  'nWts'.padStart(4),
  ...(evalGetMove ? ['gRef'.padStart(4), 'wrRf'.padStart(4), 'wrAv'.padStart(4)] : []),
  'avgL'.padStart(4),
  ' acc'.padStart(4),
  ...(ACCURACY_FILE     ? ['vacc'.padStart(4)] : []),
  ...(evalPositionsPool ? ['rms '.padStart(4), 'rAvg'.padStart(4)] : []),
  'avgW'.padStart(6),
  'tTrn'.padStart(5),
  ...(evalGetMove ? ['tTst'.padStart(5)] : []),
  'turn'.padStart(5),
].join('  '));

const t0 = Date.now();
const MAX_PRINT_INTERVAL_MS = 6 * 60 * 60 * 1000;  // cap status-print gap at 6 hours
let nextPrintAt = t0 + 1000;
let g = 0;
let totalMoves = 0;
let intervalGames = 0;
let intervalMoves = 0;
let intervalCorrect = 0, intervalNVals = 0;
let totalCorrect = 0, totalNVals = 0;
let moveElapsedMs = 0;
let intervalTrainMs = 0;
let refBudgetMs = BUDGET;
const evalHistory = [];   // per-interval game results (1/0.5/0)
const rmsHistory  = [];   // per-interval rmsErr values

while (true) {
  g++;
  const { moves, elapsedMs, correct, nVals } = trainGame(TRAIN_SIZE);
  totalMoves += moves;
  intervalGames++;
  intervalMoves += moves;
  intervalCorrect += correct;
  intervalNVals   += nVals;
  totalCorrect    += correct;
  totalNVals      += nVals;
  moveElapsedMs += elapsedMs;
  intervalTrainMs += elapsedMs;
  const timePerMoveMs = moveElapsedMs / totalMoves;

  // Force a final stats row when the game limit is reached.
  if (LIMIT_GAMES > 0 && g >= LIMIT_GAMES) nextPrintAt = 0;

  if (Date.now() >= nextPrintAt) {
    const tTestStart = Date.now();
    let latestWR = null, avgWR = null, resultsBatchLen = 0;
    if (evalGetMove) {
      const resultsBatch = [];
      while (true) {
        const { results } = evalVsReference(EVAL_SIZE, evalGetMove, 2, refBudgetMs);
        for (const r of results) resultsBatch.push(r);
        const tTestMs   = Date.now() - tTestStart;
        if (tTestMs > 0.3 * intervalTrainMs) break;
        if (resultsBatch.length >= 998) break;
      }
      for (const r of resultsBatch) evalHistory.push(r);

      latestWR = resultsBatch.reduce((s, r) => s + r, 0) / resultsBatch.length;
      const half = Math.max(1, Math.floor(evalHistory.length / 2));
      avgWR = evalHistory.slice(-half).reduce((s, r) => s + r, 0) / half;
      resultsBatchLen = resultsBatch.length;
    }

    const avgLen  = intervalMoves / intervalGames;
    const tGameMs = intervalTrainMs / intervalGames;
    const avgAcc  = totalNVals > 0 ? totalCorrect / totalNVals : 0;
    intervalGames = 0;
    intervalMoves = 0;
    intervalCorrect = 0; intervalNVals = 0;
    let vaccCol = null;
    if (ACCURACY_FILE) {
      const { accuracy } = evalValueAccuracy(ACCURACY_FILE, { weights, specs }, { nGames: ACCURACY_GAMES });
      vaccCol = Util.fmtRatio4(accuracy);
    }
    let rmsCol = null, rmsAvgCol = null;
    if (evalPositionsPool) {
      const { rmsErr } = evalPositionsSample(game => ({ move: search(game, { weights, specs, preparedSpecs: prepSpecs }) }), evalPositionsPool, POSITIONS_N || evalPositionsPool.length, 0);
      rmsHistory.push(rmsErr);
      const rmsHalf = Math.max(1, Math.floor(rmsHistory.length / 2));
      const rmsAvg  = rmsHistory.slice(-rmsHalf).reduce((s, r) => s + r, 0) / rmsHalf;
      rmsCol    = Util.fmt4(rmsErr);
      rmsAvgCol = Util.fmt4(rmsAvg);
    }
    const wAvg = wUpdateCount > 0 ? wAbsSum / wUpdateCount : 0;

    const tTestMs   = Date.now() - tTestStart;
    const elapsedMs = Date.now() - t0;
    const trainMs   = intervalTrainMs;
    intervalTrainMs = 0;
    const nextMs    = elapsedMs;
    console.log([
      Util.fmt4(g),
      Util.fmtMs(elapsedMs),
      Util.fmtMs(tGameMs),
      Util.fmt4(weights.size),
      ...(evalGetMove ? [Util.fmt4(resultsBatchLen), Util.fmtRatio4(latestWR), Util.fmtRatio4(avgWR)] : []),
      Util.fmt4(avgLen),
      Util.fmtRatio4(avgAcc),
      ...(vaccCol ? [vaccCol]               : []),
      ...(rmsCol  ? [rmsCol, rmsAvgCol]     : []),
      wAvg.toFixed(4).padStart(6),
      Util.fmtMs(trainMs),
      ...(evalGetMove ? [Util.fmtMs(tTestMs)] : []),
      Util.fmtMs(timePerMoveMs),
    ].join('  '));
    saveWeights(SAVE_PATH, { weights, specs, preparedSpecs: prepSpecs });
    nextPrintAt = Math.min(t0 + Math.round(nextMs * 1.4), Date.now() + MAX_PRINT_INTERVAL_MS);
  }

  if (LIMIT_GAMES > 0 && g >= LIMIT_GAMES) {
    console.log(`Reached --limit ${LIMIT_GAMES} games — saved ${SAVE_PATH}`);
    break;
  }
}
