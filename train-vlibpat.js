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
// Features: pattern1 + pattern2 + pattern3 (ladder-aware encoding), all cells.
//
// Status is printed at an exponentially increasing interval (× 1.5 each time).
//
// Runs indefinitely (Ctrl-C to stop).  Weights are saved at every print.

const path = require('path');
const { Game2, BLACK, PASS } = require('./game2.js');
const { evaluateFeatures, extractFeatures, prepareSpecs, loadWeights, saveWeights } = require('./vlibpat.js');
const { Game3, game3FromGame2 } = require('./game3.js');
const { search } = require('./ai/vlibpat.js');
const { search: abSearch } = require('./ab-search3.js');
const NPat = require('./npat-lib.js');
const { loadPositions, evalPositionsSample } = require('./evalmovedetails.js');
const { evalValueAccuracy } = require('./eval-value-accuracy.js');
const Util = require('./util.js');
const fs = require('fs');

// ── Arguments ─────────────────────────────────────────────────────────────────

const opts       = Util.parseArgs(process.argv.slice(2));
const TRAIN_SIZE = parseInt(opts['train-size']  || opts.size || '9',  10);
const EVAL_SIZE  = parseInt(opts['eval-size']   || opts.size || '13', 10);
const SAVE_PATH  = opts.save  || `out/vlibpat-${Math.random().toString(36).slice(2, 10)}.js`;
const LOAD_PATH  = opts.load  || null;
const EVAL_AGENT = opts.eval  || '';     // empty disables in-training reference test games
const EXT_AGENT  = opts.ext   || '';     // off-policy move source: (1-epsilon) fraction of moves come from this agent
const LIMIT_GAMES = opts.limit !== undefined ? parseInt(opts.limit, 10) : 0;
const EPSILON    = Math.min(parseFloat(opts.epsilon      || '0.1'), 0.9999);
const ON_POLICY  = Math.min(parseFloat(opts['on-policy'] || '1'), 0.9999);  // share of moves from own search1ply when --ext is set
const POSITIONS_FILE  = opts['positions-file']   || null;
const POSITIONS_N     = parseInt(opts['positions-n'] || '0', 10);
const ACCURACY_FILE   = opts['accuracy-file']    || null;
const ACCURACY_GAMES  = parseInt(opts['accuracy-games'] || '100', 10);
const LR         = parseFloat(opts['lr']       || '0.3');
const MOMENTUM   = parseFloat(opts['momentum'] || '0.0');
const EMA_ALPHA  = parseFloat(opts['ema-alpha'] || '0.95');  // per-call decay (period=50, ≈hpatterns per-game 0.999)
const BUDGET     = parseFloat(opts['budget']   || '1');
const LAMBDA     = parseFloat(opts['lambda']   || '0.0');
const EVAL_TOPK  = Math.max(1, parseInt(opts['eval-topk'] || '20', 10));
const EVAL_DITHER = 0.002;

// ── Features ───────────────────────────

let specs = [
  { size: 2 },                      // ladder-aware (7-state)
  { size: 3 },                      // ladder-aware (7-state)
  { size: 3, ladder: false },       // plain (3-state)
];
let prepSpecs = prepareSpecs(specs);

// ── Weight table ──────────────────────────────────────────────────────────────

let weights    = new Map();  // pattern key (int32) → weight (float)
let weightsEMA = new Map();  // Polyak-averaged shadow (saved on disk for eval)
let weightsEMAInit = false;  // first applyEMA seeds EMA = weights
let velocity = new Map();  // SGD momentum: vel_k ← β·vel_k + g_k

// Polyak / SWA averaging.  Updates weightsEMA in-place to track a smoothed
// version of weights:
//   weightsEMA[k] = alpha * weightsEMA[k] + (1 - alpha) * weights[k]
function applyEMA(alpha) {
  if (!weightsEMAInit) {
    for (const [k, v] of weights) weightsEMA.set(k, v);
    weightsEMAInit = true;
    return;
  }
  // Update existing EMA entries; seed new keys (added since last applyEMA).
  for (const [k, v] of weights) {
    const e = weightsEMA.get(k);
    if (e === undefined) weightsEMA.set(k, v);
    else weightsEMA.set(k, alpha * e + (1 - alpha) * v);
  }
}

// ── Training helpers ──────────────────────────────────────────────────────────

// Absolute terminal outcome: 1=BLACK wins, 0=WHITE wins.
function absoluteOutcome(game) {
//  return game.estimateWinner() === BLACK ? 1 : 0;
  return game.calcWinner() === BLACK ? 1 : 0;
}

// SGD (+ optional momentum):
//   g_k    = (target − V) / n · polarity_k
//   vel_k ← β · vel_k + g_k
//   Δw_k   = lr · vel_k
// β=0 (default) → plain SGD.  β>0 smooths gradients and accelerates in
// consistent directions while still decaying to 0 when targets are met.
function tdUpdate(features, target, lr) {
  const n = features.count;
  if (n === 0) return;
  const perFeature = (target - features.val) / n;
  const { keys, pols } = features;
  if (MOMENTUM === 0) {
    const step = lr * perFeature;
    for (let i = 0; i < n; i++) {
      const k = keys[i];
      weights.set(k, (weights.get(k) ?? 0) + pols[i] * step);
    }
  } else {
    for (let i = 0; i < n; i++) {
      const k   = keys[i];
      const g   = pols[i] * perFeature;
      const vel = MOMENTUM * (velocity.get(k) ?? 0) + g;
      velocity.set(k, vel);
      weights.set(k, (weights.get(k) ?? 0) + lr * vel);
    }
  }
}

// ── Self-play training ────────────────────────────────────────────────────────

// Custom 1-ply search which bypasses non-capture moves.
function search1ply(game, game3) {
  const area = game.N * game.N;
  const isBlack = game.current === BLACK;
  let bestMove = PASS;
  let bestScore = isBlack ? -Infinity : Infinity;
  for (let coord = 0; coord < area; coord++) {
    if (!game.isLegal(coord) || game.isTrueEye(coord)) continue;
    const features = extractFeatures(game, prepSpecs, true, coord, game3);
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
    const passFeatures = extractFeatures(game, prepSpecs, false, undefined, game3);
    evaluateFeatures(passFeatures, weights);
    if (isBlack === (passFeatures.val > bestScore)) {
      bestScore = passFeatures.val;
      bestMove = PASS;
    }
  }
  return bestMove;
}

// Both colours use the policy.  Apply 2-step logistic TD inline during play.
// All targets are absolute (P(BLACK wins)).
function trainGame(N) {
  const game     = new Game2(N, false);
  const game3    = new Game3(N);   // lockstep mirror so extractFeatures skips per-move rebuild
  const maxMoves = N * N * 4;
  const tStartMs = Date.now();

  let prev2 = null, prev1 = null;  // feature sets from 2 and 1 steps ago
  let moves = 0;
  const vals = [];
  const lambdaFeats = [];
  const lamdbaLr = LAMBDA * LR;
  const tdLr = (1 - LAMBDA) * LR;

  while (!game.gameOver && moves < maxMoves) {
    const features = extractFeatures(game, prepSpecs, false, undefined, game3);
    evaluateFeatures(features, weights);
    vals.push(features.val);
    if (Math.random() < LAMBDA) {
      lambdaFeats.push(features);
    }

    if (prev2 !== null) {
      tdUpdate(prev2, features.val, tdLr);
    }

    prev2 = prev1;
    prev1 = features;
    let move;
    const r = Math.random();
    if (r < EPSILON) {
      move = game.randomLegalMove();
    } else if (extGetMove && r < EPSILON + ON_POLICY) {
      move = search1ply(game, game3);
    } else if (extGetMove) {
      move = extGetMove(game).move;
    } else {
      move = search1ply(game, game3);
    }
    game.play(move);
    game3.play(move);
    moves++;
  }

  const elapsedMs = Date.now() - tStartMs;
  const outcome   = absoluteOutcome(game);

  for (const features of [prev2, prev1]) {
    if (features === null) continue;
    tdUpdate(features, outcome, tdLr);
  }

  for (const features of lambdaFeats) {
    tdUpdate(features, outcome, lamdbaLr);
  }

  let correct = 0;
  for (const v of vals) {
    if ((v >= 0.5) === (outcome === 1)) correct++;
  }

  return { winner: game.estimateWinner(), elapsedMs, moves, correct, nVals: vals.length };
}

// ── Evaluation against a reference agent ─────────────────────────────────────

// Reference npat for top-K candidate filtering during eval.  Loaded lazily so
// any failure surfaces only when eval actually runs.  Shares the canonical
// npat-data.js weights file with ai/npat.js etc.
let _npatRefWeights = null;
const _evalNpatStateByN = new Map();
function _loadNpatRef() {
  const raw = require(path.resolve(path.join(__dirname, 'npat-data.js')));
  let has33c = false, hasP12 = false;
  for (const [k] of raw.weights) {
    if (typeof k === 'string') continue;
    if      (k >= NPat.SHAPE33C_RAW_BASE && k < NPat.P12_RAW_BASE) has33c = true;
    else if (k >= NPat.P12_RAW_BASE)                                hasP12 = true;
  }
  const w = NPat.createWeights({
    initialCapacity: Math.max(1024, raw.weights.size | 0),
    use33c: has33c, useP12: hasP12,
  });
  for (const [k, v] of raw.weights) {
    const idx = NPat.internWeight(w, k);
    w.vals[idx] = v;
  }
  return w;
}

// Trainee-side eval move selection: take the top-EVAL_TOPK candidates from
// the npat reference policy and pick the best by the trainee's value head.
function _evalEvaluate(g3) {
  const f = extractFeatures(g3, prepSpecs);
  evaluateFeatures(f, weights);
  return f.val;
}

function _evalTopKCandidates(game, game3) {
  const N = game.N;
  let state = _evalNpatStateByN.get(N);
  if (!state) { state = NPat.createState(N); _evalNpatStateByN.set(N, state); }
  NPat.policyMove(game, state, _npatRefWeights, Math, game3);
  const n = state.count;
  if (n === 0) return [];
  const k = Math.min(EVAL_TOPK, n);
  const probs = state.probs;
  const order = new Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  if (k < n) order.sort((a, b) => probs[b] - probs[a]);
  const candidates = new Array(k);
  for (let j = 0; j < k; j++) candidates[j] = state.moves[order[j]];
  return candidates;
}

function evalSearch(game) {
  if (game.gameOver) return PASS;
  if (_npatRefWeights === null) _npatRefWeights = _loadNpatRef();
  const game3 = game3FromGame2(game);
  const candidates = _evalTopKCandidates(game, game3);
  if (candidates.length === 0) return PASS;
  return abSearch(game3, 1, _evalEvaluate, EVAL_DITHER, {
    getCandidates: () => candidates,
  });
}

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
        idx = evalSearch(game);
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
    // Seed EMA shadow from the loaded weights so subsequent applyEMA continues
    // averaging on top of the persisted (already-EMA) values.
    weightsEMA = new Map(weights);
    weightsEMAInit = true;
    console.log(`Loaded ${weights.size} weights from ${LOAD_PATH}`);
  } else {
    console.warn(`Warning: --load file not found: ${LOAD_PATH}`);
  }
}


console.log(`LR=${LR}  momentum=${MOMENTUM}  epsilon=${EPSILON}  on-policy=${ON_POLICY}  ema-alpha=${EMA_ALPHA}  train-size=${TRAIN_SIZE}  eval-size=${EVAL_SIZE}  eval-topk=${EVAL_TOPK}  ref=${EVAL_AGENT || '(none)'}  ext=${EXT_AGENT || '(none)'}  lambda=${LAMBDA}`);
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

// Apply EMA every EMA_PERIOD games to amortize the O(weight_count) cost.
// Default alpha=0.95 with period=50 ⇒ time constant ≈ 1 000 games — matches
// hpatterns per-game alpha=0.999.  Use --ema-alpha to tune (per-call):
//   alpha=0.99  → 5 000-game tc
//   alpha=0.9975→ 20 000-game tc
const EMA_PERIOD = 50;

while (true) {
  g++;
  const { moves, elapsedMs, correct, nVals } = trainGame(TRAIN_SIZE);
  if (g % EMA_PERIOD === 0) applyEMA(EMA_ALPHA);
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

    const avgLen   = intervalMoves / intervalGames;
    const tGameMs  = intervalTrainMs / intervalGames;
    const avgAcc   = totalNVals > 0 ? totalCorrect / totalNVals : 0;
    intervalGames = 0;
    intervalMoves = 0;
    intervalCorrect = 0; intervalNVals = 0;
    let vaccCell = null;
    if (ACCURACY_FILE) {
      const { accuracy } = evalValueAccuracy(ACCURACY_FILE, { weights, specs }, { nGames: ACCURACY_GAMES });
      vaccCell = Util.fmtRatio4(accuracy);
    }
    let rmsCell = null, rmsAvgCell = null;
    if (evalPositionsPool) {
      const { rmsErr } = evalPositionsSample(game => ({ move: evalSearch(game) }), evalPositionsPool, POSITIONS_N || evalPositionsPool.length, 0);
      rmsHistory.push(rmsErr);
      const rmsHalf = Math.max(1, Math.floor(rmsHistory.length / 2));
      const rmsAvg  = rmsHistory.slice(-rmsHalf).reduce((s, r) => s + r, 0) / rmsHalf;
      rmsCell    = Util.fmt4(rmsErr);
      rmsAvgCell = Util.fmt4(rmsAvg);
    }
    let wAbsSum = 0;
    for (const w of weights.values()) {
      wAbsSum += Math.abs(w);
    }
    const wAvg = weights.size > 0 ? wAbsSum / weights.size : 0;

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
      ...(vaccCell   ? [vaccCell]                : []),
      ...(rmsCell    ? [rmsCell, rmsAvgCell]     : []),
      wAvg.toFixed(4).padStart(6),
      Util.fmtMs(trainMs),
      ...(evalGetMove ? [Util.fmtMs(tTestMs)] : []),
      Util.fmtMs(timePerMoveMs),
    ].join('  '));
    // Persist Polyak-averaged weights for eval (slight rewind on resume).
    // Falls back to live weights before the first applyEMA.
    const saveSrc = weightsEMAInit ? weightsEMA : weights;
    saveWeights(SAVE_PATH, { weights: saveSrc, specs, preparedSpecs: prepSpecs });
    nextPrintAt = t0 + Math.round(nextMs * 1.4);
  }

  if (LIMIT_GAMES > 0 && g >= LIMIT_GAMES) {
    console.log(`Reached --limit ${LIMIT_GAMES} games — saved ${SAVE_PATH}`);
    break;
  }
}
