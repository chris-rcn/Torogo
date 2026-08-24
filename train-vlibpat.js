#!/usr/bin/env node
'use strict';

// train-patterns.js — learn pattern weights via logistic TD(2) self-play.
//
// Value function (absolute, P(BLACK wins)):
//   V(s) = σ( Σ  polarity_i · w[key_i] )
//
// Update rule — logistic TD, 2-step lookahead (same player to move):
//   Δw_k = (LR / n_features) · (target − V) · polarity_k   [logistic / CE]
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
const { loadPositions, evalPositions, evalPositionsSample } = require('./evalmovedetails.js');
const { loadCases, evalCases } = require('./evalladders2.js');
const { evalValueAccuracy } = require('./eval-value-accuracy.js');
const Util = require('./util.js');
const fs = require('fs');

// ── Arguments ─────────────────────────────────────────────────────────────────

const opts       = Util.parseArgs(process.argv.slice(2), [], ['accuracy-file', 'accuracy-games', 'budget', 'smooth-weights', 'epsilon', 'eval', 'eval-size', 'ext', 'ladder-file', 'lambda', 'limit', 'load', 'lr', 'md-file', 'momentum', 'on-policy', 'positions-file', 'positions-n', 'save', 'size', 'spec', 'tactics', 'train-size']);
const TRAIN_SIZE = parseInt(opts['train-size']  || opts.size || '9',  10);
const EVAL_SIZE  = parseInt(opts['eval-size']   || opts.size || '13', 10);
const SAVE_PATH  = opts.save  || `out/vlibpat-${Math.random().toString(36).slice(2, 10)}.js`;
const LOAD_PATH  = opts.load  || null;
const EVAL_AGENT = opts.eval  || '';     // empty disables in-training reference test games
const EXT_AGENT  = opts.ext   || '';     // off-policy move source: (1-epsilon) fraction of moves come from this agent
const LIMIT_GAMES = opts.limit !== undefined ? parseInt(opts.limit, 10) : 0;
const EPSILON    = parseFloat(opts.epsilon      || '0.1');
const ON_POLICY  = parseFloat(opts['on-policy'] || '1');   // share of non-random moves from own search1ply (vs --ext)
const POSITIONS_FILE  = opts['positions-file']   || null;
const POSITIONS_N     = parseInt(opts['positions-n'] || '0', 10);
const MD_FILE         = opts['md-file']          || null;   // evalmovedetails positions for the single-pass mdRms column
const ACCURACY_FILE   = opts['accuracy-file']    || null;
const ACCURACY_GAMES  = parseInt(opts['accuracy-games'] || '100', 10);
const LR         = parseFloat(opts['lr']       || '0.3');
const MOMENTUM   = parseFloat(opts['momentum'] || '0.0');
const EMA_ALPHA  = parseFloat(opts['smooth-weights'] || '0.9');  // Polyak EMA every EMA_PERIOD games; 0 = off
const BUDGET     = parseFloat(opts['budget']   || '1');
const LAMBDA     = parseFloat(opts['lambda']   || '0.0');
const LADDER_FILE = opts['ladder-file'] || null;   // evalladders2 suite for the ladr column
const EVAL_DITHER = 0.002;

// ── Features ───────────────────────────

// --spec selects the component list: comma tokens, "1"/"2"/"3" = ladder-aware
// of that size, "1p"/"3p" = plain (no ladder), "2F"/"3F" = ladder-aware with
// the alive state split by liberty count (alive4 at 4+ libs; 9-state),
// "2V"/"3V" = the same split at 5+ libs (alive5).
// Default: full 3-component spec.
let specs;
if (opts.spec) {
  specs = opts.spec.split(',').map(tok => {
    if (tok === '1p') return { size: 1, ladder: false };
    if (tok === '2p') return { size: 2, ladder: false };
    if (tok === '3p') return { size: 3, ladder: false };
    if (tok === '2F') return { size: 2, four: true };
    if (tok === '3F') return { size: 3, four: true };
    if (tok === '2V') return { size: 2, five: true };
    if (tok === '3V') return { size: 3, five: true };
    if (tok === '1')  return { size: 1 };
    if (tok === '2')  return { size: 2 };
    if (tok === '3')  return { size: 3 };
    console.error(`--spec: unknown token '${tok}' (use: 1p/2p/3p/2F/3F/2V/3V/1/2/3)`);
    process.exit(1);
  });
} else {
  specs = [
    { size: 2 },                      // ladder-aware (7-state)
    { size: 3, ladder: false },       // plain (3-state)
    { size: 3 },                      // ladder-aware (7-state)
  ];
}

// Tactical backend for ladder-aware encoding.  ladder2 = 1-2 lib classification
// (legacy default).  tactics3 = budgeted search up to 1-3 libs.
let tacticsOpts = { tactics: opts.tactics || 'ladder2' };
let prepSpecs = prepareSpecs(specs, tacticsOpts);

// ── Weight table ──────────────────────────────────────────────────────────────

let weights    = new Map();  // pattern key (int32) → weight (float)
let weightsEMA = new Map();  // Polyak-averaged shadow (saved on disk for eval)
let weightsEMAInit = false;  // first applyEMA seeds EMA = weights
let velocity = new Map();  // SGD momentum: vel_k ← β·vel_k + g_k
let wAbsSum = 0, wUpdateCount = 0;  // per-interval |weight| sum/count over weight updates (avgW; reset each print)

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
      const w = (weights.get(k) ?? 0) + pols[i] * step;
      weights.set(k, w);
      wAbsSum += Math.abs(w); wUpdateCount++;
    }
  } else {
    for (let i = 0; i < n; i++) {
      const k   = keys[i];
      const g   = pols[i] * perFeature;
      const vel = MOMENTUM * (velocity.get(k) ?? 0) + g;
      velocity.set(k, vel);
      const w = (weights.get(k) ?? 0) + lr * vel;
      weights.set(k, w);
      wAbsSum += Math.abs(w); wUpdateCount++;
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
  const game     = new Game2(N);   // free initial stone (applyFirstMove=true)
  const game3    = game3FromGame2(game);   // lockstep mirror incl. the opening stone
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
    if (Math.random() < EPSILON) {
      move = game.randomLegalMove();
    } else if (extGetMove && Math.random() > ON_POLICY) {
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

  return { winner: game.estimateWinner(), elapsedMs, moves };
}

// ── Evaluation against a reference agent ─────────────────────────────────────

// Trainee value of a position (leaf evaluator for the eval search).
// Eval ≡ save: the eval matches and acc measure the model save would write —
// the EMA shadow when enabled (and initialized), else the live weights.
function _evalEvaluate(g3) {
  const f = extractFeatures(g3, prepSpecs);
  evaluateFeatures(f, (EMA_ALPHA > 0 && weightsEMAInit) ? weightsEMA : weights);
  return f.val;
}

// Eval-game / rms move selection: the trainee's value head over the FULL move
// set, so the eval measures the value function itself.
function evalMove(game) {
  if (game.gameOver) return PASS;
  const game3 = game3FromGame2(game);
  return abSearch(game3, 1, _evalEvaluate, EVAL_DITHER);
}

// Play nGames of policy vs agent, alternating colours.
// Returns { results } where each element is 1 (policy win), 0 (agent win), or 0.5 (draw).
function evalVsReference(N, refGetMove, nGames, budget) {
  const results = [];
  let accCorrect = 0, accN = 0;   // per-position winner prediction (test-side acc)

  for (let g = 0; g < nGames; g++) {
    const policyIsBlack = (g % 2 === 0);
    const game     = new Game2(N);   // free initial stone (applyFirstMove=true)
    // Random opening: 3 random legal moves to diversify positions (same as
    // selfplay.js --rand-moves default).
    for (let r = 0; r < 3 && !game.gameOver; r++) game.play(game.randomLegalMove());
    const maxMoves = N * N * 4;
    let   moves    = 0;

    const gameVals = [];
    while (!game.gameOver && moves++ < maxMoves) {
      gameVals.push(_evalEvaluate(game3FromGame2(game)));
      let idx;
      if ((game.current === BLACK) === policyIsBlack) {
        idx = evalMove(game);
      } else {
        const mv = refGetMove(game, budget);
        idx = mv.move !== undefined ? mv.move : PASS;
      }
      if (!game.play(idx)) {
        console.log("Illegal move!");
      }
    }

    const winner = game.calcWinner();
    for (const v of gameVals) if ((v >= 0.5) === (winner === BLACK)) accCorrect++;
    accN += gameVals.length;
    if ((winner === BLACK) === policyIsBlack) {
      results.push(1);
    } else {
      results.push(0);
    }
  }

  return { results, accCorrect, accN };
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

// Move-quality suite (evalmovedetails): a single full pass scoring the trainee's
// value head against --md-file at each status print (the `mdRms` column).
const mdPositions = MD_FILE ? loadPositions(MD_FILE) : null;
if (mdPositions) console.log(`md positions: ${MD_FILE} (${mdPositions.length} positions)`);

if (LOAD_PATH) {
  if (fs.existsSync(LOAD_PATH)) {
    let loadedOpts;
    ({ weights, specs, opts: loadedOpts, preparedSpecs: prepSpecs } = loadWeights(LOAD_PATH));
    // File's tactics setting is authoritative — the weights were trained for
    // that backend's per-cell encoding.  Flag a mismatch with the CLI flag.
    if (opts.tactics && loadedOpts && loadedOpts.tactics !== opts.tactics) {
      console.warn(`Warning: --tactics ${opts.tactics} ignored; loaded file uses ${loadedOpts.tactics}`);
    }
    tacticsOpts = loadedOpts || tacticsOpts;
    // Seed EMA shadow from the loaded weights so subsequent applyEMA continues
    // averaging on top of the persisted values (only when EMA is enabled).
    if (EMA_ALPHA > 0) {
      weightsEMA = new Map(weights);
      weightsEMAInit = true;
    }
    console.log(`Loaded ${weights.size} weights from ${LOAD_PATH}`);
  } else {
    console.warn(`Warning: --load file not found: ${LOAD_PATH}`);
  }
}


console.log(`LR=${LR}  momentum=${MOMENTUM}  epsilon=${EPSILON}  on-policy=${ON_POLICY}  smooth-weights=${EMA_ALPHA}  train-size=${TRAIN_SIZE}  eval-size=${EVAL_SIZE}  ref=${EVAL_AGENT || '(none)'}  ext=${EXT_AGENT || '(none)'}  lambda=${LAMBDA}`);
console.log(`Out: ${SAVE_PATH}${LOAD_PATH ? `  (resumed from ${LOAD_PATH})` : ''}${evalPositionsPool ? `  positions: ${evalPositionsPool.length} batch=${POSITIONS_N || 'all'}` : ''}`);
console.log(`Specs: ${JSON.stringify(specs)}`);
console.log(`Tactics: ${tacticsOpts.tactics}`);
console.log();

// Ladder suite (evalladders2): score the trainee's own 1-ply argmax (search1ply)
// against --ladder-file at each status print (the `ladr` column).
const ladderCases = LADDER_FILE ? loadCases(LADDER_FILE) : null;
const ladderAgent = gm => ({ move: gm.gameOver ? PASS : search1ply(gm, game3FromGame2(gm)) });
if (ladderCases) console.log(`ladder suite: ${LADDER_FILE} (${ladderCases.length} cases)`);

// Print header.
console.log([
  // Training columns (left).
  'game'.padStart(4),
  'tElp'.padStart(5),
  'tGm '.padStart(5),
  'nWts'.padStart(4),
  'avgL'.padStart(4),
  'avgW'.padStart(6),
  'tTran'.padStart(5),
  'tTurn'.padStart(5),
  // Test / eval columns (right).
  // winRatio: "wr(g)/avg(ga)" — wr/avg fmtRatio4, g/ga fmt4 game counts (this
  // interval's, and the rolling-half window).  Fixed 21 chars wide.
  ...(evalGetMove ? ['winRatio'.padStart(21), ' acc'.padStart(4)] : []),
  ...(ladderCases ? ['ladr'.padStart(4)] : []),
  ...(ACCURACY_FILE     ? ['vacc'.padStart(4)] : []),
  ...(evalPositionsPool ? ['rms '.padStart(4), 'rAvg'.padStart(4)] : []),
  ...(mdPositions ? ['mdRms'.padStart(5)] : []),
  ...(evalGetMove ? ['tTest'.padStart(5)] : []),
].join('  '));

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

// Apply EMA every EMA_PERIOD games to amortize the O(weight_count) cost.
// Window ≈ EMA_PERIOD/(1-alpha) games: --smooth-weights 0.9 ≈ 1k games, 0.99 ≈ 10k.
const EMA_PERIOD = 100;

while (true) {
  g++;
  const { moves, elapsedMs } = trainGame(TRAIN_SIZE);
  if (EMA_ALPHA > 0 && g % EMA_PERIOD === 0) applyEMA(EMA_ALPHA);
  totalMoves += moves;
  intervalGames++;
  intervalMoves += moves;
  moveElapsedMs += elapsedMs;
  intervalTrainMs += elapsedMs;
  const timePerMoveMs = moveElapsedMs / totalMoves;

  // Force a final stats row when the game limit is reached.
  if (LIMIT_GAMES > 0 && g >= LIMIT_GAMES) nextPrintAt = 0;

  if (Date.now() >= nextPrintAt) {
    const tTestStart = Date.now();
    let latestWR = null, avgWR = null, resultsBatchLen = 0, evalHalf = 0, evalAccC = 0, evalAccN = 0;
    if (evalGetMove) {
      const resultsBatch = [];
      while (true) {
        const { results, accCorrect, accN } = evalVsReference(EVAL_SIZE, evalGetMove, 2, refBudgetMs);
        for (const r of results) resultsBatch.push(r);
        evalAccC += accCorrect; evalAccN += accN;
        const tTestMs   = Date.now() - tTestStart;
        if (tTestMs > 0.3 * intervalTrainMs) break;
        if (resultsBatch.length >= 998) break;
      }
      for (const r of resultsBatch) evalHistory.push(r);

      latestWR = resultsBatch.reduce((s, r) => s + r, 0) / resultsBatch.length;
      evalHalf = Math.max(1, Math.floor(evalHistory.length / 2));
      avgWR = evalHistory.slice(-evalHalf).reduce((s, r) => s + r, 0) / evalHalf;
      resultsBatchLen = resultsBatch.length;
    }

    const avgLen   = intervalMoves / intervalGames;
    const tGameMs  = intervalTrainMs / intervalGames;
    intervalGames = 0;
    intervalMoves = 0;
    let vaccCell = null;
    if (ACCURACY_FILE) {
      const { accuracy } = evalValueAccuracy(ACCURACY_FILE, { weights, specs }, { nGames: ACCURACY_GAMES });
      vaccCell = Util.fmtRatio4(accuracy);
    }
    let rmsCell = null, rmsAvgCell = null;
    if (evalPositionsPool) {
      const { rmsErr } = evalPositionsSample(game => ({ move: evalMove(game) }), evalPositionsPool, POSITIONS_N || evalPositionsPool.length, 0);
      rmsHistory.push(rmsErr);
      const rmsHalf = Math.max(1, Math.floor(rmsHistory.length / 2));
      const rmsAvg  = rmsHistory.slice(-rmsHalf).reduce((s, r) => s + r, 0) / rmsHalf;
      rmsCell    = Util.fmt4(rmsErr);
      rmsAvgCell = Util.fmt4(rmsAvg);
    }
    // Ladder suite (evalladders2): the live weights' own 1-ply argmax on the
    // --ladder-file cases — whether the value can RANK moves at tactical points,
    // not just score positions (the two dissociate).
    let ladrRatio = null;
    if (ladderCases) {
      const { passed, total } = evalCases(ladderCases, ladderAgent, { budgetMs: 1, oversample: 1 });
      ladrRatio = total ? passed / total : 0;
    }

    // Move-quality RMS: one full pass of evalmovedetails over --md-file.
    let mdRmsCell = null;
    if (mdPositions) {
      const { rmsErr } = evalPositions(game => ({ move: evalMove(game) }), mdPositions, 0);
      mdRmsCell = Util.fmtRatio4(rmsErr).padStart(5);
    }

    // avgW: mean |weight| encountered across weight updates (frequency-weighted
    // active weights), not the mean over all stored weights.
    const wAvg = wUpdateCount > 0 ? wAbsSum / wUpdateCount : 0;
    wAbsSum = 0; wUpdateCount = 0;   // per-interval avgW: reset at each print

    const tTestMs   = Date.now() - tTestStart;
    const elapsedMs = Date.now() - t0;
    const trainMs   = intervalTrainMs;
    intervalTrainMs = 0;
    const nextMs    = elapsedMs;
    console.log([
      // Training columns (left).
      Util.fmt4i(g),
      Util.fmtMs(elapsedMs),
      Util.fmtMs(tGameMs),
      Util.fmt4i(weights.size),
      Util.fmt4(avgLen),
      wAvg.toFixed(4).padStart(6),
      Util.fmtMs(trainMs),
      Util.fmtMs(timePerMoveMs),
      // Test / eval columns (right).
      ...(evalGetMove ? [(`${Util.fmtRatio4(latestWR)}(${Util.fmt4i(resultsBatchLen)})` +
                          `/${Util.fmtRatio4(avgWR)}(${Util.fmt4i(evalHalf)})`).padStart(21),
                         Util.fmtRatio4(evalAccN > 0 ? evalAccC / evalAccN : 0)] : []),
      ...(ladrRatio !== null ? [Util.fmtRatio4(ladrRatio)] : []),
      ...(vaccCell   ? [vaccCell]                : []),
      ...(rmsCell    ? [rmsCell, rmsAvgCell]     : []),
      ...(mdRmsCell ? [mdRmsCell]               : []),
      ...(evalGetMove ? [Util.fmtMs(tTestMs)] : []),
    ].join('  '));
    // Persist Polyak-averaged weights for eval (slight rewind on resume).
    // Falls back to live weights before the first applyEMA.
    const saveSrc = (EMA_ALPHA > 0 && weightsEMAInit) ? weightsEMA : weights;
    saveWeights(SAVE_PATH, { weights: saveSrc, specs, opts: tacticsOpts, preparedSpecs: prepSpecs });
    nextPrintAt = t0 + Math.round(nextMs * 1.4);
  }

  if (LIMIT_GAMES > 0 && g >= LIMIT_GAMES) {
    console.log(`Reached --limit ${LIMIT_GAMES} games — saved ${SAVE_PATH}`);
    break;
  }
}
