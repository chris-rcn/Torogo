#!/usr/bin/env node
'use strict';

// train-vpatterns.js — learn pattern weights via TD(λ) self-play.
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
const { loadPositions, evalPositions, evalPositionsSample } = require('./evalmovedetails.js');
const { loadCases, evalCases } = require('./evalladders2.js');
const { evalValueAccuracy } = require('./eval-value-accuracy.js');
const Util = require('./util.js');
const fs = require('fs');

// ── Arguments ─────────────────────────────────────────────────────────────────

const opts       = Util.parseArgs(process.argv.slice(2), [], ['accuracy-file', 'accuracy-games', 'budget', 'epsilon', 'eval', 'eval-size', 'ext', 'ladder-file', 'limit', 'load', 'lr', 'md-file', 'on-policy', 'positions-file', 'positions-n', 'save', 'size', 'spec', 'start-phase', 'train-size']);
const TRAIN_SIZE = parseInt(opts['train-size']  || opts.size || '9',  10);
const EVAL_SIZE  = parseInt(opts['eval-size']   || opts.size || '13', 10);
const SAVE_PATH  = opts.save  || `out/vpatterns-${Math.random().toString(36).slice(2, 10)}.js`;
const LOAD_PATH  = opts.load  || null;
const EVAL_AGENT = opts.eval  || '';     // empty disables in-training reference test games
const EXT_AGENT  = opts.ext   || '';     // off-policy move source: (1-epsilon) fraction of moves come from this agent
const LIMIT_GAMES = opts.limit !== undefined ? parseInt(opts.limit, 10) : 0;
const EPSILON    = parseFloat(opts.epsilon      || '0.1');
const ON_POLICY  = parseFloat(opts['on-policy'] || '1');   // share of non-random moves from own search1ply (vs --ext)
const START_PHASE = parseFloat(opts['start-phase'] || '0');  // random stones until this board phase, then normal training
const POSITIONS_FILE  = opts['positions-file']   || null;
const MD_FILE         = opts['md-file']          || null;   // evalmovedetails positions for the single-pass mdRms column
const LADDER_FILE     = opts['ladder-file']      || null;   // evalladders2 suite to score each status print (the ladr column)
const POSITIONS_N     = parseInt(opts['positions-n'] || '0', 10);
const ACCURACY_FILE   = opts['accuracy-file']    || null;
const ACCURACY_GAMES  = parseInt(opts['accuracy-games'] || '100', 10);
const LR         = parseFloat(opts['lr']       || '0.3');
const BUDGET     = parseFloat(opts['budget']   || '1');

// ── Features ───────────────────────────

// --spec: comma list of "size:maxLibs[f]" tokens (e.g. "1:6,2:6,3:6").  size is
// 1-3; maxLibs caps the per-cell liberty count (1 = presence only).  A trailing
// 'f' freezes that spec's weights (loaded values stay fixed; gradient updates
// skipped).  Default: sizes 1/2/3 at maxLibs 6.  With --load, --spec overrides
// the checkpoint's specs: shared specs keep their trained weights (freeze the
// carried-over ones with 'f' for a curriculum), new specs start at zero.
let specs;
const FROZEN = new Set();   // spec tags ((maxLibs << 2) | size) excluded from updates
if (opts.spec) {
  specs = opts.spec.split(',').map(tok => {
    const [s, mRaw] = tok.split(':');
    const size = parseInt(s, 10);
    const frozen = /f$/.test(mRaw);
    const maxLibs = parseInt(frozen ? mRaw.slice(0, -1) : mRaw, 10);
    if (!(size >= 1 && size <= 3) || !(maxLibs >= 1)) {
      console.error(`--spec: bad token '${tok}' (expected size:maxLibs[f], size 1-3, maxLibs >= 1)`);
      process.exit(1);
    }
    if (frozen) FROZEN.add((maxLibs << 2) | size);
    return { size, maxLibs };
  });
} else {
  specs = [
    { size: 1, maxLibs: 6 },
    { size: 2, maxLibs: 6 },
    { size: 3, maxLibs: 6 },
  ];
}
let prepSpecs = prepareSpecs(specs);

// ── Weight table ──────────────────────────────────────────────────────────────

let weights  = new Map();  // pattern key (int32) → weight (float)
let wAbsSum = 0, wUpdateCount = 0;  // per-interval |weight| sum/count over feature updates (avgW; reset each print)

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
  const { keys, pols, tags } = features;
  const hasFrozen = FROZEN.size > 0;
  let nActive = n;
  if (hasFrozen) {
    nActive = 0;
    for (let i = 0; i < n; i++) if (!FROZEN.has(tags[i])) nActive++;
    if (nActive === 0) return;
  }
  const step = lr * (target - features.val) / nActive;
  for (let i = 0; i < n; i++) {
    if (hasFrozen && FROZEN.has(tags[i])) continue;
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
  const game     = new Game2(N);   // free initial stone (applyFirstMove=true)
  const maxMoves = N * N * 4;
  const tStartMs = Date.now();

  // --start-phase: fill the board with random stones up to the target phase
  // before training begins.  The prefix is untrained (no features recorded).
  while (game.phase() < START_PHASE && !game.gameOver) game.play(game.randomLegalMove());

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

  return { winner: game.estimateWinner(), elapsedMs, moves };
}

// ── Evaluation against a reference agent ─────────────────────────────────────

// Play nGames of policy vs agent, alternating colours.
// Returns { results } where each element is 1 (policy win), 0 (agent win), or 0.5 (draw).
function evalVsReference(N, refGetMove, nGames, budget) {
  const results = [];
  let totalMoves = 0;
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
      const f = extractFeatures(game, prepSpecs);
      evaluateFeatures(f, weights);
      gameVals.push(f.val);
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
    totalMoves += moves;
    for (const v of gameVals) if ((v >= 0.5) === (winner === BLACK)) accCorrect++;
    accN += gameVals.length;
    if ((winner === BLACK) === policyIsBlack) {
      results.push(1);
    } else {
      results.push(0);
    }
  }

  return { results, moves: totalMoves, accCorrect, accN };
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

// Move-quality suite (evalmovedetails): a single full pass scoring the trainee
// against --md-file at each status print (the `mdRms` column — RMS win-ratio
// gap to the top move).
const mdPositions = MD_FILE ? loadPositions(MD_FILE) : null;
if (mdPositions) console.log(`md positions: ${MD_FILE} (${mdPositions.length} positions)`);

// Ladder suite (evalladders2): score the trainee's own 1-ply argmax (search1ply)
// against --ladder-file at each status print (the `ladr` column).
const ladderCases = LADDER_FILE ? loadCases(LADDER_FILE) : null;
const ladderAgent = gm => ({ move: gm.gameOver ? PASS : search1ply(gm) });
if (ladderCases) console.log(`ladder suite: ${LADDER_FILE} (${ladderCases.length} cases)`);

if (LOAD_PATH) {
  if (fs.existsSync(LOAD_PATH)) {
    // Compare canonical fields, not whole objects (spec objects can carry
    // derived properties that would false-positive the comparison).
    const specKey = ss => ss.map(x => `${x.size}:${x.maxLibs}`).join(',');
    const cliSpecs = opts.spec ? specs : null;
    ({ weights, specs, preparedSpecs: prepSpecs } = loadWeights(LOAD_PATH));
    if (cliSpecs !== null && specKey(cliSpecs) !== specKey(specs)) {
      // --spec overrides the checkpoint's specs.  Loaded weights are kept —
      // each spec hashes patterns with its own mixer (same collision
      // assumption as any multi-spec run), so shared specs continue from
      // their trained values (typically frozen with 'f'), new specs start at
      // zero, and dropped specs' weights stay in the table, never extracted.
      console.warn(`WARNING: --spec overrides checkpoint specs (${specKey(specs)} -> ${specKey(cliSpecs)}); shared specs keep their weights.`);
      specs = cliSpecs;
      prepSpecs = prepareSpecs(specs);
    }
    console.log(`Loaded ${weights.size} weights from ${LOAD_PATH}`);
  } else {
    console.warn(`Warning: --load file not found: ${LOAD_PATH}`);
  }
}


console.log(`LR=${LR}  epsilon=${EPSILON}  on-policy=${ON_POLICY}  start-phase=${START_PHASE}  train-size=${TRAIN_SIZE}  eval-size=${EVAL_SIZE}  ref=${EVAL_AGENT || '(none)'}  ext=${EXT_AGENT || '(none)'}`);
console.log(`Out: ${SAVE_PATH}${LOAD_PATH ? `  (resumed from ${LOAD_PATH})` : ''}${evalPositionsPool ? `  positions: ${evalPositionsPool.length} batch=${POSITIONS_N || 'all'}` : ''}`);
console.log(`Specs: ${JSON.stringify(specs)}${FROZEN.size > 0 ? `  frozen: [${specs.filter(sp => FROZEN.has((sp.maxLibs << 2) | sp.size)).map(sp => `${sp.size}:${sp.maxLibs}`).join(',')}]` : ''}`);
console.log();

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
  // tTest = whole eval pass; the trailing turn = wall-clock per move of the
  // reference MATCHES only (both sides' moves), vs the left turn = training.
  ...(evalGetMove ? ['tTest'.padStart(5), 'tTurn'.padStart(5)] : []),
].join('  '));

const t0 = Date.now();
const MAX_PRINT_INTERVAL_MS = 6 * 60 * 60 * 1000;  // cap status-print gap at 6 hours
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

  // Force a final stats row when the game limit is reached.
  if (LIMIT_GAMES > 0 && g >= LIMIT_GAMES) nextPrintAt = 0;

  if (Date.now() >= nextPrintAt) {
    const tTestStart = Date.now();
    let latestWR = null, avgWR = null, resultsBatchLen = 0, evalHalf = 0;
    let evalMatchMs = 0, evalMatchMoves = 0, evalAccC = 0, evalAccN = 0;
    if (evalGetMove) {
      const resultsBatch = [];
      while (true) {
        const { results, moves, accCorrect, accN } = evalVsReference(EVAL_SIZE, evalGetMove, 2, refBudgetMs);
        for (const r of results) resultsBatch.push(r);
        evalMatchMoves += moves;
        evalAccC += accCorrect; evalAccN += accN;
        evalMatchMs = Date.now() - tTestStart;
        if (evalMatchMs > 0.3 * intervalTrainMs) break;
        if (resultsBatch.length >= 998) break;
      }
      for (const r of resultsBatch) evalHistory.push(r);

      latestWR = resultsBatch.reduce((s, r) => s + r, 0) / resultsBatch.length;
      evalHalf = Math.max(1, Math.floor(evalHistory.length / 2));
      avgWR = evalHistory.slice(-evalHalf).reduce((s, r) => s + r, 0) / evalHalf;
      resultsBatchLen = resultsBatch.length;
    }

    const avgLen  = intervalMoves / intervalGames;
    const tGameMs = intervalTrainMs / intervalGames;
    intervalGames = 0;
    intervalMoves = 0;
    let ladrCol = null;
    if (ladderCases) {
      const { passed, total } = evalCases(ladderCases, ladderAgent, { budgetMs: 1, oversample: 1 });
      ladrCol = Util.fmtRatio4(total ? passed / total : 0);
    }
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
    let mdRmsCol = null;
    if (mdPositions) {
      const { rmsErr } = evalPositions(game => ({ move: search(game, { weights, specs, preparedSpecs: prepSpecs }) }), mdPositions, 0);
      mdRmsCol = Util.fmtRatio4(rmsErr).padStart(5);
    }
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
      ...(ladrCol ? [ladrCol]               : []),
      ...(vaccCol ? [vaccCol]               : []),
      ...(rmsCol  ? [rmsCol, rmsAvgCol]     : []),
      ...(mdRmsCol ? [mdRmsCol]             : []),
      ...(evalGetMove ? [Util.fmtMs(tTestMs),
                         Util.fmtMs(evalMatchMoves > 0 ? evalMatchMs / evalMatchMoves : 0)] : []),
    ].join('  '));
    saveWeights(SAVE_PATH, { weights, specs, preparedSpecs: prepSpecs });
    nextPrintAt = Math.min(t0 + Math.round(nextMs * 1.4), Date.now() + MAX_PRINT_INTERVAL_MS);
  }

  if (LIMIT_GAMES > 0 && g >= LIMIT_GAMES) {
    console.log(`Reached --limit ${LIMIT_GAMES} games — saved ${SAVE_PATH}`);
    break;
  }
}
