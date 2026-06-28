#!/usr/bin/env node
'use strict';

// train-npat.js — REINFORCE-style policy iterator over nine-pattern features.
//
// Each candidate move is represented by 9 canonical 3×3 pattern IDs (the nine
// windows that overlap the move — see npat-lib.js).  A linear logit policy
//   π(a|s) = softmax( Σ_{p ∈ f_a} w[p] )
// is trained by REINFORCE on the game's terminal outcome.
//
// Per-step update (for the mover at step t):
//   Δw[p] = lr · A_t · ( count(p, f_{a_t}) − Σ_i π(i|s_t) · count(p, f_i) )
// where A_t = R_t − b   (R_t ∈ {−1, 0, +1} from mover's perspective, b is an
// EMA of past rewards as a REINFORCE baseline (variance reduction).
//
// Weights are stored sparsely in a Map<pid, weight>.  They are saved at every
// progress print (interval grows geometrically).  Ctrl-C to stop.
//
// Usage:
//   node train-npat.js [--size 9] [--lr 0.02] [--reward-ema 0.99]
//                      [--load path] [--save path] [--eval-agent random]

const path = require('path');
const fs   = require('fs');

const { Game2, BLACK, PASS } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');
const NPat = require('./npat-lib.js');
const { loadCases, evalCases } = require('./evalladders2.js');
const { loadPositions, evalPositions } = require('./evalmovedetails.js');
const Util = require('./util.js');

// ── Arguments ─────────────────────────────────────────────────────────────────

const opts       = Util.parseArgs(process.argv.slice(2),
  ['help', 'use-p1', 'use-p5', 'use-p8', 'use-p9', 'use-p12', 'use-p13', 'captures'],
  ['size', 'train-size', 'eval-size', 'lr', 'reward-ema', 'weight-decay', 'temperature', 'eval', 'eval-agent', 'ladder-file', 'md-file', 'load', 'save']);
const TRAIN_SIZE = parseInt(opts['train-size'] || opts.size || '9', 10);
const EVAL_SIZE  = parseInt(opts['eval-size']  || opts.size || '13', 10);
const LR         = parseFloat(opts.lr || '0.02');
const REWARD_EMA = parseFloat(opts['reward-ema'] || '0.99');   // EMA decay for the reward baseline (variance reduction); 0 disables
const WEIGHT_DECAY = parseFloat(opts['weight-decay'] || '0.000002');  // decoupled L2 shrink per update (0 = no decay); bounds logit growth
const TEMPERATURE = Math.max(0, parseFloat(opts.temperature || '1'));
const USE_P1     = !!opts['use-p1'];                           // enable 1-cell ladder/tactical feature
const USE_P5     = !!opts['use-p5'];                               // P5 conjunction: 4-nbr shape × tact × capBucket × atari × selfAtari
const USE_P8     = !!opts['use-p8'];                           // enable 8-cell 3×3 window
const USE_P9     = !!opts['use-p9'];                           // enable p1 × p8
const USE_P12    = !!opts['use-p12'];                          // enable 12-cell diamond shape window
const USE_P13    = !!opts['use-p13'];                          // enable p1 × p12
const USE_CAPTURES = !!opts['captures'];                       // size-graded capture buckets (10 weights, sizes 1..10)
const EVAL_AGENT = opts.eval || opts['eval-agent'] || 'random';
const LADDER_FILE = opts['ladder-file'] || null;   // evalladders2 suite to score each status print
const MD_FILE     = opts['md-file'] || null;       // evalmovedetails positions for the single-pass mdRms column
const SAVE_PATH  = opts.save || `out/npat-${Math.random().toString(36).slice(2, 10)}.js`;
const LOAD_PATH  = opts.load || null;

// ── Weights ───────────────────────────────────────────────────────────────────

let weights = NPat.createWeights({ useP1: USE_P1, useP8: USE_P8, useP12: USE_P12, useP9: USE_P9, useP13: USE_P13, useCaptures: USE_CAPTURES, useP5: USE_P5 });
let ema     = 0;                     // EMA of terminal outcome from mover's perspective
let totalUpdates = 0;                // cumulative weight-update count across resumed runs

// ── Persistence ───────────────────────────────────────────────────────────────

// Save base64-packed as Int32 keys followed by int16-quantised vals: a per-file
// symmetric scale maps [-maxAbs, maxAbs] onto int16, weight ≈ qval / scale.
// Far smaller and faster for V8 to parse than a literal Map of ~100k entries.
// Loaded by npat-lib.js modelWeights (which also still reads legacy Map files).
function saveWeights(filePath, w, meta) {
  const count = w.map.size;
  let maxAbs = 0;
  for (const [, denseIdx] of w.map) { const a = Math.abs(w.vals[denseIdx]); if (a > maxAbs) maxAbs = a; }
  const scale = maxAbs > 0 ? 32767 / maxAbs : 1;

  const keys  = new Int32Array(count);
  const qvals = new Int16Array(count);
  let i = 0;
  for (const [rawId, denseIdx] of w.map) {
    keys[i] = rawId;
    let q = Math.round(w.vals[denseIdx] * scale);
    if (q > 32767) q = 32767; else if (q < -32768) q = -32768;
    qvals[i] = q;
    i++;
  }
  // Int32 keys (count*4 bytes) then Int16 qvals (count*2); count*4 is 2-aligned.
  const buf = Buffer.alloc(count * 6);
  Buffer.from(keys.buffer,  keys.byteOffset,  count * 4).copy(buf, 0);
  Buffer.from(qvals.buffer, qvals.byteOffset, count * 2).copy(buf, count * 4);
  const b64 = buf.toString('base64');
  const cfgStr = JSON.stringify(w.cfg);   // explicit active-family set
  const src = [
    "'use strict';",
    '// Auto-generated by train-npat.js — do not edit by hand.',
    '// Weights int16-quantised: weight = qvals[i] / scale.',
    'const npatModel = (() => {',
    `  const count = ${count};`,
    `  const scale = ${scale};`,
    `  const cfg = ${cfgStr};`,
    `  const ema = ${+meta.ema.toFixed(6)};`,
    `  const totalUpdates = ${Math.round(meta.totalUpdates)};`,
    `  const tactStoneLimit = ${NPat.TACT_STONE_LIMIT};`,
    `  const b64 = '${b64}';`,
    "  const bytes = typeof Buffer !== 'undefined'",
    "    ? Buffer.from(b64, 'base64')",
    "    : Uint8Array.from(atob(b64), c => c.charCodeAt(0));",
    "  const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + count * 6);",
    "  const keys  = new Int32Array(buf, 0, count);",
    "  const qvals = new Int16Array(buf, count * 4, count);",
    "  return { ema, totalUpdates, tactStoneLimit, count, scale, cfg, keys, qvals };",
    "})();",
    "if (typeof module !== 'undefined') module.exports = npatModel;",
    "else window.npatModel = npatModel;",
  ].join('\n') + '\n';
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, src);
}

function loadWeights(filePath) {
  delete require.cache[require.resolve(path.resolve(filePath))];
  const raw = require(path.resolve(filePath));
  if (raw.tactStoneLimit !== undefined && raw.tactStoneLimit !== NPat.TACT_STONE_LIMIT) {
    throw new Error(
      `TACT_STONE_LIMIT mismatch: file was trained with ${raw.tactStoneLimit}, ` +
      `current is ${NPat.TACT_STONE_LIMIT}. ` +
      `Re-run with NPAT_STONE_LIMIT=${raw.tactStoneLimit}.`
    );
  }
  const mw = NPat.modelWeights(raw);
  const w = NPat.createWeights({
    initialCapacity: Math.max(1024, mw.count | 0),
    useP1: USE_P1, useP8: USE_P8, useP12: USE_P12, useP9: USE_P9, useP13: USE_P13, useCaptures: USE_CAPTURES, useP5: USE_P5,
  });
  mw.forEach((rawId, val) => {
    w.vals[NPat.internWeight(w, rawId)] = val;
  });
  // Older save files truncated totalUpdates to int32; clamp away negatives.
  const tu = raw.totalUpdates ?? 0;
  return { weights: w, ema: raw.ema ?? 0, totalUpdates: tu < 0 ? 0 : tu, tactStoneLimit: raw.tactStoneLimit };
}

// ── Training ──────────────────────────────────────────────────────────────────

// Play one self-play game and apply REINFORCE updates after it ends.
// Returns game-level stats for logging.
function trainGame(N) {
  const game  = new Game2(N);          // free initial stone (applyFirstMove=true)
  const game3 = game3FromGame2(game);  // lockstep mirror for ladder analysis
  const maxMoves = N * N * 4;
  const tStart   = Date.now();

  // Per-step record: enough to replay the gradient with the known outcome.
  // We store (player, chosenIndex, per-shape patIds, tact, probs, count).
  // To minimise allocations we use per-step state snapshots.
  const steps = [];
  const state = NPat.createState(N);

  let moves = 0;
  while (!game.gameOver && moves < maxMoves) {
    const player = game.current;
    const choice = NPat.policyMove(game, state, weights, Math, game3, TEMPERATURE);

    if (choice.index >= 0) {
      // Snapshot features + distribution (copy just the slice we need).
      // We reuse state.touched across snapshots as the REINFORCE scratch
      // buffer — reinforceUpdate zeros each touched slot after applying.
      const n = state.count;
      const patIdsP8 = new Int32Array(n);
      patIdsP8.set(state.patIdsP8.subarray(0, n));
      const patIdsP12 = new Int32Array(n);
      patIdsP12.set(state.patIdsP12.subarray(0, n));
      const patIdsP9 = new Int32Array(n);
      patIdsP9.set(state.patIdsP9.subarray(0, n));
      const patIdsP13 = new Int32Array(n);
      patIdsP13.set(state.patIdsP13.subarray(0, n));
      const tact = new Uint8Array(n * NPat.N_TACT_SLOTS);
      tact.set(state.tact.subarray(0, n * NPat.N_TACT_SLOTS));
      const probs = new Float64Array(n);
      probs.set(state.probs.subarray(0, n));
      const captures = new Uint8Array(n);
      captures.set(state.captures.subarray(0, n));
      const patIdsP5 = new Int32Array(n);
      patIdsP5.set(state.patIdsP5.subarray(0, n));
      steps.push({ player, chosenIndex: choice.index, count: n, patIdsP8, patIdsP12, patIdsP9, patIdsP13, patIdsP5, tact, probs, captures, touched: state.touched });
    }

    game.play(choice.move);
    game3.play(choice.move);
    moves++;
  }

  // Terminal outcome: +1 if BLACK wins else -1 (use calcWinner; ties → 0).
  const winner = game.calcWinner();
  const outcomeBlack = winner === BLACK ? 1 : (winner === 0 ? 0 : -1);

  // REINFORCE: apply Δw per step with advantage = R − rewardBaseline, where R
  // is from the MOVER'S perspective.
  let stepsApplied = 0;
  let weightUpdates = 0;
  for (const s of steps) {
    const R = s.player === BLACK ? outcomeBlack : -outcomeBlack;
    const adv = REWARD_EMA > 0 ? (R - ema) : R;
    weightUpdates += NPat.reinforceUpdate(s, s.chosenIndex, adv, weights, LR, WEIGHT_DECAY) || 0;
    stepsApplied++;
  }

  if (REWARD_EMA > 0 && steps.length > 0) {
    // Update EMA towards the black-perspective outcome (so it's consistent).
    // Note: we subtract ema from mover-perspective R in the step update, which
    // is a small approximation — the baseline is still unbiased only if it is
    // independent of mover.  In practice the asymmetry washes out quickly.
    ema = REWARD_EMA * ema + (1 - REWARD_EMA) * outcomeBlack;
  }

  // Sum of max(softmax) across the snapshotted steps (top-1 confidence).
  let maxProbSum = 0, maxProbN = 0;
  for (const s of steps) {
    let mx = 0;
    const p = s.probs;
    const n = s.count;
    for (let i = 0; i < n; i++) if (p[i] > mx) mx = p[i];
    maxProbSum += mx;
    maxProbN++;
  }

  return {
    elapsedMs: Date.now() - tStart,
    moves,
    stepsApplied,
    weightUpdates,
    outcomeBlack,
    maxProbSum,
    maxProbN,
  };
}

// ── Evaluation against a reference agent ─────────────────────────────────────

function evalVsReference(N, refGetMove, nGames) {
  const state = NPat.createState(N);
  const results = [];
  for (let g = 0; g < nGames; g++) {
    const policyIsBlack = (g % 2 === 0);
    const game  = new Game2(N);          // free initial stone (applyFirstMove=true)
    const game3 = game3FromGame2(game);  // lockstep mirror for ladder analysis
    // Random opening: 4 random legal moves to diversify positions.
    for (let r = 0; r < 4 && !game.gameOver; r++) {
      const mv = game.randomLegalMove();
      game.play(mv);
      game3.play(mv);
    }
    const maxMoves = N * N * 4;
    let m = 0;
    while (!game.gameOver && m++ < maxMoves) {
      let idx;
      if ((game.current === BLACK) === policyIsBlack) {
        idx = NPat.greedyMove(game, state, weights, game3);
      } else {
        const mv = refGetMove(game);
        idx = mv && mv.move !== undefined ? mv.move : PASS;
      }
      game.play(idx);
      game3.play(idx);
    }
    const winner = game.calcWinner();
    results.push((winner === BLACK) === policyIsBlack ? 1 : 0);
  }
  return results;
}

// ── CLI setup ─────────────────────────────────────────────────────────────────

if (opts.help) {
  console.log(`Usage: node train-npat.js [options]
  --size N         | --train-size N (default 9) | --eval-size N (default 13)
  --lr F           learning rate (default 0.02)
  --reward-ema F   EMA decay for the reward baseline (variance reduction).
                   Default 0.99; 0 disables.
  --weight-decay F decoupled L2 shrink applied to each touched weight per
                   update (w -= lr*F*w); bounds logit growth.  Default 0 (off).
  --temperature F  softmax sampling temperature for training moves;
                   0 = argmax, 1 = standard softmax (default 1)
  --use-p1         enable the 1-cell ladder/tactical feature (default off)
  --use-p5         enable the P5 conjunction feature: one weight per
                   (canonical 4-neighbour shape × tactical-mask × capture-size
                   bucket 0..10 × puts-enemy-in-atari × self-atari) (default off)
  --use-p8         enable the centered 3×3 shape window (default off)
  --use-p9         enable the p1 × p8 (3×3-shape × tactical-mask) window (default off)
  --use-p12        enable the 12-cell diamond shape window (default off)
  --use-p13        enable the p1 × p12 (diamond × tactical-mask) window (default off)
  --captures       enable size-graded capture buckets (10 weights, one per
                   capture size 1..10; size >=10 fires the 10th bucket) (default off)
  --eval-agent S   reference agent in ai/ (default random)
  --ladder-file P  evalladders2 suite scored each status print (ladr column)
  --md-file P      evalmovedetails positions scored each status print; RMS gap
                   of the greedy npat policy to best (mdRms column)
  --load PATH      resume from saved weights
  --save PATH      where to save (default out/npat-<rand>.js)
`);
  process.exit(0);
}

const { getMove: evalGetMove } =
  require(path.join(__dirname, 'ai', EVAL_AGENT + '.js'));

if (LOAD_PATH) {
  if (fs.existsSync(LOAD_PATH)) {
    const loaded = loadWeights(LOAD_PATH);
    weights = loaded.weights;
    ema     = loaded.ema;
    totalUpdates = loaded.totalUpdates;
    console.log(`Loaded ${weights.size} weights from ${LOAD_PATH}  (ema=${ema.toFixed(3)}, totalUpdates=${totalUpdates})`);
  } else {
    console.warn(`Warning: --load file not found: ${LOAD_PATH}`);
  }
}

console.log(`lr=${LR}  reward-ema=${REWARD_EMA}  weight-decay=${WEIGHT_DECAY}  temperature=${TEMPERATURE}`);
console.log(`features: p1=${USE_P1 ? 'ON' : 'off'}  p5=${USE_P5 ? 'ON' : 'off'}  p8=${USE_P8 ? 'ON' : 'off'}  p9=${USE_P9 ? 'ON' : 'off'}  p12=${USE_P12 ? 'ON' : 'off'}  p13=${USE_P13 ? 'ON' : 'off'}  captures=${USE_CAPTURES ? 'ON' : 'off'}`);
console.log(`train-size=${TRAIN_SIZE}  eval-size=${EVAL_SIZE}  ref=${EVAL_AGENT}`);
console.log(`Out: ${SAVE_PATH}${LOAD_PATH ? `  (resumed from ${LOAD_PATH})` : ''}`);

// Ladder test: when --ladder-file is given, score the trainee's greedy npat
// policy against the evalladders2 suite at each status print (the `ladr` column).
const ladderCases = LADDER_FILE ? loadCases(LADDER_FILE) : null;
const _ladderStates = new Map();
function npatLadderMove(game) {
  let st = _ladderStates.get(game.N);
  if (!st) { st = NPat.createState(game.N); _ladderStates.set(game.N, st); }
  return { move: NPat.greedyMove(game, st, weights, game3FromGame2(game)) };
}
if (ladderCases) console.log(`ladder suite: ${LADDER_FILE} (${ladderCases.length} cases)`);

// Move-quality suite (evalmovedetails): a single full pass scoring the trainee's
// greedy npat policy against --md-file at each status print (the `mdRms` column).
const mdPositions = MD_FILE ? loadPositions(MD_FILE) : null;
if (mdPositions) console.log(`md positions: ${MD_FILE} (${mdPositions.length} positions)`);
console.log();

console.log([
  'game'.padStart(4),
  'tElp'.padStart(5),
  'tGm '.padStart(5),
  'nWts'.padStart(4),
  'avgW'.padStart(6),
  'u/pa'.padStart(4),
  'maxP'.padStart(4),
  // winRatio: "wr(g)/avg(ga)" — wr/avg are fmtRatio4, g/ga are fmt4 game counts
  // (this interval's, and the rolling-half window).  Fixed 21 chars wide.
  'winRatio'.padStart(21),
  ...(ladderCases ? ['ladr'.padStart(5)] : []),
  ...(mdPositions ? ['mdRms'.padStart(5)] : []),
].join('  '));

// ── Main loop ─────────────────────────────────────────────────────────────────

const t0 = Date.now();
let nextPrintAt = t0 + 1000;
let lastPrintAt = t0;
let g = 0;
let maxProbSumWindow = 0, maxProbNWindow = 0;
let lastPrintG = 0;
let elapsedMsAcc = 0;
const evalHistory = [];   // per-game eval results (1 / 0); rolling-half avgWR uses this

while (true) {
  g++;
  const r = trainGame(TRAIN_SIZE);
  totalUpdates += r.weightUpdates;
  maxProbSumWindow += r.maxProbSum;
  maxProbNWindow   += r.maxProbN;
  elapsedMsAcc     += r.elapsedMs;

  if (Date.now() >= nextPrintAt) {
    // Count only interned pids that have received a non-zero gradient: pids
    // are interned on first sight in extractFeatures, so weights.size counts
    // every pattern ever seen — the "learned" count is the non-zero subset.
    let wAbsSum = 0, wNonZero = 0;
    const vals = weights.vals;
    const wN   = weights.size;
    for (let i = 0; i < wN; i++) {
      const a = Math.abs(vals[i]);
      if (a === 0) continue;
      wNonZero++;
      wAbsSum += a;
    }
    const wAvg    = wNonZero > 0 ? wAbsSum / wNonZero : 0;
    const updPerPat = wNonZero > 0 ? totalUpdates / wNonZero : 0;

    const nextMs = Date.now() - t0;

    const maxPAvg = maxProbNWindow > 0 ? maxProbSumWindow / maxProbNWindow : 0;

    // Eval vs reference for up to 20% of the interval's training time, capped
    // at 1000 games.  greedy npat policy on this side vs evalGetMove on the
    // other side, colours alternating.
    const intervalMs   = Date.now() - lastPrintAt;
    const evalBudgetMs = intervalMs * 0.2;
    const evalStart    = Date.now();
    let evalWins = 0, evalGames = 0;
    while (evalGames < 1000 && Date.now() - evalStart < evalBudgetMs) {
      const results = evalVsReference(EVAL_SIZE, evalGetMove, 1);
      evalHistory.push(results[0]);
      evalWins += results[0];
      evalGames++;
    }
    const avgHalf = Math.max(1, Math.floor(evalHistory.length / 2));
    const avgWR   = evalHistory.length > 0
      ? evalHistory.slice(-avgHalf).reduce((s, r) => s + r, 0) / avgHalf
      : 0;

    // Ladder suite score (trainee greedy npat policy vs the evalladders2 file).
    let ladrStr = null;
    if (ladderCases) {
      const { passed, total } = evalCases(ladderCases, npatLadderMove, { budgetMs: 1, oversample: 1 });
      ladrStr = Util.fmtRatio4(total ? passed / total : 0);
    }

    // Move-quality RMS: one full pass of evalmovedetails over --md-file, scoring
    // the trainee's greedy npat policy (gap-to-best in win-ratio units).
    let mdRmsStr = null;
    if (mdPositions) {
      const { rmsErr } = evalPositions(npatLadderMove, mdPositions, 0);
      mdRmsStr = Util.fmtRatio4(rmsErr).padStart(5);
    }

    const elapsedMs      = Date.now() - t0;
    const cycleN         = g - lastPrintG;
    const cycleAvgGameMs = cycleN > 0 ? elapsedMsAcc / cycleN : 0;

    console.log([
      Util.fmt4i(g),
      Util.fmtMs(elapsedMs),
      Util.fmtMs(cycleAvgGameMs),
      Util.fmt4i(wNonZero),
      wAvg.toFixed(4).padStart(6),
      Util.fmt4(updPerPat),
      Util.fmtRatio4(maxPAvg),
      (`${Util.fmtRatio4(evalGames > 0 ? evalWins / evalGames : 0)}(${Util.fmt4i(evalGames)})` +
       `/${Util.fmtRatio4(avgWR)}(${Util.fmt4i(avgHalf)})`).padStart(21),
      ...(ladrStr !== null ? [ladrStr] : []),
      ...(mdRmsStr !== null ? [mdRmsStr] : []),
    ].join('  '));
    maxProbSumWindow = 0; maxProbNWindow = 0;
    elapsedMsAcc = 0;
    lastPrintG = g;

    saveWeights(SAVE_PATH, weights, { ema, totalUpdates });
    nextPrintAt = t0 + Math.round(nextMs * 1.4);
    lastPrintAt = Date.now();
  }
}
