#!/usr/bin/env node
'use strict';

// train-hpatterns.js — learn pattern weights via logistic TD(2) self-play.
//
// Value function (absolute, P(BLACK wins)):
//   V(s) = σ( Σ  polarity_i · w[key_i] )
//
// Update rule — logistic TD, 2-step lookahead (same player to move):
//   Δw_k = (LR / n_features) · (target − V) · polarity_k
//   target = V(s_{t+2})   for t+2 < game length  [bootstrap]
//          = 1 / 0.5 / 0  for terminal            [BLACK wins / draw / WHITE wins]
//
// Features: hierarchical pattern hashes, all sizes 2×2 … maxSize×maxSize.
// Canonicalised via algorithmic D4 + color-flip; hash collisions accepted.
//
// Runs indefinitely (Ctrl-C to stop).  Weights are saved at every print.

const path = require('path');
const fs   = require('fs');
const { Game2, BLACK, PASS } = require('./game2.js');
const { createModel, extractFeatures, evaluateFeatures, applyEMA, weightsMap } = require('./hpatterns.js');
const { loadCases, evalCases } = require('./evalladders2.js');
const { loadPositions, evalPositions } = require('./evalmovedetails.js');
const Util = require('./util.js');

// ── Arguments ─────────────────────────────────────────────────────────────────

const opts       = Util.parseArgs(process.argv.slice(2), [], ['smooth-weights', 'epsilon', 'eval', 'eval-size', 'ext', 'ladder-file', 'limit', 'load', 'lr', 'md-file', 'momentum', 'on-policy', 'save', 'size', 'spec', 'train-size']);
const TRAIN_SIZE = parseInt(opts['train-size']  || opts.size || '9',  10);
const EVAL_SIZE  = parseInt(opts['eval-size']   || opts.size || opts['train-size'] || '13', 10);
const SAVE_PATH  = opts.save  || `out/hpatterns-${Math.random().toString(36).slice(2, 10)}.js`;
const LOAD_PATH  = opts.load  || null;
const EVAL_AGENT = opts.eval || '';   // empty disables in-training reference test games
const EXT_AGENT  = opts.ext  || '';   // off-policy move source: 80% of moves come from this agent
const LADDER_FILE = opts['ladder-file'] || null;   // evalladders2 suite for the ladr column
const MD_FILE     = opts['md-file'] || null;       // evalmovedetails positions for the mdRms column
const EPSILON    = Math.min(parseFloat(opts.epsilon   || '0.1'),  1);
const ON_POLICY  = Math.min(parseFloat(opts['on-policy'] || '1'), 1);  // share of non-random moves from own search1ply (vs --ext)
const LR         = parseFloat(opts.lr               || '0.3');
const MOMENTUM   = parseFloat(opts.momentum         || '0.0');
const EMA_ALPHA  = parseFloat(opts['smooth-weights']     || '0.9');  // Polyak EMA, applied every EMA_PERIOD games; 0 = off.
                                                              // Window ≈ EMA_PERIOD/(1-alpha) games: 0.9 ≈ 1k, 0.99 ≈ 10k.
const EMA_PERIOD = 100;
// spec: "size:maxStones" pairs, e.g. "2:4,3:8,4:16". Sizes absent from spec are not
// extracted; a bare size ("4" or "4:") means unlimited (maxStones = size²).
const SPEC_RAW = opts.spec || '2:4';
const LIMIT_GAMES = opts.limit !== undefined ? parseInt(opts.limit, 10) : 0;
// Spec format: "size:max[f],..." — trailing 'f' freezes weights at that size
// (loaded values stay fixed; gradient updates skipped).
const SPEC = {};
const FROZEN = new Set();
for (const part of SPEC_RAW.split(',')) {
  const [k, vRaw] = part.split(':');
  const sz = parseInt(k, 10);
  const frozen = vRaw !== undefined && /f$/.test(vRaw);
  // Bare size ("--spec 4") means no stone limit: saturate at sz², the whole window.
  const digits = frozen ? vRaw.slice(0, -1) : vRaw;
  const v = digits ? parseInt(digits, 10) : sz * sz;
  SPEC[sz] = v;
  if (frozen) FROZEN.add(sz);
}
let MAX_SIZE   = Math.max(...Object.keys(SPEC).map(Number));
let MAX_STONES = SPEC;

// ── Model ─────────────────────────────────────────────────────────────────────

let model = createModel(MAX_STONES, MAX_SIZE);
const velocity = new Map();  // SGD momentum
let wAbsSum = 0, wUpdateCount = 0;  // per-interval |weight| sum/count over weight updates (avgW; reset each print)

// ── Persistence ───────────────────────────────────────────────────────────────

// Save base64-packed as Int32 keys followed by int16-quantised vals: a per-file
// symmetric scale maps [-maxAbs, maxAbs] onto int16, weight ≈ qval / scale.
// Far smaller and faster for V8 to parse than a literal Map.  Loaded by
// hpatterns.js weightsMap (which also still reads legacy Map files).
function saveModel(filePath, m) {
  const maxStonesStr = JSON.stringify(m.maxStones);
  const maxSizeStr   = m.maxSize === Infinity ? 'Infinity' : m.maxSize;
  // Persist Polyak-averaged weights when available (better for eval, slight
  // rewind on resume).  Falls back to live weights before the first applyEMA.
  const useEMA = m.weightsEMAInit;
  const source = useEMA ? m.weightsEMA : m.weights;   // Map<key, float>

  const count = source.size;
  let maxAbs = 0;
  for (const v of source.values()) { const a = v < 0 ? -v : v; if (a > maxAbs) maxAbs = a; }
  const scale = maxAbs > 0 ? 32767 / maxAbs : 1;

  const keys  = new Int32Array(count);
  const qvals = new Int16Array(count);
  let i = 0;
  for (const [k, v] of source) {
    keys[i] = k;
    let q = Math.round(v * scale);
    if (q > 32767) q = 32767; else if (q < -32768) q = -32768;
    qvals[i] = q;
    i++;
  }
  // Int32 keys (count*4 bytes) then Int16 qvals (count*2); count*4 is 2-aligned.
  const buf = Buffer.alloc(count * 6);
  Buffer.from(keys.buffer,  keys.byteOffset,  count * 4).copy(buf, 0);
  Buffer.from(qvals.buffer, qvals.byteOffset, count * 2).copy(buf, count * 4);
  const b64 = buf.toString('base64');

  const src = [
    "'use strict';",
    '// Auto-generated by train-hpatterns.js — do not edit by hand.',
    '// Weights int16-quantised: weight = qvals[i] / scale.',
    'const hpatternsModel = (() => {',
    `  const count = ${count};`,
    `  const scale = ${scale};`,
    `  const maxStones = ${maxStonesStr};`,
    `  const maxSize = ${maxSizeStr};`,
    `  const weightsAreEMA = ${useEMA};`,
    `  const b64 = '${b64}';`,
    "  const bytes = typeof Buffer !== 'undefined'",
    "    ? Buffer.from(b64, 'base64')",
    "    : Uint8Array.from(atob(b64), c => c.charCodeAt(0));",
    "  const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + count * 6);",
    "  const keys  = new Int32Array(buf, 0, count);",
    "  const qvals = new Int16Array(buf, count * 4, count);",
    "  return { maxStones, maxSize, weightsAreEMA, count, scale, keys, qvals };",
    "})();",
    "if (typeof module !== 'undefined') module.exports = hpatternsModel;",
    "else window.hpatternsModel = hpatternsModel;",
  ].join('\n') + '\n';
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, src);
}

function loadModel(filePath) {
  const raw = require(path.resolve(filePath));
  return createModelWithWeights(raw.maxStones, raw.maxSize === Infinity ? Infinity : raw.maxSize,
    weightsMap(raw));
}

// createModel doesn't accept an initial weights map; patch it in.  Also seed
// the EMA shadow from the loaded values so subsequent applyEMA continues to
// average on top of the persisted (already-EMA) weights.
function createModelWithWeights(maxStones, maxSize, weights) {
  const m = createModel(maxStones, maxSize);
  m.weights = weights;
  if (EMA_ALPHA > 0) {          // shadow only exists when EMA is enabled
    m.weightsEMA = new Map(weights);
    m.weightsEMAInit = true;
  }
  return m;
}

// ── Training helpers ──────────────────────────────────────────────────────────

function absoluteOutcome(game) {
  return game.calcWinner() === BLACK ? 1 : 0;
}

function tdUpdate(features, target, lr) {
  const n = features.count;
  if (n === 0) return;
  const { keys, pols, sizes } = features;
  const hasFrozen = FROZEN.size > 0;
  let nActive = n;
  if (hasFrozen) {
    nActive = 0;
    for (let i = 0; i < n; i++) if (!FROZEN.has(sizes[i])) nActive++;
    if (nActive === 0) return;
  }
  const perFeature = (target - features.val) / nActive;
  if (MOMENTUM === 0) {
    const step = lr * perFeature;
    for (let i = 0; i < n; i++) {
      if (hasFrozen && FROZEN.has(sizes[i])) continue;
      const k = keys[i];
      const w = (model.weights.get(k) ?? 0) + pols[i] * step;
      model.weights.set(k, w);
      wAbsSum += Math.abs(w); wUpdateCount++;
    }
  } else {
    for (let i = 0; i < n; i++) {
      if (hasFrozen && FROZEN.has(sizes[i])) continue;
      const k   = keys[i];
      const g   = pols[i] * perFeature;
      const vel = MOMENTUM * (velocity.get(k) ?? 0) + g;
      velocity.set(k, vel);
      const w = (model.weights.get(k) ?? 0) + lr * vel;
      model.weights.set(k, w);
      wAbsSum += Math.abs(w); wUpdateCount++;
    }
  }
}

// ── 1-ply search ──────────────────────────────────────────────────────────────

// Shallow depth for two-pass search: fast first pass to score all moves cheaply.
const SHALLOW_DEPTH = 3;
// Temperature for softmax-based full-eval selection (value space, 0–1).
// Moves with player-score within ~TEMP of the best are evaluated with high probability;
// moves further away are exponentially less likely to be fully evaluated.
// The best shallow move always has p=1 and is always fully evaluated.
const SHALLOW_TEMP  = 0.05;

function search1ply(game, maxSearch, w = model.weights) {
  const area    = game.N * game.N;
  const isBlack = game.current === BLACK;

  // Collect legal non-eye moves; include PASS when relevant.
  // PASS = -1 < 0, so extractFeatures treats it as non-speculative (correct: PASS doesn't change board).
  const coords = [];
  for (let coord = 0; coord < area; coord++) {
    if (!game.isLegal(coord) || game.isTrueEye(coord)) continue;
    coords.push(coord);
  }
  if (game.consecutivePasses > 0 || game.emptyCount < area / 2) coords.push(PASS);
  if (coords.length === 0) return PASS;

  // If already at or below shallow depth, single pass suffices.
  const doTwoPass = maxSearch === undefined || maxSearch > SHALLOW_DEPTH;
  const depth1    = doTwoPass ? SHALLOW_DEPTH : maxSearch;

  // ── Pass 1: shallow evaluation ─────────────────────────────────────────────
  const vals1  = new Float64Array(coords.length);
  let   best1  = isBlack ? -Infinity : Infinity;
  let   bestMove = coords[0];

  for (let i = 0; i < coords.length; i++) {
    const val = evaluateFeatures(extractFeatures(game, model, depth1, coords[i]), w);
    vals1[i]  = val;
    if (isBlack ? val > best1 : val < best1) { best1 = val; bestMove = coords[i]; }
  }

  if (!doTwoPass) return bestMove;

  // ── Pass 2: full evaluation, softmax-selected ──────────────────────────────
  // p(full eval) = exp((player_score − best_player_score) / SHALLOW_TEMP)
  // Best shallow move has diff=0 → p=1, so it is always fully evaluated.
  let bestVal = isBlack ? -Infinity : Infinity;

  for (let i = 0; i < coords.length; i++) {
    const diff = isBlack ? vals1[i] - best1 : best1 - vals1[i];
    if (Math.random() >= Math.exp(diff / SHALLOW_TEMP)) continue;
    const val = evaluateFeatures(extractFeatures(game, model, maxSearch, coords[i]), w);
    if (isBlack ? val > bestVal : val < bestVal) { bestVal = val; bestMove = coords[i]; }
  }

  return bestMove;
}

// ── Self-play training ────────────────────────────────────────────────────────

function trainGame(N) {
  const game     = new Game2(N);   // free initial stone (applyFirstMove=true)
  const maxMoves = N * N * 4;
  const tStartMs = Date.now();

  let prev2 = null, prev1 = null;
  let moves = 0;
  const vals = [];
  let maxSearch = MAX_SIZE;  // shrinks when a level has no eligible patterns; reset on capture/new game

  while (!game.gameOver && moves < maxMoves) {
    const f = extractFeatures(game, model, maxSearch);
    f.val = evaluateFeatures(f, model.weights);
    vals.push(f.val);

    if (prev2 !== null) tdUpdate(prev2, f.val, LR);

    prev2 = prev1;
    prev1 = { keys: f.keys.slice(0, f.count), pols: f.pols.slice(0, f.count), sizes: f.sizes.slice(0, f.count), count: f.count, val: f.val };

    // Move source: EPSILON random; of the (1-epsilon) remainder, ON_POLICY share
    // from own search1ply and the rest from --ext.  Independent draws, so
    // ON_POLICY acts on the remainder (consistent with train-vpatterns/vlibpat).
    let move;
    if (Math.random() < EPSILON) {
      move = game.randomLegalMove();
    } else if (extGetMove && Math.random() > ON_POLICY) {
      move = extGetMove(game).move;
    } else {
      move = search1ply(game, maxSearch);
    }
    const hasCaptures = move !== PASS && game.captureList(move).length > 0;
    game.play(move);

    // After a capture, stones are removed so higher levels may become eligible again.
    maxSearch = hasCaptures ? MAX_SIZE : f.topLevel;
    moves++;
  }

  const outcome = absoluteOutcome(game);

  if (prev2 !== null) tdUpdate(prev2, outcome, LR);
  if (prev1 !== null) tdUpdate(prev1, outcome, LR);

  return { elapsedMs: Date.now() - tStartMs, moves };
}

// ── Evaluation against a reference agent ─────────────────────────────────────

function evalVsReference(N, refGetMove, nGames) {
  const results = [];
  // Eval ≡ save: matches and acc measure the model save would write — the
  // EMA shadow when enabled (and initialized), else the live weights.
  const evalW = (EMA_ALPHA > 0 && model.weightsEMAInit) ? model.weightsEMA : model.weights;
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
      gameVals.push(evaluateFeatures(extractFeatures(game, model, MAX_SIZE), evalW));
      let idx;
      if ((game.current === BLACK) === policyIsBlack) {
        idx = search1ply(game, undefined, evalW);
      } else {
        const mv = refGetMove(game);
        idx = mv.move !== undefined ? mv.move : PASS;
      }
      game.play(idx);
    }

    const winner = game.calcWinner();
    for (const v of gameVals) if ((v >= 0.5) === (winner === BLACK)) accCorrect++;
    accN += gameVals.length;
    results.push((winner === BLACK) === policyIsBlack ? 1 : 0);
  }
  return { results, accCorrect, accN };
}

// ── CLI setup ─────────────────────────────────────────────────────────────────

const evalGetMove = EVAL_AGENT
  ? require(path.join(__dirname, 'ai', EVAL_AGENT + '.js')).getMove
  : null;
const extGetMove = EXT_AGENT
  ? require(path.join(__dirname, 'ai', EXT_AGENT + '.js')).getMove
  : null;

if (LOAD_PATH) {
  if (fs.existsSync(LOAD_PATH)) {
    const raw = require(path.resolve(LOAD_PATH));
    // Union of file spec and CLI spec; for sizes present in both, take the larger stone limit.
    MAX_STONES = Object.assign({}, raw.maxStones);
    for (const [k, v] of Object.entries(SPEC))
      MAX_STONES[k] = Math.max(MAX_STONES[k] ?? 0, v);
    MAX_SIZE = Math.max(...Object.keys(MAX_STONES).map(Number));
    model = createModelWithWeights(MAX_STONES, MAX_SIZE, weightsMap(raw));
    console.log(`Loaded ${model.weights.size} weights from ${LOAD_PATH}`);
  } else {
    console.warn(`Warning: --load file not found: ${LOAD_PATH}`);
  }
}

console.log(`LR=${LR}  momentum=${MOMENTUM}  epsilon=${EPSILON}  on-policy=${ON_POLICY}  smooth-weights=${EMA_ALPHA}  train-size=${TRAIN_SIZE}  eval-size=${EVAL_SIZE}  ref=${EVAL_AGENT || '(none)'}  ext=${EXT_AGENT || '(none)'}`);
console.log(`spec=${SPEC_RAW}${FROZEN.size > 0 ? `  frozen=[${[...FROZEN].join(',')}]` : ''}`);
console.log(`Out: ${SAVE_PATH}${LOAD_PATH ? `  (resumed from ${LOAD_PATH})` : ''}`);

// Ladder suite (evalladders2): score the trainee's own 1-ply argmax against
// --ladder-file at each status print (the `ladr` column).
const ladderCases = LADDER_FILE ? loadCases(LADDER_FILE) : null;
const ladderAgent = gm => ({ move: gm.gameOver ? PASS : search1ply(gm) });
if (ladderCases) console.log(`ladder suite: ${LADDER_FILE} (${ladderCases.length} cases)`);

// Move-quality suite (evalmovedetails): a single full pass scoring the
// trainee's 1-ply argmax against --md-file at each status print (the `mdRms`
// column — RMS win-ratio gap to the top move).
const mdPositions = MD_FILE ? loadPositions(MD_FILE) : null;
const mdAgent = gm => ({ move: gm.gameOver ? PASS : search1ply(gm) });
if (mdPositions) console.log(`md positions: ${MD_FILE} (${mdPositions.length} positions)`);
console.log();

// Training columns (left).
const headerCols = [
  'game'.padStart(4),
  'tGm '.padStart(5),
  'nWts'.padStart(4),
  'avgL'.padStart(4),
  'avgW'.padStart(6),
  'tTran'.padStart(5),
  'tTurn'.padStart(5),
];
// Test / eval columns (right).
// winRatio: "wr(g)/avg(ga)" — wr/avg fmtRatio4, g/ga fmt4 game counts (this
// interval's, and the rolling-half window).  Fixed 21 chars wide.
if (evalGetMove) headerCols.push('winRatio'.padStart(21), ' acc'.padStart(4));
if (ladderCases) headerCols.push('ladr'.padStart(4));
if (mdPositions) headerCols.push('mdRms'.padStart(5));
headerCols.push('tTest'.padStart(5));
console.log(headerCols.join('  '));

// ── Main loop ─────────────────────────────────────────────────────────────────

const t0 = Date.now();
let nextPrintAt = t0 + 1000;
let g = 0;
let totalMoves = 0;
let intervalGames = 0, intervalMoves = 0;
let moveElapsedMs = 0;
let intervalTrainMs = 0;
const evalHistory = [];

while (true) {
  g++;
  const { moves, elapsedMs } = trainGame(TRAIN_SIZE);
  // Polyak / SWA: fold live weights into the shadow every EMA_PERIOD games.
  if (EMA_ALPHA > 0 && g % EMA_PERIOD === 0) applyEMA(model, EMA_ALPHA);
  totalMoves      += moves;
  intervalGames++;
  intervalMoves   += moves;
  moveElapsedMs   += elapsedMs;  intervalTrainMs += elapsedMs;

  // Force the print/save block to fire on the limit-reaching iteration so
  // the final stats are emitted before we break.
  const limitReached = LIMIT_GAMES > 0 && g >= LIMIT_GAMES;
  if (limitReached) nextPrintAt = 0;

  if (Date.now() >= nextPrintAt) {
    const tTestStart = Date.now();
    let batch = null, latestWR = 0, avgWR = 0, evalHalf = 0, evalAccC = 0, evalAccN = 0;
    if (evalGetMove) {
      batch = [];
      while (true) {
        const { results, accCorrect, accN } = evalVsReference(EVAL_SIZE, evalGetMove, 2);
        for (const r of results) batch.push(r);
        evalAccC += accCorrect; evalAccN += accN;
        const tMs = Date.now() - tTestStart;
        if (tMs > 0.3 * intervalTrainMs || batch.length >= 998) break;
      }
      for (const r of batch) evalHistory.push(r);
      latestWR  = batch.reduce((s, r) => s + r, 0) / batch.length;
      evalHalf  = Math.max(1, Math.floor(evalHistory.length / 2));
      avgWR     = evalHistory.slice(-evalHalf).reduce((s, r) => s + r, 0) / evalHalf;
    }
    const avgLen    = intervalMoves / intervalGames;
    const tGameMs   = intervalTrainMs / intervalGames;
    const tpMove    = moveElapsedMs / totalMoves;

    intervalGames   = 0;
    intervalMoves   = 0;

    const ws   = model.weights.size;
    // avgW: mean |weight| encountered across weight updates (frequency-weighted
    // active weights), not the mean over all stored weights.
    const wAvg = wUpdateCount > 0 ? wAbsSum / wUpdateCount : 0;
    wAbsSum = 0; wUpdateCount = 0;   // per-interval avgW: reset at each print

    const trainMs   = intervalTrainMs;
    intervalTrainMs = 0;

    // Ladder suite score (trainee's own 1-ply argmax vs --ladder-file).
    let ladrRatio = null;
    if (ladderCases) {
      const { passed, total } = evalCases(ladderCases, ladderAgent, { budgetMs: 1, oversample: 1 });
      ladrRatio = total ? passed / total : 0;
    }

    // Move-quality RMS: one full pass of evalmovedetails over --md-file.
    let mdRms = null;
    if (mdPositions) {
      mdRms = evalPositions(mdAgent, mdPositions, 0).rmsErr;
    }

    // Total eval time this print: reference matches + ladder suite + moveDetails.
    const tTestMs = Date.now() - tTestStart;

    // Training columns (left).
    const cols = [
      Util.fmt4i(g),
      Util.fmtMs(tGameMs),
      Util.fmt4i(ws),
      Util.fmt4(avgLen),
      wAvg.toFixed(4).padStart(6),
      Util.fmtMs(trainMs),
      Util.fmtMs(tpMove),
    ];
    // Test / eval columns (right).
    if (evalGetMove) cols.push((`${Util.fmtRatio4(latestWR)}(${Util.fmt4i(batch.length)})` +
                                `/${Util.fmtRatio4(avgWR)}(${Util.fmt4i(evalHalf)})`).padStart(21),
                               Util.fmtRatio4(evalAccN > 0 ? evalAccC / evalAccN : 0));
    if (ladrRatio !== null) cols.push(Util.fmtRatio4(ladrRatio));
    if (mdRms !== null) cols.push(Util.fmtRatio4(mdRms).padStart(5));
    cols.push(Util.fmtMs(tTestMs));
    console.log(cols.join('  '));

    saveModel(SAVE_PATH, model);
    // Schedule the next print.  Keep the geometric growth on total elapsed
    // time, but also require the next interval to spend at least as long
    // TRAINING as this test cycle took — otherwise a slow test (e.g. a large
    // --md-file pass) outruns the geometric target and forces one training
    // game per print.  tTestMs is measured after the full test, so it captures
    // the ref/ladder/md eval cost the premature schedule used to miss.
    const nowMs = Date.now();
    const geometricAt = t0 + Math.round((nowMs - t0) * 1.4);
    nextPrintAt = Math.max(geometricAt, nowMs + tTestMs);
  }

  if (limitReached) {
    console.log();
    console.log(`Reached --limit ${LIMIT_GAMES} games — saved ${SAVE_PATH}`);
    break;
  }
}
