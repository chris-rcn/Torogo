#!/usr/bin/env node
'use strict';

// train-hpat-playout-eval.js — supervised training of hpatterns weights on a
// gen-playout-evals data file (static playout-value evaluator).
//
// Value function (absolute, P(BLACK wins)) — same as train-hpatterns:
//   V(s) = σ( Σ  polarity_i · w[key_i] )
//
// Update rule — logistic regression on the corpus label:
//   Δw_k = (LR / n_features) · (target − V) · polarity_k
//   target = the record's winRatio (P(side-to-move wins)) mapped to P(BLACK wins)
//
// Data lines (gen-playout-evals.js):  <bsize> <phase> <move1,...> <winRatio>
// Every record is replay-validated at load; the recorded phase is checked
// against the replayed position (an integrity check, since it is redundant).
// The first --test-pos records form a held-out test set (teMSE column); the
// rest are the train pool, visited in a fresh shuffle each epoch.
//
// Runs indefinitely (Ctrl-C to stop) unless --limit is given.  Weights are
// saved at every print.

const path = require('path');
const fs   = require('fs');
const { Game2, BLACK, PASS, KOMI, parseMove } = require('./game2.js');
const { createModel, extractFeatures, evaluateFeatures, applyEMA, weightsMap,
        saturatedOnly, zFromBuffers, deltaZ } = require('./hpatterns.js');
const { loadCases, evalCases } = require('./evalladders2.js');
const { loadPositions, evalPositions } = require('./evalmovedetails.js');
const Util = require('./util.js');

// ── Arguments ─────────────────────────────────────────────────────────────────

const opts = Util.parseArgs(process.argv.slice(2), ['no-add', 'help'],
  ['data', 'test-pos', 'smooth-weights', 'eval', 'eval-size', 'ladder-file',
   'limit', 'load', 'lr', 'md-file', 'momentum', 'save', 'spec']);
if (opts.help || !opts.data) {
  console.log(`Usage: node train-hpat-playout-eval.js --data <file> [options]

Supervised trainer: fits hpatterns weights to the playout-value labels of a
gen-playout-evals data file (logistic regression on P(BLACK wins)).  Runs
indefinitely unless --limit is given; the checkpoint is written at every print.

  --data FILE       gen-playout-evals data file (required)
  --test-pos N      first N records form the held-out test set for the teMSE
                    column (default 1000, clamped to half the file)
  --limit N         stop after N training positions (default 0 = run indefinitely)

  --spec S          sizes to extract, as "size:maxStones" pairs (default 2:4).
                    A bare size ("4" or "4:") means no stone limit
                    (maxStones = size^2); a trailing 'f' freezes that size's
                    loaded weights, e.g. '2:4f,3:8'
  --lr F            step size for the update (default 0.3)
  --momentum F      SGD momentum (default 0 = off)
  --smooth-weights A  Polyak EMA decay, applied every 1000 positions; 0 = off
                    (default 0.9).  The EMA weights are what gets saved once it
                    has run, and what teMSE / eval games measure

  --load PATH       resume from a checkpoint.  Its spec is unioned with --spec
                    (larger stone limit wins per size)
  --no-add          fine-tune ONLY the patterns already in the loaded model:
                    a feature whose key is absent is skipped entirely and takes
                    no share of the error.  Needs a non-empty --load
  --save PATH       checkpoint path (default out/hpat-pe-<random>.js)

  --eval AGENT      ai/<name>.js played as the reference in test games
                    (default: none, which disables the test games)
  --eval-size N     evaluation board size (default 13)
  --ladder-file F   evalladders2 suite scored each print (ladr column)
  --md-file F       evalmovedetails positions scored each print (mdRms column)
  --help            show this message`);
  process.exit(opts.help ? 0 : 1);
}

const DATA_PATH  = opts.data;
const EVAL_SIZE  = parseInt(opts['eval-size'] || '13', 10);
const SAVE_PATH  = opts.save || `out/hpat-pe-${Math.random().toString(36).slice(2, 10)}.js`;
const LOAD_PATH  = opts.load || null;
const NO_ADD     = opts['no-add'] === true;
const EVAL_AGENT = opts.eval || '';
const LADDER_FILE = opts['ladder-file'] || null;
const MD_FILE     = opts['md-file'] || null;
const LR         = parseFloat(opts.lr       || '0.3');
const MOMENTUM   = parseFloat(opts.momentum || '0.0');
const EMA_ALPHA  = parseFloat(opts['smooth-weights'] || '0.9');
const EMA_PERIOD = 1000;   // positions between applyEMA folds
const SPEC_RAW   = opts.spec || '2:4';
const LIMIT_POS  = opts.limit !== undefined ? parseInt(opts.limit, 10) : 0;
const TEST_POS_RAW = opts['test-pos'] !== undefined ? parseInt(opts['test-pos'], 10) : 1000;

// Eval games use the game2 per-size komi (there is no self-play here, so no
// komi controller either).
const EVAL_KOMI = KOMI(EVAL_SIZE);

// Spec format: "size:max[f],..." — trailing 'f' freezes weights at that size.
const SPEC = {};
const FROZEN = new Set();
for (const part of SPEC_RAW.split(',')) {
  const [k, vRaw] = part.split(':');
  const sz = parseInt(k, 10);
  const frozen = vRaw !== undefined && /f$/.test(vRaw);
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
let wAbsSum = 0, wUpdateCount = 0;  // per-interval |weight| sum/count (avgW; reset each print)

// ── Persistence (same checkpoint format as train-hpatterns) ──────────────────

function saveModel(filePath, m) {
  const maxStonesStr = JSON.stringify(m.maxStones);
  const maxSizeStr   = m.maxSize === Infinity ? 'Infinity' : m.maxSize;
  const useEMA = m.weightsEMAInit;
  const source = useEMA ? m.weightsEMA : m.weights;   // Map<key, float>

  const count = source.size;
  let maxAbs = 0;
  source.forEach((k, v) => { const a = v < 0 ? -v : v; if (a > maxAbs) maxAbs = a; });
  const scale = maxAbs > 0 ? 32767 / maxAbs : 1;

  const keys  = new Int32Array(count);
  const qvals = new Int16Array(count);
  let i = 0;
  source.forEach((k, v) => {
    keys[i] = k;
    let q = Math.round(v * scale);
    if (q > 32767) q = 32767; else if (q < -32768) q = -32768;
    qvals[i] = q;
    i++;
  });
  const buf = Buffer.alloc(count * 6);
  Buffer.from(keys.buffer,  keys.byteOffset,  count * 4).copy(buf, 0);
  Buffer.from(qvals.buffer, qvals.byteOffset, count * 2).copy(buf, count * 4);
  const b64 = buf.toString('base64');

  const src = [
    "'use strict';",
    '// Auto-generated by train-hpat-playout-eval.js — do not edit by hand.',
    '// Weights int16-quantised: weight = qvals[i] / scale.',
    'const hpatternsModel = (() => {',
    `  const count = ${count};`,
    `  const scale = ${scale};`,
    `  const maxStones = ${maxStonesStr};`,
    `  const maxSize = ${maxSizeStr};`,
    `  const komi = ${EVAL_KOMI};`,
    `  const trainMs = ${PRIOR_TRAIN_MS + (Date.now() - t0)};`,
    `  const weightsAreEMA = ${useEMA};`,
    `  const b64 = '${b64}';`,
    "  const bytes = typeof Buffer !== 'undefined'",
    "    ? Buffer.from(b64, 'base64')",
    "    : Uint8Array.from(atob(b64), c => c.charCodeAt(0));",
    "  const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + count * 6);",
    "  const keys  = new Int32Array(buf, 0, count);",
    "  const qvals = new Int16Array(buf, count * 4, count);",
    "  return { maxStones, maxSize, komi, trainMs, weightsAreEMA, count, scale, keys, qvals };",
    "})();",
    "if (typeof module !== 'undefined') module.exports = hpatternsModel;",
    "else window.hpatternsModel = hpatternsModel;",
  ].join('\n') + '\n';
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, src);
}

function createModelWithWeights(maxStones, maxSize, weights) {
  const m = createModel(maxStones, maxSize);
  m.weights = weights;
  if (EMA_ALPHA > 0) {
    m.weightsEMA = weights.clone();
    m.weightsEMAInit = true;
  }
  return m;
}

// ── Training update (same rule as train-hpatterns) ────────────────────────────

function tdUpdate(features, target, lr) {
  const n = features.count;
  if (n === 0) return;
  const { keys, pols, sizes } = features;
  const hasFrozen = FROZEN.size > 0;
  const skip = (i) => (hasFrozen && FROZEN.has(sizes[i])) ||
                      (NO_ADD && model.weights.get(keys[i]) === undefined);
  let nActive = n;
  if (hasFrozen || NO_ADD) {
    nActive = 0;
    for (let i = 0; i < n; i++) if (!skip(i)) nActive++;
    if (nActive === 0) return;
  }
  const perFeature = (target - features.val) / nActive;
  if (MOMENTUM === 0) {
    const step = lr * perFeature;
    for (let i = 0; i < n; i++) {
      if (skip(i)) continue;
      const k = keys[i];
      const w = (model.weights.get(k) ?? 0) + pols[i] * step;
      model.weights.set(k, w);
      wAbsSum += Math.abs(w); wUpdateCount++;
    }
  } else {
    for (let i = 0; i < n; i++) {
      if (skip(i)) continue;
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

// ── 1-ply search (for eval games / ladder / md suites only) ──────────────────

const TIE_BREAK = 1e-9;
function tieBreak() { return (Math.random() - 0.5) * TIE_BREAK; }

const SHALLOW_DEPTH = 3;
const SHALLOW_TEMP  = 0.05;
let INCREMENTAL = false;

function _withFallbackBufs(fn) {
  if (!model._fbBufs) {
    model._fbBufs  = model._hBufs.map(b => new Int32Array(b.length));
    model._fbBufsI = model._hBufsInv.map(b => new Int32Array(b.length));
  }
  const sN = model._hBufs, sI = model._hBufsInv;
  model._hBufs = model._fbBufs; model._hBufsInv = model._fbBufsI;
  const r = fn();
  model._hBufs = sN; model._hBufsInv = sI;
  return r;
}

function _specVal(game, coord, depth, zBase, w) {
  const d = deltaZ(game, model, w, coord, depth);
  if (!Number.isNaN(d)) return 1 / (1 + Math.exp(-(zBase + d)));
  return _withFallbackBufs(() =>
    evaluateFeatures(extractFeatures(game, model, depth, coord), w));
}

function search1ply(game, maxSearch, w = model.weights) {
  const area    = game.N * game.N;
  const isBlack = game.current === BLACK;

  const coords = [];
  for (let coord = 0; coord < area; coord++) {
    if (!game.isLegal(coord) || game.isTrueEye(coord)) continue;
    coords.push(coord);
  }
  if (game.consecutivePasses > 0 || game.emptyCount < area / 2) coords.push(PASS);
  if (coords.length === 0) return PASS;

  const doTwoPass = maxSearch === undefined || maxSearch > SHALLOW_DEPTH;
  const depth1    = doTwoPass ? SHALLOW_DEPTH : maxSearch;

  let zCum = null, zShallow = 0, zFull = 0;
  if (INCREMENTAL) {
    extractFeatures(game, model, maxSearch);
    zCum = zFromBuffers(model, w, game.N, maxSearch);
    const top = zCum.length - 1;
    zShallow = zCum[Math.min(Math.max(depth1, 2), top)];
    zFull    = zCum[top];
  }

  const vals1  = new Float64Array(coords.length);
  let   best1  = isBlack ? -Infinity : Infinity;
  let   bestMove = coords[0];

  for (let i = 0; i < coords.length; i++) {
    const val = INCREMENTAL
      ? _specVal(game, coords[i], depth1, zShallow, w)
      : evaluateFeatures(extractFeatures(game, model, depth1, coords[i]), w);
    vals1[i]  = val;
    if (isBlack ? val + tieBreak() > best1 : val + tieBreak() < best1) { best1 = val; bestMove = coords[i]; }
  }

  if (!doTwoPass) return bestMove;

  let bestVal = isBlack ? -Infinity : Infinity;
  for (let i = 0; i < coords.length; i++) {
    const diff = isBlack ? vals1[i] - best1 : best1 - vals1[i];
    if (Math.random() >= Math.exp(diff / SHALLOW_TEMP)) continue;
    const val = INCREMENTAL
      ? _specVal(game, coords[i], maxSearch, zFull, w)
      : evaluateFeatures(extractFeatures(game, model, maxSearch, coords[i]), w);
    if (isBlack ? val + tieBreak() > bestVal : val + tieBreak() < bestVal) { bestVal = val; bestMove = coords[i]; }
  }

  return bestMove;
}

// ── Evaluation against a reference agent (same as train-hpatterns) ───────────

function evalVsReference(N, refGetMove, nGames) {
  const results = [];
  const evalW = (EMA_ALPHA > 0 && model.weightsEMAInit) ? model.weightsEMA : model.weights;
  for (let g = 0; g < nGames; g++) {
    const policyIsBlack = (g % 2 === 0);
    const game     = new Game2(N);   // free initial stone (applyFirstMove=true)
    for (let r = 0; r < 3 && !game.gameOver; r++) game.play(game.randomLegalMove());
    const maxMoves = N * N * 4;
    let   moves    = 0;
    while (!game.gameOver && moves++ < maxMoves) {
      let idx;
      if ((game.current === BLACK) === policyIsBlack) {
        idx = search1ply(game, undefined, evalW);
      } else {
        const mv = refGetMove(game);
        idx = mv.move !== undefined ? mv.move : PASS;
      }
      game.play(idx);
    }
    results.push((game.calcWinner() === BLACK) === policyIsBlack ? 1 : 0);
  }
  return { results };
}

// ── Data loading ──────────────────────────────────────────────────────────────

// Records: { size, moves: Int16Array, targetB }.  Every record is replayed once
// here: the replay validates the move sequence, supplies side-to-move for the
// target mapping, and cross-checks the recorded phase (redundant by
// construction, so a mismatch means a corrupt or foreign file).
const records = [];
{
  let malformed = 0, badReplay = 0, badPhase = 0;
  for (const line of fs.readFileSync(DATA_PATH, 'utf8').split('\n')) {
    if (!line || line[0] === '#') continue;
    const p = line.split(/\s+/);
    if (p.length !== 4) { malformed++; continue; }
    const size  = parseInt(p[0], 10);
    const phase = parseFloat(p[1]);
    const wr    = parseFloat(p[3]);
    if (!Number.isFinite(size) || !Number.isFinite(phase) ||
        !(wr >= 0 && wr <= 1)) { malformed++; continue; }
    const toks = p[2].split(',');
    const moves = new Int16Array(toks.length);
    let ok = true;
    for (let i = 0; i < toks.length; i++) {
      const m = parseMove(toks[i], size);
      if (!Number.isInteger(m) || m < PASS || m >= size * size) { ok = false; break; }
      moves[i] = m;
    }
    if (!ok) { malformed++; continue; }

    const game = new Game2(size);
    for (let i = 0; i < moves.length && ok; i++) ok = game.play(moves[i]);
    if (!ok) { badReplay++; continue; }
    // Recorded phase is toFixed(3): allow the rounding half-quantum plus slack.
    if (Math.abs(game.phase() - phase) > 0.0006) { badPhase++; continue; }

    const targetB = game.current === BLACK ? wr : 1 - wr;   // P(BLACK wins)
    records.push({ size, moves, targetB });
  }
  if (malformed || badReplay || badPhase) {
    console.error(`data: dropped ${malformed} malformed, ${badReplay} failed-replay, ` +
                  `${badPhase} phase-mismatch line(s)`);
  }
  if (records.length === 0) { console.error(`data '${DATA_PATH}' contains no valid records`); process.exit(1); }
}

// Held-out test head (teMSE), then the train pool.
const TEST_POS = Math.min(TEST_POS_RAW, records.length >> 1);
const testRecs  = records.slice(0, TEST_POS);
const trainRecs = records.slice(TEST_POS);

function replayRecord(rec) {
  const game = new Game2(rec.size);
  for (let i = 0; i < rec.moves.length; i++) game.play(rec.moves[i]);
  return game;
}

// Full pass over the test head with the save-eval weights (EMA when running).
function testMSE() {
  if (testRecs.length === 0) return null;
  const evalW = (EMA_ALPHA > 0 && model.weightsEMAInit) ? model.weightsEMA : model.weights;
  let se = 0;
  for (const rec of testRecs) {
    const v = evaluateFeatures(extractFeatures(replayRecord(rec), model, MAX_SIZE), evalW);
    se += (rec.targetB - v) * (rec.targetB - v);
  }
  return se / testRecs.length;
}

// ── CLI setup ─────────────────────────────────────────────────────────────────

const evalGetMove = EVAL_AGENT
  ? require(path.join(__dirname, 'ai', EVAL_AGENT + '.js')).getMove
  : null;

// Cumulative training wall time across all legs (restored from checkpoint).
let PRIOR_TRAIN_MS = 0;

if (LOAD_PATH) {
  if (fs.existsSync(LOAD_PATH)) {
    const raw = require(path.resolve(LOAD_PATH));
    MAX_STONES = Object.assign({}, raw.maxStones);
    for (const [k, v] of Object.entries(SPEC))
      MAX_STONES[k] = Math.max(MAX_STONES[k] ?? 0, v);
    MAX_SIZE = Math.max(...Object.keys(MAX_STONES).map(Number));
    model = createModelWithWeights(MAX_STONES, MAX_SIZE, weightsMap(raw));
    PRIOR_TRAIN_MS = raw.trainMs ?? 0;
    console.log(`Loaded ${model.weights.size} weights from ${LOAD_PATH}`);
  } else {
    console.warn(`Warning: --load file not found: ${LOAD_PATH}`);
  }
}

INCREMENTAL = saturatedOnly(model.maxStones);

if (NO_ADD && model.weights.size === 0) {
  console.error('error: --no-add needs a loaded model to fine-tune (pass --load with a non-empty checkpoint)');
  process.exit(1);
}

console.log(`data: ${DATA_PATH} (${records.length} records: ${testRecs.length} test, ${trainRecs.length} train)`);
console.log(`LR=${LR}  momentum=${MOMENTUM}  smooth-weights=${EMA_ALPHA}  eval-size=${EVAL_SIZE}  ref=${EVAL_AGENT || '(none)'}`);
console.log(`eval-komi=${EVAL_KOMI} (fixed)`);
console.log(`spec=${SPEC_RAW}${FROZEN.size > 0 ? `  frozen=[${[...FROZEN].join(',')}]` : ''}${NO_ADD ? `  no-add (fine-tuning the loaded ${model.weights.size} patterns only)` : ''}`);
console.log(`Out: ${SAVE_PATH}${LOAD_PATH ? `  (resumed from ${LOAD_PATH})` : ''}`);

const ladderCases = LADDER_FILE ? loadCases(LADDER_FILE) : null;
const ladderAgent = gm => ({ move: gm.gameOver ? PASS : search1ply(gm) });
if (ladderCases) console.log(`ladder suite: ${LADDER_FILE} (${ladderCases.length} cases)`);

const mdPositions = MD_FILE ? loadPositions(MD_FILE) : null;
const mdAgent = gm => ({ move: gm.gameOver ? PASS : search1ply(gm) });
if (mdPositions) console.log(`md positions: ${MD_FILE} (${mdPositions.length} positions)`);
console.log();

// Training columns (left), then test / eval columns (right).  Every cell is
// padded to its column width, so headers and data stay aligned regardless of
// the individual formatters' string lengths.
const COLS = ['T', 'TT', 'pos', 'epoch', 'tPos', 'nWts', 'avgW', 'trMSE', 'teMSE',
              ...(evalGetMove ? ['winRatio'] : []),
              ...(ladderCases ? ['ladr'] : []),
              ...(mdPositions ? ['mdRms'] : [])];
const COLW = [5, 5, 5, 5, 5, 4, 6, 6, 6,
              ...(evalGetMove ? [21] : []),
              ...(ladderCases ? [4] : []),
              ...(mdPositions ? [5] : [])];
const printRow = cells => console.log(cells.map((c, i) => String(c).padStart(COLW[i])).join('  '));
printRow(COLS);

// ── Main loop ─────────────────────────────────────────────────────────────────

const t0 = Date.now();
const MAX_PRINT_GAP_MS = 4 * 3600 * 1000;   // 4 h
const MAX_EVAL_GAMES = 2000;
let nextPrintAt = t0 + 1000, lastPrintAt = t0;
let nPos = 0, epoch = 0;
let intervalPos = 0, intervalTrainMs = 0, trSE = 0, trSEN = 0;
const evalHistory = [];

const order = trainRecs.map((_, i) => i);
function shuffleOrder() {
  for (let i = order.length - 1; i > 0; i--) {
    const j = (Math.random() * (i + 1)) | 0;
    const t = order[i]; order[i] = order[j]; order[j] = t;
  }
}

let done = false;
while (!done) {
  epoch++;
  shuffleOrder();
  for (const oi of order) {
    const tStartMs = Date.now();
    const rec  = trainRecs[oi];
    const game = replayRecord(rec);
    const f = extractFeatures(game, model, MAX_SIZE);
    f.val = evaluateFeatures(f, model.weights);
    trSE += (rec.targetB - f.val) * (rec.targetB - f.val); trSEN++;
    tdUpdate(f, rec.targetB, LR);
    nPos++; intervalPos++;
    intervalTrainMs += Date.now() - tStartMs;

    if (EMA_ALPHA > 0 && nPos % EMA_PERIOD === 0) applyEMA(model, EMA_ALPHA);

    const limitReached = LIMIT_POS > 0 && nPos >= LIMIT_POS;
    if (limitReached) { done = true; nextPrintAt = 0; }

    if (Date.now() >= nextPrintAt) {
      const tTestStart = Date.now();
      let batch = null, latestWR = 0, avgWR = 0, evalHalf = 0;
      if (evalGetMove) {
        batch = [];
        while (true) {
          const { results } = evalVsReference(EVAL_SIZE, evalGetMove, 2);
          for (const r of results) batch.push(r);
          const tMs = Date.now() - tTestStart;
          if (tMs > 0.3 * intervalTrainMs || batch.length >= MAX_EVAL_GAMES) break;
        }
        for (const r of batch) evalHistory.push(r);
        latestWR  = batch.reduce((s, r) => s + r, 0) / batch.length;
        evalHalf  = Math.max(1, Math.floor(evalHistory.length / 2));
        avgWR     = evalHistory.slice(-evalHalf).reduce((s, r) => s + r, 0) / evalHalf;
      }

      const tPosMs = intervalTrainMs / Math.max(1, intervalPos);
      const trMSE  = trSEN > 0 ? trSE / trSEN : 0;
      const teMSE  = testMSE();

      const ws   = model.weights.size;
      const wAvg = wUpdateCount > 0 ? wAbsSum / wUpdateCount : 0;
      wAbsSum = 0; wUpdateCount = 0;
      intervalPos = 0; intervalTrainMs = 0; trSE = 0; trSEN = 0;

      let ladrRatio = null;
      if (ladderCases) {
        const { passed, total } = evalCases(ladderCases, ladderAgent, { budgetMs: 1, oversample: 1 });
        ladrRatio = total ? passed / total : 0;
      }
      let mdRms = null;
      if (mdPositions) {
        mdRms = evalPositions(mdAgent, mdPositions, 0).rmsErr;
      }
      const tTestMs = Date.now() - tTestStart;

      const cols = [
        Util.fmtMs(Date.now() - t0),
        Util.fmtMs(PRIOR_TRAIN_MS + (Date.now() - t0)),
        Util.fmt4i(nPos),
        Util.fmt4i(epoch),
        Util.fmtMs(tPosMs),
        Util.fmt4i(ws),
        wAvg.toFixed(4),
        trMSE.toFixed(4),
        (teMSE !== null ? teMSE.toFixed(4) : '-'),
      ];
      if (evalGetMove) cols.push(`${Util.fmtRatio4(latestWR)}(${Util.fmt4i(batch.length)})` +
                                 `/${Util.fmtRatio4(avgWR)}(${Util.fmt4i(evalHalf)})`);
      if (ladrRatio !== null) cols.push(Util.fmtRatio4(ladrRatio));
      if (mdRms !== null) cols.push(Util.fmtRatio4(mdRms));
      printRow(cols);

      saveModel(SAVE_PATH, model);
      const nowMs = Date.now();
      const geometricAt = t0 + Math.round((nowMs - t0) * 1.3);
      const cappedAt = Math.min(geometricAt, nowMs + MAX_PRINT_GAP_MS);
      nextPrintAt = Math.max(cappedAt, nowMs + tTestMs);
      lastPrintAt = Date.now();
    }
    if (done) break;
  }
}

console.log();
console.log(`Reached --limit ${LIMIT_POS} positions — saved ${SAVE_PATH}`);
