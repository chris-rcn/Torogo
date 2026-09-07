#!/usr/bin/env node
'use strict';

// train-vpat-playout-eval.js — supervised training of vpatterns weights on a
// gen-agent-evals data file (static playout-value evaluator).
//
// Value function (absolute, P(BLACK wins)):
//   V(s) = σ( Σ  polarity_i · w[key_i] )
//
// Update rule — logistic regression on the corpus label:
//   Δw_k = (LR / n_features) · (target − V) · polarity_k
//
// Data lines (gen-agent-evals.js):  <bsize> <phase> <move1,...> <winRatio>
// Loading only parses (tokens, move syntax, band filter on the recorded
// phase); replay validation is deferred to each record's FIRST replay and
// aborts loudly on a corrupt record.
//
// Runs indefinitely (Ctrl-C to stop) unless --epochs is given.  Weights are
// saved at every print; a new best teMSE also writes the -best checkpoint.

const path = require('path');
const fs   = require('fs');
const { Game2, BLACK, PASS, KOMI, parseMove } = require('./game2.js');
const { prepareSpecs, extractFeatures, evaluateFeatures,
        loadWeights, saveWeights, makeWeights, specTag } = require('./vpatterns.js');
const { search } = require('./ai/vpatsearch.js');
const { loadCases, evalCases } = require('./evalladders2.js');
const { loadPositions, evalPositions } = require('./evalmovedetails.js');
const Util = require('./util.js');

// ── Arguments ─────────────────────────────────────────────────────────────────

const opts = Util.parseArgs(process.argv.slice(2), ['no-add', 'help'],
  ['data', 'test-file', 'test-pos', 'bias-file', 'min-phase', 'max-phase', 'smooth-weights', 'eval', 'eval-size',
   'ladder-file', 'epochs', 'load', 'lr', 'lr-decay', 'max-weights', 'md-file', 'save', 'spec']);
if (opts.help || !opts.data) {
  console.log(`Usage: node train-vpat-playout-eval.js --data <file> [options]

Supervised trainer: fits vpatterns weights to the playout-value labels of a
gen-agent-evals data file (logistic regression on P(BLACK wins)).  Runs
indefinitely unless --epochs is given; the checkpoint is written at every
print, and each new best teMSE also writes the -best checkpoint.

  --data FILE       gen-agent-evals data file (required)
  --bias-file F     bias test set (measure-trunc-bias --emit): per print,
                    the EMA weights are scored over its cached endpoint
                    pairs, adding b2 and bias columns — the truncation bias
                    floor E[b^2] and its shared lean, the deployment
                    quantities teMSE cannot see.  Reported only; -best
                    stays teMSE-selected
  --test-file F     separate data file supplying the held-out test set:
                    band-filtered, then all of it (or capped at --test-pos);
                    --data is then entirely train pool
  --test-pos N      test-set size cap (default: with --test-file, all of it;
                    otherwise 0 = no test set, no teMSE).  Without
                    --test-file: the first N records of --data become the
                    test head (clamped to half the file)
  --min-phase F     train only on records with phase >= F (default 0)
  --max-phase F     train only on records with phase <= F (default 1);
                    the band filters the train pool and the --test-file set
  --epochs N        stop after N full passes over the train pool
                    (default 0 = run indefinitely)

  --spec S          comma list of "size:maxLibs[f]" tokens (size 1-4, or
                    34 = the 3x4/4x3 rectangle pair, both orientations;
                    maxLibs 1 = presence only, or L = ladder-coded cells
                    (vlibpat 7-state tactical alphabet; game3 pass per
                    position, not incremental); trailing 'f' freezes that
                    spec's loaded weights).  Default 1:6,2:6,3:6
  --lr F            step size for the update (default 0.3)
  --lr-decay F      multiply LR by this factor at the end of each epoch
                    (default 0.9; 1 = no decay)
  --smooth-weights A  Polyak EMA decay, applied every 1000 positions; 0 = off
                    (default 0.9).  The EMA weights are what gets saved, and
                    what teMSE / ladder / md / eval games measure
  --max-weights N   stop admitting NEW patterns once the weight table holds
                    N entries; existing weights keep training.  0 = unlimited

  --load PATH       resume from a checkpoint.  --spec overrides its specs
                    (shared specs keep their weights; freeze with 'f')
  --no-add          fine-tune ONLY the patterns already in the loaded model
  --save PATH       checkpoint path (default out/vpat-pe-<random>.js)

  --eval AGENT      ai/<name>.js played as the reference in test games
                    (default: none, which disables the test games)
  --eval-size N     evaluation board size (default 13)
  --ladder-file F   evalladders2 suite scored each print (ladr column)
  --md-file F       evalmovedetails positions scored each print (mdRms column)
  --help            show this message`);
  process.exit(opts.help ? 0 : 1);
}

const DATA_PATH  = opts.data;
const TEST_FILE  = opts['test-file'] || null;
const EVAL_SIZE  = parseInt(opts['eval-size'] || '13', 10);
const SAVE_PATH  = opts.save || `out/vpat-pe-${Math.random().toString(36).slice(2, 10)}.js`;
const LOAD_PATH  = opts.load || null;
const NO_ADD     = opts['no-add'] === true;
const EVAL_AGENT = opts.eval || '';
const LADDER_FILE = opts['ladder-file'] || null;
const MD_FILE     = opts['md-file'] || null;
let LR           = parseFloat(opts.lr       || '0.3');
const LR_DECAY   = parseFloat(opts['lr-decay'] || '0.9');
const EMA_ALPHA  = parseFloat(opts['smooth-weights'] || '0.9');
const EMA_PERIOD = 1000;   // positions between applyEMA folds
const MAX_WEIGHTS = opts['max-weights'] !== undefined ? parseInt(opts['max-weights'], 10) : 0;
const EPOCHS     = opts.epochs !== undefined ? parseInt(opts.epochs, 10) : 0;
const TEST_POS_RAW = opts['test-pos'] !== undefined ? parseInt(opts['test-pos'], 10)
                                                    : (opts['test-file'] ? Infinity : 0);
const MIN_PHASE  = opts['min-phase'] !== undefined ? parseFloat(opts['min-phase']) : 0;
const MAX_PHASE  = opts['max-phase'] !== undefined ? parseFloat(opts['max-phase']) : 1;
if (MIN_PHASE < 0 || MAX_PHASE > 1 || MIN_PHASE > MAX_PHASE) {
  console.error('--min-phase/--max-phase must satisfy 0 <= min <= max <= 1');
  process.exit(1);
}
const EVAL_KOMI = KOMI(EVAL_SIZE);

// ── Specs ─────────────────────────────────────────────────────────────────────

// vpatterns spec grammar: "size:maxLibs[f]" (see train-vpatterns).
let specs;
const FROZEN = new Set();   // spec tags ((maxLibs << 3) | size) excluded from updates
if (opts.spec) {
  specs = opts.spec.split(',').map(tok => {
    const [s, mRaw] = tok.split(':');
    const size = parseInt(s, 10);
    const frozen = /f$/.test(mRaw);
    const body = frozen ? mRaw.slice(0, -1) : mRaw;
    // 'L' = the ladder-coded family (vlibpat 7-state tactical alphabet),
    // internally maxLibs 0.  Not incremental: unusable with deltaZ consumers.
    const maxLibs = body === 'L' ? 0 : parseInt(body, 10);
    if (!((size >= 1 && size <= 4) || size === 34) || !(maxLibs >= 1 || body === 'L')) {
      console.error(`--spec: bad token '${tok}' (expected size:maxLibs[f] or size:L[f], size 1-4 or 34, maxLibs >= 1)`);
      process.exit(1);
    }
    if (frozen) FROZEN.add(specTag({ size, maxLibs }));
    return { size, maxLibs };
  });
} else {
  specs = [{ size: 1, maxLibs: 6 }, { size: 2, maxLibs: 6 }, { size: 3, maxLibs: 6 }];
}
let prepSpecs = prepareSpecs(specs);
const specKey = sp => sp.map(x => `${x.size}:${x.maxLibs === 0 ? 'L' : x.maxLibs}`).join(',');

// ── Model ─────────────────────────────────────────────────────────────────────

let weights = makeWeights();
let weightsEMA = makeWeights();
let weightsEMAInit = false;
let wAbsSum = 0, wUpdateCount = 0;   // per-interval avgW (reset each print)

function applyEMA(alpha) {
  if (!weightsEMAInit) {
    weights.forEach((k, v) => weightsEMA.set(k, v));
    weightsEMAInit = true;
    return;
  }
  weights.forEach((k, v) => {
    const e = weightsEMA.get(k);
    weightsEMA.set(k, e === undefined ? v : alpha * e + (1 - alpha) * v);
  });
}

// Eval ≡ save: every test harness (teMSE, ladder/md suites, reference games)
// measures the weights the model save would write.
function saveEvalW() {
  return (EMA_ALPHA > 0 && weightsEMAInit) ? weightsEMA : weights;
}

// Logistic update; a feature is skipped (and takes no share of the error)
// when its spec is frozen, or — under --no-add / at the --max-weights cap —
// when its key is not already in the table.
function tdUpdate(features, target, lr) {
  const n = features.count;
  if (n === 0) return;
  const { keys, pols, tags } = features;
  const hasFrozen = FROZEN.size > 0;
  const atCap = MAX_WEIGHTS > 0 && weights.size >= MAX_WEIGHTS;
  const skip = (i) => (hasFrozen && FROZEN.has(tags[i])) ||
                      ((NO_ADD || atCap) && weights.get(keys[i]) === undefined);
  let nActive = n;
  if (hasFrozen || NO_ADD || atCap) {
    nActive = 0;
    for (let i = 0; i < n; i++) if (!skip(i)) nActive++;
    if (nActive === 0) return;
  }
  const step = lr * (target - features.val) / nActive;
  for (let i = 0; i < n; i++) {
    if (skip(i)) continue;
    const k = keys[i];
    const w = (weights.get(k) ?? 0) + pols[i] * step;
    weights.set(k, w);
    wAbsSum += Math.abs(w);
    wUpdateCount++;
  }
}

// ── Data loading ──────────────────────────────────────────────────────────────

function loadRecords(filePath, exemptFirst = 0) {
  const t0 = Date.now();
  process.stdout.write(`Loading: ${filePath} (${(fs.statSync(filePath).size / 1e6).toFixed(0)}MB)...`);
  const recs = [];
  let malformed = 0, outsideBand = 0;
  const processLine = (line) => {
    if (!line || line[0] === '#') return;
    const p = line.split(/\s+/);
    if (p.length !== 4) { malformed++; return; }
    const size  = parseInt(p[0], 10);
    const phase = parseFloat(p[1]);
    const wr    = parseFloat(p[3]);
    if (!Number.isFinite(size) || !Number.isFinite(phase) ||
        !(wr >= 0 && wr <= 1)) { malformed++; return; }
    if (recs.length >= exemptFirst &&
        (phase < MIN_PHASE || phase > MAX_PHASE)) { outsideBand++; return; }
    const toks = p[2].split(',');
    const moves = new Int16Array(toks.length);
    let ok = true;
    for (let i = 0; i < toks.length; i++) {
      const m = parseMove(toks[i], size);
      if (!Number.isInteger(m) || m < PASS || m >= size * size) { ok = false; break; }
      moves[i] = m;
    }
    if (!ok) { malformed++; return; }
    // Side to move is pure parity: WHITE moves first after the free initial
    // stone, and every play (pass included) flips.
    const targetB = (moves.length % 2 === 1) ? wr : 1 - wr;   // P(BLACK wins)
    recs.push({ size, moves, targetB, v: false });
  };
  const fd = fs.openSync(filePath, 'r');
  const buf = Buffer.alloc(1 << 22);   // 4MB chunks
  let rem = '';
  for (;;) {
    const n = fs.readSync(fd, buf, 0, buf.length, null);
    if (n === 0) break;
    const lines = (rem + buf.toString('utf8', 0, n)).split('\n');
    rem = lines.pop();
    for (const line of lines) processLine(line);
  }
  fs.closeSync(fd);
  if (rem) processLine(rem);
  console.log(` Done.` +
    (malformed ? `  Dropped ${malformed} malformed lines.` : '') +
    `  Loaded ${recs.length} records in ${((Date.now() - t0) / 1000).toFixed(1)}s.`);
  recs.outsideBand = outsideBand;
  if (recs.length === 0) { console.error(`data '${filePath}' contains no valid records`); process.exit(1); }
  return recs;
}

const records = loadRecords(DATA_PATH, TEST_FILE ? 0 : TEST_POS_RAW);
const _testAll = TEST_FILE ? loadRecords(TEST_FILE) : null;
const testRecs  = TEST_FILE
  ? _testAll.slice(0, TEST_POS_RAW)
  : records.slice(0, Math.min(TEST_POS_RAW, records.length >> 1));
if (TEST_FILE && testRecs.length === 0) {
  console.error(`no test records in phase band [${MIN_PHASE}, ${MAX_PHASE}]`);
  process.exit(1);
}
const trainRecs = TEST_FILE ? records : records.slice(testRecs.length);
if (trainRecs.length === 0) {
  console.error(`no train records in phase band [${MIN_PHASE}, ${MAX_PHASE}]`);
  process.exit(1);
}

// First replay of each record validates it (deferred from load); band-
// filtered records also cross-check the recorded phase column via the band.
const BAND_ACTIVE = MIN_PHASE > 0 || MAX_PHASE < 1;
function replayRecord(rec, bandExempt = false) {
  const game = new Game2(rec.size);
  const moves = rec.moves;
  if (rec.v) {
    for (let i = 0; i < moves.length; i++) game.play(moves[i]);
    return game;
  }
  for (let i = 0; i < moves.length; i++) {
    if (!game.play(moves[i])) {
      console.error(`corrupt record: replay failed at move ${i + 1}/${moves.length} (size ${rec.size})`);
      process.exit(1);
    }
  }
  if (BAND_ACTIVE && !bandExempt) {
    const ph = game.phase();
    if (ph < MIN_PHASE - 0.0006 || ph > MAX_PHASE + 0.0006) {
      console.error(`corrupt record: replayed phase ${ph.toFixed(3)} outside band ` +
                    `[${MIN_PHASE}, ${MAX_PHASE}] (size ${rec.size}, ${moves.length} moves)`);
      process.exit(1);
    }
  }
  rec.v = true;
  return game;
}

// ── Bias test set (measure-trunc-bias --emit artifact) ───────────────────────
// Endpoint features are model-independent given the spec: replay + extract
// once at load, then each print is just 2n evaluateFeatures passes.
const BIAS_FILE = opts['bias-file'] || null;
let biasPairs = null;
// Called AFTER --load has resolved the final specs/prepSpecs: the cached
// features must be extracted under the spec the weights are keyed by.
function loadBiasPairs() {
  if (!BIAS_FILE) return;
  const lines = fs.readFileSync(BIAS_FILE, 'utf8').split('\n');
  const header = lines.find(l => l.startsWith('# bias-pairs:'));
  if (header) console.log(header.slice(2));
  biasPairs = [];
  let biasDropped = 0;
  const t0b = Date.now();
  for (const line of lines) {
    if (!line || line[0] === '#') continue;
    const p = line.trim().split(/\s+/);
    if (p.length !== 7) continue;
    const size = parseInt(p[0], 10);
    // Pairs whose ENDPOINT phase falls outside the training band are dropped:
    // under the band-matching convention the training band IS the consulted
    // band, and out-of-band pairs add checkpoint-dependent extrapolation
    // noise to varB (and to -best selection).
    const ph = parseFloat(p[1]);
    if (ph < MIN_PHASE || ph > MAX_PHASE) { biasDropped++; continue; }
    const rec = { pa: parseFloat(p[5]), pb: parseFloat(p[6]) };
    for (const [key, col] of [['f1', 3], ['f2', 4]]) {
      const g = new Game2(size);
      for (const t of p[col].split(',')) {
        if (!g.play(parseMove(t, size))) {
          console.error(`bias-file: replay failed (${BIAS_FILE})`);
          process.exit(1);
        }
      }
      const f = extractFeatures(g, prepSpecs);
      rec[key] = { keys: f.keys.slice(0, f.count), pols: f.pols.slice(0, f.count), count: f.count };
    }
    biasPairs.push(rec);
  }
  console.log(`bias pairs: ${biasPairs.length} loaded+extracted from ${BIAS_FILE} in ${((Date.now() - t0b) / 1000).toFixed(1)}s` +
    (biasDropped ? ` (${biasDropped} outside band [${MIN_PHASE}, ${MAX_PHASE}] dropped)` : ''));
}


function biasStats() {
  if (!biasPairs) return null;
  const evalW = saveEvalW();
  let prod = 0, lean = 0;
  for (const rec of biasPairs) {
    const v1 = evaluateFeatures(rec.f1, evalW);
    const v2 = evaluateFeatures(rec.f2, evalW);
    prod += (v1 - rec.pa) * (v2 - rec.pb);
    lean += (v1 + v2) / 2 - (rec.pa + rec.pb) / 2;
  }
  return { b2: prod / biasPairs.length, lean: lean / biasPairs.length };
}

function testMSE() {
  if (testRecs.length === 0) return null;
  const evalW = saveEvalW();
  let se = 0;
  for (const rec of testRecs) {
    const v = evaluateFeatures(extractFeatures(replayRecord(rec, !TEST_FILE), prepSpecs), evalW);
    se += (rec.targetB - v) * (rec.targetB - v);
  }
  return se / testRecs.length;
}

// ── CLI setup ─────────────────────────────────────────────────────────────────

const evalGetMove = EVAL_AGENT
  ? require(path.join(__dirname, 'ai', EVAL_AGENT + '.js')).getMove
  : null;

if (LOAD_PATH) {
  if (fs.existsSync(LOAD_PATH)) {
    const cliSpecs = opts.spec ? specs : null;
    const loaded = loadWeights(LOAD_PATH);
    ({ weights, specs, preparedSpecs: prepSpecs } = loaded);
    if (cliSpecs !== null && specKey(cliSpecs) !== specKey(specs)) {
      console.warn(`WARNING: --spec overrides checkpoint specs (${specKey(specs)} -> ${specKey(cliSpecs)}); shared specs keep their weights.`);
      specs = cliSpecs;
      prepSpecs = prepareSpecs(specs);
    }
    if (EMA_ALPHA > 0) {   // continue averaging on top of the persisted values
      weightsEMA = weights.clone();
      weightsEMAInit = true;
    }
    console.log(`Loaded ${weights.size} weights from ${LOAD_PATH}`);
  } else {
    console.warn(`Warning: --load file not found: ${LOAD_PATH}`);
  }
}
if (NO_ADD && weights.size === 0) {
  console.error('error: --no-add needs a loaded model to fine-tune (pass --load with a non-empty checkpoint)');
  process.exit(1);
}

{
  const band = BAND_ACTIVE ? ` (band [${MIN_PHASE}, ${MAX_PHASE}]: ${records.outsideBand} outside dropped)` : '';
  console.log(`data: ${DATA_PATH} (${records.length} records: ` +
    (TEST_FILE ? `all train` : `${testRecs.length} test, ${trainRecs.length} train`) + `)` + band);
  if (TEST_FILE) console.log(`test: ${TEST_FILE} (${testRecs.length} records` +
    (Number.isFinite(TEST_POS_RAW) ? `, capped at ${TEST_POS_RAW}` : ``) +
    (BAND_ACTIVE ? `, ${_testAll.outsideBand} outside band dropped)` : `)`));
}
loadBiasPairs();
console.log(`LR=${LR}  lr-decay=${LR_DECAY}  smooth-weights=${EMA_ALPHA}  max-weights=${MAX_WEIGHTS || '(unlimited)'}  eval-size=${EVAL_SIZE}  ref=${EVAL_AGENT || '(none)'}`);
console.log(`Specs: ${JSON.stringify(specs)}${FROZEN.size > 0 ? `  frozen: [${specs.filter(sp => FROZEN.has(specTag(sp))).map(sp => `${sp.size}:${sp.maxLibs}`).join(',')}]` : ''}${NO_ADD ? `  no-add` : ''}`);
// '-best' goes before the file extension, whatever it is (x.js -> x-best.js,
// x.txt -> x-best.txt); an extensionless path gets it appended.
const BEST_PATH = (() => {
  const pp = path.parse(SAVE_PATH);
  return path.join(pp.dir, `${pp.name}-best${pp.ext}`);
})();
let bestMetric = Infinity;
const hasBest = TEST_FILE || TEST_POS_RAW > 0;
console.log(`Out: ${SAVE_PATH}${hasBest || biasPairs ? ` (best: ${BEST_PATH})` : ''}${LOAD_PATH ? `  (resumed from ${LOAD_PATH})` : ''}`);

const ladderCases = LADDER_FILE ? loadCases(LADDER_FILE) : null;
const testAgent = gm => ({ move: gm.gameOver ? PASS
  : search(gm, { weights: saveEvalW(), specs, preparedSpecs: prepSpecs }) });
if (ladderCases) console.log(`ladder suite: ${LADDER_FILE} (${ladderCases.length} cases)`);
const mdPositions = MD_FILE ? loadPositions(MD_FILE) : null;
if (mdPositions) console.log(`md positions: ${MD_FILE} (${mdPositions.length} positions)`);

// ── Evaluation against a reference agent ─────────────────────────────────────

function evalVsReference(N, refGetMove, nGames) {
  const results = [];
  const evalW = saveEvalW();
  const m = { weights: evalW, specs, preparedSpecs: prepSpecs };
  for (let g = 0; g < nGames; g++) {
    const policyIsBlack = (g % 2 === 0);
    const game = new Game2(N);   // free initial stone
    for (let r = 0; r < 3 && !game.gameOver; r++) game.play(game.randomLegalMove());
    const maxMoves = N * N * 4;
    let moves = 0;
    while (!game.gameOver && moves++ < maxMoves) {
      let idx;
      if ((game.current === BLACK) === policyIsBlack) {
        idx = search(game, m);
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

// ── Columns ───────────────────────────────────────────────────────────────────

const COLS = ['T', 'pos', 'epoch', 'LR', 'tPos', 'nWts', 'avgW', 'trMSE', 'teMSE',
              ...(biasPairs ? ['b2', 'bias', 'varB'] : []),
              ...(evalGetMove ? ['winRatio'] : []),
              ...(ladderCases ? ['ladr'] : []),
              ...(mdPositions ? ['mdRms'] : [])];
const COLW = [5, 5, 5, 7, 5, 4, 6, 6, 7,
              ...(biasPairs ? [8, 7, 8] : []),
              ...(evalGetMove ? [21] : []),
              ...(ladderCases ? [4] : []),
              ...(mdPositions ? [5] : [])];
const printRow = cells => console.log(cells.map((c, i) => String(c).padStart(COLW[i])).join('  '));
printRow(COLS);

// ── Main loop ─────────────────────────────────────────────────────────────────

const t0 = Date.now();
// Print schedule: geometric in POSITIONS — first row at 10k, then total
// position count grows 1.4x per row, with a 4h time backstop.
const PRINT_START_POS  = 10000;
const MAX_PRINT_GAP_MS = 4 * 3600 * 1000;
const MAX_EVAL_GAMES = 2000;
let nextPrintPos = PRINT_START_POS, nextPrintAt = t0 + MAX_PRINT_GAP_MS;
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

function statusPrint() {
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
  const bs = biasPairs ? biasStats() : null;
  const varB = bs ? bs.b2 - bs.lean * bs.lean : null;
  // One -best, selected by the most deployment-relevant metric available:
  // varB (the truncation floor) when a bias file is loaded, else teMSE.
  const metric = varB !== null ? varB : teMSE;
  const isBest = metric !== null && metric < bestMetric;
  if (isBest) bestMetric = metric;

  const ws   = weights.size;
  const wAvg = wUpdateCount > 0 ? wAbsSum / wUpdateCount : 0;
  wAbsSum = 0; wUpdateCount = 0;
  intervalPos = 0; intervalTrainMs = 0; trSE = 0; trSEN = 0;

  let ladrRatio = null;
  if (ladderCases) {
    const { passed, total } = evalCases(ladderCases, testAgent, { budgetMs: 1, oversample: 1 });
    ladrRatio = total ? passed / total : 0;
  }
  let mdRms = null;
  if (mdPositions) {
    mdRms = evalPositions(testAgent, mdPositions, 0).rmsErr;
  }

  const cols = [
    Util.fmtMs(Date.now() - t0),
    Util.fmt4i(nPos),
    Util.fmt4i(epoch),
    LR >= 0.001 ? LR.toFixed(4) : LR.toExponential(1),
    Util.fmtMs(tPosMs),
    Util.fmt4i(ws),
    wAvg.toFixed(4),
    trMSE.toFixed(4),
    (teMSE !== null ? teMSE.toFixed(4) + (isBest && varB === null ? '*' : ' ') : '-'),
  ];
  if (bs) {
    // varB = the floor that survives a constant (step-2) correction; when a
    // bias file is loaded it is the -best selector, so the '*' lives here.
    cols.push(bs.b2.toFixed(5), (bs.lean >= 0 ? '+' : '') + bs.lean.toFixed(3),
              varB.toFixed(5) + (isBest ? '*' : ' '));
  }
  if (evalGetMove) cols.push(`${Util.fmtRatio4(latestWR)}(${Util.fmt4i(batch.length)})` +
                             `/${Util.fmtRatio4(avgWR)}(${Util.fmt4i(evalHalf)})`);
  if (ladrRatio !== null) cols.push(Util.fmtRatio4(ladrRatio));
  if (mdRms !== null) cols.push(Util.fmtRatio4(mdRms));
  printRow(cols);

  saveWeights(SAVE_PATH, { weights: saveEvalW(), specs, preparedSpecs: prepSpecs });
  if (isBest) saveWeights(BEST_PATH, { weights: saveEvalW(), specs, preparedSpecs: prepSpecs });
  nextPrintPos = Math.max(Math.ceil(nPos * 1.4), nPos + 1);
  nextPrintAt  = Date.now() + MAX_PRINT_GAP_MS;
}

let done = false;
while (!done) {
  epoch++;
  shuffleOrder();
  for (const oi of order) {
    const tStartMs = Date.now();
    const rec  = trainRecs[oi];
    const game = replayRecord(rec);
    const f = extractFeatures(game, prepSpecs);
    evaluateFeatures(f, weights);
    trSE += (rec.targetB - f.val) * (rec.targetB - f.val); trSEN++;
    tdUpdate(f, rec.targetB, LR);
    nPos++; intervalPos++;
    intervalTrainMs += Date.now() - tStartMs;

    if (EMA_ALPHA > 0 && nPos % EMA_PERIOD === 0) applyEMA(EMA_ALPHA);
    if (nPos >= nextPrintPos || Date.now() >= nextPrintAt) statusPrint();
  }
  LR *= LR_DECAY;
  if (EPOCHS > 0 && epoch >= EPOCHS) done = true;
}

if (intervalPos > 0) statusPrint();   // final partial interval
