#!/usr/bin/env node
'use strict';

// train-vpat-supervised.js — supervised training of vpatterns weights on a
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
// Each checkpoint's header records its provenance: git state, resolved
// settings, every input with its own header (chaining through --load), and
// the progress and metrics of the row that wrote it.

const path = require('path');
const fs   = require('fs');
const { Game2, BLACK, PASS, KOMI, parseMove } = require('./game2.js');
const { prepareSpecs, extractFeatures, evaluateFeatures,
        loadWeights, saveWeights, makeWeights, specTag, specString } = require('./vpatterns.js');
const { search } = require('./ai/vpatsearch.js');
const { loadCases, evalCases } = require('./evalladders2.js');
const { loadPositions, evalPositions } = require('./evalmovedetails.js');
const Util = require('./util.js');

// ── Arguments ─────────────────────────────────────────────────────────────────

const opts = Util.parseArgs(process.argv.slice(2), ['no-add', 'help', 'no-cache-features'],
  ['data', 'test-file', 'test-pos', 'train-pos', 'bias-file', 'min-phase', 'max-phase', 'smooth-weights', 'eval', 'eval-size',
   'ladder-file', 'epochs', 'load', 'lr', 'lr-decay', 'max-weights', 'md-file', 'save', 'spec', 'delta']);
if (opts.help || !opts.data) {
  console.log(`Usage: node train-vpat-supervised.js --data <file> [options]

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
                    stays teMSE-selected.  Requires --delta
  --delta D         deployment truncation delta: the bias is measured at it, and
                    it is baked into every saved checkpoint's 'trunc' block (so
                    puct-trunc / mc-ppat read it as a default).  The
                    recorded prefixes are truncated to ceil(D*area) moves past
                    the start; the bias file's header delta is only its MAXIMUM,
                    D must not exceed it.  Required with --bias-file
  --test-file F     data file supplying the held-out test set (band-filtered);
                    all of --data is the train pool.  This is the ONLY source of
                    the test set / teMSE — without --test-file there is none
  --test-pos N      cap the --test-file test set to N records (default: all);
                    requires --test-file
  --train-pos N     cap the training pool to the first N records (default: all)
  --min-phase F     train only on records with phase >= F (default: --delta
                    if given, else 0 — the endpoint phase is never below delta)
  --max-phase F     train only on records with phase <= F (default 1);
                    the band filters the train pool and the --test-file set
  --epochs N        stop after N full passes over the train pool
                    (default 0 = run indefinitely)

  --spec S          comma list of "size:maxLibs[f]" tokens (size 1-4, or
                    23 = the 2x3/3x2 and 34 = the 3x4/4x3 rectangle
                    pairs, both orientations;
                    maxLibs 1 = presence only, or L = ladder-coded cells
                    (vlibpat 7-state tactical alphabet; game3 pass per
                    position, not incremental); trailing 'f' freezes that
                    spec's loaded weights).  Required, unless --load
                    supplies the specs from a checkpoint
  --lr F            step size for the update (default 0.2)
  --lr-decay F      multiply LR by this factor at the end of each epoch
                    (default 1 = no decay)
  --smooth-weights A  Polyak EMA decay, applied every 1000 positions; 0 = off
                    (default 0.9).  The EMA weights are what gets saved, and
                    what teMSE / ladder / md / eval games measure
  --max-weights N   stop admitting NEW patterns once the weight table holds
                    N entries; existing weights keep training.  0 = unlimited

  --load PATH       resume from a checkpoint.  --spec overrides its specs
                    (shared specs keep their weights; freeze with 'f')
  --no-add          fine-tune ONLY the patterns already in the loaded model
  --no-cache-features  do NOT cache extracted features across epochs (replay +
                    re-extract every position every epoch).  Caching is on by
                    default (test eager, train lazy) and auto-off with frozen specs
  --save PATH       checkpoint path (default out/vpat-<random>.js)

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
const SAVE_PATH  = opts.save || `out/vpat-${Math.random().toString(36).slice(2, 10)}.js`;
const LOAD_PATH  = opts.load || null;
const NO_ADD     = opts['no-add'] === true;
const EVAL_AGENT = opts.eval || '';
const LADDER_FILE = opts['ladder-file'] || null;
const MD_FILE     = opts['md-file'] || null;
let LR           = parseFloat(opts.lr       || '0.2');
const LR_DECAY   = parseFloat(opts['lr-decay'] || '1');
const EMA_ALPHA  = parseFloat(opts['smooth-weights'] || '0.9');
const EMA_PERIOD = 1000;   // positions between applyEMA folds
const MAX_WEIGHTS = opts['max-weights'] !== undefined ? parseInt(opts['max-weights'], 10) : 0;
const EPOCHS     = opts.epochs !== undefined ? parseInt(opts.epochs, 10) : 0;
// The test set comes only from --test-file; --test-pos caps it (default: all).
const TEST_POS_RAW = opts['test-pos'] !== undefined ? parseInt(opts['test-pos'], 10) : Infinity;
if (opts['test-pos'] !== undefined && !opts['test-file']) {
  console.error('--test-pos requires --test-file (the test set comes only from --test-file)');
  process.exit(1);
}
// --train-pos caps the number of training positions used (default: all of --data).
const TRAIN_POS_RAW = opts['train-pos'] !== undefined ? parseInt(opts['train-pos'], 10) : Infinity;
if (opts['train-pos'] !== undefined && !(TRAIN_POS_RAW >= 1)) {
  console.error(`--train-pos must be a positive integer, got ${opts['train-pos']}`);
  process.exit(1);
}
// --delta D: the DEPLOYMENT truncation delta to measure the bias at, and to bake
// into saved checkpoints' 'trunc' block.  The bias artifact is emitted at a
// MAXIMUM delta (its header); any D up to that is valid — the recorded prefixes
// are truncated to ceil(D*area) moves past the start, exactly as the deployed
// agent does.  Required with --bias-file.
const DELTA = opts.delta !== undefined ? parseFloat(opts.delta) : null;
if (opts['bias-file'] && DELTA === null) {
  console.error("--bias-file requires --delta: the deployment delta to measure the bias at " +
                "(the bias file's header delta is only its maximum)");
  process.exit(1);
}
if (DELTA !== null && !opts['bias-file']) {
  console.error('--delta only applies with --bias-file'); process.exit(1);
}
if (DELTA !== null && !(DELTA >= 0 && DELTA < 1)) {
  console.error('--delta: expected 0 <= D < 1'); process.exit(1);
}
// The truncation endpoint phase is leaf + delta >= delta, so nothing below delta
// is ever consulted (or trained on): default --min-phase to delta when it is set.
const MIN_PHASE  = opts['min-phase'] !== undefined ? parseFloat(opts['min-phase'])
                 : (DELTA !== null ? DELTA : 0);
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
// Health model for health-coded specs / the C survival attribute; the library
// no longer reads the environment itself.
const HEALTH_PATH = (typeof process !== 'undefined' && process.env.HEALTH_DATA) || '';
if (opts.spec) {
  specs = opts.spec.split(',').map(tok => {
    if (tok[0] === 't') {
      let body = tok.slice(1);
      let phaseBins = 0;
      const pm = /^p(\d+)$/.exec(body);
      if (pm) { phaseBins = parseInt(pm[1], 10); body = ''; }
      if (body !== '' || (pm && !(phaseBins >= 1 && phaseBins <= 64))) {
        console.error(`--spec: bad t token '${tok}' (expected t or t p<1-64>, e.g. tp9)`);
        process.exit(1);
      }
      return phaseBins > 1 ? { turn: true, phaseBins } : { turn: true };
    }
    const [s, mRaw] = tok.split(':');
    const size = parseInt(s, 10);
    const frozen = /f$/.test(mRaw);
    let body = frozen ? mRaw.slice(0, -1) : mRaw;
    // optional phase-bin suffix pN (e.g. 2:1p2): per-spec phase-salted keys
    let patBins = 0;
    const pbm = /p(\d+)$/.exec(body);
    if (pbm) { patBins = parseInt(pbm[1], 10); body = body.slice(0, -pbm[0].length); }
    // 'H<N>' = the health-coded family: each stone takes its chain's survival
    // bucket (1..N under the frozen chain-survival model) in place of its
    // liberty count.  Internally maxLibs = -N; needs HEALTH_DATA, and like
    // 'L' it is not incremental.
    const hm = /^H(\d+)$/.exec(body);
    if (hm) {
      const hb = parseInt(hm[1], 10);
      if (!((size >= 1 && size <= 4) || size === 8 || size === 34 || size === 23) || !(hb >= 2 && hb <= 15)) {
        console.error(`--spec: bad token '${tok}' (expected size:H<N>, size 1-4, 8, 23 or 34, N 2-15)`);
        process.exit(1);
      }
      if (frozen) FROZEN.add(specTag({ size, maxLibs: -hb }));
      return patBins > 1 ? { size, maxLibs: -hb, phaseBins: patBins } : { size, maxLibs: -hb };
    }
    // 'L' = the ladder-coded family (vlibpat 7-state tactical alphabet),
    // internally maxLibs 0.  Not incremental: unusable with deltaZ consumers.
    // Strict: parseInt would silently accept trailing garbage ('2H' -> 2), so
    // a mistyped family marker would train as a liberty spec instead of
    // erroring.  Only 'L', 'H<N>' (handled above) or bare digits are valid.
    if (body !== 'L' && !/^\d+$/.test(body)) {
      console.error(`--spec: bad token '${tok}' (expected size:maxLibs, size:L or size:H<N>; got '${body}')`);
      process.exit(1);
    }
    const maxLibs = body === 'L' ? 0 : parseInt(body, 10);
    if (!((size >= 1 && size <= 4) || size === 8 || size === 34 || size === 23) || !(maxLibs >= 1 || body === 'L')) {
      console.error(`--spec: bad token '${tok}' (expected size:maxLibs[f] or size:L[f], size 1-4, 8, 23 or 34, maxLibs >= 1)`);
      process.exit(1);
    }
    if (frozen) FROZEN.add(specTag({ size, maxLibs }));
    return patBins > 1 ? { size, maxLibs, phaseBins: patBins } : { size, maxLibs };
  });
} else if (!LOAD_PATH) {
  console.error('error: --spec is required (no default); or resume the specs from a checkpoint with --load');
  process.exit(1);
}
// else: no --spec but --load given — the specs come from the checkpoint below.
let prepSpecs = specs ? prepareSpecs(specs, { health: HEALTH_PATH }) : null;
// Every trainer extraction replays a Game2 only (replayRecord), so a ladder
// spec must rebuild the Game3 for its tactical pass — expected here.  Acknowledge
// it (game3RebuildOk=true) so extractFeatures' rebuild warning stays reserved
// for genuine surprises (an agent path that should have passed a synced Game3).
const extractRebuildOk = (game) => extractFeatures(game, prepSpecs, undefined, undefined, undefined, undefined, true);
const specKey = sp => sp.map(x => x.turn ? 't' : `${x.size}:${x.maxLibs === 0 ? 'L' : x.maxLibs}`).join(',');

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

function loadRecords(filePath) {
  const recs = [];
  let malformed = 0, outsideBand = 0;
  const headers = new Map();   // distinct '#' header line -> count, for provenance
  const processLine = (line) => {
    if (line && line[0] === '#') { headers.set(line, (headers.get(line) || 0) + 1); return; }
    if (!line) return;
    const p = line.split(/\s+/);
    if (p.length !== 4) { malformed++; return; }
    const size  = parseInt(p[0], 10);
    const phase = parseFloat(p[1]);
    const wr    = parseFloat(p[3]);
    if (!Number.isFinite(size) || !Number.isFinite(phase) ||
        !(wr >= 0 && wr <= 1)) { malformed++; return; }
    if (phase < MIN_PHASE || phase > MAX_PHASE) { outsideBand++; return; }
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
  recs.outsideBand = outsideBand;
  recs.malformed = malformed;
  recs.headers = headers;
  if (recs.length === 0) { console.error(`data '${filePath}' contains no valid records`); process.exit(1); }
  return recs;
}

// All of --data is the train pool; the test set comes only from --test-file.
const records = loadRecords(DATA_PATH);
const trainRecs = records.slice(0, TRAIN_POS_RAW);
const _testAll = TEST_FILE ? loadRecords(TEST_FILE) : null;
const testRecs  = TEST_FILE ? _testAll.slice(0, TEST_POS_RAW) : [];
if (TEST_FILE && testRecs.length === 0) {
  console.error(`no test records in phase band [${MIN_PHASE}, ${MAX_PHASE}]`);
  process.exit(1);
}
if (trainRecs.length === 0) {
  console.error(`no train records in phase band [${MIN_PHASE}, ${MAX_PHASE}]`);
  process.exit(1);
}

// First replay of each record validates it (deferred from load); band-
// filtered records also cross-check the recorded phase column via the band.
const BAND_ACTIVE = MIN_PHASE > 0 || MAX_PHASE < 1;
function replayRecord(rec) {
  const game = new Game2(rec.size, true);
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
  if (BAND_ACTIVE) {
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
let biasInfo = null;   // { dropped, secs, ppat, minPhase, source } for the startup banner
let biasHeaders = [];  // the artifact's '#' lines, for provenance
// Called AFTER --load has resolved the final specs/prepSpecs: the cached
// features must be extracted under the spec the weights are keyed by.  The
// artifact is rescored at --delta: each recorded prefix is truncated to
// n0 + ceil(DELTA*area) moves (start moves + the D-descent) and the endpoint
// features + phase are taken there — sound for any DELTA up to the artifact's
// own (maximum) delta, since the references belong to the start.
function loadBiasPairs() {
  if (!BIAS_FILE) return;
  const lines = fs.readFileSync(BIAS_FILE, 'utf8').split('\n');
  const header = lines.find(l => l.startsWith('# bias-pairs:')) || '';
  biasHeaders = lines.filter(l => l[0] === '#');
  const field = (k) => { const m = header.match(new RegExp(`${k}:\\s*(\\S+)`)); return m ? m[1] : null; };
  const maxDelta = field('delta') !== null ? parseFloat(field('delta')) : null;
  if (maxDelta !== null && DELTA > maxDelta + 1e-9) {
    console.error(`--delta ${DELTA} exceeds the bias artifact's maximum delta ${maxDelta} (${BIAS_FILE})`);
    process.exit(1);
  }
  biasPairs = [];
  let biasDropped = 0;
  const t0b = Date.now();
  // Replay the first `cut` moves of a recorded prefix, returning the board.
  const replayCut = (size, moves, cut) => {
    const toks = moves.split(',');
    if (cut > toks.length) {
      console.error(`bias-file: --delta ${DELTA} needs ${cut} moves but a row has only ${toks.length} ` +
                    `(delta exceeds the artifact) — ${BIAS_FILE}`);
      process.exit(1);
    }
    const g = new Game2(size, true);
    for (let i = 0; i < cut; i++) {
      if (!g.play(parseMove(toks[i], size))) {
        console.error(`bias-file: replay failed (${BIAS_FILE})`);
        process.exit(1);
      }
    }
    return g;
  };
  for (const line of lines) {
    if (!line || line[0] === '#') continue;
    const p = line.trim().split(/\s+/);
    if (p.length !== 7) continue;
    const size = parseInt(p[0], 10);
    const area = size * size;
    const n0 = p[2] === '-' ? 0 : p[2].split(',').length;   // start moves before the descent
    const cut = n0 + Math.ceil(DELTA * area);
    // Endpoint at DELTA (both prefixes share the move count, hence the phase).
    const g1 = replayCut(size, p[3], cut);
    // Pairs whose D-ENDPOINT phase falls outside the training band are dropped:
    // under the band-matching convention the training band IS the consulted
    // band, and out-of-band pairs add checkpoint-dependent extrapolation
    // noise to varB (and to -best selection).
    const ph = 1 - g1.emptyCount / area;
    if (ph < MIN_PHASE || ph > MAX_PHASE) { biasDropped++; continue; }
    const g2 = replayCut(size, p[4], cut);
    const rec = { ph, pa: parseFloat(p[5]), pb: parseFloat(p[6]) };
    for (const [key, g] of [['f1', g1], ['f2', g2]]) {
      const f = extractRebuildOk(g);
      rec[key] = { keys: f.keys.slice(0, f.count), pols: f.pols.slice(0, f.count), count: f.count };
    }
    biasPairs.push(rec);
  }
  biasInfo = {
    dropped: biasDropped,
    secs: (Date.now() - t0b) / 1000,
    ppat: field('ppat'), minPhase: field('ppat-min-phase'),
    source: field('source') ? path.parse(field('source')).name : null,
  };
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

// Truncation delta baked into every saved checkpoint's 'trunc' block (from
// --delta), so consumers (puct-trunc, mc-ppat) read it as a default.
// undefined when --delta was not given.
const TRUNC_META = DELTA !== null ? { delta: DELTA } : undefined;

function testMSE() {
  if (testRecs.length === 0) return null;
  const evalW = saveEvalW();
  let se = 0;
  if (teKeys) {
    for (let i = 0; i < testRecs.length; i++) {
      const v = evalSplit(teKeys, teOff[i], teNPos[i], teOff[i + 1] - teOff[i], evalW);
      const d = testRecs[i].targetB - v;
      se += d * d;
    }
  } else {
    for (const rec of testRecs) {
      const v = evaluateFeatures(extractRebuildOk(replayRecord(rec)), evalW);
      se += (rec.targetB - v) * (rec.targetB - v);
    }
  }
  return se / testRecs.length;
}

// ── CLI setup ─────────────────────────────────────────────────────────────────

const evalGetMove = EVAL_AGENT
  ? require(path.join(__dirname, 'ai', EVAL_AGENT + '.js')).getMove
  : null;

let loadHeaders = null, loadWeightCount = 0;   // the resumed checkpoint's own header, for provenance
if (LOAD_PATH) {
  if (fs.existsSync(LOAD_PATH)) {
    const cliSpecs = opts.spec ? specs : null;
    const loaded = loadWeights(LOAD_PATH, HEALTH_PATH);
    loadHeaders = fs.readFileSync(LOAD_PATH, 'utf8').split('\n', 200).filter(l => l.startsWith('//'));
    loadWeightCount = loaded.weights.size;
    ({ weights, specs, preparedSpecs: prepSpecs } = loaded);
    if (cliSpecs !== null && specKey(cliSpecs) !== specKey(specs)) {
      console.warn(`WARNING: --spec overrides checkpoint specs (${specKey(specs)} -> ${specKey(cliSpecs)}); shared specs keep their weights.`);
      specs = cliSpecs;
      prepSpecs = prepareSpecs(specs, { health: HEALTH_PATH, ladderMinChain: prepSpecs.ladderMinChain });
    }
    if (EMA_ALPHA > 0) {   // continue averaging on top of the persisted values
      weightsEMA = weights.clone();
      weightsEMAInit = true;
    }
  } else {
    console.warn(`Warning: --load file not found: ${LOAD_PATH}`);
  }
}
if (!specs) {
  console.error('error: no spec resolved — pass --spec, or --load a checkpoint that supplies one');
  process.exit(1);
}
if (NO_ADD && weights.size === 0) {
  console.error('error: --no-add needs a loaded model to fine-tune (pass --load with a non-empty checkpoint)');
  process.exit(1);
}

// ── Feature cache ────────────────────────────────────────────────────────────
// Replaying a record and re-extracting its pattern features is the bulk of the
// per-position cost, and it is identical every epoch.  So convert each record's
// features to a flat list of weight keys ONCE and reuse it thereafter.  Every
// polarity is +/-1, so a record's keys are stored split into a +1 run followed
// by a -1 run — no per-feature polarity byte — and the value is
//   V = sigma( sum_{+keys} w[k]  -  sum_{-keys} w[k] ).
// Test data is cached EAGERLY (small, and the first status row — before any
// training — needs it); train data LAZILY, filled on first touch during epoch 1
// into an estimate-and-grow buffer (a single-epoch run then pays almost no
// conversion overhead, and the fill cost lands honestly inside epoch 1's time).
//
// The compact cache keeps no spec tags, so it cannot honour a frozen spec's
// per-feature skip; caching is disabled when any spec is frozen (and by
// --no-cache-features).  The +/-1 assumption is checked as records are cached.
const CACHE_FEATURES = !opts['no-cache-features'] && FROZEN.size === 0;

// Copy a feature set's keys into dst[at..], +1 keys first then -1 keys, and
// return the number of +1 keys (nPos); total written == f.count.  Throws on a
// polarity that is not +/-1 (such a spec is not cacheable).
function splitInto(f, dst, at) {
  const { keys, pols, count } = f;
  let p = at, nPos = 0;
  for (let i = 0; i < count; i++) if (pols[i] === 1) { dst[p++] = keys[i]; nPos++; }
  for (let i = 0; i < count; i++) {
    const pol = pols[i];
    if (pol === -1) dst[p++] = keys[i];
    else if (pol !== 1) throw new Error(`feature cache: polarity ${pol} is not +/-1 (spec not cacheable)`);
  }
  return nPos;
}

// Evaluate a polarity-split key run: keysArr[start .. start+nPos) are +1,
// keysArr[start+nPos .. start+cnt) are -1.
function evalSplit(keysArr, start, nPos, cnt, w) {
  let z = 0;
  const posEnd = start + nPos, end = start + cnt;
  for (let j = start; j < posEnd; j++) z += w.get(keysArr[j]) ?? 0;
  for (let j = posEnd; j < end;    j++) z -= w.get(keysArr[j]) ?? 0;
  return 1 / (1 + Math.exp(-z));
}

// Test cache (eager): built now so the baseline status row can read it.  Sizes
// are known exactly, so a single flat buffer plus off[n+1] and nPos[n].
let teKeys = null, teOff = null, teNPos = null;
if (CACHE_FEATURES && testRecs.length > 0) {
  const fs = testRecs.map(rec => extractRebuildOk(replayRecord(rec)));
  let total = 0;
  for (const f of fs) total += f.count;
  teKeys = new Int32Array(total);
  teOff  = new Int32Array(testRecs.length + 1);
  teNPos = new Int32Array(testRecs.length);
  let at = 0;
  for (let i = 0; i < fs.length; i++) {
    teOff[i]  = at;
    teNPos[i] = splitInto(fs[i], teKeys, at);
    at += fs[i].count;
  }
  teOff[testRecs.length] = at;
}

// Train cache (lazy): filled in shuffled order during epoch 1, so fill order
// != index order and each record needs its own off/cnt/nPos.  off[i] == -1
// marks a record not yet cached.  Buffer size is estimated from a small sample
// and grown x1.5 on overflow.
let trKeys = null;   // Int32Array (estimate-and-grow)
let trEnd  = 0;      // fill pointer / count of cached keys so far
const trOff  = CACHE_FEATURES ? new Int32Array(trainRecs.length).fill(-1) : null;
const trNPos = CACHE_FEATURES ? new Int32Array(trainRecs.length) : null;
const trCnt  = CACHE_FEATURES ? new Int32Array(trainRecs.length) : null;
if (CACHE_FEATURES) {
  const K = Math.min(500, trainRecs.length);
  let sum = 0;
  for (let i = 0; i < K; i++) sum += extractRebuildOk(replayRecord(trainRecs[i])).count;
  const avg = K > 0 ? sum / K : 64;
  trKeys = new Int32Array(Math.max(1, Math.ceil(trainRecs.length * avg * 1.2)));
}
function trGrow(need) {
  if (trEnd + need <= trKeys.length) return;
  let cap = trKeys.length;
  while (cap < trEnd + need) cap = Math.ceil(cap * 1.5);
  const bigger = new Int32Array(cap);
  bigger.set(trKeys.subarray(0, trEnd));
  trKeys = bigger;
}
// Lazy fill: copy record oi's split keys into the flat buffer.
function cacheTrain(oi, f) {
  trGrow(f.count);
  trNPos[oi] = splitInto(f, trKeys, trEnd);
  trCnt[oi]  = f.count;
  trOff[oi]  = trEnd;
  trEnd += f.count;
}
// Cached logistic update — the polarity-split twin of tdUpdate (frozen never
// applies here, since caching is off when any spec is frozen).
function tdUpdateCached(oi, target, V, lr) {
  const start = trOff[oi], nP = trNPos[oi], cnt = trCnt[oi];
  if (cnt === 0) return;
  const posEnd = start + nP, end = start + cnt;
  const atCap = MAX_WEIGHTS > 0 && weights.size >= MAX_WEIGHTS;
  const gated = NO_ADD || atCap;
  let nActive = cnt;
  if (gated) {
    nActive = 0;
    for (let j = start; j < end; j++) if (weights.get(trKeys[j]) !== undefined) nActive++;
    if (nActive === 0) return;
  }
  const step = lr * (target - V) / nActive;
  for (let j = start; j < posEnd; j++) {
    const k = trKeys[j];
    if (gated && weights.get(k) === undefined) continue;
    const w = (weights.get(k) ?? 0) + step;
    weights.set(k, w); wAbsSum += Math.abs(w); wUpdateCount++;
  }
  for (let j = posEnd; j < end; j++) {
    const k = trKeys[j];
    if (gated && weights.get(k) === undefined) continue;
    const w = (weights.get(k) ?? 0) - step;
    weights.set(k, w); wAbsSum += Math.abs(w); wUpdateCount++;
  }
}

loadBiasPairs();
// '-best' goes before the file extension, whatever it is (x.js -> x-best.js,
// x.txt -> x-best.txt); an extensionless path gets it appended.
const BEST_PATH = (() => {
  const pp = path.parse(SAVE_PATH);
  return path.join(pp.dir, `${pp.name}-best${pp.ext}`);
})();
let bestMetric = Infinity;
const hasBest = TEST_FILE || !!biasPairs;
const ladderCases = LADDER_FILE ? loadCases(LADDER_FILE) : null;
const mdPositions = MD_FILE ? loadPositions(MD_FILE) : null;
const testAgent = gm => ({ move: gm.gameOver ? PASS
  : search(gm, { weights: saveEvalW(), specs, preparedSpecs: prepSpecs }) });

// ── Startup banner: one aligned "label: value" line per group ────────────────
const bline = (label, content) => console.log(label.padEnd(8) + content);
const f4 = (n) => Util.fmt4i(n).trim();   // compact integer count (3418 -> "3418", 360427 -> "360K")
bline('data:', `${DATA_PATH}  ${f4(trainRecs.length)} train` +
  (Number.isFinite(TRAIN_POS_RAW) ? ` (capped ${f4(TRAIN_POS_RAW)})` : ``) +
  (BAND_ACTIVE ? `  band [${MIN_PHASE}, ${MAX_PHASE}]` : ``) +
  (records.malformed ? `  ${f4(records.malformed)} malformed` : ``) +
  (records.outsideBand ? `  ${f4(records.outsideBand)} out-of-band` : ``));
if (TEST_FILE) bline('test:', `${TEST_FILE}  ${f4(testRecs.length)} records` +
  (Number.isFinite(TEST_POS_RAW) ? ` (capped ${f4(TEST_POS_RAW)})` : ``) +
  (BAND_ACTIVE && _testAll.outsideBand ? `  ${f4(_testAll.outsideBand)} out-of-band` : ``));
if (biasPairs) bline('bias:', `${BIAS_FILE}  ${f4(biasPairs.length)} pairs, delta ${DELTA}` +
  (biasInfo.ppat ? `  (ppat ${biasInfo.ppat}${biasInfo.minPhase ? ` @${biasInfo.minPhase}` : ``}` +
                   `${biasInfo.source ? `, src ${biasInfo.source}` : ``})` : ``) +
  (biasInfo.dropped ? `  ${f4(biasInfo.dropped)} out-of-band` : ``) +
  `  [${biasInfo.secs.toFixed(1)}s]`);
bline('model:', `${specString(specs)}` +
  (FROZEN.size > 0 ? `  frozen [${specString(specs.filter(sp => FROZEN.has(specTag(sp))))}]` : ``) +
  (NO_ADD ? `  no-add` : ``) +
  (LOAD_PATH && weights.size > 0 ? `  (resumed ${f4(weights.size)} weights from ${LOAD_PATH})` : ``));
bline('train:', `lr ${LR}, lr-decay ${LR_DECAY}, smooth-weights ${EMA_ALPHA}, ` +
  `max-weights ${MAX_WEIGHTS ? f4(MAX_WEIGHTS) : 'unlimited'}, eval-size ${EVAL_SIZE}` +
  (EVAL_AGENT ? `, ref ${EVAL_AGENT}` : ``));
bline('cache:', CACHE_FEATURES
  ? `features (test eager${testRecs.length ? ` ${f4(teKeys.length)} keys` : ``}, train lazy)`
  : (FROZEN.size > 0 ? `off (frozen specs)` : `off (--no-cache-features)`));
if (ladderCases) bline('ladder:', `${LADDER_FILE}  ${f4(ladderCases.length)} cases`);
if (mdPositions) bline('md:', `${MD_FILE}  ${f4(mdPositions.length)} positions`);
bline('out:', `${SAVE_PATH}${hasBest ? `  (best ${BEST_PATH})` : ``}`);

// ── Evaluation against a reference agent ─────────────────────────────────────

function evalVsReference(N, refGetMove, nGames) {
  const results = [];
  const evalW = saveEvalW();
  const m = { weights: evalW, specs, preparedSpecs: prepSpecs };
  for (let g = 0; g < nGames; g++) {
    const policyIsBlack = (g % 2 === 0);
    const game = new Game2(N, true);   // free initial stone
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

// ── Provenance header ────────────────────────────────────────────────────────
// Every checkpoint records how it was made: code state, resolved settings
// (defaults included), each input's size and its own '#' / '//' header (so
// provenance chains through --load), and the progress + metrics of the row
// that wrote it.

const START_ISO = new Date().toISOString();
const GIT_STATE = (() => {
  const cp = require('child_process');
  const git = (a) => cp.execFileSync('git', a, { cwd: __dirname, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] });
  try {
    const head = git(['rev-parse', '--short', 'HEAD']).trim();
    // porcelain lines are 'XY path'; the status columns may be blank, so no trim
    const dirty = git(['status', '--porcelain', '--untracked-files=no']).split('\n').filter(Boolean).map(l => l.slice(3));
    return dirty.length ? `${head}, modified: ${dirty.join(' ')}` : `${head}, clean`;
  } catch (e) { return `unavailable (${e.message.split('\n')[0]})`; }
})();
const PROV_MAX_HEADERS = 40;   // distinct header lines kept per input
function headerLines(lines, indent) {
  const out = [];
  let n = 0;
  for (const [line, count] of lines) {
    if (n++ === PROV_MAX_HEADERS) { out.push(`${indent}... ${lines.length - PROV_MAX_HEADERS} more distinct header lines`); break; }
    out.push(`${indent}${line}${count > 1 ? `  (x${count})` : ''}`);
  }
  return out;
}
const PROV_STATIC = [
  `started: ${START_ISO}`,
  `git: ${GIT_STATE}`,
  `host: ${require('os').hostname()}  node: ${process.version}  cwd: ${process.cwd()}`,
  ...(HEALTH_PATH ? [`env: HEALTH_DATA=${HEALTH_PATH}`] : []),
  `spec: ${specString(specs)}` +
    (FROZEN.size > 0 ? `  frozen: ${specString(specs.filter(sp => FROZEN.has(specTag(sp))))}` : ''),
  `settings: lr: ${LR}  lr-decay: ${LR_DECAY}  smooth-weights: ${EMA_ALPHA}` +
    `  max-weights: ${MAX_WEIGHTS || 'unlimited'}  epochs: ${EPOCHS || 'unlimited'}` +
    `  band: [${MIN_PHASE}, ${MAX_PHASE}]  delta: ${DELTA !== null ? DELTA : 'none'}` +
    `  no-add: ${NO_ADD}  cache-features: ${CACHE_FEATURES}`,
  `data: ${DATA_PATH}  train: ${trainRecs.length}` +
    (Number.isFinite(TRAIN_POS_RAW) ? ` (train-pos ${TRAIN_POS_RAW})` : '') +
    `  of ${records.length} in band  malformed: ${records.malformed}  out-of-band: ${records.outsideBand}`,
  ...headerLines([...records.headers], '  '),
  ...(TEST_FILE ? [
    `test: ${TEST_FILE}  records: ${testRecs.length}` +
      (Number.isFinite(TEST_POS_RAW) ? ` (test-pos ${TEST_POS_RAW})` : '') +
      `  of ${_testAll.length} in band  malformed: ${_testAll.malformed}  out-of-band: ${_testAll.outsideBand}`,
    ...headerLines([..._testAll.headers], '  '),
  ] : ['test: none']),
  ...(biasPairs ? [
    `bias: ${BIAS_FILE}  pairs: ${biasPairs.length}  out-of-band: ${biasInfo.dropped}`,
    ...headerLines(biasHeaders.map(l => [l, 1]), '  '),
  ] : []),
  ...(LOAD_PATH ? [
    `load: ${LOAD_PATH}  ${loadHeaders ? `weights: ${loadWeightCount}` : 'NOT FOUND (fresh start)'}`,
    ...(loadHeaders || []).map(l => `  ${l}`),
  ] : ['load: none (fresh start)']),
];
const fmtM = (v, d) => v !== null && v !== undefined ? v.toFixed(d) : 'none';
function provenance(row) {
  return [
    `written: ${new Date().toISOString()}  elapsed: ${Util.fmtMs(Date.now() - t0).trim()}`,
    `progress: epoch: ${epoch}  positions: ${nPos}  lr-now: ${LR}  weights: ${weights.size}`,
    `metrics: trMSE: ${fmtM(row.trMSE, 5)}  teMSE: ${fmtM(row.teMSE, 5)}` +
      (row.varB !== null ? `  b2: ${fmtM(row.b2, 6)}  lean: ${fmtM(row.lean, 4)}  varB: ${fmtM(row.varB, 6)}` : '') +
      (row.ladr !== null ? `  ladr: ${fmtM(row.ladr, 4)}` : '') +
      (row.mdRms !== null ? `  mdRms: ${fmtM(row.mdRms, 4)}` : '') +
      (row.winRatio !== null ? `  winRatio vs ${EVAL_AGENT}: ${fmtM(row.winRatio, 4)} (${row.games} games)` : '') +
      `  best-so-far: ${fmtM(bestMetric === Infinity ? null : bestMetric, 6)} (by ${biasPairs ? 'varB' : 'teMSE'})` +
      (row.isBest ? '  this row set it' : ''),
    ...PROV_STATIC,
  ];
}

// ── Columns ───────────────────────────────────────────────────────────────────

const COLS = ['T', 'pos', 'epoch', 'LR', 'tPos', 'nWts', 'avgW', 'trMSE', 'teMSE',
              ...(biasPairs ? ['b2', 'bias', 'varB', 'varB×t'] : []),
              ...(evalGetMove ? ['winRatio'] : []),
              ...(ladderCases ? ['ladr'] : []),
              ...(mdPositions ? ['mdRms'] : [])];
const COLW = [5, 5, 5, 7, 5, 4, 6, 6, 7,
              ...(biasPairs ? [8, 7, 8, 7] : []),
              ...(evalGetMove ? [21] : []),
              ...(ladderCases ? [4] : []),
              ...(mdPositions ? [5] : [])];
const printRow = cells => console.log(cells.map((c, i) => String(c).padStart(COLW[i])).join('  '));
printRow(COLS);

// ── Main loop ─────────────────────────────────────────────────────────────────

const t0 = Date.now();
// Print schedule: geometric in POSITIONS — first row at 10k, then total
// position count grows 1.5x per row, with a 4h time backstop.
const PRINT_START_POS  = 10000;
const MAX_PRINT_GAP_MS = 4 * 3600 * 1000;
const MAX_EVAL_GAMES = 2000;
let nextPrintPos = PRINT_START_POS, nextPrintAt = t0 + MAX_PRINT_GAP_MS;
let nPos = 0, epoch = 0;
let intervalPos = 0, intervalTrainMs = 0, trSE = 0, trSEN = 0;
// tPos reports the epoch-1 overall average per-position cost, frozen once epoch
// 1 finishes: from epoch 2 on the feature cache serves positions from memory,
// so per-interval time is a fraction of the real replay+extract+update cost and
// would make both tPos and varB×t (which scales varB by it) read far too low.
let ep1TrainMs = 0, ep1Pos = 0, tPosEp1 = null;
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

  // Epoch-1 average (running while in epoch 1, then the frozen value); see the
  // ep1/tPosEp1 declaration for why per-interval time is not used.
  const tPosMs = tPosEp1 !== null ? tPosEp1 : (ep1Pos > 0 ? ep1TrainMs / ep1Pos : 0);
  const trMSE  = trSEN > 0 ? trSE / trSEN : null;   // null on the pre-training baseline row
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
    (trMSE !== null ? Util.fmtMs(tPosMs) : '-'),
    Util.fmt4i(ws),
    wAvg.toFixed(4),
    (trMSE !== null ? trMSE.toFixed(4) : '-'),
    (teMSE !== null ? teMSE.toFixed(4) + (isBest && varB === null ? '*' : ' ') : '-'),
  ];
  if (bs) {
    // varB = the floor that survives a constant (step-2) correction; when a
    // bias file is loaded it is the -best selector, so the '*' lives here.
    cols.push(bs.b2.toFixed(5), (bs.lean >= 0 ? '+' : '') + bs.lean.toFixed(3),
              varB.toFixed(5) + (isBest ? '*' : ' '));
    // varB x tPos in MICROSECONDS: the accuracy-per-millisecond figure of
    // merit.  One truncated eval is worth 0.25/varB playouts of MSE, so value
    // per unit time goes as 1/(varB * t) and the product is what to minimise —
    // a cheaper evaluator wins at equal product.  NOTE tPos is the TRAINER's
    // per-position cost (gradient work included), not the deployed eval cost,
    // so this is a proxy: comparable between rows and between runs on the same
    // machine, not an absolute.
    cols.push(trMSE !== null ? (varB * tPosMs * 1000).toFixed(4) : '-');
  }
  if (evalGetMove) cols.push(`${Util.fmtRatio4(latestWR)}(${Util.fmt4i(batch.length)})` +
                             `/${Util.fmtRatio4(avgWR)}(${Util.fmt4i(evalHalf)})`);
  if (ladrRatio !== null) cols.push(Util.fmtRatio4(ladrRatio));
  if (mdRms !== null) cols.push(Util.fmtRatio4(mdRms));
  printRow(cols);

  // Bake the deployment delta (--delta) into the checkpoint so consumers read
  // it as a default.
  saveWeights(SAVE_PATH, { weights: saveEvalW(), specs, preparedSpecs: prepSpecs, trunc: TRUNC_META },
    provenance({ trMSE, teMSE, b2: bs ? bs.b2 : null, lean: bs ? bs.lean : null, varB,
                 ladr: ladrRatio, mdRms, winRatio: evalGetMove ? latestWR : null,
                 games: batch ? batch.length : 0, isBest }));
  // The best checkpoint is byte-identical to the one just written, so copy the
  // file instead of re-serializing the whole (possibly huge) weight table.
  if (isBest) fs.copyFileSync(SAVE_PATH, BEST_PATH);
  nextPrintPos = Math.max(Math.ceil(nPos * 1.5), nPos + 1);
  nextPrintAt  = Date.now() + MAX_PRINT_GAP_MS;
}

statusPrint();                      // baseline row: initial metrics before any training
nextPrintPos = PRINT_START_POS;     // statusPrint set it to 1; resume the normal schedule

let done = false;
while (!done) {
  epoch++;
  shuffleOrder();
  for (const oi of order) {
    const tStartMs = Date.now();
    const rec  = trainRecs[oi];
    let V;
    if (trOff && trOff[oi] >= 0) {
      // Cached: skip replay + extraction, evaluate and update from flat keys.
      V = evalSplit(trKeys, trOff[oi], trNPos[oi], trCnt[oi], weights);
      tdUpdateCached(oi, rec.targetB, V, LR);
    } else {
      const game = replayRecord(rec);
      const f = extractRebuildOk(game);
      if (trOff) cacheTrain(oi, f);   // lazy fill on first touch (epoch 1)
      evaluateFeatures(f, weights);
      V = f.val;
      tdUpdate(f, rec.targetB, LR);
    }
    trSE += (rec.targetB - V) * (rec.targetB - V); trSEN++;
    nPos++; intervalPos++;
    const dt = Date.now() - tStartMs;
    intervalTrainMs += dt;
    if (epoch === 1) { ep1TrainMs += dt; ep1Pos++; }

    if (EMA_ALPHA > 0 && nPos % EMA_PERIOD === 0) applyEMA(EMA_ALPHA);
    if (nPos >= nextPrintPos || Date.now() >= nextPrintAt) statusPrint();
  }
  if (epoch === 1) tPosEp1 = ep1Pos > 0 ? ep1TrainMs / ep1Pos : null;   // freeze tPos
  LR *= LR_DECAY;
  if (EPOCHS > 0 && epoch >= EPOCHS) done = true;
}

if (intervalPos > 0) statusPrint();   // final partial interval
