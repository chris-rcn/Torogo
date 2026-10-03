'use strict';

// train-featurepol-distill.js — fine-tune a featurepol model as a PUCT prior
// from gen-fp-distill.js records, with a pairwise ranking loss.
//
// For each record, every pair of candidates (a, b) whose teacher win ratios
// differ by more than --min-z standard errors (binomial, from the record's
// playout count) contributes the logistic loss log(1 + exp(-(s_a - s_b))),
// weighted by the gap wr_a - wr_b, where s is the model's move score (the
// softmax logit).  The loss never penalises a margin for being large, so the
// conviction of the starting model's decisive moves is not capped the way
// cross-entropy to soft targets capped it.  Gradients go through
// FeaturePol.applyScoreGradient (the REINFORCE trainer's update, so lr is on
// the same scale), normalised per record by its total pair weight.
// --decay pulls touched weights toward the STARTING model's values.
//
// Each status row reports the train loss, pair accuracy on train and holdout
// records, and fp-recall.js recall/regret on the MD file, and saves the model.
//
// Usage: node train-featurepol-distill.js --data out/x.ndjson [--load ref/ref-fp2-data.js] [--save FILE] ...

const fs = require('fs');
const path = require('path');
const Util = require('./util.js');
const FP = require('./featurepol-lib.js');
const { Game2, parseMove } = require('./game2.js');
const { recallStats, loadMdRows } = require('./fp-recall.js');
const { makeRng } = require('./xorshift.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'],
  ['data', 'load', 'save', 'lr', 'decay', 'min-z', 'epochs', 'holdout', 'md-file', 'seed']);
if (opts.help || !opts.data) {
  console.log(`Usage: node train-featurepol-distill.js --data FILE [options]

Fine-tunes a featurepol model on gen-fp-distill.js records with a pairwise
ranking loss.

  --data FILE      gen-fp-distill.js NDJSON records                  (required)
  --load FILE      starting model                         (default ref/ref-fp2-data.js)
  --save FILE      output model             (default out/featurepol-distill-<pid>.js)
  --lr F           learning rate                                      (default 0.5)
  --decay F        pull toward the starting weights, per update        (default 0)
  --min-z F        pairs need a gap of F standard errors              (default 2)
  --epochs N       passes over the training records                  (default 10)
  --holdout F      fraction of records held out for pair accuracy    (default 0.1)
  --md-file FILE   MD positions for the recall columns  (default out/md-trunc30k-827.md)
  --seed N         shuffle seed                            (default: random, logged)
  --help           show this message`);
  process.exit(opts.help ? 0 : 1);
}
const LOAD    = opts.load || 'ref/ref-fp2-data.js';
const SAVE    = opts.save || `out/featurepol-distill-${process.pid}.js`;
const LR      = parseFloat(opts.lr || '0.5');
const DECAY   = parseFloat(opts.decay || '0');
const MIN_Z   = parseFloat(opts['min-z'] || '2');
const EPOCHS  = parseInt(opts.epochs || '10', 10);
const HOLDOUT = parseFloat(opts.holdout || '0.1');
const MD_FILE = opts['md-file'] || 'out/md-trunc30k-827.md';
const SEED    = opts.seed !== undefined ? parseInt(opts.seed, 10) : Util.randomSeed();
const rng     = makeRng(SEED);
if (!(LR > 0) || !(DECAY >= 0) || !(MIN_Z >= 0) || !(EPOCHS >= 1) || !(HOLDOUT >= 0 && HOLDOUT < 1)) {
  console.error('bad option value (lr > 0, decay >= 0, min-z >= 0, epochs >= 1, 0 <= holdout < 1)');
  process.exit(1);
}

// ── Model ─────────────────────────────────────────────────────────────────────
const loaded = FP.loadModel({ path: LOAD });
const weights = loaded.weights;
if (weights.spec.rankSpaces && weights.spec.rankSpaces.length) weights.rankTopN = 0;   // as puct-trunc
const base = Float32Array.from(weights.vals.subarray(0, weights.size));   // the --decay anchor
const baseOf = idx => (idx < base.length ? base[idx] : 0);

// ── Records → training examples (pairs per record) ───────────────────────────
// Each record keeps its replayable position and its valid pairs (a better than b).
function pairsOf(rec) {
  const P = rec.po, n = rec.cands.length;
  const se = w => { const p = Math.min(1 - 1 / P, Math.max(1 / P, w)); return Math.sqrt(p * (1 - p) / P); };
  const s = rec.wr.map(se);
  const pairs = [];
  for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) {
    const gap = rec.wr[a] - rec.wr[b];
    if (gap > 0 && gap > MIN_Z * Math.sqrt(s[a] * s[a] + s[b] * s[b])) pairs.push(a, b, gap);
  }
  return pairs;
}
const records = fs.readFileSync(opts.data, 'utf8').split('\n').filter(Boolean).map(l => JSON.parse(l));
const examples = [];
for (let r = 0; r < records.length; r++) {
  const rec = records[r];
  const pairs = pairsOf(rec);
  if (pairs.length === 0) continue;
  examples.push({ rec, pairs, holdout: (r * 2654435761 >>> 0) / 4294967296 < HOLDOUT });
}
const train = examples.filter(e => !e.holdout), held = examples.filter(e => e.holdout);
if (train.length === 0) { console.error(`${opts.data}: no record has a pair above --min-z ${MIN_Z}`); process.exit(1); }
const mdRows = loadMdRows(MD_FILE);

const states = new Map();
function setUp(rec) {
  const g = new Game2(rec.size, false);   // records replay from the empty board
  for (const t of rec.moves) if (!g.play(parseMove(t, rec.size))) throw new Error(`illegal replay move ${t}`);
  let st = states.get(rec.size);
  if (!st) { st = FP.createState(rec.size, weights.spec); states.set(rec.size, st); }
  FP.extractFeatures(g, st, weights);
  const at = new Map();
  for (let i = 0; i < st.count; i++) at.set(st.moves[i], i);
  const idx = rec.cands.map(c => at.get(parseMove(c, rec.size)));
  if (idx.some(i => i === undefined)) throw new Error('a recorded candidate is not a featurepol candidate after replay');
  return { st, idx };
}

let scores = new Float64Array(1024), grad = new Float64Array(1024);
function score(st) {
  if (scores.length < st.count) { scores = new Float64Array(st.count * 2); grad = new Float64Array(st.count * 2); }
  FP.scoreAll(st, weights, scores);
}

// One record: returns its weighted loss and pair counts; updates when `learn`.
function step(ex, learn) {
  const { st, idx } = setUp(ex.rec);
  score(st);
  const p = ex.pairs;
  let loss = 0, wsum = 0, right = 0, n = 0;
  if (learn) grad.fill(0, 0, st.count);
  for (let k = 0; k < p.length; k += 3) {
    const a = idx[p[k]], b = idx[p[k + 1]], w = p[k + 2];
    const d = scores[a] - scores[b];
    loss += w * Math.log1p(Math.exp(-d));
    wsum += w; n++;
    if (d > 0) right++;
    if (learn) { const g = w / (1 + Math.exp(d)); grad[a] += g; grad[b] -= g; }
  }
  if (learn) {
    for (let i = 0; i < st.count; i++) grad[i] /= wsum;
    FP.applyScoreGradient(st, grad, weights, LR, 0);
    if (DECAY > 0) {
      const vals = weights.vals, keys = st.keys, end = st.keyOff[st.count];
      for (let k = 0; k < end; k++) { const ix = keys[k]; vals[ix] -= LR * DECAY * (vals[ix] - baseOf(ix)); }
    }
  }
  return { loss, wsum, right, n };
}

function pairAccuracy(set) {
  let right = 0, n = 0;
  for (const ex of set) { const r = step(ex, false); right += r.right; n += r.n; }
  return n ? right / n : NaN;
}

function save() {
  const src = FP.serialize(weights, { spec: weights.spec.str, ema: loaded.ema, totalUpdates: loaded.totalUpdates, komi: loaded.komi });
  const producer = [path.basename(process.argv[1]), ...process.argv.slice(2)].join(' ');
  const lines = src.split('\n');
  lines.splice(1, 0, `// Generated by: ${producer}`, `// Started from: ${LOAD}`);
  Util.writeFileAtomic(SAVE, lines.join('\n'));
}

// ── Training loop ─────────────────────────────────────────────────────────────
const nPairs = examples.reduce((s, e) => s + e.pairs.length / 3, 0);
console.log(`data      ${opts.data}  (${records.length} records, ${examples.length} with pairs: ${train.length} train, ${held.length} holdout; ${nPairs} pairs at min-z ${MIN_Z})`);
console.log(`model     ${LOAD} (${weights.size} weights)  -> ${SAVE}`);
console.log(`train     lr ${LR}, decay ${DECAY}, epochs ${EPOCHS}, seed ${SEED}; md ${MD_FILE} (${mdRows.length} positions)`);
const header = 'epoch  records  trLoss   trAcc   hoAcc  mdBest  mdNear1  mdRegret  elapsed';
const t0 = Date.now();
function row(epoch, seen, lossAvg) {
  const md = recallStats(weights, mdRows);
  const pct = x => Number.isFinite(x) ? (100 * x).toFixed(1).padStart(6) : '     -';
  console.log(`${String(epoch).padStart(5)}  ${String(seen).padStart(7)}  ${Number.isFinite(lossAvg) ? lossAvg.toFixed(4) : '     -'}  ${pct(pairAccuracy(train))}  ${pct(pairAccuracy(held))}  ${pct(md.bestKept)}  ${pct(md.near1).padStart(7)}  ${md.meanRegret.toFixed(4).padStart(8)}  ${Util.fmtMs(Date.now() - t0).padStart(7)}`);
}
console.log(header);
row(0, 0, NaN);
let seen = 0;
for (let epoch = 1; epoch <= EPOCHS; epoch++) {
  const order = train.slice();
  Util.shuffle(order, rng);
  let loss = 0, wsum = 0;
  for (const ex of order) { const r = step(ex, true); loss += r.loss; wsum += r.wsum; seen++; }
  row(epoch, seen, loss / wsum);
  save();
}
console.log(`saved ${SAVE}`);
