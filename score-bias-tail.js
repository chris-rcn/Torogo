'use strict';

// score-bias-tail.js — TAIL analysis of a vpat model's truncation bias over a
// measure-trunc-bias --emit artifact.  score-bias-curve reports the bias
// curve and varB, both averages; this tool asks how the error MASS is
// distributed: what fraction of positions are wrong by a lot, and how much of
// the squared error they carry.  A rare-but-catastrophic error class (a
// misjudged group's life) can decide games while moving varB little — or can
// turn out to BE most of varB; the decomposition says which.
//
// Per row the artifact gives two evaluator endpoints (v1, v2) and a split
// playout reference (pa, pb).  The per-position bias estimate is
//   bhat = (v1+v2)/2 - (pa+pb)/2,
// corrected by the bin's mean lean (the deployed offset removes exactly
// that).  bhat carries sampling noise on top of the true b(s); the NOISE
// NULL
//   nu = (v1-v2)/2 + (pa-pb)/2
// has the same variance as that noise, zero mean, and is independent of
// b(s) — so nu's tail is what bhat's tail would look like for an evaluator
// with no positional bias at all, and the EXCESS of bhat's tail over nu's is
// real.  The sq% column is each tail's share of the total squared corrected
// bias, i.e. the decomposition of (varB + noise) by error size.
//
// Usage:
//   node score-bias-tail.js --model <vpat.js> --file <bias.txt> [--file ...]
//        [--thresholds 0.1,0.2,0.3] [--bin 0.1]

const fs = require('fs');
const { Game2, PASS } = require('./game2.js');
const VPat = require('./vpatterns.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['model', 'file', 'thresholds', 'bin', 'min-phase', 'max-phase', 'bin-by', 'delta']);
if (opts.help || !opts.model || !opts.file) {
  console.error(`Usage: node score-bias-tail.js --model <vpat.js> --file <bias-pairs.txt> [options]

Tail analysis of the truncation bias: per phase bin and pooled, the fraction
of positions whose offset-corrected bias estimate exceeds each threshold,
that tail's share of the total squared bias, and the same fractions for the
noise null (what a bias-free evaluator would show).

  --model PATH        vpat model to score (required)
  --file PATH         bias-pair artifact; repeat to pool several (required)
  --thresholds LIST   |b| thresholds (default 0.1,0.2,0.3)
  --bin W             phase bin width (default 0.1)
  --min-phase F / --max-phase F  phase band to include (default all), in the
                      --bin-by axis.  Use the DEPLOYED band so the pooled row
                      means what the agent actually evaluates; the artifact's
                      endpoints run delta past its start band
  --delta D           rescore the artifact AT DELTA D: truncate both recorded
                      prefixes ceil(D * area) moves past the start and
                      evaluate there.  Sound for any D up to the artifact's
                      own delta, because the references belong to the START
                      (delta-independent) and a prefix of a prefix is exactly
                      a shorter prefix (the policy is Markov).  One artifact
                      therefore carries the whole delta curve, PAIRED: every
                      D shares starts, references and prefix randomness, so
                      curve differences are within-row.  D=0 evaluates the
                      start itself — both truncations coincide, the return is
                      deterministic, and the noise null shrinks to reference
                      noise alone.  D above the artifact's delta is an error.
                      Requires --bin-by start (the recorded endpointPhase
                      belongs to the artifact's own delta)
  --bin-by end|start  binning axis (default end = the recorded endpoint
                      phase).  'start' bins by the playout-START phase,
                      recovered by replaying the start moves — the axis that
                      stays comparable ACROSS deltas, since b is the eval at
                      the endpoint against the truth at the start and a
                      different delta shifts every endpoint bin's start
                      population by the delta difference
  --help              show this message`);
  process.exit(opts.help ? 0 : 1);
}
const BIN = parseFloat(opts.bin || '0.1');
if (!(BIN > 0 && BIN <= 0.5)) { console.error('--bin: bad width'); process.exit(1); }
const THR = (opts.thresholds || '0.1,0.2,0.3').split(',').map(parseFloat);
if (THR.some(t => !(t > 0 && t < 1))) { console.error('--thresholds: bad list'); process.exit(1); }
const MIN_PH = parseFloat(opts['min-phase'] !== undefined ? opts['min-phase'] : '0');
const MAX_PH = parseFloat(opts['max-phase'] !== undefined ? opts['max-phase'] : '1');
const DELTA = opts.delta !== undefined ? parseFloat(opts.delta) : null;   // null = the artifact's own
if (DELTA !== null && !(DELTA >= 0 && DELTA < 1)) { console.error('--delta: bad value'); process.exit(1); }
const BIN_BY = opts['bin-by'] || (DELTA !== null ? 'start' : 'end');
if (BIN_BY !== 'end' && BIN_BY !== 'start') { console.error('--bin-by: end or start'); process.exit(1); }
if (DELTA !== null && BIN_BY !== 'start') {
  console.error('--delta bins by start (the recorded endpointPhase belongs to the artifact\'s own delta)');
  process.exit(1);
}
const files = Array.isArray(opts.file) ? opts.file : [opts.file];

const model = VPat.loadWeights(opts.model);
console.log(`model: ${opts.model}`);

function replay(size, moves, limit) {
  const g = new Game2(size, true);
  const toks = moves.split(',');
  const n = limit === undefined ? toks.length : limit;
  if (n > toks.length) {
    throw new Error(`--delta wants ${n} moves but the row has ${toks.length} — above the artifact's own delta`);
  }
  for (let i = 0; i < n; i++) {
    const t = toks[i];
    const m = t[0] === 'p' ? PASS : (parseInt(t.slice(1), 10) - 1) * size + (t.charCodeAt(0) - 97);
    if (!g.play(m)) throw new Error('bias-pair replay failed — artifact/engine mismatch');
  }
  return g;
}

// Pass 1: evaluate every row, keeping (phase, bhat, nu, prod) — the per-bin
// mean lean must be known before the tails can be offset-corrected.
const rows = [];
for (const file of files) {
  for (const line of fs.readFileSync(file, 'utf8').split('\n')) {
    if (!line || line[0] === '#') continue;
    const p = line.split(/\s+/);
    if (p.length !== 7) continue;
    const size = +p[0], pa = +p[5], pb = +p[6];
    let ph = +p[1], g0 = null;
    if (BIN_BY === 'start') {
      // Start phase is not recorded; recover it by replaying the start moves
      // (captures shift emptyCount, so a move count is not enough).
      g0 = p[2] === '-' ? new Game2(size, true) : replay(size, p[2]);
      ph = 1 - g0.emptyCount / (size * size);
    }
    if (ph < MIN_PH || ph > MAX_PH) continue;
    let v1, v2;
    if (DELTA === null) {
      v1 = VPat.evaluate(replay(size, p[3]), model);
      v2 = VPat.evaluate(replay(size, p[4]), model);
    } else if (DELTA === 0) {
      v1 = v2 = VPat.evaluate(g0, model);
    } else {
      // The recorded prefixes truncated at the D-delta point: the emit runs a
      // FIXED move count, ceil(delta * area), so ceil(D * area) moves past the
      // start is exactly the position a D-delta descent would have reached.
      const n0 = p[2] === '-' ? 0 : p[2].split(',').length;
      const cut = n0 + Math.ceil(DELTA * size * size);
      v1 = VPat.evaluate(replay(size, p[3], cut), model);
      v2 = VPat.evaluate(replay(size, p[4], cut), model);
    }
    rows.push({ ph,
                bhat: (v1 + v2) / 2 - (pa + pb) / 2,
                nu:   (v1 - v2) / 2 + (pa - pb) / 2,
                prod: (v1 - pa) * (v2 - pb) });
  }
}
if (rows.length === 0) { console.error('no pairs read'); process.exit(1); }

const binKey = ph => (Math.floor(ph / BIN) * BIN).toFixed(3);
const lean = new Map();
for (const r of rows) {
  const k = binKey(r.ph);
  const a = lean.get(k) || { n: 0, s: 0 };
  a.n++; a.s += r.bhat; lean.set(k, a);
}

// Pass 2: per-bin tails of the corrected bhat and of the noise null, and each
// tail's share of the total squared corrected bias.
const bins = new Map();
for (const r of rows) {
  const k = binKey(r.ph);
  const c = r.bhat - lean.get(k).s / lean.get(k).n;
  let a = bins.get(k);
  if (a === undefined) {
    a = { n: 0, prod: 0, sq: 0, tail: THR.map(() => 0), tailSq: THR.map(() => 0),
          nuTail: THR.map(() => 0) };
    bins.set(k, a);
  }
  a.n++; a.prod += r.prod; a.sq += c * c;
  for (let i = 0; i < THR.length; i++) {
    if (Math.abs(c) > THR[i]) { a.tail[i]++; a.tailSq[i] += c * c; }
    if (Math.abs(r.nu) > THR[i]) a.nuTail[i]++;
  }
}

const hdrTails = THR.map(t => `|b|>${t}%  sq%  nu%`.padStart(19)).join('');
console.log(`\nrows: ${rows.length}   (tails offset-corrected per bin; nu = noise null)`);
console.log((BIN_BY === 'start' ? 'startPhase' : 'endPhase  ') +
  '       n     E[b2]   mean-sq' + hdrTails);
function printRow(label, a) {
  let s = label + String(a.n).padStart(8) +
    (a.prod / a.n).toFixed(5).padStart(10) +
    (a.sq / a.n).toFixed(5).padStart(10);
  for (let i = 0; i < THR.length; i++) {
    s += (100 * a.tail[i] / a.n).toFixed(2).padStart(8) +
         (100 * a.tailSq[i] / (a.sq || 1)).toFixed(1).padStart(6) +
         (100 * a.nuTail[i] / a.n).toFixed(2).padStart(7);
  }
  console.log(s);
}
const pooled = { n: 0, prod: 0, sq: 0, tail: THR.map(() => 0), tailSq: THR.map(() => 0),
                 nuTail: THR.map(() => 0) };
for (const key of [...bins.keys()].sort((a, b) => +a - +b)) {
  const a = bins.get(key);
  printRow(`${key}-${(+key + BIN).toFixed(3)}`, a);
  pooled.n += a.n; pooled.prod += a.prod; pooled.sq += a.sq;
  for (let i = 0; i < THR.length; i++) {
    pooled.tail[i] += a.tail[i]; pooled.tailSq[i] += a.tailSq[i]; pooled.nuTail[i] += a.nuTail[i];
  }
}
printRow('pooled     ', pooled);
console.log('\nmean-sq = E[(bhat - lean)^2] = varB + sampling noise (nu^2 mean ' +
  (rows.reduce((s, r) => s + r.nu * r.nu, 0) / rows.length).toFixed(5) + ')');
