'use strict';

// score-bias-curve.js — score a vpat model over one or more bias-pair
// artifacts (measure-trunc-bias --emit files), binned by the recorded
// endpoint phase.  Per bin: the mean lean (bias), E[b^2] via the paired-
// return product, and varB = E[b^2] - bias^2 (the floor that survives a
// constant correction).
//
// With --fit A,B a least-squares line bias(ph) = a + b*ph is fitted over
// the bins whose centres lie inside [A, B] (the DEPLOYED band — offsets
// serve the gate, not the trainer) and printed in TRUNC_VALUE_OFFSET
// form.  Offsets are per (model, delta, band) and never transfer.
//
// Usage:
//   node score-bias-curve.js --model <vpat.js> --file <bias.txt> [--file ...]
//        [--fit A,B] [--bin 0.05]

const fs = require('fs');
const { Game2, PASS } = require('./game2.js');
const VPat = require('./vpatterns.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['model', 'file', 'fit', 'bin', 'delta']);
if (opts.help || !opts.model || !opts.file) {
  console.error(`Usage: node score-bias-curve.js --model <vpat.js> --file <bias-pairs.txt> [options]

Score a vpat model over a measure-trunc-bias --emit artifact, binned by
endpoint phase: bias (mean lean vs the playout references), E[b2] (paired-
return product) and varB (E[b2] - bias^2) per bin.

  --model PATH   vpat model to score (required)
  --file PATH    bias-pair artifact; repeat to pool several (required)
  --fit A,B      least-squares bias(ph) = a + b*ph over bin centres in
                 [A, B] (use the DEPLOYED band, not the training band);
                 prints the line as TRUNC_VALUE_OFFSET=a,b
  --bin W        phase bin width (default 0.05)
  --delta D      rescore the artifact AT DELTA D: truncate both recorded
                 prefixes ceil(D * area) moves past the start and evaluate
                 there (sound for any D up to the artifact's own delta — the
                 references belong to the START and are delta-independent).
                 Rows are then binned by the TRUNCATED endpoint's phase, so
                 the natural fit band is [D, gate].  D=0 evaluates the start
                 itself.  Offsets are per (model, delta, band) — a D fitted
                 here pairs only with TRUNC_PHASE_DELTA=D
  --help         show this message`);
  process.exit(opts.help ? 0 : 1);
}
const BIN = parseFloat(opts.bin || '0.05');
if (!(BIN > 0 && BIN <= 0.5)) { console.error('--bin: bad width'); process.exit(1); }
let FIT = null;
if (opts.fit !== undefined) {
  FIT = opts.fit.split(',').map(parseFloat);
  if (FIT.length !== 2 || !(FIT[0] < FIT[1])) { console.error('--fit: expected A,B with A < B'); process.exit(1); }
}
const files = Array.isArray(opts.file) ? opts.file : [opts.file];
const DELTA = opts.delta !== undefined ? parseFloat(opts.delta) : null;   // null = the artifact's own
if (DELTA !== null && !(DELTA >= 0 && DELTA < 1)) { console.error('--delta: bad value'); process.exit(1); }

const model = VPat.loadWeights(opts.model, process.env.HEALTH_DATA || '');
console.log(`model: ${opts.model}` + (DELTA !== null ? `  delta: ${DELTA}` : ''));

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

const bins = new Map();   // bin start (fixed decimals) -> { n, prod, lean }
for (const file of files) {
  for (const line of fs.readFileSync(file, 'utf8').split('\n')) {
    if (!line || line[0] === '#') continue;
    const p = line.split(/\s+/);
    if (p.length !== 7) continue;
    const size = +p[0], pa = +p[5], pb = +p[6];
    let ph = +p[1], v1, v2;
    if (DELTA === null) {
      v1 = VPat.evaluate(replay(size, p[3]), model);
      v2 = VPat.evaluate(replay(size, p[4]), model);
    } else {
      // Truncated rescoring: the emit ran a FIXED move count, ceil(delta *
      // area), so ceil(D * area) moves past the start is exactly the position
      // a D-delta descent would have reached.  The recorded endpointPhase
      // belongs to the artifact's own delta, so bin by the truncated E1's.
      const n0 = p[2] === '-' ? 0 : p[2].split(',').length;
      const cut = n0 + Math.ceil(DELTA * size * size);
      const g1 = replay(size, p[3], cut);
      ph = 1 - g1.emptyCount / (size * size);
      v1 = VPat.evaluate(g1, model);
      v2 = DELTA === 0 ? v1 : VPat.evaluate(replay(size, p[4], cut), model);
    }
    const key = (Math.floor(ph / BIN) * BIN).toFixed(3);
    const b = bins.get(key) || { n: 0, prod: 0, lean: 0 };
    b.n++; b.prod += (v1 - pa) * (v2 - pb); b.lean += (v1 + v2) / 2 - (pa + pb) / 2;
    bins.set(key, b);
  }
}
if (bins.size === 0) { console.error('no pairs read'); process.exit(1); }

console.log('endPhase         n      bias                E[b2]     varB');
const centres = [];
for (const key of [...bins.keys()].sort((a, b) => +a - +b)) {
  const b = bins.get(key);
  const lean = b.lean / b.n, b2 = b.prod / b.n;
  centres.push({ x: +key + BIN / 2, y: lean, n: b.n });
  console.log(`${key}-${(+key + BIN).toFixed(3)}` +
    String(b.n).padStart(8) +
    ((lean >= 0 ? '+' : '') + lean.toFixed(4)).padStart(10) + ` (±${(0.09 / Math.sqrt(b.n)).toFixed(4)})` +
    b2.toFixed(5).padStart(11) +
    (b2 - lean * lean).toFixed(5).padStart(9));
}

if (FIT) {
  // n-weighted least squares over bin centres in the deployed band.
  const pts = centres.filter(c => c.x >= FIT[0] && c.x <= FIT[1]);
  if (pts.length < 2) { console.error(`--fit: only ${pts.length} bin(s) in [${FIT[0]}, ${FIT[1]}]`); process.exit(1); }
  let W = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (const c of pts) { W += c.n; sx += c.n * c.x; sy += c.n * c.y; sxx += c.n * c.x * c.x; sxy += c.n * c.x * c.y; }
  const mx = sx / W, my = sy / W;
  const slope = (sxy - W * mx * my) / (sxx - W * mx * mx);
  const a = my - slope * mx;
  let maxResid = 0;
  for (const c of pts) maxResid = Math.max(maxResid, Math.abs(c.y - (a + slope * c.x)));
  console.log(`fit [${FIT[0]}, ${FIT[1]}] (${pts.length} bins, n-weighted)  maxResid: ${maxResid.toFixed(4)}`);
  console.log(`TRUNC_VALUE_OFFSET=${a.toFixed(3)},${slope.toFixed(3)}`);
}
