'use strict';

// fp-recall.js — how well a featurepol model's top-K keeps the referee's best
// moves, on movedetails (MD) positions, with K as puct-trunc computes it
// (TOP_K_A at phase 0 rising to TOP_K_B at phase 1, rounded).  The vpat<n>
// ranking is off, as in puct-trunc (FPOL_RANK_TOPN 0).
//
//   recall      the referee's best-rated move is inside the top K
//   near d      some move rated within d of the best is inside the top K
//   regret      best rated win ratio minus the best kept move's
//
// Usage: node fp-recall.js [--model A.js,B.js,...] [--baseline FILE] [--by-phase] [--md-file out/md-trunc30k]
//   Each model's recall and mean regret (± SE).  With --baseline, each model's
//   regret is also paired against the baseline's position by position (± SE).
// Library: recallStats(weights, mdRows, opts) -> { n, bestKept, near1, near2, meanRegret, regretSE, regrets, byBand }

(function () {

const FP = require('./featurepol-lib.js');
const MD = require('./movedetails-format.js');
const { parseMove } = require('./game2.js');

const BANDS = 5;

// mdRows: parsed MD rows (MD.parseRow).  opts: { topKA = 10, topKB = 40 }.
function recallStats(weights, mdRows, opts = {}) {
  const topKA = opts.topKA ?? 10, topKB = opts.topKB ?? 40;
  const savedTopN = weights.rankTopN;
  if (weights.spec.rankSpaces && weights.spec.rankSpaces.length) weights.rankTopN = 0;
  const states = new Map();
  const acc = () => ({ n: 0, bestKept: 0, near1: 0, near2: 0, regretSum: 0 });
  const all = acc(), byBand = Array.from({ length: BANDS }, acc);
  const regrets = [];   // per position, in mdRows order (positions with < 2 rated moves skipped)
  for (const p of mdRows) {
    const rated = p.candidates.filter(c => c.winRatio != null);
    if (rated.length < 2) continue;
    const g = MD.buildGame(p), N = g.N;
    let st = states.get(N);
    if (!st) { st = FP.createState(N, weights.spec); states.set(N, st); }
    FP.extractFeatures(g, st, weights);
    FP.computeSoftmax(st, weights);
    const order = Array.from({ length: st.count }, (_, i) => i)
      .sort((a, b) => st.probs[b] - st.probs[a]).map(i => st.moves[i]);
    const K = Math.round(topKA + (topKB - topKA) * g.phase());
    const kept = new Set(order.slice(0, K));
    const wr = rated.map(c => ({ m: parseMove(c.m, N), w: c.winRatio })).sort((a, b) => b.w - a.w);
    const best = wr[0].w;
    let bestKept = null;
    for (const c of wr) if (kept.has(c.m)) { bestKept = c.w; break; }
    const near = d => wr.some(c => c.w >= best - d && kept.has(c.m));
    const b = Math.min(BANDS - 1, Math.floor(g.phase() * BANDS));
    regrets.push(bestKept === null ? best : best - bestKept);
    for (const s of [all, byBand[b]]) {
      s.n++;
      if (kept.has(wr[0].m)) s.bestKept++;
      if (near(0.01)) s.near1++;
      if (near(0.02)) s.near2++;
      s.regretSum += bestKept === null ? best : best - bestKept;
    }
  }
  weights.rankTopN = savedTopN;
  const fin = s => ({ n: s.n, bestKept: s.bestKept / s.n, near1: s.near1 / s.n, near2: s.near2 / s.n,
                      meanRegret: s.regretSum / s.n });
  return { ...fin(all), regretSE: _se(regrets), regrets,
           byBand: byBand.map((s, i) => s.n ? { lo: i / BANDS, hi: (i + 1) / BANDS, ...fin(s) } : null) };
}

function _se(a) {
  const n = a.length; if (n < 2) return NaN;
  let m = 0; for (const x of a) m += x; m /= n;
  let v = 0; for (const x of a) v += (x - m) * (x - m);
  return Math.sqrt(v / (n - 1) / n);
}

function loadMdRows(file) {
  return require('fs').readFileSync(file, 'utf8').split('\n').map(l => MD.parseRow(l)).filter(Boolean);
}

module.exports = { recallStats, loadMdRows };

if (require.main === module) {
  const Util = require('./util.js');
  const path = require('path');
  const opts = Util.parseArgs(process.argv.slice(2), ['help', 'by-phase'], ['model', 'baseline', 'md-file']);
  if (opts.help) {
    console.log(`Usage: node fp-recall.js [--model A.js,B.js,...] [--baseline FILE] [options]

Recall at puct-trunc's top-K of the referee's best MD moves, and the
win ratio pruning costs (regret), for each model.

  --model LIST     comma-separated featurepol models     (default ref/ref-fp2-data.js)
  --baseline FILE  pair every model's regret against this model, position by
                   position (± SE); it is reported on its own row too
  --by-phase       add each model's phase-band breakdown
  --md-file FILE   movedetails positions                 (default out/md-trunc30k)
  --help           show this message`);
    process.exit(0);
  }
  const models = (opts.model || 'ref/ref-fp2-data.js').split(',').map(x => x.trim()).filter(Boolean);
  const mdFile = opts['md-file'] || 'out/md-trunc30k';
  const rows = loadMdRows(mdFile);
  const run = file => recallStats(FP.loadModel({ path: file }).weights, rows);
  const base = opts.baseline ? { file: opts.baseline, r: run(opts.baseline) } : null;
  const results = models.map(file => ({ file, r: file === opts.baseline ? base.r : run(file) }));
  const pct = x => (100 * x).toFixed(1) + '%';
  const n = (base || results[0]).r.n;
  console.log(`md: ${mdFile}  positions: ${n}${base ? `  baseline: ${base.file}` : ''}`);
  const w = Math.max(5, ...[...results, ...(base ? [base] : [])].map(x => path.basename(x.file).length));
  const head = `${'model'.padEnd(w)}  best kept  within 0.01  within 0.02  mean regret` + (base ? '          vs baseline' : '');
  console.log(head);
  const line = (x, isBase) => {
    const r = x.r;
    let t = `${path.basename(x.file).padEnd(w)}  ${pct(r.bestKept).padStart(9)}  ${pct(r.near1).padStart(11)}  ${pct(r.near2).padStart(11)}  ${r.meanRegret.toFixed(4)} ± ${r.regretSE.toFixed(4)}`;
    if (base && !isBase) {
      const d = r.regrets.map((v, i) => v - base.r.regrets[i]);
      let m = 0; for (const v of d) m += v; m /= d.length;
      t += `  ${m >= 0 ? '+' : ''}${m.toFixed(4)} ± ${_se(d).toFixed(4)}`;
    } else if (base) t += '  (baseline)';
    console.log(t);
  };
  if (base && !models.includes(base.file)) line(base, true);
  for (const x of results) line(x, x.file === opts.baseline);
  if (opts['by-phase']) for (const x of (base && !models.includes(base.file) ? [base, ...results] : results)) {
    console.log(`\n${path.basename(x.file)}\nphase      positions  best kept  within 0.01  within 0.02  mean regret`);
    for (const b of x.r.byBand) if (b) console.log(`${b.lo.toFixed(1)}-${b.hi.toFixed(1)}  ${String(b.n).padStart(9)}  ${pct(b.bestKept).padStart(9)}  ${pct(b.near1).padStart(11)}  ${pct(b.near2).padStart(11)}  ${b.meanRegret.toFixed(4).padStart(11)}`);
  }
}

})();
