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
// Usage: node fp-recall.js [--model ref/ref-fp2-data.js] [--md-file out/md-trunc30k]
// Library: recallStats(weights, mdRows, opts) -> { n, bestKept, near1, near2, meanRegret, byBand }

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
  return { ...fin(all), byBand: byBand.map((s, i) => s.n ? { lo: i / BANDS, hi: (i + 1) / BANDS, ...fin(s) } : null) };
}

function loadMdRows(file) {
  return require('fs').readFileSync(file, 'utf8').split('\n').map(l => MD.parseRow(l)).filter(Boolean);
}

module.exports = { recallStats, loadMdRows };

if (require.main === module) {
  const Util = require('./util.js');
  const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['model', 'md-file']);
  if (opts.help) {
    console.log(`Usage: node fp-recall.js [--model FILE] [--md-file FILE]

Recall at puct-trunc's top-K of the referee's best MD moves, and the
win ratio pruning costs.

  --model FILE     featurepol model             (default ref/ref-fp2-data.js)
  --md-file FILE   movedetails positions        (default out/md-trunc30k)
  --help           show this message`);
    process.exit(0);
  }
  const model = opts.model || 'ref/ref-fp2-data.js';
  const mdFile = opts['md-file'] || 'out/md-trunc30k';
  const { weights } = FP.loadModel({ path: model });
  const r = recallStats(weights, loadMdRows(mdFile));
  const pct = x => (100 * x).toFixed(1) + '%';
  console.log(`model: ${model}  md: ${mdFile}  positions: ${r.n}`);
  console.log(`best kept: ${pct(r.bestKept)}  within 0.01: ${pct(r.near1)}  within 0.02: ${pct(r.near2)}  mean regret: ${r.meanRegret.toFixed(4)}`);
  console.log('phase      positions  best kept  within 0.01  within 0.02  mean regret');
  for (const b of r.byBand) if (b) console.log(`${b.lo.toFixed(1)}-${b.hi.toFixed(1)}  ${String(b.n).padStart(9)}  ${pct(b.bestKept).padStart(9)}  ${pct(b.near1).padStart(11)}  ${pct(b.near2).padStart(11)}  ${b.meanRegret.toFixed(4).padStart(11)}`);
}

})();
