'use strict';

// compare-dists.js — distribution-shape diagnostic: npat softmax vs search
// root visits vs policy softmax on the same positions.
//
// For each position (from a movedetails file) computes each source's
// distribution over legal placements, then reports the average shape:
// entropy, cumulative probability mass at rank 1/2/5/10/20/40, and support
// (moves above the 1/area uniform floor).  Cross columns: how often the
// source's argmax equals npat's, and the mass the source puts on npat's
// top-5 moves.  The question this informs: what transformation makes search
// results npat-shaped while keeping search's ordering where it exists.
//
// Usage:
//   node compare-dists.js --file <movedetails.ndjson> [--limit <n>]
//                         [--playouts <n>] [--seed <n>]
//
//   --file      positions source (movedetails ndjson)   (required)
//   --limit     use only the first n positions          (default: all)
//   --playouts  puct-static root search playout count   (default: 100)
//   --seed      search rng seed                         (default: 1)
//   --emit-template <path>  also write npat's empirical mean mass-by-rank
//               curve as a JS module (for train-npats --rank-template)

const EvalMD = require('./evalmovedetails.js');
const { Game2, PASS, parseMove } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');
const NPat   = require('./npat-lib.js');
const NPats = require('./npats-lib.js');
const Util   = require('./util.js');
const { makeRng } = require('./xorshift.js');
const PuctStatic = require('./ai/puct-static.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['emit-template', 'file', 'limit', 'playouts', 'seed']);
if (opts.help || !opts.file) {
  console.log('Usage: node compare-dists.js --file <md.ndjson> [--limit <n>] [--playouts <n>] [--seed <n>]');
  process.exit(opts.help ? 0 : 1);
}

const LIMIT    = opts.limit !== undefined ? parseInt(opts.limit, 10) : Infinity;
const PLAYOUTS = parseInt(opts.playouts || '100', 10);
const SEED     = parseInt(opts.seed || '1', 10);

const positions = EvalMD.loadPositions(opts.file).slice(0, LIMIT);
console.log(`${positions.length} positions from ${opts.file}  playouts=${PLAYOUTS}`);

const npatWeights = NPat.loadModel({ name: 'compare-dists' }).weights;
const npatsModel = NPats.loadModel({ name: 'compare-dists', path: process.env.NPATS_WEIGHTS });

const RANKS = [1, 2, 5, 10, 20, 40];

// Per-source accumulators.
function makeAcc(name) {
  return { name, n: 0, entropy: 0, support: 0, cum: new Float64Array(RANKS.length),
           npatTop1: 0, npatTop5Mass: 0 };
}

// probs: descending-sorted Float64Array of the distribution (sums to 1).
// movesDesc: moves in the same descending order.
function accumulate(acc, probs, movesDesc, area, npatBest, npatTop5) {
  let H = 0, support = 0;
  const floor = 1 / area;
  for (let i = 0; i < probs.length; i++) {
    const p = probs[i];
    if (p > 0) H -= p * Math.log(p);
    if (p > floor) support++;
  }
  let cum = 0;
  for (let r = 0, k = 0; r < probs.length && k < RANKS.length; r++) {
    cum += probs[r];
    while (k < RANKS.length && r + 1 === RANKS[k]) { acc.cum[k] += cum; k++; }
  }
  // Positions with fewer moves than a rank threshold count their full mass.
  for (let k = 0; k < RANKS.length; k++) {
    if (RANKS[k] > probs.length) acc.cum[k] += 1;
  }
  acc.entropy += H;
  acc.support += support;
  if (movesDesc[0] === npatBest) acc.npatTop1++;
  let m5 = 0;
  for (let i = 0; i < movesDesc.length; i++) {
    if (npatTop5.has(movesDesc[i])) m5 += probs[i];
  }
  acc.npatTop5Mass += m5;
  acc.n++;
}

// Sort a sparse {move → prob} into descending (probs, moves) arrays.
function sortDesc(moves, probs, n) {
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => probs[b] - probs[a]);
  return {
    probs: idx.map(i => probs[i]),
    moves: idx.map(i => moves[i]),
  };
}

const accNpat   = makeAcc('npat');
const accSearch = makeAcc(`search@${PLAYOUTS}`);
const accNpats = makeAcc('npats');

// npat mean mass per rank (for --emit-template).
const rankMass  = new Float64Array(512);
let rankMassN = 0;

const npatStateByN = new Map();
const polStateByN  = new Map();
const rng = makeRng(SEED);

for (const p of positions) {
  const N = p.boardSize, area = N * N;
  const game = new Game2(N);
  for (const h of p.history) game.play(parseMove(h, N));
  const game3 = game3FromGame2(game);

  // npat distribution (also defines the cross-reference head).
  let ns = npatStateByN.get(N);
  if (!ns) { ns = NPat.createState(N); npatStateByN.set(N, ns); }
  NPat.extractFeatures(game, ns, undefined, game3, npatWeights);
  if (ns.count === 0) continue;
  NPat.computeSoftmax(ns, npatWeights);
  const npat = sortDesc(Array.from(ns.moves.slice(0, ns.count)), ns.probs, ns.count);
  const npatBest = npat.moves[0];
  const npatTop5 = new Set(npat.moves.slice(0, 5));

  // Search root visit distribution.
  const r = PuctStatic.getMove(game, 0, { playoutLimit: PLAYOUTS, rng });
  const kids = (r.children || []).filter(c => c.move !== PASS);
  if (kids.length === 0) continue;
  let vSum = 0;
  for (const c of kids) vSum += c.visits;
  const search = sortDesc(kids.map(c => c.move), kids.map(c => c.visits / vSum), kids.length);

  // NPats distribution.
  let ps = polStateByN.get(N);
  if (!ps) { ps = NPats.createState(N); polStateByN.set(N, ps); }
  const pn = NPats.computeProbs(game, ps, game3, npatsModel, 0);
  if (pn === 0) continue;
  const pol = sortDesc(Array.from(ps.moves.slice(0, pn)), ps.probs, pn);

  accumulate(accNpat,   npat.probs,   npat.moves,   area, npatBest, npatTop5);
  accumulate(accSearch, search.probs, search.moves, area, npatBest, npatTop5);
  accumulate(accNpats, pol.probs,    pol.moves,    area, npatBest, npatTop5);

  for (let r = 0; r < npat.probs.length && r < rankMass.length; r++) rankMass[r] += npat.probs[r];
  rankMassN++;
}

if (opts['emit-template']) {
  const fs = require('fs');
  let last = rankMass.length;
  while (last > 1 && rankMass[last - 1] === 0) last--;
  const template = Array.from(rankMass.slice(0, last), v => v / rankMassN);
  fs.writeFileSync(opts['emit-template'],
    `'use strict';\n` +
    `// npat mean probability mass by rank — emitted by compare-dists.js\n` +
    `// (${rankMassN} positions from ${opts.file}).  Used by train-npats\n` +
    `// --rank-template to reshape search targets to npat's calibration.\n` +
    `module.exports = ${JSON.stringify(template.map(v => +v.toPrecision(6)))};\n`);
  console.log(`template (${last} ranks) written to ${opts['emit-template']}`);
}

console.log();
console.log([
  'source'.padEnd(12),
  'H'.padStart(5),
  ...RANKS.map(r => ('top' + r).padStart(6)),
  'supp'.padStart(5),
  '=npat1'.padStart(6),
  'npat5M'.padStart(6),
].join('  '));

for (const acc of [accNpat, accSearch, accNpats]) {
  const n = acc.n;
  console.log([
    acc.name.padEnd(12),
    Util.fmt4(acc.entropy / n).padStart(5),
    ...Array.from(acc.cum).map(c => Util.fmtRatio4(c / n).padStart(6)),
    Util.fmt4(acc.support / n).padStart(5),
    Util.fmtRatio4(acc.npatTop1 / n).padStart(6),
    Util.fmtRatio4(acc.npatTop5Mass / n).padStart(6),
  ].join('  '));
}
