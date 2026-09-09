'use strict';

// train-health.js — the CHAIN HEALTH model: frozen logit weights
// predicting whether a chain survives to the end of a standard playout.
// Consumed via HEALTH_DATA by the 'size:H<N>' pattern family and by the C
// family's health-bucket slot.  The label is chain survival, not eye
// formation, and stone ninecells count as evidence alongside liberty ones.
//
// Model: P(chain survives to game end) = sigmoid(bias + sum of feature
// logits).  Features per chain, all t-hashed NINECELLS — the 3x3 region
// hash the E family uses) on a 2 + 2*--max-libs state alphabet: empty, this
// chain, and then friendly-other and enemy each split by liberty count capped
// at --max-libs.  At the default 2 that is six states — empty / this chain /
// friendly-other in atari / friendly-other with 2+ / enemy in atari / enemy
// with 2+.  Separating "this chain" from other friendly stones is what lets a
// shape mean "eye of THIS chain" (worth ~0.002 log-loss on its own); the
// liberty split says whether a neighbouring group is about to fall.  Two
// groups:
//   LIB   — one ninecell centred on each liberty of the chain.
//   STONE — one centred on each stone, salted into its own key space (the
//           same ninecell means different things around a stone and a
//           liberty).
// Trained against that exact composition, so sigma(sum) IS the survival
// estimate; the sigmoid's saturation supplies the two-eyes-suffice concavity.
//
// Liberty-PAIR keys were measured and dropped: typed by relation, tuned per
// learning rate, with coverage saturated (0.7% cold-start), they never beat
// singles alone (best 0.4578 vs 0.4581) while costing 300K weights.  An
// additive pair term cannot express "two eyes suffice" anyway — it also just
// sums, so many mediocre pairs outweigh one real conjunction.
//
// Position machinery — gen-agent-evals --prefix-delta, exactly: sample one
// random ply in [--min-phase, --max-phase] from a corpus game (the tree LEAF
// / playout-start position), then descend --delta in fullness with the
// standard playout policy (ppat-data.js, uniform below phase 0.6).  The
// ENDPOINT is where features are extracted and graded, because the endpoint
// is the position class the evaluator is actually applied to — training on
// raw corpus positions would fit the wrong distribution.  The playout then
// runs to the end to label chain survival (capture is atomic, merges
// included, so "first stone still owner-colour" decides it).
//
// Sampling is ONE endpoint position per game, so the example stream is iid
// and the training loss, scored BEFORE each update (batched score-then-update
// within a position, since one position's chains share the game's future), is
// an unbiased out-of-sample estimate — progressive validation, no test set.
// The loss column IS the generalization readout.
//
// The update is NORMALIZED by the example's active-feature count, so the
// logit moves a comparable amount whatever the chain's size; without it a
// 50-feature chain steps ~50x further than a 1-feature one.
//
// Usage: node train-health.js --corpus <games.txt> [options]
//   --corpus PATH   gen-games corpus ("<size> <move1,...>" lines), required
//   --games N       corpus games to consume (default: all)
//   --size N        board size (default 13)
//   --lr F          SGD step, constant (default 0.02).  0.05 won a 40k-game
//                   ladder, but over the full corpus the smaller step settles
//                   lower — more data, less end-state jitter.
//   --max-libs N    add a chain-level LIBERTY-COUNT one-hot for the chain
//                   being predicted, capped at N (default 8 — measured: 7 is
//                   worse, 9 no better; 0 = off).  Distinct from
//                   --other-max-libs, which caps OTHER chains' liberty counts
//                   in the ninecell alphabet.  The ninecell sum moves only
//                   linearly with liberty count, while survival is sharply
//                   non-linear in it, so this one weight per capped count
//                   lets the model fit that curve.  One extra active feature
//                   per example — it does not touch any ninecell key.
//   --max-join-libs N  BEST-SINGLE-JOIN liberty one-hot for the chain being
//                   predicted, capped at N (default 10; 0 = off): the
//                   liberty count it would have after its most favourable
//                   connecting move — my liberties, plus those of every friend
//                   adjacent to the join point, plus that point's empty
//                   neighbours, less the point itself.  Unlike the other
//                   aggregates this is COUNTERFACTUAL: no sum over current
//                   neighbourhoods can compute it at any window size.
//   --other-max-libs N  liberty-count cap for OTHER chains' stone states in
//                   the ninecell alphabet (default 2).  It does not apply to
//                   the subject chain, which is a single state.
//   --min-phase F   leaf-sample band lower bound (default 0)
//   --max-phase F   leaf-sample band upper bound (default 0.8).  With --delta
//                   this fixes the ENDPOINT distribution: the default pair
//                   gives endpoints spanning [0.2, 1.0], i.e. every phase, so
//                   the model is usable wherever it is asked — ab-search
//                   leaves, featurepol's rank feature, H-coded patterns.
//                   Outside its training band a health model extrapolates and
//                   degrades sharply, so narrow this only for a model that
//                   will be used behind a matching gate (e.g. --max-phase 0.4
//                   for a truncation evaluator gated at phase 0.6).
//   --delta F       playout descent in fullness from leaf to graded endpoint
//                   (default 0.2, the deployed truncation delta)
//   --floor F       irreducible label entropy subtracted in the 'exc' column
//                   (default: MEASURED at startup for this exact band/delta,
//                   deterministically — see measureFloor)
//                   (default 0.4493).  The label is a stochastic playout
//                   outcome, so no model can score below the mean entropy of
//                   the per-chain playout survival probability; measured on
//                   THIS distribution (leaf band [0, 0.4], delta 0.2) at 600
//                   positions x 200 completions, plug-in estimates
//                   bias-corrected and extrapolated (0.4458/0.4476/0.4491/
//                   0.4489/0.4488 at m=10/25/50/100/200).
//   --save PATH     output data file (default out/health-<random>.js)
//   --seed N        rng seed (default 23)
//   --verbose       print the end-of-run diagnostics (endpoint phase
//                   distribution, calibration deciles, highest/lowest weight
//                   shapes).  Off by default: the per-key occurrence counts
//                   and 3x3 renderings those need cost a map write per
//                   feature slot — tens of millions of them on a full run.

const fs = require('fs');
const path = require('path');
const { Game2, parseMove } = require('./game2.js');
const Util = require('./util.js');
const VPat = require('./vpatterns.js');
const PPat = require('./ppat-lib.js');
const { makeRng } = require('./xorshift.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help', 'verbose'],
  ['corpus', 'games', 'size', 'lr', 'other-max-libs', 'max-libs', 'max-join-libs', 'min-phase', 'max-phase',
   'delta', 'floor', 'save', 'seed']);
if (opts.help || !opts.corpus) {
  console.error(`Usage: node train-health.js --corpus <games.txt> [options]

Chain-survival logistic regression over the ninecells (3x3 regions) of a chain's liberties
and stones (six-state alphabet: empty, then friendly-other and enemy stones
each split by liberty count capped at --max-libs; the subject chain is one
state), at standard-playout endpoints descended 0.2 in fullness from leaves
sampled in [0, 0.4].  One iid example stream; the reported loss is prequential
(scored before each update), so it needs no test set, and 'exc' subtracts the
irreducible label entropy.

  --corpus PATH   gen-games corpus ("<size> <move1,...>" lines)
  --games N       corpus games to consume (default: all)
  --size N        board size (default 13)
  --lr F          SGD step, constant (default 0.02; 0.05 is better only on
                  short runs)
  --min-phase F   leaf band lower bound (default 0)
  --max-phase F   leaf band upper bound (default 0.8 = endpoints at every
                  phase; narrow it only for a gated consumer)
  --delta F       playout descent to the graded endpoint (default 0.2)
  --max-libs N    liberty-count one-hot for the chain being predicted, capped
                  at N (default 8 — 7 measured worse, 9 no better; 0 = off)
  --max-join-libs N  best-single-join liberty one-hot for the chain being
                  predicted, capped at N (default 10; 0 = off)
  --other-max-libs N  liberty cap for OTHER chains' stone states (default 2);
                  the subject chain is one state and is unaffected
  --floor F       irreducible label entropy for the 'exc' column (default
                  0.4493, measured on this band/delta)
  --save PATH     output data file (default out/health-<random>.js)
  --seed N        rng seed (default 23)
  --verbose       end-of-run diagnostics (phase distribution, calibration,
                  weight listings); off by default, and skipping it also skips
                  the per-key bookkeeping they need
  --help          show this message`);
  process.exit(opts.help ? 0 : 1);
}
const CORPUS = opts.corpus;
// Leaf band and descent fix the ENDPOINT distribution, and the floor belongs
// to that distribution, so it is measured per run rather than hardcoded.
const MIN_PH = parseFloat(opts['min-phase'] !== undefined ? opts['min-phase'] : '0');
const MAX_PH = parseFloat(opts['max-phase'] !== undefined ? opts['max-phase'] : '0.8');
const DELTA = parseFloat(opts.delta !== undefined ? opts.delta : '0.2');
// Floor sample size: the label is a stochastic playout outcome, so no model
// can beat the mean entropy of the per-chain survival probability.  Costs a
// few seconds at startup.  More COMPLETIONS is what buys accuracy — the
// plug-in entropy is biased low by roughly 1/(2M), and the correction is only
// first-order — while more POSITIONS just reduces variance.  At 250x40 this
// reproduces a 600x200 reference measurement (0.4493) to within ~0.001.
const FLOOR_POSITIONS = 250, FLOOR_COMPLETIONS = 40;
const SIZE = parseInt(opts.size || '13', 10);
const LR = parseFloat(opts.lr || '0.02');
const OTHER_MAX_LIBS = parseInt(opts['other-max-libs'] || '2', 10);
const MAX_LIBS = parseInt(opts['max-libs'] !== undefined ? opts['max-libs'] : '8', 10);
const MAX_JOIN_LIBS = parseInt(opts['max-join-libs'] !== undefined ? opts['max-join-libs'] : '10', 10);
// 2 + 2*maxLibs states: empty, the subject chain, then friendly-other and
// enemy each split by capped liberty count.  Splitting the SUBJECT chain by
// its own liberty count was tried and measured worse at every learning rate
// (exc 0.0539 vs 0.0515) — its liberty count is already implicit in how many
// liberty ninecells the chain emits, so the split re-encodes known information
// while halving the observations behind each key.
const N_STATES = 2 + 2 * OTHER_MAX_LIBS;
let FLOOR = opts.floor !== undefined ? parseFloat(opts.floor) : null;
const SAVE = opts.save || `out/health-${Math.random().toString(36).slice(2, 10)}.js`;
const PHASE_BINS = 4;
const area = SIZE * SIZE;
const VERBOSE = !!opts.verbose;

const rng = makeRng(parseInt(opts.seed || '23', 10) || 1);
const corpus = fs.readFileSync(CORPUS, 'utf-8').split('\n').filter(l => l && l[0] !== '#');
const GAMES = Math.min(opts.games !== undefined ? parseInt(opts.games, 10) : Infinity, corpus.length);
// The standard playout policy, as deployed (prod.js): band-trained ppat,
// uniform below phase 0.6.
const ppatModel = PPat.loadWeights(path.join(__dirname, 'ppat-data.js'));
ppatModel.uniformBelowPhase = 0.6;
const ppatState = PPat.createState(SIZE);
console.log(`train-health: corpus ${CORPUS} (${corpus.length} games, using ${GAMES})  leaf-band [${MIN_PH}, ${MAX_PH}]  delta ${DELTA}  size ${SIZE}  lr ${LR}  other-max-libs ${OTHER_MAX_LIBS} (${N_STATES}-state)  max-libs ${MAX_LIBS}  max-join-libs ${MAX_JOIN_LIBS}  save ${SAVE}`);

let bias = 0;
const weights = new Map();   // key -> logit weight
const counts = new Map();    // key -> occurrence count
const examples = new Map();  // key -> rendered 3x3 (first sighting)
// Stone-centred ninecells live in their own key space.
const STONE_SALT = 0x5bf03635 | 0;

// '.' empty; THIS chain 'X'; other chains as their capped liberty count,
// lower case for friendly (a/b) and upper for enemy (A/B).
function render(g, l, owner, gid, chainGid) {
  const nbr = g._nbr, dnbr = g._dnbr, cells = g.cells, ls = g._ls, b4 = l * 4;
  const ch = i => {
    const c = cells[i];
    if (c === 0) return '.';
    const gg = gid[i];
    const lib = Math.min(ls[gg], OTHER_MAX_LIBS);
    if (c === owner && gg === chainGid) return 'X';
    return String.fromCharCode((c === owner ? 97 : 65) + lib - 1);
  };
  return ch(dnbr[b4]) + ch(nbr[b4]) + ch(dnbr[b4 + 1]) + '/' +
         ch(nbr[b4 + 2]) + '.' + ch(nbr[b4 + 3]) + '/' +
         ch(dnbr[b4 + 2]) + ch(nbr[b4 + 1]) + ch(dnbr[b4 + 3]);
}

const makeBuf = () => ({ exShapes: [], exStart: [], exLen: [],
                         exOwner: [], exStone: [], exBin: [] });

// One position's observations: every chain contributes one example — its
// liberties' ninecells, then its stones'.
function collectObs(game, phase, buf) {
  const area = game.N * game.N;
  const cells = game.cells, nbr = game._nbr, dnbr = game._dnbr, gid = game._gid, ls = game._ls;
  const { exShapes, exStart, exLen, exOwner, exStone, exBin } = buf;
  let bin = Math.floor(phase * PHASE_BINS);
  if (bin >= PHASE_BINS) bin = PHASE_BINS - 1;
  // one pass: chain id -> its stones and (deduped) liberties
  const chains = new Map();
  for (let idx = 0; idx < area; idx++) {
    if (cells[idx] === 0) continue;
    const g0 = gid[idx];
    let c = chains.get(g0);
    if (!c) { c = { owner: cells[idx], stones: [], libs: [] }; chains.set(g0, c); }
    c.stones.push(idx);
  }
  for (let l = 0; l < area; l++) {
    if (cells[l] !== 0) continue;
    const b4 = l * 4;
    let s0 = -1, s1 = -1, s2 = -1;
    for (let d = 0; d < 4; d++) {
      const j = nbr[b4 + d];
      if (cells[j] === 0) continue;
      const gj = gid[j];
      if (gj === s0 || gj === s1 || gj === s2) continue;   // dedupe
      if (s0 < 0) s0 = gj; else if (s1 < 0) s1 = gj; else s2 = gj;
      chains.get(gj).libs.push(l);
    }
  }
  const libsByGid = MAX_JOIN_LIBS > 0 ? new Map() : null;
  if (libsByGid) for (const [g2, c2] of chains) libsByGid.set(g2, c2.libs);
  for (const [g0, c] of chains) {
    const owner = c.owner, libs = c.libs;
    const start = exShapes.length;
    // Shared with the C family's survival attribute (vpatterns.chainSurvKeys):
    // liberty ninecells first, then stone ninecells.
    VPat.chainSurvKeys(cells, nbr, dnbr, gid, ls, owner, g0, libs, c.stones,
                       OTHER_MAX_LIBS, STONE_SALT, exShapes, MAX_LIBS,
                       MAX_JOIN_LIBS, libsByGid);
    if (VERBOSE) {
      for (let a = 0; a < libs.length; a++) {
        const k = exShapes[start + a];
        if (!examples.has(k)) examples.set(k, render(game, libs[a], owner, gid, g0));
      }
      for (let a = 0; a < c.stones.length; a++) {
        const k = exShapes[start + libs.length + a];
        if (!examples.has(k)) examples.set(k, 'S:' + render(game, c.stones[a], owner, gid, g0));
      }
      // the one-hots trail the ninecells: liberty count, then best-join
      let oh = start + libs.length + c.stones.length;
      if (MAX_LIBS > 0) {
        const k = exShapes[oh++];
        if (!examples.has(k)) examples.set(k, 'libs=' + Math.min(libs.length, MAX_LIBS));
      }
      if (MAX_JOIN_LIBS > 0 && !examples.has(exShapes[oh])) examples.set(exShapes[oh], 'joinLibs');
    }
    exStart.push(start); exLen.push(exShapes.length - start);
    exOwner.push(owner); exStone.push(c.stones[0]); exBin.push(bin);
  }
}

// exc = loss - FLOOR: the only part of the loss a model can influence, since
// the label is a stochastic playout outcome (as varB is to the vpat trainer).
// Sample one endpoint the way the training loop does, from `walk` over one
// corpus line; returns the endpoint position or null if the game ended inside
// the descent.  Shared by the floor measurement and the training loop so the
// two always draw from the same distribution.
function sampleEndpoint(line, r) {
  const sp = line.indexOf(' ');
  if (parseInt(line.slice(0, sp), 10) !== SIZE) return null;
  const toks = line.slice(sp + 1).split(',');
  const walk = new Game2(SIZE);
  let game = null, nEligible = 0;
  for (let i = 0; i < toks.length; i++) {
    const ph = 1 - walk.emptyCount / area;
    if (ph > MAX_PH) break;
    if (ph >= MIN_PH) { nEligible++; if (r.random() * nEligible < 1) game = walk.clone(); }
    walk.play(parseMove(toks[i], SIZE));
  }
  if (!game) return null;
  const stopEmpty = game.emptyCount - Math.ceil(DELTA * area);
  const moveLimit = 3 * game.emptyCount + 20;
  let n = 0;
  while (!game.gameOver && game.emptyCount > stopEmpty && n < moveLimit) {
    game.play(PPat.ppatMove(game, ppatState, ppatModel, r));
    n++;
  }
  return game.emptyCount > stopEmpty ? null : game;
}

// Irreducible label entropy for THIS band/delta/policy: grade the same
// endpoint with several independent completions, and average the entropy of
// each chain's empirical survival.  A model that knew the position perfectly
// would predict exactly that, so it is the floor the 'exc' column subtracts.
// Deterministic: its own rng stream seeded from --seed, over the first usable
// corpus games.  The plug-in estimate is biased low by ~1/(2M), corrected.
function measureFloor() {
  const r = makeRng((parseInt(opts.seed || '23', 10) || 1) + 7919);
  const H = p => (p <= 0 || p >= 1) ? 0 : -(p * Math.log(p) + (1 - p) * Math.log(1 - p));
  let sumH = 0, nCh = 0, done = 0;
  for (let gi = 0; done < FLOOR_POSITIONS && gi < corpus.length; gi++) {
    const pos = sampleEndpoint(corpus[gi], r);
    if (!pos) continue;
    const reps = [], owners = [], seen = new Set();
    for (let idx = 0; idx < area; idx++) {
      if (pos.cells[idx] === 0) continue;
      const g0 = pos._gid[idx];
      if (seen.has(g0)) continue;
      seen.add(g0); reps.push(idx); owners.push(pos.cells[idx]);
    }
    const surv = new Int32Array(reps.length);
    for (let m = 0; m < FLOOR_COMPLETIONS; m++) {
      const g = pos.clone();
      let k = 0;
      const lim = 3 * g.emptyCount + 20;     // fixed bound: re-reading emptyCount would stop early
      while (!g.gameOver && k < lim) { g.play(PPat.ppatMove(g, ppatState, ppatModel, r)); k++; }
      for (let i = 0; i < reps.length; i++) if (g.cells[reps[i]] === owners[i]) surv[i]++;
    }
    for (let i = 0; i < reps.length; i++) sumH += H(surv[i] / FLOOR_COMPLETIONS);
    nCh += reps.length;
    done++;
  }
  return sumH / nCh + 1 / (2 * FLOOR_COMPLETIONS);
}

if (FLOOR === null) {
  const tF = Date.now();
  FLOOR = measureFloor();
  console.log(`floor: ${FLOOR.toFixed(4)}  (${FLOOR_POSITIONS} positions x ${FLOOR_COMPLETIONS} completions, ` +
              `bias-corrected, ${((Date.now() - tF) / 1000).toFixed(1)}s)`);
}

const COLS = ['T', 'games', 'ex', 'nWts', 'loss', 'exc'];
const COLW = [5, 5, 5, 5, 7, 7];
const printRow = cells => console.log(cells.map((c, i) => String(c).padStart(COLW[i])).join('  '));
printRow(COLS);

const t0 = Date.now();
// Print schedule: geometric in EXAMPLES — first row at 10k, then 1.4x per row.
let nextPrintEx = 10000;
let nEx = 0, nSurv = 0, nSkip = 0;
let ivSum = 0, ivN = 0;   // interval accumulators (reset at every print row)
const binN = new Float64Array(PHASE_BINS), binS = new Float64Array(PHASE_BINS);
// calibration (prequential): predicted decile -> [n, survived]
const calN = new Float64Array(10), calS = new Float64Array(10);
for (let gi = 0; gi < GAMES; gi++) {
  const line = corpus[gi];
  const sp = line.indexOf(' ');
  const rSize = parseInt(line.slice(0, sp), 10);
  if (rSize !== SIZE) { console.error(`corpus size ${rSize} != --size ${SIZE} (game ${gi + 1})`); process.exit(1); }
  const toks = line.slice(sp + 1).split(',');
  // leaf: one ply reservoir-sampled among those whose ACTUAL phase is in
  // [MIN_PH, MAX_PH] (captures make ply count overstate fullness, so the
  // band must be tested on the position, as gen-agent-evals does)
  const walk = new Game2(SIZE);
  let game = null, nEligible = 0;
  for (let i = 0; i < toks.length; i++) {
    const ph = 1 - walk.emptyCount / area;
    if (ph > MAX_PH) break;
    if (ph >= MIN_PH) {
      nEligible++;
      if (rng.random() * nEligible < 1) game = walk.clone();
    }
    walk.play(parseMove(toks[i], SIZE));
  }
  if (!game) { nSkip++; continue; }
  // descend DELTA in fullness with the standard playout; the endpoint is the
  // graded position (captures delay the gain, as deployed)
  const stopEmpty = game.emptyCount - Math.ceil(DELTA * area);
  const moveLimit = 3 * game.emptyCount + 20;
  let n = 0;
  while (!game.gameOver && game.emptyCount > stopEmpty && n < moveLimit) {
    game.play(PPat.ppatMove(game, ppatState, ppatModel, rng));
    n++;
  }
  if (game.emptyCount > stopEmpty) { nSkip++; continue; }   // game ended inside the descent
  const buf = makeBuf();
  collectObs(game, 1 - game.emptyCount / area, buf);
  // continue the playout to the end: chain survival is the label
  let n2 = 0;
  const lim2 = 3 * game.emptyCount + 20;
  while (!game.gameOver && n2 < lim2) { game.play(PPat.ppatMove(game, ppatState, ppatModel, rng)); n2++; }
  // score-then-update: all of this position's chains share the game's future,
  // so every example is scored with pre-position weights before any update
  const cells = game.cells;
  const { exShapes, exStart, exLen, exOwner, exStone, exBin } = buf;
  const nCh = exStart.length;
  const preP = new Float64Array(nCh);
  for (let i = 0; i < nCh; i++) {
    const lo = exStart[i], hi = lo + exLen[i];
    let z = bias;
    for (let j = lo; j < hi; j++) z += weights.get(exShapes[j]) || 0;
    preP[i] = 1 / (1 + Math.exp(-z));
  }
  for (let i = 0; i < nCh; i++) {
    const y = cells[exStone[i]] === exOwner[i] ? 1 : 0;
    const p = preP[i];
    const lo = exStart[i], hi = lo + exLen[i];
    // normalize by active-feature count (+1 for the bias)
    const g = LR * (y - p) / (hi - lo + 1);
    bias += g;
    for (let j = lo; j < hi; j++) weights.set(exShapes[j], (weights.get(exShapes[j]) || 0) + g);
    if (VERBOSE) for (let j = lo; j < hi; j++) counts.set(exShapes[j], (counts.get(exShapes[j]) || 0) + 1);
    ivSum += -(y ? Math.log(p + 1e-12) : Math.log(1 - p + 1e-12));
    ivN++;
    binN[exBin[i]]++; binS[exBin[i]] += y;
    let dec = Math.floor(p * 10); if (dec > 9) dec = 9;
    calN[dec]++; calS[dec] += y;
    nEx++; nSurv += y;
  }
  if (nEx >= nextPrintEx) {
    printRow([Util.fmtMs(Date.now() - t0), Util.fmt4i(gi + 1), Util.fmt4i(nEx),
              Util.fmt4i(weights.size), (ivSum / ivN).toFixed(4),
              (ivSum / ivN - FLOOR).toFixed(4)]);
    nextPrintEx = Math.max(Math.ceil(nEx * 1.4), nEx + 1);
    ivSum = 0; ivN = 0;
    saveModel();
  }
}
// Flush the final partial interval: the geometric schedule's next threshold
// usually lies past the end of the run, so without this the last stretch of
// training never appears as a row.
if (ivN > 0) {
  printRow([Util.fmtMs(Date.now() - t0), Util.fmt4i(GAMES), Util.fmt4i(nEx),
            Util.fmt4i(weights.size), (ivSum / ivN).toFixed(4),
            (ivSum / ivN - FLOOR).toFixed(4)]);
}
const secs = (Date.now() - t0) / 1000;
console.log(`done: ${GAMES} games in ${secs.toFixed(1)}s (${(secs / GAMES * 1000).toFixed(1)}ms/game, ${nSkip} skipped: game ended inside the descent)`);
console.log(`examples: ${nEx}  survival ratio: ${(nSurv / nEx).toFixed(4)}  weights: ${weights.size}  bias: ${bias.toFixed(4)}`);
console.log(`final-interval log-loss (prequential): ${(ivSum / Math.max(1, ivN)).toFixed(5)}  (above floor ${FLOOR.toFixed(4)}: ${(ivSum / Math.max(1, ivN) - FLOOR).toFixed(5)})`);
if (VERBOSE) {
  console.log('graded-endpoint phase distribution:');
  for (let b = 0; b < PHASE_BINS; b++)
    console.log(`  [${(b / PHASE_BINS).toFixed(2)},${((b + 1) / PHASE_BINS).toFixed(2)}): examples ${binN[b]}  survival ratio: ${binN[b] ? (binS[b] / binN[b]).toFixed(4) : '-'}`);
  console.log('calibration, prequential (predicted decile: actual survival ratio / n):');
  for (let d = 0; d < 10; d++)
    if (calN[d]) console.log(`  [${(d / 10).toFixed(1)},${((d + 1) / 10).toFixed(1)}): ${(calS[d] / calN[d]).toFixed(3)} / ${calN[d]}`);

  const ranked = [...weights.entries()].filter(([h]) => (counts.get(h) || 0) >= 1000);
  ranked.sort((a, b) => b[1] - a[1]);
  const show = (label, rows) => {
    console.log(label);
    for (const [h, w] of rows)
      console.log(`  w: ${w >= 0 ? '+' : ''}${w.toFixed(3)}  n: ${counts.get(h)}  ${examples.get(h)}`);
  };
  show('highest logit weights (n >= 1000):', ranked.slice(0, 12));
  show('lowest logit weights (n >= 1000):', ranked.slice(-12).reverse());
}

function saveModel() {
const entries = [...weights.entries()].map(([h, w]) => `[${h},${+w.toFixed(5)}]`);
const src = [
  "'use strict';",
  '// Auto-generated by train-health.js — do not edit by hand.',
  `// corpus ${CORPUS}  leaf-band [${MIN_PH}, ${MAX_PH}]  delta ${DELTA}  size ${SIZE}  lr ${LR}`,
  `// P(chain survives) = sigmoid(bias + sum of weights[ninecellHash(lib)] over the`,
  `// chain's liberties + sum of weights[ninecellHash(stone) ^ 0x${(STONE_SALT >>> 0).toString(16)}] over its`,
  `// stones, all on the ${N_STATES}-state alphabet: empty, this chain (one`,
  `// state), then friendly-other and enemy each split by liberty count capped`,
  `// at maxLibs ${OTHER_MAX_LIBS})`,
  `const chainSurvModel = { corpus: ${JSON.stringify(CORPUS)}, size: ${SIZE},`,
  `  // The endpoint distribution this model was fitted on: leaves sampled in`,
  `  // [minPhase, maxPhase], descended delta in fullness.  Outside it the model`,
  `  // extrapolates and its survival estimates degrade sharply, so a consumer`,
  `  // that sees every phase (ab-search, rank features) needs a full-range fit.`,
  `  minPhase: ${MIN_PH}, maxPhase: ${MAX_PH}, delta: ${DELTA}, floor: ${+FLOOR.toFixed(5)},`,
  `  examples: ${nEx}, stoneSalt: ${STONE_SALT}, otherMaxLibs: ${OTHER_MAX_LIBS},`,
  `  maxLibs: ${MAX_LIBS}, maxJoinLibs: ${MAX_JOIN_LIBS}, bias: ${+bias.toFixed(5)},`,
  `  weights: new Map([${entries.join(',')}]) };  // key -> logit weight`,
  "if (typeof module !== 'undefined') module.exports = chainSurvModel;",
  'else window.chainSurvModel = chainSurvModel;',
].join('\n') + '\n';
fs.writeFileSync(SAVE, src);
}
saveModel();
console.log(`saved: ${SAVE} (${weights.size} weights)`);
