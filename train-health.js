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
//                   in the ninecell alphabet.  The ninecell sum moves only
//                   linearly with liberty count, while survival is sharply
//                   non-linear in it, so this one weight per capped count
//                   lets the model fit that curve.  One extra active feature
//                   per example — it does not touch any ninecell key.
//   --friend-health-max-buckets N  friendHealthMax: a one-hot over the health
//                   of the HEALTHIEST joinable friend, bucketed uniformly in p
//                   into N levels (default 7; 0 = off).  Measured on the full
//                   corpus at leaf-band [0, 0.8]: exc 0.0295 at 7 buckets
//                   against 0.0336 with the feature off, 12% of the excess
//                   loss for seven extra weights.  The ladder is monotone in
//                   N and saturates at 7-8 (8 buys 0.0001, inside the row
//                   jitter).  The friend relation runs
//                   through a shared liberty — two friendly chains in contact
//                   would be one chain — so it is the same neighbour set the
//                   best-single-join one-hot considers, but valued by whether
//                   that friend is alive rather than by what it adds in
//                   liberties.  A max is the right aggregate for friends
//                   (one strong friend suffices to escape); for foes it would
//                   not be.  N is free to raise: one weight per bucket, and
//                   the ninecell key space is untouched.
//   --foe-health-min-buckets N  foeHealthMin: a one-hot over the health of the
//                   WEAKEST enemy chain in contact, bucketed uniformly in p
//                   into N levels (default 5; 0 = off).  Measured on the full
//                   corpus at leaf-band [0, 0.8], on top of friend 7: exc
//                   0.0285 against 0.0306 without it, saturating at 5 buckets
//                   (4 gives 0.0287, 8 gives 0.0285).  A different channel from
//                   the mean, not a coarser view of it — the mean says how
//                   enclosed a chain is, the minimum says whether there is
//                   something nearby worth attacking, and the two disagree
//                   exactly where it matters (a doomed enemy stone inside the
//                   subject's own area: min 0.033, contact-weighted mean
//                   0.838).  Not contact-weighted: every enemy in contact is a
//                   candidate however short the shared border.
//   --iterations N  forward passes of the propagation (default 2), used only
//                   when a neighbour-health feature is on.  Every chain starts
//                   at health 0.5; pass k rescores every chain with pass k-1's
//                   neighbour healths.  The LAST pass's key is the one the
//                   example carries, so the gradient sees the neighbour values
//                   as fixed inputs — no gradient flows back through them.
//                   At N=1 every chain still holds INIT_HEALTH, so the
//                   bucket is constant and the feature is just a second bias;
//                   N=2 is the first setting that propagates anything.
//   --max-join-libs N  BEST-SINGLE-JOIN liberty one-hot for the chain being
//                   predicted, capped at N (default 10; 0 = off): the
//                   liberty count it would have after its most favourable
//                   connecting move — my liberties, plus those of every friend
//                   adjacent to the join point, plus that point's empty
//                   neighbours, less the point itself.  Unlike the other
//                   aggregates this is COUNTERFACTUAL: no sum over current
//                   neighbourhoods can compute it at any window size.
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
const HL = require('./health-lib.js');
const PPat = require('./ppat-lib.js');
const { makeRng } = require('./xorshift.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help', 'verbose'],
  ['corpus', 'games', 'size', 'lr', 'max-libs', 'max-stones', 'max-join-libs',
   'friend-health-max-buckets', 'foe-health-min-buckets', 'stone-ninecells',
   'liberty-ninecells', 'liberty-pairs', 'iterations', 'min-phase', 'max-phase',
   'delta', 'floor', 'save', 'seed']);
if (opts.help || !opts.corpus) {
  console.error(`Usage: node train-health.js --corpus <games.txt> [options]

Chain-survival logistic regression over the ninecells (3x3 regions) of a chain's
liberties and stones, each cell coded by COLOR alone (empty / the chain's color /
the other color), exact-indexed and canonicalised over D4, at standard-playout
endpoints descended --delta in fullness from leaves sampled in the band.  One iid example stream; the reported loss is prequential
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
  --stone-ninecells 0|1  emit a ninecell per STONE (default 1)
  --liberty-ninecells 0|1  emit a ninecell per LIBERTY (default 1).  The two
                  halves are the bulk of this model's cost and are partly
                  redundant; at least one must stay on
  --max-stones N  chain SIZE one-hot, capped at N (default 0 = off).  One key
                  per chain, no hashing
  --liberty-pairs N  one key per unordered pair of NON-ADJACENT liberties, for
                  chains with at most N liberties (default 0 = off).  The two-eye
                  signal is a conjunction and cannot come from the additive sum
  --max-join-libs N  best-single-join liberty one-hot for the chain being
                  predicted, capped at N (default 10; 0 = off)
  --friend-health-max-buckets N  one-hot over the health of the healthiest
                  joinable friend, bucketed uniformly in p into N levels
                  (default 7 — monotone in N, saturated by 7-8; 0 = off);
                  needs --iterations >= 2 to say anything
  --foe-health-min-buckets N  one-hot over the health of the weakest enemy
                  chain in contact, bucketed uniformly in p into N levels
                  (default 5 — saturated by 5; 0 = off)
  --iterations N  forward propagation passes (default 2); all chains start at
                  INIT_HEALTH and the last pass's neighbour values are fixed
                  inputs to the gradient step
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
// The descent is a fixed number of MOVES, not a fullness check repeated after
// every move: delta is expressed as a fullness fraction only so one number
// carries across board sizes, and once the size is known the prefix should be
// a constant length whatever it captures.  Matches the deployed agent
// (ai/puct-ppat-fp-trunc.js) exactly.
const PREFIX_LEN = Math.ceil(DELTA * SIZE * SIZE);

const LR = parseFloat(opts.lr || '0.02');
const MAX_LIBS = parseInt(opts['max-libs'] !== undefined ? opts['max-libs'] : '8', 10);
// Chain SIZE one-hot, the sibling of --max-libs.  One key per chain and no
// hashing, so it is free next to the ninecells; default 0 (off).
const MAX_STONES = parseInt(opts['max-stones'] !== undefined ? opts['max-stones'] : '0', 10);
const MAX_JOIN_LIBS = parseInt(opts['max-join-libs'] !== undefined ? opts['max-join-libs'] : '10', 10);
const FHM_BUCKETS = parseInt(opts['friend-health-max-buckets'] !== undefined
  ? opts['friend-health-max-buckets'] : '7', 10);
const FOE_MIN_BUCKETS = parseInt(opts['foe-health-min-buckets'] !== undefined
  ? opts['foe-health-min-buckets'] : '5', 10);
const NEIGHBOUR_ON = FHM_BUCKETS > 0 || FOE_MIN_BUCKETS > 0;
const ITERATIONS = parseInt(opts.iterations !== undefined ? opts.iterations : '2', 10);
// What a chain's neighbours are assumed to be worth before the first
// propagation pass.  Measured on the full corpus (friend 7, iterations 2):
// 0.5 gave exc 0.0302 and the true base rate 0.7541 gave 0.0310, a gap inside
// the row jitter, so 0.6 splits them.  Recorded in the saved model because
// inference has to start from the same place.
const INIT_HEALTH = 0.6;
// Whether a chain emits a ninecell per STONE as well as per liberty.  Profiled
// 2026-09-11: the ninecell hashing is ~13% of an evaluator's total run time and
// the stone hashes are most of it — a chain emits one per stone, and stones
// outnumber liberties on a full board.  0 keeps only the liberty ninecells.
// Recorded in the saved model, since scoring must emit the same key set.
const STONE_NINECELLS = (opts['stone-ninecells'] !== undefined
  ? parseInt(opts['stone-ninecells'], 10) : 1) !== 0;
// The mirror: keep only the per-STONE ninecells.  Dropping stones while keeping
// liberties was measured near-free, which says stones are redundant GIVEN
// liberties — not that liberties carry the signal.  This is the knob that tells
// the two apart.
const LIBERTY_NINECELLS = (opts['liberty-ninecells'] !== undefined
  ? parseInt(opts['liberty-ninecells'], 10) : 1) !== 0;
// One key per unordered pair of NON-ADJACENT liberties, for chains with at most
// this many liberties (0 = off).  Two eyes mean life, which is a conjunction of
// two separated shapes; a sum over per-liberty ninecells is additive and cannot
// express it.  The gate matters because pairs grow as n^2 and a chain with many
// liberties is not one whose life is in question.
const LIBERTY_PAIRS = parseInt(opts['liberty-pairs'] !== undefined ? opts['liberty-pairs'] : '0', 10);
// Same field names a health model uses, so health-lib's neighbourHealthKeys and
// propagateHealth take this and a scored model interchangeably.
const NB_CFG = { friendHealthMaxBuckets: FHM_BUCKETS, foeHealthMinBuckets: FOE_MIN_BUCKETS,
                 iterations: ITERATIONS, initHealth: INIT_HEALTH };
// 2 + 2*maxLibs states: empty, the subject chain, then friendly-other and
// enemy each split by capped liberty count.  Splitting the SUBJECT chain by
// its own liberty count was tried and measured worse at every learning rate
// (exc 0.0539 vs 0.0515) — its liberty count is already implicit in how many
// liberty ninecells the chain emits, so the split re-encodes known information
// while halving the observations behind each key.
let FLOOR = opts.floor !== undefined ? parseFloat(opts.floor) : null;
const SAVE = opts.save || `out/health-${Math.random().toString(36).slice(2, 10)}.js`;
const PHASE_BINS = 4;
const area = SIZE * SIZE;
const VERBOSE = !!opts.verbose;

const rng = makeRng(parseInt(opts.seed || '23', 10) || 1);
// The corpus is held as a BUFFER and indexed by line offset, not split into
// strings: half a gigabyte is past V8's maximum string length (0x1fffffe8), so
// readFileSync(path, 'utf-8') throws outright, and one string per game would
// cost more than the file itself.  corpusLine(i) decodes a single line on
// demand — the loops touch each line once.
const corpusBuf = fs.readFileSync(CORPUS);
const lineAt = [], lineEnd = [];
for (let i = 0, n = corpusBuf.length; i < n; ) {
  let j = corpusBuf.indexOf(10, i);
  if (j < 0) j = n;
  if (j > i && corpusBuf[i] !== 35) { lineAt.push(i); lineEnd.push(j); }   // 35 = '#'
  i = j + 1;
}
const corpusCount = lineAt.length;
const corpusLine = i => corpusBuf.toString('utf8', lineAt[i], lineEnd[i]);
const GAMES = Math.min(opts.games !== undefined ? parseInt(opts.games, 10) : Infinity, corpusCount);
// The standard playout policy, as deployed (prod.js): band-trained ppat,
// uniform below phase 0.6.
const ppatModel = PPat.loadWeights(path.join(__dirname, 'ppat-data.js'));
ppatModel.uniformBelowPhase = 0.6;
const ppatState = PPat.createState(SIZE);
console.log(`train-health: corpus ${CORPUS} (${corpusCount} games, using ${GAMES})  leaf-band [${MIN_PH}, ${MAX_PH}]  delta ${DELTA}  size ${SIZE}  lr ${LR}  ninecell 3-state (color)  max-libs ${MAX_LIBS}${MAX_STONES > 0 ? `  max-stones ${MAX_STONES}` : ''}  max-join-libs ${MAX_JOIN_LIBS}` +
            (STONE_NINECELLS ? '' : '  stone-ninecells 0') +
            (LIBERTY_NINECELLS ? '' : '  liberty-ninecells 0') +
            (LIBERTY_PAIRS > 0 ? `  liberty-pairs ${LIBERTY_PAIRS}` : '') +
            (FHM_BUCKETS > 0 ? `  friend-health-max-buckets ${FHM_BUCKETS}` : '') +
            (FOE_MIN_BUCKETS > 0 ? `  foe-health-min-buckets ${FOE_MIN_BUCKETS}` : '') +
            (NEIGHBOUR_ON ? `  iterations ${ITERATIONS}` : '') +
            `  save ${SAVE}`);

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
    return c === owner ? 'x' : 'O';
  };
  return ch(dnbr[b4]) + ch(nbr[b4]) + ch(dnbr[b4 + 1]) + '/' +
         ch(nbr[b4 + 2]) + '.' + ch(nbr[b4 + 3]) + '/' +
         ch(dnbr[b4 + 2]) + ch(nbr[b4 + 1]) + ch(dnbr[b4 + 3]);
}

// exNb holds the neighbour-health keys, NB_SLOTS per chain (0 = none), kept
// outside the exShapes range because they are not a function of the position
// alone: the propagation rewrites them every pass, and a chain with no joinable
// friend, or none in contact with an enemy, has no such feature at all.
const makeBuf = () => ({ exShapes: [], exStart: [], exLen: [],
                         exOwner: [], exStone: [], exBin: [], exNb: [], exLive: [] });

// Propagation scratch: at most one chain per point.  The neighbour relations
// and the key computation come from vpatterns, so what is trained here and what
// is scored there cannot drift.
const _baseZ = new Float64Array(area);
const NB_SLOTS = HL.NB_SLOTS;   // friendHealthMax, foeHealthMin
const _nb = HL.makeNeighbourhoods();

// One position's observations: every chain contributes one example — its
// liberties' ninecells, then its stones'.
function collectObs(game, phase, buf) {
  const cells = game.cells, nbr = game._nbr, dnbr = game._dnbr, gid = game._gid, ls = game._ls;
  const { exShapes, exStart, exLen, exOwner, exStone, exBin, exNb, exLive } = buf;
  let bin = Math.floor(phase * PHASE_BINS);
  if (bin >= PHASE_BINS) bin = PHASE_BINS - 1;
  // Chain enumeration, the friend relation and the key set all come from
  // vpatterns, so what is trained here and what is scored there cannot drift.
  // chainsOf's chain order IS the example order, so a record's .idx indexes
  // the example arrays directly.
  const { chains, byGid } = HL.chainsOf(cells, nbr, gid);
  // Chains PROVEN uncapturable are pinned to health 1 rather than predicted,
  // exactly as health-lib.chainHealthAll does at scoring time.  Their gradient
  // is then zero on its own (y = p = 1), so they stop dragging the liberty
  // one-hot: a 19-stone group with two eyes is not evidence that 2 liberties
  // is survivable.
  HL.markLiveChains(cells, nbr, gid, dnbr, chains, byGid);
  for (const c of chains) {
    const owner = c.c, libs = c.libs, g0 = c.gid;
    const start = exShapes.length;
    // health-lib.chainSurvKeys: liberty ninecells first, then stone ninecells
    // (the latter only when --stone-ninecells is on).
    HL.chainSurvKeys(cells, nbr, dnbr, gid, owner, g0, libs, c.stones,
                       STONE_SALT, exShapes, MAX_LIBS,
                       MAX_JOIN_LIBS, byGid, STONE_NINECELLS, LIBERTY_NINECELLS,
                       MAX_STONES, LIBERTY_PAIRS);
    if (VERBOSE) {
      if (LIBERTY_NINECELLS) for (let a = 0; a < libs.length; a++) {
        const k = exShapes[start + a];
        if (!examples.has(k)) examples.set(k, render(game, libs[a], owner, gid, g0));
      }
      if (STONE_NINECELLS) {
        const base = start + (LIBERTY_NINECELLS ? libs.length : 0);
        for (let a = 0; a < c.stones.length; a++) {
          const k = exShapes[base + a];
          if (!examples.has(k)) examples.set(k, 'S:' + render(game, c.stones[a], owner, gid, g0));
        }
      }
      // the one-hots trail the ninecells: liberty count, then best-join
      let oh = start + (LIBERTY_NINECELLS ? libs.length : 0)
                     + (STONE_NINECELLS ? c.stones.length : 0);
      if (MAX_STONES > 0) {
        const k = exShapes[oh++];
        if (!examples.has(k)) examples.set(k, 'stones=' + Math.min(c.stones.length, MAX_STONES));
      }
      if (MAX_LIBS > 0) {
        const k = exShapes[oh++];
        if (!examples.has(k)) examples.set(k, 'libs=' + Math.min(libs.length, MAX_LIBS));
      }
      if (MAX_JOIN_LIBS > 0 && !examples.has(exShapes[oh])) examples.set(exShapes[oh], 'joinLibs');
    }
    for (let k = 0; k < NB_SLOTS; k++) exNb.push(0);
    exLive.push(c.live);
    exStart.push(start); exLen.push(exShapes.length - start);
    exOwner.push(owner); exStone.push(c.stones[0]); exBin.push(bin);
  }
  if (NEIGHBOUR_ON) propagate(buf, cells, nbr, gid, chains, byGid);
}

// Iterative neighbour-health propagation.  Every chain starts at health 0.5;
// each pass rescores every chain from the PREVIOUS pass's neighbour healths,
// with the weights as they stand before this position is trained on.  The
// final pass leaves its keys in exFH, and those are what the example carries,
// so the gradient step treats the neighbour values as fixed inputs — nothing
// flows back through them.
function propagate(buf, cells, nbr, gid, chains, byGid) {
  const { exShapes, exStart, exLen, exNb, exLive } = buf;
  const nCh = exStart.length;
  HL.neighbourhoodsOf(cells, nbr, gid, chains, byGid, FHM_BUCKETS > 0, FOE_MIN_BUCKETS > 0, _nb);
  // The per-chain base logit: bias plus the chain's own ninecells and one-hots.
  // Fixed for the whole propagation — the weights do not move until this
  // position has been trained on — so it is summed once rather than per pass.
  for (let i = 0; i < nCh; i++) {
    const lo = exStart[i], hi = lo + exLen[i];
    let z = bias;
    for (let j = lo; j < hi; j++) z += weights.get(exShapes[j]) || 0;
    _baseZ[i] = z;
  }
  HL.propagateHealth(NB_CFG, weights, _baseZ, exLive, nCh, _nb, exNb);
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
  let n = 0;
  while (!game.gameOver && n < PREFIX_LEN) {
    game.play(PPat.ppatMove(game, ppatState, ppatModel, r));
    n++;
  }
  return n < PREFIX_LEN ? null : game;
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
  for (let gi = 0; done < FLOOR_POSITIONS && gi < corpusCount; gi++) {
    const pos = sampleEndpoint(corpusLine(gi), r);
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

// 'tPos' = INFERENCE time per position, tallied CUMULATIVELY over every
// position so far (unlike loss/exc, which are per-interval): what
// a consumer pays to score one position — chain enumeration, ninecell keys,
// the propagation passes and the final sigmoid — excluding the playouts and
// the weight updates, which are the trainer's cost and not the model's.  It
// runs slightly high against a deployed consumer (the trainer appends keys to
// growing arrays where vpatterns reuses a scratch), and --verbose inflates it
// further with the per-key bookkeeping.
const COLS = ['T', 'games', 'ex', 'nWts', 'tPos', 'loss', 'exc'];
const COLW = [5, 5, 5, 5, 5, 7, 8];
// '*' marks a row whose exc is the lowest so far.  Nothing is saved on it —
// the checkpoint is written every row regardless; the marker is only so the
// run's best interval is visible while it scrolls.
let bestExc = Infinity;
const excCell = exc => {
  const best = exc < bestExc;
  if (best) bestExc = exc;
  return exc.toFixed(4) + (best ? '*' : ' ');
};
const printRow = cells => console.log(cells.map((c, i) => String(c).padStart(COLW[i])).join('  '));
printRow(COLS);

// Label the neighbour-health buckets up front: their keys are opaque by the
// time propagate() hands them back, and a bucket's midpoint maps to exactly
// that bucket.
if (VERBOSE) {
  for (let b = 0; b < FHM_BUCKETS; b++)
    examples.set(HL.chainFriendHealthKey((b + 0.5) / FHM_BUCKETS, FHM_BUCKETS), 'friendHealthMax=' + b);
  for (let b = 0; b < FOE_MIN_BUCKETS; b++)
    examples.set(HL.chainFoeMinHealthKey((b + 0.5) / FOE_MIN_BUCKETS, FOE_MIN_BUCKETS), 'foeHealthMin=' + b);
}

const t0 = Date.now();
// Print schedule: geometric in EXAMPLES — first row at 10k, then 1.4x per row.
let nextPrintEx = 10000;
let nEx = 0, nSurv = 0, nSkip = 0;
let ivSum = 0, ivN = 0;   // interval accumulators (reset at every print row)
let infMs = 0, infPos = 0;   // inference time and positions, cumulative
const binN = new Float64Array(PHASE_BINS), binS = new Float64Array(PHASE_BINS);
// calibration (prequential): predicted decile -> [n, survived]
const calN = new Float64Array(10), calS = new Float64Array(10);
for (let gi = 0; gi < GAMES; gi++) {
  const line = corpusLine(gi);
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
  // descend PREFIX_LEN moves with the standard playout; the endpoint is the
  // graded position (as deployed)
  let n = 0;
  while (!game.gameOver && n < PREFIX_LEN) {
    game.play(PPat.ppatMove(game, ppatState, ppatModel, rng));
    n++;
  }
  if (n < PREFIX_LEN) { nSkip++; continue; }   // game ended inside the descent
  const buf = makeBuf();
  const tInf0 = performance.now();
  collectObs(game, 1 - game.emptyCount / area, buf);
  infMs += performance.now() - tInf0;
  // continue the playout to the end: chain survival is the label
  let n2 = 0;
  const lim2 = 3 * game.emptyCount + 20;
  while (!game.gameOver && n2 < lim2) { game.play(PPat.ppatMove(game, ppatState, ppatModel, rng)); n2++; }
  // score-then-update: all of this position's chains share the game's future,
  // so every example is scored with pre-position weights before any update
  const cells = game.cells;
  const { exShapes, exStart, exLen, exOwner, exStone, exBin, exNb, exLive } = buf;
  const nCh = exStart.length;
  const tInf1 = performance.now();
  const preP = new Float64Array(nCh);
  for (let i = 0; i < nCh; i++) {
    const lo = exStart[i], hi = lo + exLen[i];
    let z = bias;
    for (let j = lo; j < hi; j++) z += weights.get(exShapes[j]) || 0;
    const b = i * NB_SLOTS;
    for (let k = 0; k < NB_SLOTS; k++) { const key = exNb[b + k]; if (key !== 0) z += weights.get(key) || 0; }
    preP[i] = exLive[i] ? 1 : 1 / (1 + Math.exp(-z));
  }
  infMs += performance.now() - tInf1;
  infPos++;
  for (let i = 0; i < nCh; i++) {
    const y = cells[exStone[i]] === exOwner[i] ? 1 : 0;
    const p = preP[i];
    const lo = exStart[i], hi = lo + exLen[i];
    const b = i * NB_SLOTS;
    let nActive = 0;
    for (let k = 0; k < NB_SLOTS; k++) if (exNb[b + k] !== 0) nActive++;
    // normalize by active-feature count (+1 for the bias)
    const g = LR * (y - p) / (hi - lo + 1 + nActive);
    bias += g;
    for (let j = lo; j < hi; j++) weights.set(exShapes[j], (weights.get(exShapes[j]) || 0) + g);
    for (let k = 0; k < NB_SLOTS; k++) {
      const key = exNb[b + k];
      if (key !== 0) weights.set(key, (weights.get(key) || 0) + g);
    }
    if (VERBOSE) {
      for (let j = lo; j < hi; j++) counts.set(exShapes[j], (counts.get(exShapes[j]) || 0) + 1);
      for (let k = 0; k < NB_SLOTS; k++) {
        const key = exNb[b + k];
        if (key !== 0) counts.set(key, (counts.get(key) || 0) + 1);
      }
    }
    ivSum += -(y ? Math.log(p + 1e-12) : Math.log(1 - p + 1e-12));
    ivN++;
    binN[exBin[i]]++; binS[exBin[i]] += y;
    let dec = Math.floor(p * 10); if (dec > 9) dec = 9;
    calN[dec]++; calS[dec] += y;
    nEx++; nSurv += y;
  }
  if (nEx >= nextPrintEx) {
    printRow([Util.fmtMs(Date.now() - t0), Util.fmt4i(gi + 1), Util.fmt4i(nEx),
              Util.fmt4i(weights.size), Util.fmtMs(infMs / infPos),
              (ivSum / ivN).toFixed(4), excCell(ivSum / ivN - FLOOR)]);
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
            Util.fmt4i(weights.size), Util.fmtMs(infMs / infPos),
            (ivSum / ivN).toFixed(4), excCell(ivSum / ivN - FLOOR)]);
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
  `// P(chain survives) = sigmoid(bias + sum of weights[ninecellId(lib)] over the`,
  `// chain's liberties` + (STONE_NINECELLS
     ? ` + sum of weights[ninecellId(stone) ^ 0x${(STONE_SALT >>> 0).toString(16)}] over its stones`
     : ` (stone ninecells OFF)`) + `,`,
  `// on the 3-state alphabet: empty, the chain's color, the other color)`,
  `const chainSurvModel = { corpus: ${JSON.stringify(CORPUS)}, size: ${SIZE},`,
  `  // The endpoint distribution this model was fitted on: leaves sampled in`,
  `  // [minPhase, maxPhase], descended delta in fullness.  Outside it the model`,
  `  // extrapolates and its survival estimates degrade sharply, so a consumer`,
  `  // that sees every phase (ab-search, rank features) needs a full-range fit.`,
  `  minPhase: ${MIN_PH}, maxPhase: ${MAX_PH}, delta: ${DELTA}, floor: ${+FLOOR.toFixed(5)},`,
  `  examples: ${nEx}, stoneSalt: ${STONE_SALT},`,
  `  maxLibs: ${MAX_LIBS}, maxJoinLibs: ${MAX_JOIN_LIBS},`,
  `  // Neighbour-health features: friendHealthMax over the healthiest joinable`,
  `  // friend, foeHealthMin over the weakest enemy chain in contact, both`,
  `  // bucketed uniformly in p.  Their keys depend on the OTHER chains' health, so`,
  `  // scoring needs the same iterative propagation (all chains from initHealth,`,
  `  // 'iterations' passes) the trainer ran — health-lib.chainHealthAll refuses`,
  `  // such a model rather than score it with the feature missing.`,
  `  stoneNinecells: ${STONE_NINECELLS}, libertyNinecells: ${LIBERTY_NINECELLS}, maxStones: ${MAX_STONES},`,
  `  libertyPairs: ${LIBERTY_PAIRS},`,
  // Ninecell encoding generation.  1 was the symmetric-combine hash used
  // 2026-09-09..11 (20.6% D4 fidelity at 6 states); 2 is the exact index +
  // canonicalisation table over the 3-state color alphabet.  The key spaces are
  // unrelated, so a generation-1 model scored by this code reads other weights.
  `  ninecellScheme: 2,`,
  `  friendHealthMaxBuckets: ${FHM_BUCKETS}, foeHealthMinBuckets: ${FOE_MIN_BUCKETS},`,
  `  iterations: ${ITERATIONS}, initHealth: ${INIT_HEALTH},`,
  `  bias: ${+bias.toFixed(5)},`,
  `  weights: new Map([${entries.join(',')}]) };  // key -> logit weight`,
  "if (typeof module !== 'undefined') module.exports = chainSurvModel;",
  'else window.chainSurvModel = chainSurvModel;',
].join('\n') + '\n';
fs.writeFileSync(SAVE, src);
}
saveModel();
console.log(`saved: ${SAVE} (${weights.size} weights)`);
