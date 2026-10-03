'use strict';

// fpol-fold.js — fold a featurepol model into a smaller spec with the same scores.
//
// A source space whose terms are all inside one target space is FOLDED: its
// weight is added to the target key a move gets, since that key determines the
// source space's key.  E.g. stones8,adjLib4,stones8+adjLib4,stones12b folds to
// stones8+adjLib4,stones12b: w'(stones8+adjLib4 key) = w(cross) + w(stones8) +
// w(adjLib4).  Spaces in both specs carry over unchanged.
//
// No feature is computed here: the key mapping is learned by replaying
// self-play positions through featurepol's own extraction (moves sampled from
// the source model's temperature-1 softmax, 10% uniformly random), recording
// every candidate move's per-space keys.  Pairs seen there fold exactly;
// target keys never seen keep only their own weight (reported).  Stops on a
// mapping conflict (a target key seen with two different folded keys), on a
// target space emitting more than one key per move, or on a rank space that
// would need folding.  The saved model is then checked against the source on
// held-out games.
//
// Usage: node fpol-fold.js --in MODEL --spec TARGET [--out FILE] [--games 2000]
//                          [--size 13] [--seed N] [--check-games 200]

const path = require('path');
const Util = require('./util.js');
const FP = require('./featurepol-lib.js');
const { Game2, PASS } = require('./game2.js');
const { makeRng } = require('./xorshift.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['in', 'spec', 'out', 'games', 'size', 'seed', 'check-games']);
if (opts.help || !opts.in || !opts.spec) {
  console.log(`Usage: node fpol-fold.js --in MODEL --spec TARGET [options]

Folds a featurepol model into a smaller spec, learning the key mapping from
self-play through featurepol's own extraction.

  --in MODEL          source featurepol model                          (required)
  --spec TARGET       target spec, e.g. stones8+adjLib4,stones12b      (required)
  --out FILE          output model                       (default <in>-fold.js)
  --games N           self-play games for the key mapping            (default 2000)
  --size N            board size                                        (default 13)
  --seed N            rng seed                              (default: random, logged)
  --check-games N     held-out games comparing source and folded     (default 200)
  --help              show this message`);
  process.exit(opts.help ? 0 : 1);
}
const OUT    = opts.out || opts.in.replace(/\.js$/, '') + '-fold.js';
const GAMES  = parseInt(opts.games || '2000', 10);
const SIZE   = parseInt(opts.size || '13', 10);
const CHECK  = parseInt(opts['check-games'] || '200', 10);
const SEED   = opts.seed !== undefined ? parseInt(opts.seed, 10) : Util.randomSeed();
const EPSILON = 0.1;
const die = msg => { console.error(`fpol-fold: ${msg}`); process.exit(1); };
if (!(GAMES >= 1) || !(SIZE >= 2) || !(CHECK >= 0)) die('--games and --size must be positive, --check-games >= 0');
const rng = makeRng(SEED);

// ── Plan: which source spaces carry over, which fold into which target ──────
const src = FP.loadModel({ path: opts.in });
const sw = src.weights;
if (sw.spec.rankSpaces && sw.spec.rankSpaces.length) sw.rankTopN = 0;   // as puct-trunc (rank spaces emit nothing)
const tgtSpec = FP.parseSpec(opts.spec);
const termsOf = str => new Set(str.split('+').map(t => t.trim()));
const srcIdx = new Map(sw.spec.spaces.map((sp, i) => [sp.str, i]));
for (const tp of tgtSpec.spaces) if (!srcIdx.has(tp.str)) die(`target space '${tp.str}' is not a space of the source spec '${sw.spec.str}'`);
const tgtStrs = new Set(tgtSpec.spaces.map(sp => sp.str));
const folds = [];   // { s: source space index, t: source index of its target space }
for (let i = 0; i < sw.spec.spaces.length; i++) {
  const sp = sw.spec.spaces[i];
  if (tgtStrs.has(sp.str)) continue;
  if (sp.usesRank || sp.listFn) die(`source space '${sp.str}' is a rank or list space and cannot be folded`);
  const st = termsOf(sp.str);
  const into = tgtSpec.spaces.filter(tp => [...st].every(t => termsOf(tp.str).has(t)));
  if (into.length !== 1) die(`source space '${sp.str}' needs exactly one target space containing its terms; found ${into.length}`);
  folds.push({ s: i, t: srcIdx.get(into[0].str) });
}
const targetsUsed = [...new Set(folds.map(f => f.t))];
console.log(`source    ${opts.in}  spec '${sw.spec.str}' (${sw.size} weights)`);
console.log(`target    spec '${tgtSpec.str}'`);
for (const t of targetsUsed) console.log(`fold      ${folds.filter(f => f.t === t).map(f => `'${sw.spec.spaces[f.s].str}'`).join(' + ')} -> '${sw.spec.spaces[t].str}'`);
const carried = tgtSpec.spaces.filter(tp => !targetsUsed.includes(srcIdx.get(tp.str))).map(tp => `'${tp.str}'`);
if (carried.length) console.log(`carry     ${carried.join(', ')} unchanged`);
console.log(`corpus    ${GAMES} self-play games at size ${SIZE}, temperature 1, epsilon ${EPSILON}, seed ${SEED}`);

// ── Corpus: learn target key -> folded keys, by replaying through extraction ─
const nS = sw.spec.spaces.length;
const state = FP.createState(SIZE, sw.spec, { components: true });
// mapping[targetIdx]: Map(target dense key -> Map(source space index -> signature string of its dense keys))
const mapping = new Map(targetsUsed.map(t => [t, new Map()]));
let movesSeen = 0, positions = 0, gatedLoss = 0;
const keyRange = (i, s) => {   // [start, end) of move i's keys in source space s
  const start = state.spaceOff[i * nS + s];
  const end = s + 1 < nS ? state.spaceOff[i * nS + s + 1] : state.keyOff[i + 1];
  return [start, end];
};
function pickMove(game, st, weights) {
  if (rng.random() < EPSILON || st.count === 0) return game.randomLegalMove(rng);
  FP.computeSoftmax(st, weights);
  let r = rng.random(), i = 0;
  for (; i < st.count - 1; i++) { r -= st.probs[i]; if (r <= 0) break; }
  return st.moves[i];
}
for (let g = 0; g < GAMES; g++) {
  const game = new Game2(SIZE, true);
  for (let m = 0; m < 3 * SIZE * SIZE && !game.gameOver; m++) {
    FP.extractFeatures(game, state, sw);
    positions++;
    for (let i = 0; i < state.count; i++) {
      movesSeen++;
      for (const t of targetsUsed) {
        const [ts, te] = keyRange(i, t);
        if (te - ts > 1) die(`target space '${sw.spec.spaces[t].str}' emitted ${te - ts} keys for one move (folding needs one)`);
        const tKeys = mapping.get(t);
        for (const f of folds) if (f.t === t) {
          const [ss, se] = keyRange(i, f.s);
          if (te === ts) { if (se > ss) gatedLoss++; continue; }   // target gated off, folded space fired
          const sig = Array.from(state.keys.subarray(ss, se)).sort((a, b) => a - b).join(',');
          const tk = state.keys[ts];
          let bySpace = tKeys.get(tk);
          if (!bySpace) { bySpace = new Map(); tKeys.set(tk, bySpace); }
          const prev = bySpace.get(f.s);
          if (prev === undefined) bySpace.set(f.s, sig);
          else if (prev !== sig) die(`mapping conflict: a '${sw.spec.spaces[t].str}' key appeared with two different '${sw.spec.spaces[f.s].str}' key sets`);
        }
      }
    }
    game.play(pickMove(game, state, sw));
  }
}

// ── Build the folded weights ──────────────────────────────────────────────────
const rawOf = new Int32Array(sw.size);   // dense index -> raw key
sw.map.forEach((key, idx) => { rawOf[idx] = key; });
const foldedAway = new Set();             // dense keys of folded source spaces seen in the corpus
for (const [, tKeys] of mapping) for (const [, bySpace] of tKeys) for (const [, sig] of bySpace)
  if (sig) for (const k of sig.split(',')) foldedAway.add(+k);
const tw = FP.createWeights({ spec: tgtSpec, initialCapacity: sw.size,
  ladderMinChain: sw.ladderMinChain, t3MinChain: sw.t3MinChain, t3DepthLimit: sw.t3DepthLimit, t3NodeLimit: sw.t3NodeLimit });
if (sw.vpatModel) tw.vpatModel = sw.vpatModel;
for (let d = 0; d < sw.size; d++) {
  if (foldedAway.has(d)) continue;
  tw.vals[FP.internKey(tw, rawOf[d])] = sw.vals[d];
}
let targetKeysFolded = 0;
for (const [, tKeys] of mapping) for (const [tk, bySpace] of tKeys) {
  let add = 0;
  for (const [, sig] of bySpace) if (sig) for (const k of sig.split(',')) add += sw.vals[+k];
  const ix = FP.internKey(tw, rawOf[tk]);
  tw.vals[ix] += add;
  targetKeysFolded++;
}
const save = () => {
  const lines = FP.serialize(tw, { spec: tgtSpec.str, ema: src.ema, totalUpdates: src.totalUpdates, komi: src.komi }).split('\n');
  lines.splice(1, 0, `// Generated by: ${[path.basename(process.argv[1]), ...process.argv.slice(2)].join(' ')}`,
                     `// Folded from: ${opts.in} (spec '${sw.spec.str}')`);
  Util.writeFileAtomic(OUT, lines.join('\n'));
};
save();
console.log(`mapping   ${positions} positions, ${movesSeen} candidate moves; ${targetKeysFolded} target keys took folded weight` +
            (gatedLoss ? `; ${gatedLoss} moves had a folded space fire while its target was gated off` : ''));
console.log(`saved     ${OUT} (${tw.size} weights; ${foldedAway.size} folded-away source keys dropped)`);

// ── Check: saved folded model against the source on held-out games ──────────
if (CHECK > 0) {
  const fw = FP.loadModel({ path: OUT }).weights;
  if (fw.spec.rankSpaces && fw.spec.rankSpaces.length) fw.rankTopN = 0;
  const sSt = FP.createState(SIZE, sw.spec), fSt = FP.createState(SIZE, fw.spec, { components: true });
  // Which folded-model keys took folded weight (raw keys), to split the check.
  const foldedRaw = new Set();
  for (const [, tKeys] of mapping) for (const [tk] of tKeys) foldedRaw.add(rawOf[tk]);
  const fRaw = new Int32Array(fw.size); fw.map.forEach((key, idx) => { fRaw[idx] = key; });
  const fTargets = targetsUsed.map(t => fw.spec.spaces.findIndex(sp => sp.str === sw.spec.spaces[t].str));
  const fnS = fw.spec.spaces.length;
  const allFolded = i => fTargets.every(t => {
    const k = fSt.keys[fSt.spaceOff[i * fnS + t]];
    return k < fRaw.length && foldedRaw.has(fRaw[k]);
  });
  let nF = 0, withinF = 0, maxF = 0;
  const sSc = new Float64Array(SIZE * SIZE + 1), fSc = new Float64Array(SIZE * SIZE + 1);
  let n = 0, within = 0, sumAbs = 0, maxAbs = 0, top = 0, posN = 0;
  for (let g = 0; g < CHECK; g++) {
    const game = new Game2(SIZE, true);
    for (let m = 0; m < 3 * SIZE * SIZE && !game.gameOver; m++) {
      FP.extractFeatures(game, sSt, sw); FP.scoreAll(sSt, sw, sSc);
      FP.extractFeatures(game, fSt, fw); FP.scoreAll(fSt, fw, fSc);
      if (sSt.count !== fSt.count) die('source and folded models disagree on the candidate moves');
      let bs = 0, bf = 0;
      for (let i = 0; i < sSt.count; i++) {
        if (sSt.moves[i] !== fSt.moves[i]) die('candidate order differs between source and folded extraction');
        const d = Math.abs(fSc[i] - sSc[i]);
        n++; sumAbs += d; if (d > maxAbs) maxAbs = d; if (d <= 0.01) within++;
        if (allFolded(i)) { nF++; if (d <= 0.01) withinF++; if (d > maxF) maxF = d; }
        if (sSc[i] > sSc[bs]) bs = i; if (fSc[i] > fSc[bf]) bf = i;
      }
      if (sSt.count) { posN++; if (bs === bf) top++; }
      game.play(pickMove(game, sSt, sw));
    }
  }
  console.log(`check     ${CHECK} held-out games, ${posN} positions, ${n} candidate moves: score within 0.01 ${(100 * within / n).toFixed(2)}%, ` +
              `mean |diff| ${(sumAbs / n).toFixed(4)}, max |diff| ${maxAbs.toFixed(3)}; same top move ${(100 * top / posN).toFixed(2)}%`);
  console.log(`          moves whose target keys were all folded: ${(100 * nF / n).toFixed(2)}% of moves, within 0.01 ${(100 * withinF / nF).toFixed(2)}%, max |diff| ${maxF.toFixed(4)}`);
}
