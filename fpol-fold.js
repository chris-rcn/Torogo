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
// target keys never seen keep only their own weight.  Stops on a mapping
// conflict (a target key seen with two different folded keys), on a target
// space emitting more than one key per move, or on a rank space that would
// need folding.
//
// Runs until --limit games (default: unlimited).  Each status row (after 1, 2,
// 3, 5, 7, 10, ... games, x1.4) rebuilds the folded weights, saves them, and
// checks the SAVED model against the source on a fixed held-out set.
//
// Usage: node fpol-fold.js --in MODEL --spec TARGET [--out FILE] [--limit N]
//                          [--size 13] [--seed N] [--check-games 100]

const path = require('path');
const Util = require('./util.js');
const FP = require('./featurepol-lib.js');
const { Game2, PASS } = require('./game2.js');
const { makeRng } = require('./xorshift.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['in', 'spec', 'out', 'limit', 'size', 'seed', 'check-games']);
if (opts.help || !opts.in || !opts.spec) {
  console.log(`Usage: node fpol-fold.js --in MODEL --spec TARGET [options]

Folds a featurepol model into a smaller spec, learning the key mapping from
self-play through featurepol's own extraction.

  --in MODEL          source featurepol model                          (required)
  --spec TARGET       target spec, e.g. stones8+adjLib4,stones12b      (required)
  --out FILE          output model                       (default <in>-fold.js)
  --limit N           stop after N corpus games                (default: unlimited)
  --size N            board size                                        (default 13)
  --seed N            rng seed                              (default: random, logged)
  --check-games N     held-out games checked each row                (default 100)
  --help              show this message`);
  process.exit(opts.help ? 0 : 1);
}
const OUT    = opts.out || opts.in.replace(/\.js$/, '') + '-fold.js';
const LIMIT  = opts.limit !== undefined ? parseInt(opts.limit, 10) : Infinity;
const SIZE   = parseInt(opts.size || '13', 10);
const CHECK  = parseInt(opts['check-games'] || '100', 10);
const SEED   = opts.seed !== undefined ? parseInt(opts.seed, 10) : Util.randomSeed();
const EPSILON = 0.1;
const die = msg => { console.error(`fpol-fold: ${msg}`); process.exit(1); };
if (!(LIMIT >= 1) || !(SIZE >= 2) || !(CHECK >= 1)) die('--limit, --size and --check-games must be positive');
const rng = makeRng(SEED), checkRng = makeRng((SEED ^ 0x9e3779b9) >>> 0 || 1);

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
console.log(`corpus    self-play at size ${SIZE}, temperature 1, epsilon ${EPSILON}, seed ${SEED}${LIMIT < Infinity ? `, limit ${LIMIT} games` : ''}`);
console.log(`out       ${OUT}`);

// ── Corpus state: target key -> folded keys, learned through extraction ────
const nS = sw.spec.spaces.length;
const state = FP.createState(SIZE, sw.spec, { components: true });
// mapping[target space]: Map(target dense key -> Map(source space index -> signature of its dense keys))
const mapping = new Map(targetsUsed.map(t => [t, new Map()]));
let movesSeen = 0, positions = 0, gatedLoss = 0;
const keyRange = (st, i, s, n) => {   // [start, end) of move i's keys in space s
  const start = st.spaceOff[i * n + s];
  return [start, s + 1 < n ? st.spaceOff[i * n + s + 1] : st.keyOff[i + 1]];
};
function pickMove(game, st, weights, r) {
  if (r.random() < EPSILON || st.count === 0) return game.randomLegalMove(r);
  FP.computeSoftmax(st, weights);
  let x = r.random(), i = 0;
  for (; i < st.count - 1; i++) { x -= st.probs[i]; if (x <= 0) break; }
  return st.moves[i];
}
function recordPosition() {
  positions++;
  for (let i = 0; i < state.count; i++) {
    movesSeen++;
    for (const t of targetsUsed) {
      const [ts, te] = keyRange(state, i, t, nS);
      if (te - ts > 1) die(`target space '${sw.spec.spaces[t].str}' emitted ${te - ts} keys for one move (folding needs one)`);
      const tKeys = mapping.get(t);
      for (const f of folds) if (f.t === t) {
        const [ss, se] = keyRange(state, i, f.s, nS);
        if (te === ts) { if (se > ss) gatedLoss++; continue; }   // target gated off while a folded space fired
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
}
function playCorpusGame() {
  const game = new Game2(SIZE, true);
  for (let m = 0; m < 3 * SIZE * SIZE && !game.gameOver; m++) {
    FP.extractFeatures(game, state, sw);
    recordPosition();
    game.play(pickMove(game, state, sw, rng));
  }
}

// ── Held-out set: fixed games, the source's scores cached ───────────────────
// Its own rng stream, played before the corpus, so it never overlaps it.
const held = [];   // { moves: Int16Array (game moves), pos: [{ cands: Int16Array, scores: Float32Array }] }
{
  const st = FP.createState(SIZE, sw.spec), sc = new Float64Array(SIZE * SIZE + 1);
  for (let g = 0; g < CHECK; g++) {
    const game = new Game2(SIZE, true), mv = [], pos = [];
    for (let m = 0; m < 3 * SIZE * SIZE && !game.gameOver; m++) {
      FP.extractFeatures(game, st, sw); FP.scoreAll(st, sw, sc);
      pos.push({ cands: Int16Array.from(st.moves.subarray(0, st.count)), scores: Float32Array.from(sc.subarray(0, st.count)) });
      const m1 = pickMove(game, st, sw, checkRng);
      game.play(m1); mv.push(m1);
    }
    held.push({ moves: Int16Array.from(mv), pos });
  }
}
const heldPositions = held.reduce((s, h) => s + h.pos.length, 0);
const heldMoves = held.reduce((s, h) => s + h.pos.reduce((a, p) => a + p.cands.length, 0), 0);
console.log(`check     ${CHECK} held-out games: ${heldPositions} positions, ${heldMoves} candidate moves`);

// ── Folded weights from the current mapping ──────────────────────────────────
function buildFolded() {
  const rawOf = new Int32Array(sw.size);   // source dense index -> raw key
  sw.map.forEach((key, idx) => { rawOf[idx] = key; });
  const foldedAway = new Set();             // dense keys of folded source spaces seen so far
  for (const [, tKeys] of mapping) for (const [, bySpace] of tKeys) for (const [, sig] of bySpace)
    if (sig) for (const k of sig.split(',')) foldedAway.add(+k);
  const tw = FP.createWeights({ spec: tgtSpec, initialCapacity: sw.size,
    ladderMinChain: sw.ladderMinChain, t3MinChain: sw.t3MinChain, t3DepthLimit: sw.t3DepthLimit, t3NodeLimit: sw.t3NodeLimit });
  if (sw.vpatModel) tw.vpatModel = sw.vpatModel;
  for (let d = 0; d < sw.size; d++) if (!foldedAway.has(d)) tw.vals[FP.internKey(tw, rawOf[d])] = sw.vals[d];
  const foldedRaw = new Set();
  for (const [, tKeys] of mapping) for (const [tk, bySpace] of tKeys) {
    let add = 0;
    for (const [, sig] of bySpace) if (sig) for (const k of sig.split(',')) add += sw.vals[+k];
    tw.vals[FP.internKey(tw, rawOf[tk])] += add;
    foldedRaw.add(rawOf[tk]);
  }
  return { tw, foldedRaw };
}
function save(tw) {
  const lines = FP.serialize(tw, { spec: tgtSpec.str, ema: src.ema, totalUpdates: src.totalUpdates, komi: src.komi }).split('\n');
  lines.splice(1, 0, `// Generated by: ${[path.basename(process.argv[1]), ...process.argv.slice(2)].join(' ')}`,
                     `// Folded from: ${opts.in} (spec '${sw.spec.str}')`);
  Util.writeFileAtomic(OUT, lines.join('\n'));
}

// The SAVED model (so the check includes save rounding) on the held-out set.
function check(foldedRaw) {
  const resolved = require.resolve(path.resolve(OUT));
  delete require.cache[resolved];                     // re-read the file just written
  const fw = FP.loadModel({ path: OUT }).weights;
  if (fw.spec.rankSpaces && fw.spec.rankSpaces.length) fw.rankTopN = 0;
  const fSt = FP.createState(SIZE, fw.spec, { components: true }), fSc = new Float64Array(SIZE * SIZE + 1);
  const at = new Int32Array(SIZE * SIZE).fill(-1);   // move -> folded candidate index (candidate ORDER can differ:
                                                     // it follows the board's empty-cell list, which randomLegalMove reorders)
  const fRaw = new Int32Array(fw.size); fw.map.forEach((key, idx) => { fRaw[idx] = key; });
  const fTargets = targetsUsed.map(t => fw.spec.spaces.findIndex(sp => sp.str === sw.spec.spaces[t].str));
  const fnS = fw.spec.spaces.length;
  let n = 0, nF = 0, within = 0, sumAbs = 0, maxAbs = 0, top = 0, posN = 0;
  for (const h of held) {
    const game = new Game2(SIZE, true);
    for (let p = 0; p < h.pos.length; p++) {
      const { cands, scores } = h.pos[p];
      FP.extractFeatures(game, fSt, fw); FP.scoreAll(fSt, fw, fSc);
      if (fSt.count !== cands.length) die('source and folded models disagree on the candidate moves');
      for (let i = 0; i < fSt.count; i++) at[fSt.moves[i]] = i;
      let bs = 0, bf = 0;
      for (let j = 0; j < cands.length; j++) {
        const i = at[cands[j]];
        if (i < 0) die('a source candidate is not a folded candidate');
        const d = Math.abs(fSc[i] - scores[j]);
        n++; sumAbs += d; if (d > maxAbs) maxAbs = d; if (d <= 0.01) within++;
        if (fTargets.every(t => { const k = fSt.keys[fSt.spaceOff[i * fnS + t]]; return k < fRaw.length && foldedRaw.has(fRaw[k]); })) nF++;
        if (scores[j] > scores[bs]) bs = j;
        if (fSc[i] > fSc[at[cands[bf]]]) bf = j;
      }
      for (let i = 0; i < fSt.count; i++) at[fSt.moves[i]] = -1;
      if (cands.length) { posN++; if (bs === bf) top++; }
      game.play(h.moves[p]);
    }
  }
  return { cover: nF / n, exact: within / n, meanAbs: sumAbs / n, maxAbs, top: top / posN, weights: fw.size };
}

// ── Run ───────────────────────────────────────────────────────────────────────
console.log('');
const COLS = [['games', 5], ['positions', 9], ['moves', 6], ['tKeys', 7], ['nWts', 5], ['gated', 6], ['cover', 6],
              ['exact', 6], ['meanDif', 8], ['maxDif', 7], ['top1', 6], ['tCheck', 7], ['elapsed', 8]];
console.log(COLS.map(([h, w]) => h.padStart(w)).join('  '));
const t0 = Date.now();
let games = 0, nextRow = 1;
while (games < LIMIT) {
  playCorpusGame();
  games++;
  if (games >= nextRow || games >= LIMIT) {
    let tKeys = 0; for (const [, m] of mapping) tKeys += m.size;
    const { tw, foldedRaw } = buildFolded();
    save(tw);
    const tc0 = Date.now();
    const c = check(foldedRaw);
    const tCheck = Date.now() - tc0;
    const cells = [Util.fmt4i(games), Util.fmt4i(positions), Util.fmt4i(movesSeen), Util.fmt4i(tKeys), Util.fmt4i(c.weights),
                   Util.fmt4i(gatedLoss), Util.fmtRatio4(c.cover), Util.fmtRatio4(c.exact), c.meanAbs.toFixed(4),
                   c.maxAbs.toFixed(3), Util.fmtRatio4(c.top), Util.fmtMs(tCheck), Util.fmtMs(Date.now() - t0)];
    console.log(cells.map((v, k) => String(v).padStart(COLS[k][1])).join('  '));
    nextRow = Math.max(games + 1, Math.ceil(nextRow * 1.4));
  }
}
