'use strict';

// vpat-fold: fold one term of a two-term composite vpat model into the other,
// producing a flat single-term model.
//
// The fold works when the SOURCE term's per-cell window is derivable from the
// DEST term's per-cell window content — i.e. the source is a sub-window and/or a
// coarser encoding of the dest, sharing the anchor.  Three cases:
//   2:M -> 3:M   the 2×2 is the top-left corner sub-window of the co-anchored 3×3
//   3:k -> 3:K   (k<K) the same 3×3 window, ml-k a per-cell liberty-clamp of ml-K
//   1:m -> d:M   (m<=M, d = 2 or 3; liberty family) a single cell of the d×d
// The source's per-cell contribution folds into the dest key:
//
//     w_dst'[kd] = w_dst[kd] + mean_i( pol_s·pol_d·w_src[ks] : kd(i) = kd )
//
// except for a 1:m source, which is split evenly over the dest windows: each
// window takes (1/d²)·Σ over its d² cells of pol_s·pol_d·w_src.  On the torus a
// cell lies in exactly d² windows, so the total is preserved, and the share is
// a symmetric function of the window's content — exact even though the dest
// key (D4-invariant) cannot say which of its cells is the anchor.
//
// EXACT wherever the (lossy) dest key uniquely determines the source content, and
// the occurrence-weighted mean is the L2-optimal flat value where the dest key
// collides.  The residual — measured here as z_flat vs z_composite, not assumed —
// bottoms out at the dest key's fidelity, plus (for a size-reducing fold only) a
// small colour-symmetric-dest leak: cells whose dest window is colour-symmetric
// emit no dest key, so a non-symmetric source term there is dropped.  A
// coarsening fold (3:k->3:K) has no such leak (ml-k symmetry follows from ml-K
// symmetry), so its only residual is the collision floor.
//
// Streams the whole corpus, printing a geometric row schedule; each row rebuilds
// the flat model, writes it to --save, and measures it on a held-out eval slice.
//
// Usage:
//   node vpat-fold.js --model out/vpat-<composite>.js --position-agent rfs \
//        --source S:M --dest S:M [--save PATH] [--games N] [--eval-games N] \
//        [--ply-stride K] [--size N]

const path = require('path');
const VPatterns = require('./vpatterns.js');
const { Game2, PASS } = require('./game2.js');
const { makeRng } = require('./xorshift.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'],
  ['model', 'position-agent', 'games', 'budget', 'save', 'source', 'dest', 'eval-games', 'ply-stride', 'size']);

if (opts.help || !opts.model || !opts['position-agent'] || !opts.source || !opts.dest) {
  console.log(`vpat-fold: fold one term of a two-term composite vpat model into the other.

Fold positions come from a self-play agent (--position-agent), so no game corpus
is needed.  The z_flat-vs-z_composite measurement uses a held-out set from the
same agent.

  --model PATH       vpat model containing the source and dest terms; any other
                     terms (e.g. a phased turn conditioner) are carried through
  --position-agent X ai/<X>.js self-play generates the fold positions.  Required.
  --games N          self-play games to generate (default: unlimited)
  --budget MS        per-move budget for the position agent (default 100)
  --source S:M       term to fold away (required)
  --dest   S:M       term to fold into  (required)
  --save PATH        output model (input minus the source term; default out/vpat-fold-<rand>.js)
  --eval-games N     held-out games for the z_flat-vs-z_composite measurement (default 300)
  --ply-stride K     sample every K-th position within a game (default 5)
  --size N           board size (default 13)

Foldable when source.size <= dest.size and source.maxLibs <= dest.maxLibs
(the source window is a sub-window and/or a coarser encoding of the dest).
Sizes: source 1, 2 or 3 (size 1: liberty-coded only), dest 2 or 3.`);
  process.exit(opts.help ? 0 : 1);
}

const MODEL_PATH = opts.model;
const SAVE_PATH  = opts.save || `out/vpat-fold-${Math.random().toString(36).slice(2, 10)}.js`;
const EVAL_GAMES = parseInt(opts['eval-games'] || '300', 10);
const POS_AGENT  = opts['position-agent'];
const GEN_GAMES  = opts['games'] !== undefined ? parseInt(opts['games'], 10) : Infinity;
const GEN_BUDGET = parseInt(opts['budget'] || '100', 10);
const PLY_STRIDE = Math.max(1, parseInt(opts['ply-stride'] || '5', 10));
const SIZE       = parseInt(opts.size || '13', 10);

// ── Load, resolve the source/dest terms, validate foldability ────────────────
const comp = VPatterns.loadWeights(MODEL_PATH);
const specs = comp.specs;
const tok = sp => VPatterns.specToken(sp);
function fail(msg) { console.error('Error: ' + msg); process.exit(1); }

const SRC = specs.find(sp => tok(sp) === opts.source);
const DST = specs.find(sp => tok(sp) === opts.dest);
if (!SRC) fail(`--source '${opts.source}' is not a term of the model (has '${VPatterns.specString(specs)}').`);
if (!DST) fail(`--dest '${opts.dest}' is not a term of the model (has '${VPatterns.specString(specs)}').`);
if (tok(SRC) === tok(DST)) fail('source and dest are the same term.');
if ((SRC.phaseBins || 1) > 1 || (DST.phaseBins || 1) > 1)
  fail(`source/dest are phase-binned; the fold cannot reconstruct their per-bin salts. ` +
       `(Other, non-fold terms may be phased — they are carried through unchanged.)`);
if (!(SRC.size === 1 || SRC.size === 2 || SRC.size === 3) || !(DST.size === 2 || DST.size === 3))
  fail('supported: source size 1, 2 or 3 into dest size 2 or 3.');
if (SRC.size === 1 && !(SRC.maxLibs > 0)) fail('a size-1 source must be liberty-coded (maxLibs > 0).');
// Same encoding family only: maxLibs>0 = liberty counts, 0 = ladder codes (L).
// The fold maps the source sub-window onto the co-anchored dest window within
// ONE alphabet, so cross-family folds (e.g. an L window into a liberty term)
// are meaningless.  Health (H<n>, maxLibs<0) stays unsupported: its bucketed
// alphabet isn't a clean coarsening, so co-anchoring derivability is unverified.
const fam = ml => ml > 0 ? 'lib' : ml === 0 ? 'ladder' : 'health';
if (fam(SRC.maxLibs) === 'health' || fam(DST.maxLibs) === 'health') fail('health (H<n>) terms are not supported by the fold.');
if (fam(SRC.maxLibs) !== fam(DST.maxLibs)) fail(`source ${tok(SRC)} and dest ${tok(DST)} use different encoding families (${fam(SRC.maxLibs)} vs ${fam(DST.maxLibs)}); cannot fold across them.`);
if (SRC.size > DST.size || SRC.maxLibs > DST.maxLibs)
  fail(`source ${tok(SRC)} is not derivable from dest ${tok(DST)} — need source.size <= dest.size and source.maxLibs <= dest.maxLibs.`);

const PS = 0;   // src/dest are unphased (asserted above), so their keys carry no salt
const SRC_TAG = (VPatterns.tagBaseOf(SRC.maxLibs) << 3) | VPatterns.sizeCode(SRC.size);
const DST_TAG = (VPatterns.tagBaseOf(DST.maxLibs) << 3) | VPatterns.sizeCode(DST.size);
const SRC1 = SRC.size === 1;   // per-stone source keys, no window planes (see the header)
const SHN = SRC.size === 2 ? 'h2N' : 'h3N', SHI = SRC.size === 2 ? 'h2I' : 'h3I';
const DHN = DST.size === 2 ? 'h2N' : 'h3N', DHI = DST.size === 2 ? 'h2I' : 'h3I';
function keyFor(hN, hI, tag) { return (VPatterns.mixTag(hN < hI ? hN : hI, tag) ^ PS) | 0; }

// Size-1 source: each stone's signed state, its colour times its chain's
// liberty count capped at SRC.maxLibs (the extraction's raw value; ml 1 =
// presence), and its key (libs + 131·tagBase, unmixed, as extractFeatures emits).
const SRC1_BASE = 131 * VPatterns.tagBaseOf(SRC.maxLibs);
function src1State(g, i) {
  const c = g.cells[i];
  if (c === 0) return 0;
  if (SRC.maxLibs === 1) return c;
  const libs = g._ls[g._gid[i]];
  return c * (libs < SRC.maxLibs ? libs : SRC.maxLibs);
}
function src1Key(s) { return ((s > 0 ? s : -s) + SRC1_BASE) | 0; }
// Cells of the DST.size window anchored (top-left) at i, on the torus.
const DW = DST.size;
function windowCells(i, N, out) {
  const y = (i / N) | 0, x = i % N;
  let n = 0;
  for (let dy = 0; dy < DW; dy++) {
    const r = ((y + dy) % N) * N;
    for (let dx = 0; dx < DW; dx++) out[n++] = r + (x + dx) % N;
  }
  return out;
}
const _win = new Int32Array(DW * DW);

// Output = the input minus the folded-away source term (dest absorbs it); every
// other term (e.g. a phased turn conditioner) is carried through unchanged.
const flatSpecs = specs.filter(sp => tok(sp) !== tok(SRC));
const flatPrep = VPatterns.prepareSpecs(flatSpecs);

console.log(`composite: ${path.basename(MODEL_PATH)}  specs='${VPatterns.specString(specs)}'  weights=${comp.weights.size}`);
console.log(`fold: ${tok(SRC)} -> ${tok(DST)}   agent=${POS_AGENT} (${GEN_GAMES === Infinity ? 'unlimited' : GEN_GAMES} games)   ` +
            `ply-stride=${PLY_STRIDE}`);
console.log(`out: ${SAVE_PATH}`);

// Position-agent getMove: 'random' is a uniform playout over legal non-true-eye
// moves; anything else is the ai/<name>.js agent (factory-aware).
const posGetMove = POS_AGENT === 'random'
  ? (game, _b, o) => {
      const rng = o.rng, ec = game.emptyCount, emC = game._emptyCells;
      let pick = PASS, nValid = 0;
      for (let ei = 0; ei < ec; ei++) { const idx = emC[ei]; if (game.isLegal(idx) && !game.isTrueEye(idx) && rng.random() * (++nValid) < 1) pick = idx; }
      return { move: pick };
    }
  : (() => { const m = require(path.join(__dirname, 'ai', POS_AGENT + '.js')); return (typeof m.create === 'function' ? m.create(Util.makeCfg()) : m).getMove; })();

// One self-play game from the position agent, sampled every PLY_STRIDE plies.
// Ends on two passes / no legal move, or a 2*area ply cap.
function* agentGamePositions(rng) {
  const g = new Game2(SIZE, false), plyCap = SIZE * SIZE * 2;
  let passes = 0;
  for (let m = 0; m < plyCap; m++) {
    const mv = posGetMove(g, GEN_BUDGET, { rng }).move;
    if (mv === PASS) { if (++passes >= 2) break; g.play(PASS); }
    else { passes = 0; if (!g.isLegal(mv)) break; g.play(mv); }
    if (m % PLY_STRIDE === 0) yield g;
  }
}

// Game sources (each yields one per-game position generator).  Distinct RNG
// seeds keep the agent-generated fold, eval, and self-check sets disjoint.
function* foldGames()      { const rng = makeRng(1001); for (let i = 0; i < GEN_GAMES; i++) yield agentGamePositions(rng); }
function* evalGames()      { const rng = makeRng(2002); for (let i = 0; i < EVAL_GAMES; i++) yield agentGamePositions(rng); }
function* selfCheckGames() { const rng = makeRng(3003); for (let i = 0; i < 40;         i++) yield agentGamePositions(rng); }

// ── Self-check: per-cell source/dest keys must match the extraction's emit ────
function multiset(arr) { const m = new Map(); for (const k of arr) m.set(k, (m.get(k) || 0) + 1); return m; }
function eqMultiset(a, b) { if (a.size !== b.size) return false; for (const [k, v] of a) if (b.get(k) !== v) return false; return true; }
(function selfCheck() {
  let checked = 0;
  for (const gamePositions of selfCheckGames()) {
    for (const g of gamePositions) {
      const f = VPatterns.extractFeatures(g, comp.preparedSpecs);
      const sp = SRC1 ? null : comp.preparedSpecs._planes.get(SRC.maxLibs), dp = comp.preparedSpecs._planes.get(DST.maxLibs);
      const cap = g.N * g.N, mySrc = [], myDst = [];
      for (let i = 0; i < cap; i++) {
        if (SRC1) { const st = src1State(g, i); if (st !== 0) mySrc.push(src1Key(st)); }
        else if (sp[SHN][i] !== sp[SHI][i]) mySrc.push(keyFor(sp[SHN][i], sp[SHI][i], SRC_TAG));
        if (dp[DHN][i] !== dp[DHI][i]) myDst.push(keyFor(dp[DHN][i], dp[DHI][i], DST_TAG));
      }
      const exSrc = [], exDst = [];
      for (let j = 0; j < f.count; j++) { if (f.tags[j] === SRC_TAG) exSrc.push(f.keys[j] | 0); else if (f.tags[j] === DST_TAG) exDst.push(f.keys[j] | 0); }
      if (!eqMultiset(multiset(mySrc), multiset(exSrc)) || !eqMultiset(multiset(myDst), multiset(exDst))) {
        console.error('SELF-CHECK FAILED: per-cell keys diverge from extraction.'); process.exit(1);
      }
      if (++checked >= 60) { console.log(`self-check: per-cell keys match extraction on ${checked} positions`); return; }
    }
  }
})();

// ── Pre-extract the held-out eval set: composite z (fixed) + flat dest keys ──
const evalSet = [];
for (const gamePositions of evalGames()) {
  for (const g of gamePositions) {
    const fc = VPatterns.extractFeatures(g, comp.preparedSpecs);
    VPatterns.evaluateFeatures(fc, comp.weights);
    const ff = VPatterns.extractFeatures(g, flatPrep);   // flat model's keys (carried terms + folded dest)
    evalSet.push({ zcomp: fc.z, keys: ff.keys.slice(0, ff.count), pols: ff.pols.slice(0, ff.count), tags: ff.tags.slice(0, ff.count) });
  }
}
console.log(`eval positions: ${evalSet.length}`);

// ── Fold state ───────────────────────────────────────────────────────────────
// acc: destKey -> { wd, sum, cnt } ; flatVal(a) = wd + sum/cnt
const acc = new Map();
// Type-3 weights (neither source nor dest): carried through verbatim.  Keys are
// identified by their extraction tag, since a raw key doesn't expose its tag.
const carry = new Map();
let cells = 0, lost = 0;

function foldPosition(g) {
  const f = VPatterns.extractFeatures(g, comp.preparedSpecs);
  for (let j = 0; j < f.count; j++) {
    const t = f.tags[j];
    if (t !== SRC_TAG && t !== DST_TAG) {
      const k = f.keys[j] | 0;
      if (!carry.has(k)) { const w = comp.weights.get(k); if (w !== undefined) carry.set(k, w); }
    }
  }
  const dp = comp.preparedSpecs._planes.get(DST.maxLibs);
  const dHN = dp[DHN], dHI = dp[DHI];
  const cap = g.N * g.N;
  if (SRC1) {
    // Each stone's source weight, then each dest window takes 1/d² of its cells' sum.
    const ws = new Float64Array(cap);
    for (let i = 0; i < cap; i++) {
      const st = src1State(g, i);
      if (st !== 0) ws[i] = (st > 0 ? 1 : -1) * (comp.weights.get(src1Key(st)) ?? 0);
    }
    for (let i = 0; i < cap; i++) {
      const dN = dHN[i], dI = dHI[i];
      windowCells(i, g.N, _win);
      let sum = 0, any = false;
      for (let j = 0; j < _win.length; j++) { const c = _win[j]; if (g.cells[c] !== 0) { any = true; sum += ws[c]; } }
      if (dN === dI) { if (any) lost++; continue; }   // colour-symmetric dest: no dest key
      const dk = keyFor(dN, dI, DST_TAG);
      const pd = dN < dI ? 1 : -1;
      let a = acc.get(dk);
      if (!a) { a = { wd: comp.weights.get(dk) ?? 0, sum: 0, cnt: 0 }; acc.set(dk, a); }
      a.sum += pd * sum / _win.length; a.cnt++; cells++;
    }
    return;
  }
  const sp = comp.preparedSpecs._planes.get(SRC.maxLibs);
  const sHN = sp[SHN], sHI = sp[SHI];
  for (let i = 0; i < cap; i++) {
    const dN = dHN[i], dI = dHI[i], sN = sHN[i], sI = sHI[i];
    const hasS = sN !== sI;
    if (dN === dI) { if (hasS) lost++; continue; }   // colour-symmetric dest: no dest key
    const dk = keyFor(dN, dI, DST_TAG);
    let inc = 0;
    if (hasS) { const ps = sN < sI ? 1 : -1, pd = dN < dI ? 1 : -1; inc = ps * pd * (comp.weights.get(keyFor(sN, sI, SRC_TAG)) ?? 0); }
    let a = acc.get(dk);
    if (!a) { a = { wd: comp.weights.get(dk) ?? 0, sum: 0, cnt: 0 }; acc.set(dk, a); }
    a.sum += inc; a.cnt++; cells++;
  }
}

// ── Row: rebuild + save the flat model, measure z_flat vs z_composite ─────────
const t0 = Date.now();
console.log();
console.log([
  'games'.padStart(6), 'pos'.padStart(6), 'nWts'.padStart(6), 'cover'.padStart(6),
  'lost'.padStart(6), 'zRMS'.padStart(6), 'vRMS'.padStart(7), 'tElp'.padStart(6),
].join('  '));

function row(games, positionsFolded) {
  const fw = VPatterns.makeWeights(Math.max(1024, (acc.size + carry.size) * 2));
  for (const [k, w] of carry) fw.set(k | 0, w);                 // type-3 terms, verbatim
  for (const [dk, a] of acc) { const w = a.wd + a.sum / a.cnt; if (+w.toFixed(6) !== 0) fw.set(dk | 0, w); }
  VPatterns.saveWeights(SAVE_PATH, { specs: flatSpecs, preparedSpecs: flatPrep, weights: fw, komi: comp.komi });

  let se = 0, vse = 0, covered = 0, total = 0;
  for (const e of evalSet) {
    let z = 0;
    for (let j = 0; j < e.keys.length; j++) {
      const k = e.keys[j], w = fw.get(k);
      if (w !== undefined) z += e.pols[j] * w;
      if (e.tags[j] === DST_TAG) { total++; if (acc.has(k)) covered++; }   // coverage over dest keys only
    }
    const d = z - e.zcomp; se += d * d;
    const vc = 1 / (1 + Math.exp(-e.zcomp)), vf = 1 / (1 + Math.exp(-z)); vse += (vf - vc) * (vf - vc);
  }
  const n = evalSet.length;
  console.log([
    Util.fmt4i(games).padStart(6),
    Util.fmt4i(positionsFolded).padStart(6),
    Util.fmt4i(acc.size).padStart(6),
    Util.fmtRatio4(covered / total).padStart(6),
    Util.fmtRatio4(lost / (cells + lost)).padStart(6),
    Util.fmt4(Math.sqrt(se / n)).padStart(6),
    Util.fmtRatio4(Math.sqrt(vse / n)).padStart(7),
    Util.fmtMs(Date.now() - t0).padStart(6),
  ].join('  '));
}

// ── Stream the fold source, geometric row schedule by games folded ───────────
let positionsFolded = 0, gamesFolded = 0, nextRow = 1, printedAt = -1;
for (const gamePositions of foldGames()) {
  for (const g of gamePositions) { foldPosition(g); positionsFolded++; }
  const games = ++gamesFolded;
  if (games >= nextRow) {
    row(games, positionsFolded);
    printedAt = games;
    nextRow = Math.max(Math.ceil(nextRow * 1.5), nextRow + 1);
  }
}
if (gamesFolded !== printedAt) row(gamesFolded, positionsFolded);
