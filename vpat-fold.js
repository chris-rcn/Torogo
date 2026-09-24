'use strict';

// vpat-fold: fold one term of a two-term composite vpat model into the other,
// producing a flat single-term model.
//
// The fold works when the SOURCE term's per-cell window is derivable from the
// DEST term's per-cell window content — i.e. the source is a sub-window and/or a
// coarser encoding of the dest, sharing the anchor.  Two cases:
//   2:M -> 3:M   the 2×2 is the top-left corner sub-window of the co-anchored 3×3
//   3:k -> 3:K   (k<K) the same 3×3 window, ml-k a per-cell liberty-clamp of ml-K
// The source's per-cell contribution folds into the dest key:
//
//     w_dst'[kd] = w_dst[kd] + mean_i( pol_s·pol_d·w_src[ks] : kd(i) = kd )
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
//   node vpat-fold.js --model out/vpat-<composite>.js --games out/games.txt \
//        [--source S:M --dest S:M] [--save PATH] [--eval-games N] [--ply-stride K] [--size N]
//   --source/--dest default to the 2:M / 3:M terms of a 2:M,3:M composite.

const fs = require('fs');
const path = require('path');
const VPatterns = require('./vpatterns.js');
const { Game2, parseMove } = require('./game2.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'],
  ['model', 'games', 'save', 'source', 'dest', 'eval-games', 'ply-stride', 'size']);

if (opts.help || !opts.model || !opts.games) {
  console.log(`vpat-fold: fold one term of a two-term composite vpat model into the other.

  --model PATH       composite vpat model (exactly two terms: the source and dest)
  --games PATH       game corpus (all games are folded, minus the eval slice)
  --source S:M       term to fold away (default: the 2:M term of a 2:M,3:M model)
  --dest   S:M       term to fold into  (default: the 3:M term of a 2:M,3:M model)
  --save PATH        output flat single-term model (default out/vpat-fold-<rand>.js)
  --eval-games N     held-out games for the z_flat-vs-z_composite measurement (default 300)
  --ply-stride K     sample every K-th position within a game (default 2)
  --size N           board size (default 13)

Foldable when source.size <= dest.size and source.maxLibs <= dest.maxLibs
(the source window is a sub-window and/or a coarser encoding of the dest).`);
  process.exit(opts.help ? 0 : 1);
}

const MODEL_PATH = opts.model;
const GAMES_PATH = opts.games;
const SAVE_PATH  = opts.save || `out/vpat-fold-${Math.random().toString(36).slice(2, 10)}.js`;
const EVAL_GAMES = parseInt(opts['eval-games'] || '300', 10);
const PLY_STRIDE = Math.max(1, parseInt(opts['ply-stride'] || '2', 10));
const SIZE       = parseInt(opts.size || '13', 10);

// ── Load, resolve the source/dest terms, validate foldability ────────────────
const comp = VPatterns.loadWeights(MODEL_PATH);
if (comp.preparedSpecs.hasPhasedPatterns) {
  console.error('Error: phase-binned specs are not supported by the fold (per-tag phase salts).');
  process.exit(1);
}
const specs = comp.specs;
const tok = sp => VPatterns.specToken(sp);
function fail(msg) { console.error('Error: ' + msg); process.exit(1); }

let SRC, DST;
if (opts.source || opts.dest) {
  if (!opts.source || !opts.dest) fail('pass both --source and --dest, or neither.');
  SRC = specs.find(sp => tok(sp) === opts.source);
  DST = specs.find(sp => tok(sp) === opts.dest);
  if (!SRC) fail(`--source '${opts.source}' is not a term of the model (has '${VPatterns.specString(specs)}').`);
  if (!DST) fail(`--dest '${opts.dest}' is not a term of the model (has '${VPatterns.specString(specs)}').`);
} else {
  const s2 = specs.find(s => s.size === 2), s3 = specs.find(s => s.size === 3);
  if (specs.length !== 2 || !s2 || !s3 || s2.maxLibs !== s3.maxLibs)
    fail(`no --source/--dest and the model is not a plain 2:M,3:M composite ('${VPatterns.specString(specs)}').`);
  SRC = s2; DST = s3;
}
if (specs.length !== 2) fail(`the composite must have exactly the two terms {source, dest}; got '${VPatterns.specString(specs)}'.`);
if (tok(SRC) === tok(DST)) fail('source and dest are the same term.');
if (!(SRC.size === 2 || SRC.size === 3) || !(DST.size === 2 || DST.size === 3)) fail('only size 2 and 3 terms are supported.');
if (!(SRC.maxLibs > 0 && DST.maxLibs > 0)) fail('only positive-maxLibs terms are supported (not L / H<n>).');
if (SRC.size > DST.size || SRC.maxLibs > DST.maxLibs)
  fail(`source ${tok(SRC)} is not derivable from dest ${tok(DST)} — need source.size <= dest.size and source.maxLibs <= dest.maxLibs.`);

const PS = 0;   // no phase bins asserted above
const SRC_TAG = (VPatterns.tagBaseOf(SRC.maxLibs) << 3) | VPatterns.sizeCode(SRC.size);
const DST_TAG = (VPatterns.tagBaseOf(DST.maxLibs) << 3) | VPatterns.sizeCode(DST.size);
const SHN = SRC.size === 2 ? 'h2N' : 'h3N', SHI = SRC.size === 2 ? 'h2I' : 'h3I';
const DHN = DST.size === 2 ? 'h2N' : 'h3N', DHI = DST.size === 2 ? 'h2I' : 'h3I';
function keyFor(hN, hI, tag) { return (VPatterns.mixTag(hN < hI ? hN : hI, tag) ^ PS) | 0; }

const flatSpecs = [{ size: DST.size, maxLibs: DST.maxLibs }];
const flatPrep = VPatterns.prepareSpecs(flatSpecs);

console.log(`composite: ${path.basename(MODEL_PATH)}  specs='${VPatterns.specString(specs)}'  weights=${comp.weights.size}`);
console.log(`fold: ${tok(SRC)} -> ${tok(DST)}   corpus=${path.basename(GAMES_PATH)}   ply-stride=${PLY_STRIDE}`);
console.log(`out: ${SAVE_PATH}`);

// ── Corpus ───────────────────────────────────────────────────────────────────
const lines = fs.readFileSync(GAMES_PATH, 'utf8').split('\n').filter(l => l && l[0] !== '#');
if (lines.length <= EVAL_GAMES) fail(`corpus has only ${lines.length} games, need > --eval-games (${EVAL_GAMES}).`);
const evalLines = lines.slice(0, EVAL_GAMES);
const foldLines = lines.slice(EVAL_GAMES);
console.log(`corpus games: ${lines.length}  (eval ${evalLines.length}, fold ${foldLines.length})`);

// Yield sampled positions (Game2) from a game line, one every PLY_STRIDE plies.
function* positions(line) {
  const parts = line.trim().split(/\s+/);
  const N = parseInt(parts[0], 10);
  if (N !== SIZE) return;
  const moves = parts[1] ? parts[1].split(',') : [];
  const g = new Game2(N, false);
  for (let m = 0; m < moves.length; m++) {
    const idx = parseMove(moves[m], N);
    if (!g.isLegal(idx)) return;
    g.play(idx);
    if (m % PLY_STRIDE === 0) yield g;
  }
}

// ── Self-check: per-cell source/dest keys must match the extraction's emit ────
function multiset(arr) { const m = new Map(); for (const k of arr) m.set(k, (m.get(k) || 0) + 1); return m; }
function eqMultiset(a, b) { if (a.size !== b.size) return false; for (const [k, v] of a) if (b.get(k) !== v) return false; return true; }
(function selfCheck() {
  let checked = 0;
  for (const line of lines.slice(0, 40)) {
    for (const g of positions(line)) {
      const f = VPatterns.extractFeatures(g, comp.preparedSpecs);
      const sp = comp.preparedSpecs._planes.get(SRC.maxLibs), dp = comp.preparedSpecs._planes.get(DST.maxLibs);
      const cap = g.N * g.N, mySrc = [], myDst = [];
      for (let i = 0; i < cap; i++) {
        if (sp[SHN][i] !== sp[SHI][i]) mySrc.push(keyFor(sp[SHN][i], sp[SHI][i], SRC_TAG));
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
for (const line of evalLines) {
  for (const g of positions(line)) {
    const fc = VPatterns.extractFeatures(g, comp.preparedSpecs);
    VPatterns.evaluateFeatures(fc, comp.weights);
    const ff = VPatterns.extractFeatures(g, flatPrep);   // flat dest-term keys (weights irrelevant)
    evalSet.push({ zcomp: fc.z, keys: ff.keys.slice(0, ff.count), pols: ff.pols.slice(0, ff.count) });
  }
}
console.log(`eval positions: ${evalSet.length}`);

// ── Fold state ───────────────────────────────────────────────────────────────
// acc: destKey -> { wd, sum, cnt } ; flatVal(a) = wd + sum/cnt
const acc = new Map();
let cells = 0, lost = 0;

function foldPosition(g) {
  VPatterns.extractFeatures(g, comp.preparedSpecs);
  const sp = comp.preparedSpecs._planes.get(SRC.maxLibs), dp = comp.preparedSpecs._planes.get(DST.maxLibs);
  const sHN = sp[SHN], sHI = sp[SHI], dHN = dp[DHN], dHI = dp[DHI];
  const cap = g.N * g.N;
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
  const fw = VPatterns.makeWeights(Math.max(1024, acc.size * 2));
  for (const [dk, a] of acc) { const w = a.wd + a.sum / a.cnt; if (+w.toFixed(6) !== 0) fw.set(dk | 0, w); }
  VPatterns.saveWeights(SAVE_PATH, { specs: flatSpecs, preparedSpecs: flatPrep, weights: fw, komi: comp.komi });

  let se = 0, vse = 0, covered = 0, total = 0;
  for (const e of evalSet) {
    let z = 0;
    for (let j = 0; j < e.keys.length; j++) {
      const a = acc.get(e.keys[j]); total++;
      if (a) { covered++; z += e.pols[j] * (a.wd + a.sum / a.cnt); }
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

// ── Stream the corpus, geometric row schedule by games folded ────────────────
let positionsFolded = 0, nextRow = 1, printedAt = -1;
for (let gi = 0; gi < foldLines.length; gi++) {
  for (const g of positions(foldLines[gi])) { foldPosition(g); positionsFolded++; }
  const games = gi + 1;
  if (games >= nextRow) {
    row(games, positionsFolded);
    printedAt = games;
    nextRow = Math.max(Math.ceil(nextRow * 1.5), nextRow + 1);
  }
}
if (foldLines.length !== printedAt) row(foldLines.length, positionsFolded);
