'use strict';

// vpat-fold: fold a composite '2:M,3:M' vpat model into a flat '3:M' model.
//
// The 2×2 and 3×3 families each extract one window per cell; the 2×2 anchored at
// cell i is the top-left corner sub-window of the 3×3 anchored at i.  So the
// composite's per-cell contribution  pol3·w3[k3] + pol2·w2[k2]  folds into a
// single 3×3 weight:
//
//     w3'[k3] = w3[k3] + mean_i( pol2·pol3·w2[k2] : k3(i) = k3 )
//
// EXACT wherever the (lossy) 3×3 key uniquely determines its co-anchored 2×2, and
// the occurrence-weighted mean is the L2-optimal flat value where k3 collides.
// The residual — measured here as z_flat vs z_composite, not assumed — bottoms
// out at the 3×3 key's fidelity plus a small colour-symmetric-3×3 leak (cells
// whose 3×3 is colour-symmetric emit no k3, so their 2×2 term is dropped).
//
// Streams the whole corpus, printing a geometric row schedule; each row rebuilds
// the flat model, writes it to --save, and measures it on a held-out eval slice,
// so the RMS-vs-games curve is visible and the output file is always current.
//
// Usage:
//   node vpat-fold.js --model out/vpat-<composite>.js --games out/games.txt \
//        [--save out/vpat-<x>.js] [--eval-games N] [--ply-stride K] [--size N]

const fs = require('fs');
const path = require('path');
const VPatterns = require('./vpatterns.js');
const { Game2, parseMove } = require('./game2.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'],
  ['model', 'games', 'save', 'eval-games', 'ply-stride', 'size']);

if (opts.help || !opts.model || !opts.games) {
  console.log(`vpat-fold: fold a composite 2:M,3:M vpat model into a flat 3:M model.

  --model PATH       composite vpat model (specs must be exactly 2:M,3:M)
  --games PATH       game corpus (all games are folded, minus the eval slice)
  --save PATH        output flat 3:M model (default out/vpat-fold-<rand>.js)
  --eval-games N     held-out games for the z_flat-vs-z_composite measurement (default 300)
  --ply-stride K     sample every K-th position within a game (default 2)
  --size N           board size (default 13)`);
  process.exit(opts.help ? 0 : 1);
}

const MODEL_PATH = opts.model;
const GAMES_PATH = opts.games;
const SAVE_PATH  = opts.save || `out/vpat-fold-${Math.random().toString(36).slice(2, 10)}.js`;
const EVAL_GAMES = parseInt(opts['eval-games'] || '300', 10);
const PLY_STRIDE = Math.max(1, parseInt(opts['ply-stride'] || '2', 10));
const SIZE       = parseInt(opts.size || '13', 10);

// ── Load and validate the composite ──────────────────────────────────────────
const comp = VPatterns.loadWeights(MODEL_PATH);
if (comp.preparedSpecs.hasPhasedPatterns) {
  console.error('Error: phase-binned specs are not supported by the fold (per-tag phase salts).');
  process.exit(1);
}
const specs = comp.specs;
const s2 = specs.find(s => s.size === 2), s3 = specs.find(s => s.size === 3);
if (specs.length !== 2 || !s2 || !s3 || s2.maxLibs !== s3.maxLibs) {
  console.error(`Error: --model specs must be exactly 2:M,3:M with a shared M; got '${VPatterns.specString(specs)}'.`);
  process.exit(1);
}
const M = s2.maxLibs;
const tagBase = VPatterns.tagBaseOf(M);
const TAG2 = (tagBase << 3) | VPatterns.sizeCode(2);
const TAG3 = (tagBase << 3) | VPatterns.sizeCode(3);
const PS = 0;   // no phase bins asserted above

const flatSpecs = [{ size: 3, maxLibs: M }];
const flatPrep = VPatterns.prepareSpecs(flatSpecs);

console.log(`composite: ${path.basename(MODEL_PATH)}  specs='${VPatterns.specString(specs)}'  weights=${comp.weights.size}`);
console.log(`fold: 2:${M} -> 3:${M}   corpus=${path.basename(GAMES_PATH)}   ply-stride=${PLY_STRIDE}`);
console.log(`out: ${SAVE_PATH}`);

// Per-cell output keys/polarity — mirrors extractFeatures' size-2/3 emit exactly
// (verified in selfCheck below).
function key2(n2, i2) { return (VPatterns.mixTag(n2 < i2 ? n2 : i2, TAG2) ^ PS) | 0; }
function key3(n3, i3) { return (VPatterns.mixTag(n3 < i3 ? n3 : i3, TAG3) ^ PS) | 0; }

// ── Corpus ───────────────────────────────────────────────────────────────────
const lines = fs.readFileSync(GAMES_PATH, 'utf8').split('\n').filter(l => l && l[0] !== '#');
if (lines.length <= EVAL_GAMES) { console.error(`Error: corpus has only ${lines.length} games, need > --eval-games (${EVAL_GAMES}).`); process.exit(1); }
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

// ── Self-check: per-cell keys must match the extraction's emitted keys ────────
function multiset(arr) { const m = new Map(); for (const k of arr) m.set(k, (m.get(k) || 0) + 1); return m; }
function eqMultiset(a, b) { if (a.size !== b.size) return false; for (const [k, v] of a) if (b.get(k) !== v) return false; return true; }
(function selfCheck() {
  let checked = 0;
  for (const line of lines.slice(0, 40)) {
    for (const g of positions(line)) {
      const f = VPatterns.extractFeatures(g, comp.preparedSpecs);
      const pl = comp.preparedSpecs._planes.get(M);
      const cap = g.N * g.N, my2 = [], my3 = [];
      for (let i = 0; i < cap; i++) {
        if (pl.h2N[i] !== pl.h2I[i]) my2.push(key2(pl.h2N[i], pl.h2I[i]));
        if (pl.h3N[i] !== pl.h3I[i]) my3.push(key3(pl.h3N[i], pl.h3I[i]));
      }
      const ex2 = [], ex3 = [];
      for (let j = 0; j < f.count; j++) { if (f.tags[j] === TAG2) ex2.push(f.keys[j] | 0); else if (f.tags[j] === TAG3) ex3.push(f.keys[j] | 0); }
      if (!eqMultiset(multiset(my2), multiset(ex2)) || !eqMultiset(multiset(my3), multiset(ex3))) {
        console.error('SELF-CHECK FAILED: per-cell keys diverge from extraction.'); process.exit(1);
      }
      if (++checked >= 60) { console.log(`self-check: per-cell keys match extraction on ${checked} positions`); return; }
    }
  }
})();

// ── Pre-extract the held-out eval set: composite z (fixed) + flat 3×3 keys ────
const evalSet = [];
for (const line of evalLines) {
  for (const g of positions(line)) {
    const fc = VPatterns.extractFeatures(g, comp.preparedSpecs);
    VPatterns.evaluateFeatures(fc, comp.weights);
    const ff = VPatterns.extractFeatures(g, flatPrep);   // flat 3:M keys (weights irrelevant)
    evalSet.push({ zcomp: fc.z, keys: ff.keys.slice(0, ff.count), pols: ff.pols.slice(0, ff.count) });
  }
}
console.log(`eval positions: ${evalSet.length}`);

// ── Fold state ───────────────────────────────────────────────────────────────
// acc: key3 -> { w3, sum, cnt } ; flatVal(a) = w3 + sum/cnt
const acc = new Map();
let cells = 0, lost = 0;

function foldPosition(g) {
  VPatterns.extractFeatures(g, comp.preparedSpecs);
  const pl = comp.preparedSpecs._planes.get(M);
  const cap = g.N * g.N;
  for (let i = 0; i < cap; i++) {
    const n3 = pl.h3N[i], i3 = pl.h3I[i], n2 = pl.h2N[i], i2 = pl.h2I[i];
    const has2 = n2 !== i2;
    if (n3 === i3) { if (has2) lost++; continue; }   // colour-symmetric 3×3: no k3
    const k3 = key3(n3, i3);
    let inc = 0;
    if (has2) { const pol2 = n2 < i2 ? 1 : -1, pol3 = n3 < i3 ? 1 : -1; inc = pol2 * pol3 * (comp.weights.get(key2(n2, i2)) ?? 0); }
    let a = acc.get(k3);
    if (!a) { a = { w3: comp.weights.get(k3) ?? 0, sum: 0, cnt: 0 }; acc.set(k3, a); }
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
  // Build + save the flat 3:M model.
  const fw = VPatterns.makeWeights(Math.max(1024, acc.size * 2));
  for (const [k3, a] of acc) { const w = a.w3 + a.sum / a.cnt; if (+w.toFixed(6) !== 0) fw.set(k3 | 0, w); }
  VPatterns.saveWeights(SAVE_PATH, { specs: flatSpecs, preparedSpecs: flatPrep, weights: fw, komi: comp.komi });

  // Measure on the cached eval set (look the flat value straight out of acc).
  let se = 0, vse = 0, covered = 0, total = 0;
  for (const e of evalSet) {
    let z = 0;
    for (let j = 0; j < e.keys.length; j++) {
      const a = acc.get(e.keys[j]); total++;
      if (a) { covered++; z += e.pols[j] * (a.w3 + a.sum / a.cnt); }
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
