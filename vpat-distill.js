'use strict';

// vpat-distill: fold a composite '2:M,3:M' vpat model into a flat '3:M' model.
//
// The 2×2 and 3×3 families each extract one window per cell; the 2×2 anchored at
// cell i is the top-left corner sub-window of the 3×3 anchored at i.  So the
// composite's per-cell contribution  pol3·w3[k3] + pol2·w2[k2]  can be folded
// into a single 3×3 weight:
//
//     w3'[k3] = w3[k3] + mean_i( pol2·pol3·w2[k2] : k3(i) = k3 )
//
// This is EXACT wherever the (lossy) 3×3 key k3 uniquely determines its
// co-anchored 2×2, and the occurrence-weighted mean is the L2-optimal flat value
// where k3 collides on differing 2×2s.  A flat 3×3 model therefore cannot beat
// the 3×3 key's own fidelity — the fold's residual is measured here directly as
// z_flat vs z_composite, not assumed.  A second, smaller leak: at cells where the
// 3×3 is colour-symmetric (no k3 emitted) but the 2×2 is not, that 2×2
// contribution has no 3×3 key to ride and is dropped (reported as lost-frac).
//
// Usage:
//   node vpat-distill.js --model out/vpat-<composite>.js --games out/games.txt \
//        [--save out/vpat-<x>.js] [--fold-games N] [--eval-games N] [--ply-stride K]

const fs = require('fs');
const path = require('path');
const VPatterns = require('./vpatterns.js');
const { Game2, parseMove } = require('./game2.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'],
  ['model', 'games', 'save', 'fold-games', 'eval-games', 'ply-stride', 'size']);

if (opts.help || !opts.model || !opts.games) {
  console.log(`vpat-distill: fold a composite 2:M,3:M vpat model into a flat 3:M model.

  --model PATH       composite vpat model (specs must be exactly 2:M,3:M)
  --games PATH       game corpus to fold/measure over
  --save PATH        output flat 3:M model (default out/vpat-distill-<rand>.js)
  --fold-games N     games used to build the fold (default 1500)
  --eval-games N     held-out games to measure z_flat vs z_composite (default 400)
  --ply-stride K     sample every K-th position within a game (default 3)
  --size N           board size (default 13)`);
  process.exit(opts.help ? 0 : 1);
}

const MODEL_PATH  = opts.model;
const GAMES_PATH  = opts.games;
const SAVE_PATH   = opts.save || `out/vpat-distill-${Math.random().toString(36).slice(2, 10)}.js`;
const FOLD_GAMES  = parseInt(opts['fold-games'] || '1500', 10);
const EVAL_GAMES  = parseInt(opts['eval-games'] || '400', 10);
const PLY_STRIDE  = Math.max(1, parseInt(opts['ply-stride'] || '3', 10));
const SIZE        = parseInt(opts.size || '13', 10);

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

console.log(`composite: ${path.basename(MODEL_PATH)}  specs='${VPatterns.specString(specs)}'  weights=${comp.weights.size}`);
console.log(`fold: 2:${M} -> 3:${M}   tag2=${TAG2} tag3=${TAG3}   corpus=${path.basename(GAMES_PATH)}`);

// Per-cell output keys and polarities from the extraction planes.  Mirrors
// extractFeatures' size-2/size-3 emit exactly (verified below in selfCheck).
function key2(n2, i2) { return (VPatterns.mixTag(n2 < i2 ? n2 : i2, TAG2) ^ PS) | 0; }
function key3(n3, i3) { return (VPatterns.mixTag(n3 < i3 ? n3 : i3, TAG3) ^ PS) | 0; }

// ── Corpus ───────────────────────────────────────────────────────────────────
const lines = fs.readFileSync(GAMES_PATH, 'utf8').split('\n').filter(l => l && l[0] !== '#');
console.log(`corpus games: ${lines.length}`);

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
function eqMultiset(a, b) {
  if (a.size !== b.size) return false;
  for (const [k, v] of a) if (b.get(k) !== v) return false;
  return true;
}
function selfCheck() {
  let checked = 0;
  for (const line of lines.slice(0, 40)) {
    for (const g of positions(line)) {
      const f = VPatterns.extractFeatures(g, comp.preparedSpecs);
      const pl = comp.preparedSpecs._planes.get(M);
      const cap = g.N * g.N;
      const my2 = [], my3 = [];
      for (let i = 0; i < cap; i++) {
        if (pl.h2N[i] !== pl.h2I[i]) my2.push(key2(pl.h2N[i], pl.h2I[i]));
        if (pl.h3N[i] !== pl.h3I[i]) my3.push(key3(pl.h3N[i], pl.h3I[i]));
      }
      const ex2 = [], ex3 = [];
      for (let j = 0; j < f.count; j++) {
        if (f.tags[j] === TAG2) ex2.push(f.keys[j] | 0);
        else if (f.tags[j] === TAG3) ex3.push(f.keys[j] | 0);
      }
      if (!eqMultiset(multiset(my2), multiset(ex2)) || !eqMultiset(multiset(my3), multiset(ex3))) {
        console.error(`SELF-CHECK FAILED at a position (my2=${my2.length}/ex2=${ex2.length}, my3=${my3.length}/ex3=${ex3.length}).`);
        process.exit(1);
      }
      if (++checked >= 60) { console.log(`self-check: per-cell keys match extraction on ${checked} positions ✓`); return; }
    }
  }
  console.log(`self-check: per-cell keys match extraction on ${checked} positions ✓`);
}
selfCheck();

// ── Fold pass ────────────────────────────────────────────────────────────────
// acc: key3 -> { w3, sum, cnt }  (sum/cnt = mean pol2·pol3·w2 for that k3)
const acc = new Map();
let cells = 0, lost = 0, lostMag = 0, positionsFolded = 0;
const foldLines = lines.slice(0, FOLD_GAMES);
for (let gi = 0; gi < foldLines.length; gi++) {
  for (const g of positions(foldLines[gi])) {
    VPatterns.extractFeatures(g, comp.preparedSpecs);
    const pl = comp.preparedSpecs._planes.get(M);
    const cap = g.N * g.N;
    for (let i = 0; i < cap; i++) {
      const n3 = pl.h3N[i], i3 = pl.h3I[i];
      const n2 = pl.h2N[i], i2 = pl.h2I[i];
      const has2 = n2 !== i2;
      if (n3 === i3) {                       // 3×3 colour-symmetric: no k3 to carry a 2×2
        if (has2) { lost++; lostMag += Math.abs(comp.weights.get(key2(n2, i2)) ?? 0); }
        continue;
      }
      const k3 = key3(n3, i3);
      let inc = 0;
      if (has2) {
        const pol2 = n2 < i2 ? 1 : -1, pol3 = n3 < i3 ? 1 : -1;
        inc = pol2 * pol3 * (comp.weights.get(key2(n2, i2)) ?? 0);
      }
      let a = acc.get(k3);
      if (!a) { a = { w3: comp.weights.get(k3) ?? 0, sum: 0, cnt: 0 }; acc.set(k3, a); }
      a.sum += inc; a.cnt++;
      cells++;
    }
    positionsFolded++;
  }
  if ((gi + 1) % 500 === 0) console.log(`  folded ${gi + 1}/${foldLines.length} games, ${positionsFolded} positions, ${acc.size} keys`);
}
console.log(`fold: ${positionsFolded} positions, ${cells} cells, ${acc.size} distinct 3×3 keys`);
console.log(`lost (3×3-symmetric cells carrying a 2×2): ${lost} (${(100 * lost / (cells + lost)).toFixed(3)}% of cells, sum|w2|=${lostMag.toFixed(2)})`);

// ── Build + save the flat 3:M model ──────────────────────────────────────────
const flatWeights = VPatterns.makeWeights(Math.max(1024, acc.size * 2));
for (const [k3, a] of acc) {
  const w = a.w3 + a.sum / a.cnt;
  if (+w.toFixed(6) !== 0) flatWeights.set(k3 | 0, w);
}
const flatSpecs = [{ size: 3, maxLibs: M }];
const flatModel = { specs: flatSpecs, preparedSpecs: VPatterns.prepareSpecs(flatSpecs), weights: flatWeights, komi: comp.komi };
VPatterns.saveWeights(SAVE_PATH, flatModel);
console.log(`saved flat 3:${M}: ${SAVE_PATH}  (${flatWeights.size} weights)`);

// ── Measure: z_flat vs z_composite on held-out games ─────────────────────────
const flat = VPatterns.loadWeights(SAVE_PATH);
const evalLines = lines.slice(FOLD_GAMES, FOLD_GAMES + EVAL_GAMES);
let nE = 0, seSum = 0, aeSum = 0, maxAbs = 0, vSe = 0, coveredCells = 0, totalCells = 0;
for (const line of evalLines) {
  for (const g of positions(line)) {
    const fc = VPatterns.extractFeatures(g, comp.preparedSpecs);
    VPatterns.evaluateFeatures(fc, comp.weights);   // sets fc.z
    const zcomp = fc.z;
    const ff = VPatterns.extractFeatures(g, flat.preparedSpecs);
    VPatterns.evaluateFeatures(ff, flat.weights);
    const zflat = ff.z;
    // coverage: fraction of the flat model's 3×3 cell-keys present in its table
    for (let j = 0; j < ff.count; j++) { totalCells++; if (flat.weights.get(ff.keys[j] | 0) !== undefined) coveredCells++; }
    const d = zflat - zcomp;
    seSum += d * d; aeSum += Math.abs(d); if (Math.abs(d) > maxAbs) maxAbs = Math.abs(d);
    const vc = 1 / (1 + Math.exp(-zcomp)), vf = 1 / (1 + Math.exp(-zflat));
    vSe += (vf - vc) * (vf - vc);
    nE++;
  }
}
console.log(`\n── z_flat vs z_composite over ${nE} held-out positions ──`);
console.log(`z    RMS: ${Math.sqrt(seSum / nE).toFixed(5)}   MAE: ${(aeSum / nE).toFixed(5)}   max|Δ|: ${maxAbs.toFixed(4)}`);
console.log(`V=σ(z) RMS: ${Math.sqrt(vSe / nE).toFixed(6)}   (win-prob units)`);
console.log(`flat 3×3 cell-key coverage on eval: ${(100 * coveredCells / totalCells).toFixed(2)}%`);
