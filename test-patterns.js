#!/usr/bin/env node
'use strict';

// test-patterns.js — correctness tests for vpatterns.js

const { Game2, BLACK, WHITE } = require('./game2.js');
const { rawState, canonicalize, extractFeatures: _extractFeatures,
        prepareSpecs, evaluateFeatures,
        PERMS_2x2, PERMS_3x3 } = require('./vpatterns.js');
const _prepCache = new Map();
function extractFeatures(game, specs, ...rest) {
  if (!_prepCache.has(specs)) _prepCache.set(specs, prepareSpecs(specs));
  const f = _extractFeatures(game, _prepCache.get(specs), ...rest);
  // Convert typed arrays back to array-of-objects for test assertions.
  const arr = [];
  for (let i = 0; i < f.count; i++) arr.push({ key: f.keys[i], polarity: f.pols[i] });
  return arr;
}

let pass = 0, fail = 0;

function check(cond, msg) {
  if (cond) { pass++; }
  else       { fail++; console.error('  FAIL:', msg); }
}

function section(name) { console.log(`\n── ${name} ──`); }

// ── rawState ──────────────────────────────────────────────────────────────────

section('rawState');
{
  const g = new Game2(5, false);
  check(rawState(g, 4, 0) === 0, 'rawState: empty cell → 0');
}
{
  // B@12 = center of 5×5, 4 liberties; with maxLibs=4, rawState = +4.
  const g = new Game2(5, false);
  g.play(12);
  const s = rawState(g, 4, 12);
  check(s === 4, `rawState: B@12 → +4, got ${s}`);
}
{
  // W@7=(1,2) has B@12 as its down-neighbour → 3 empty neighbours → rawState = -3.
  const g = new Game2(5, false);
  g.play(12); g.play(7);
  const s = rawState(g, 4, 7);
  check(s === -3, `rawState: W@7 → -3, got ${s}`);
}


// ── evaluateFeatures ──────────────────────────────────────────────────────────

section('evaluateFeatures');
{
  // No features → z=0 → σ(0) = 0.5.
  const v = evaluateFeatures({ keys: new Int32Array(0), pols: new Int8Array(0), count: 0 }, new Map());
  check(v === 0.5, `evaluateFeatures: empty → 0.5, got ${v}`);
}
{
  // One feature with polarity=+1 and weight=10 → σ(10) > 0.99.
  const f = { keys: new Int32Array([1]), pols: new Int8Array([1]), count: 1 };
  const w = new Map([[1, 10]]);
  const v = evaluateFeatures(f, w);
  check(v > 0.99, `evaluateFeatures: strong positive weight → >0.99, got ${v}`);
}

// ── extractFeatures – size 1 ──────────────────────────────────────────────────

section('extractFeatures – size 1');
{
  const SPEC = [{ size: 1, maxLibs: 1 }];

  // Empty board → no features.
  const g = new Game2(9, false);
  check(extractFeatures(g, SPEC).length === 0, 'size-1: empty board → no features');
}
{
  const SPEC = [{ size: 1, maxLibs: 1 }];

  // Single B stone → one feature with polarity=+1.
  const g = new Game2(9, false);
  g.play(40);
  const f = extractFeatures(g, SPEC);
  check(f.length === 1 && f[0].polarity === 1,
    `size-1: B stone → 1 feature polarity=+1, got length=${f.length} polarity=${f[0]?.polarity}`);
}
{
  const SPEC = [{ size: 1, maxLibs: 1 }];

  // Color symmetry: B and W at same position give same key, opposite polarity.
  const gB = new Game2(9, false);
  gB.play(40);
  const fB = extractFeatures(gB, SPEC);

  // B@0 (far corner), W@40 — cells not adjacent, so liberty counts unaffected.
  const gW = new Game2(9, false);
  gW.play(0); gW.play(40);
  const fW = extractFeatures(gW, SPEC).filter(f => f.polarity === -1);

  check(fB.length === 1 && fW.length === 1 && fB[0].key === fW[0].key,
    `size-1: color symmetry — keyB=${fB[0]?.key} keyW=${fW[0]?.key}`);
}

// ── extractFeatures – size 2 ──────────────────────────────────────────────────

section('extractFeatures – size 2');
{
  const SPEC = [{ size: 2, maxLibs: 1 }];

  // Empty board → no features.
  const g = new Game2(9, false);
  check(extractFeatures(g, SPEC).length === 0, 'size-2: empty board → no features');
}
{
  const SPEC = [{ size: 2, maxLibs: 1 }];

  // Single B stone → exactly 4 overlapping 2×2 windows; all 4 positions within a
  // 2×2 window form a single D4 orbit, so all features get the same canonical key
  // and polarity.
  const g = new Game2(9, false);
  g.play(40);
  const f = extractFeatures(g, SPEC);
  const keys = new Set(f.map(x => x.key));
  const pols = new Set(f.map(x => x.polarity));
  check(f.length === 4, 'size-2: single B stone → 4 features');
  check(keys.size === 1, 'size-2: single B stone → all same key (D4 symmetry)');
  check(pols.size === 1, 'size-2: single B stone → all same polarity');
}
{
  const SPEC = [{ size: 2, maxLibs: 1 }];

  // Color symmetry: B and W at the same position give same key, opposite polarity.
  const gB = new Game2(9, false);
  gB.play(40);
  const fB = extractFeatures(gB, SPEC);
  const keyB = fB[0].key, polB = fB[0].polarity;

  // B@0 (far from 40), W@40.
  const gW = new Game2(9, false);
  gW.play(0); gW.play(40);
  const fW = extractFeatures(gW, SPEC).filter(f => f.polarity === -polB);

  check(fW.length === 4 && fW[0].key === keyB,
    `size-2: color symmetry — keyB=${keyB} keyW=${fW[0]?.key}`);
}

// ── extractFeatures – size 3 ──────────────────────────────────────────────────

section('extractFeatures – size 3');
{
  const SPEC = [{ size: 3, maxLibs: 1 }];

  // Empty board → no features.
  const g = new Game2(9, false);
  check(extractFeatures(g, SPEC).length === 0, 'size-3: empty board → no features');
}
{
  const SPEC = [{ size: 3, maxLibs: 1 }];

  // Single B stone → exactly 9 overlapping 3×3 windows.  The 9 positions within a
  // 3×3 window form 3 D4 orbits (4 corners, 4 edge-midpoints, 1 center), producing
  // exactly 3 distinct canonical keys.
  const g = new Game2(9, false);
  g.play(40);
  const f = extractFeatures(g, SPEC);
  const keys = new Set(f.map(x => x.key));
  check(f.length === 9,    'size-3: single B stone → 9 features');
  check(keys.size === 3,   'size-3: single B stone → 3 distinct keys (3 D4 orbits)');
}
{
  const SPEC = [{ size: 3, maxLibs: 1 }];

  // Orbit consistency: 4 corner windows (B at different corners) all give same key
  // and polarity.  On a 9×9 board with B@40=(4,4), corner anchors are at
  // (2,2)=20, (2,4)=22, (4,2)=38, (4,4)=40.
  const g = new Game2(9, false);
  g.play(40);
  const all = extractFeatures(g, SPEC);
  // The 3 keys correspond to corners, edges, and center; sort by frequency to find them.
  const freq = new Map();
  for (const f of all) freq.set(f.key, (freq.get(f.key) || 0) + 1);
  const sorted = [...freq.entries()].sort((a, b) => b[1] - a[1]);
  // corners and edges both appear 4 times; center appears once
  check(sorted[2][1] === 1, 'size-3: center orbit has exactly 1 feature');
  check(sorted[0][1] === 4 && sorted[1][1] === 4,
    'size-3: corner and edge orbits each have exactly 4 features');
}
{
  const SPEC = [{ size: 3, maxLibs: 1 }];

  // Color symmetry: for each orbit, B and W at the same position give same key,
  // opposite polarity.  Check by comparing the full sets of keys.
  const gB = new Game2(9, false);
  gB.play(40);
  const fB = extractFeatures(gB, SPEC);

  const gW = new Game2(9, false);
  gW.play(0); gW.play(40);  // B@0 (far from 40), W@40
  // Features from W@40 have opposite polarity to the corresponding B@40 features.
  // B@0 produces its own set of keys; the intersection with B@40 keys verifies symmetry.
  const keysB = new Set(fB.map(f => f.key));
  const fW = extractFeatures(gW, SPEC).filter(f => keysB.has(f.key));
  const keysW = new Set(fW.map(f => f.key));
  check(keysB.size === keysW.size && [...keysB].every(k => keysW.has(k)),
    'size-3: color symmetry — B and W produce same set of keys');
}

// ── maxLibs key isolation ─────────────────────────────────────────────────────

section('maxLibs key isolation');
{
  // No key produced by maxLibs=1 specs should collide with any from maxLibs=2.
  const g = new Game2(5, false);
  g.play(12);  // B@12, center, 4 liberties
  const sizes = [1, 2, 3];
  const f1 = extractFeatures(g, sizes.map(size => ({ size, maxLibs: 1 })));
  const f2 = extractFeatures(g, sizes.map(size => ({ size, maxLibs: 2 })));
  const keys1 = new Set(f1.map(f => f.key));
  const keys2 = new Set(f2.map(f => f.key));
  const collisions = [...keys1].filter(k => keys2.has(k));
  check(collisions.length === 0,
    `${collisions.length} collision(s) between maxLibs=1 and maxLibs=2: [${collisions.slice(0, 5)}]`);
}

// ── Full enumeration counts ───────────────────────────────────────────────────

section('full enumeration counts');
{
  // Enumerate all 3^n raw cell patterns and count distinct canonical keys.
  // Uses canonicalize directly with the same mixers as extractFeatures
  // (131 for 2×2, 537 for 3×3, both with maxLibs=1).
  // Expected: 2×2 → 8, 3×3 → 1418.

  const keys2 = new Set();
  const cells4 = new Array(4);
  for (let mask = 0; mask < 81 /* 3^4 */; mask++) {
    let m = mask;
    for (let i = 0; i < 4; i++) { cells4[i] = (m % 3) - 1; m = (m / 3) | 0; }
    const r = canonicalize(cells4, PERMS_2x2, 131);
    if (r !== null) keys2.add(r.key);
  }
  check(keys2.size === 8, `2×2 full enum: expected 8, got ${keys2.size}`);

  const keys3 = new Set();
  const cells9 = new Array(9);
  for (let mask = 0; mask < 19683 /* 3^9 */; mask++) {
    let m = mask;
    for (let i = 0; i < 9; i++) { cells9[i] = (m % 3) - 1; m = (m / 3) | 0; }
    const r = canonicalize(cells9, PERMS_3x3, 537);
    if (r !== null) keys3.add(r.key);
  }
  check(keys3.size === 1418, `3×3 full enum: expected 1418, got ${keys3.size}`);
}

// ── Summary ───────────────────────────────────────────────────────────────────

console.log(`\n${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
