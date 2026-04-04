#!/usr/bin/env node
'use strict';

// test-hpatterns.js — correctness tests for hpatterns.js

const { Game2, BLACK, WHITE } = require('./game2.js');
const { createModel, extractFeatures, evaluateFeatures, evaluate } = require('./hpatterns.js');

let pass = 0, fail = 0;

function check(cond, msg) {
  if (cond) { pass++; }
  else       { fail++; console.error('  FAIL:', msg); }
}

function section(name) { console.log(`\n── ${name} ──`); }

// helper: collect features into array-of-objects for assertions
function featureArr(f) {
  const out = [];
  for (let i = 0; i < f.count; i++) out.push({ key: f.keys[i], pol: f.pols[i] });
  return out;
}

// ── evaluateFeatures ──────────────────────────────────────────────────────────

section('evaluateFeatures');
{
  const f = { keys: new Int32Array(0), pols: new Int8Array(0), count: 0 };
  check(evaluateFeatures(f, new Map()) === 0.5, 'empty features → 0.5');
}
{
  const f = { keys: new Int32Array([1]), pols: new Int8Array([1]), count: 1 };
  check(evaluateFeatures(f, new Map([[1, 10]])) > 0.99, 'strong positive weight → >0.99');
}
{
  const f = { keys: new Int32Array([1]), pols: new Int8Array([-1]), count: 1 };
  check(evaluateFeatures(f, new Map([[1, 10]])) < 0.01, 'strong negative-polarity weight → <0.01');
}

// ── empty board ───────────────────────────────────────────────────────────────

section('empty board');
{
  const m = createModel({2:4, 3:4, 4:4}, 4);
  const g = new Game2(9, false);
  const f = extractFeatures(g, m);
  check(f.count === 0, 'empty 9×9 → 0 features');
}
{
  const m = createModel({2:4, 3:4, 4:4}, 4);
  const g = new Game2(5, false);
  const f = extractFeatures(g, m);
  check(f.count === 0, 'empty 5×5 → 0 features');
}

// ── single stone ──────────────────────────────────────────────────────────────

section('single stone');
{
  const m = createModel({2:1}, 2);
  const g = new Game2(9, false);
  g.play(40);
  const f = featureArr(extractFeatures(g, m));
  // One B stone → exactly 4 overlapping 2×2 windows, all with the same canonKey.
  check(f.length === 4, `B@40, maxSize=2: expect 4 features, got ${f.length}`);
  const keys = new Set(f.map(x => x.key));
  check(keys.size === 1, 'all 4 features share one canonKey (D4 symmetry)');
  const pols = new Set(f.map(x => x.pol));
  check(pols.size === 1, 'all 4 features have the same polarity');
}
{
  const m = createModel({2:1}, 2);
  const g = new Game2(9, false);
  g.play(40);
  const f = featureArr(extractFeatures(g, m));
  // B stone features should contribute positively (polarity = +1).
  check(f[0].pol === 1, `B@40 polarity should be +1, got ${f[0].pol}`);
}

// ── color symmetry ────────────────────────────────────────────────────────────

section('color symmetry');
{
  // B@40 and W@40 with same canonMap → same canonKeys, opposite polarities.
  const m = createModel({2:4, 3:4}, 3);

  const gB = new Game2(9, false);
  gB.play(40);
  const fB = featureArr(extractFeatures(gB, m));

  // Place W directly (no second stone from game mechanics).
  const gW = new Game2(9, false);
  gW.cells[40] = WHITE;
  const fW = featureArr(extractFeatures(gW, m));

  check(fB.length === fW.length, `same feature count: B=${fB.length} W=${fW.length}`);

  const mapB = new Map(fB.map(f => [f.key, f.pol]));
  const mapW = new Map(fW.map(f => [f.key, f.pol]));

  let symOk = fB.length > 0;
  for (const [k, p] of mapB) {
    if (!mapW.has(k) || mapW.get(k) !== -p) { symOk = false; break; }
  }
  check(symOk, 'B and W produce same canonKeys with opposite polarities');
}

// ── maxStones filter ──────────────────────────────────────────────────────────

section('maxStones filter');
{
  // With limit=1, a 2×2 window with 2 stones is excluded.
  const m1 = createModel({2:1}, 2);
  const m2 = createModel({2:2}, 2);
  const g = new Game2(9, false);
  g.play(40); g.play(0);  // B@40, W@0 — far apart on 9×9
  // play B somewhere else to avoid W being excluded
  // Actually: after g.play(40) B plays, g.play(0) — wait, g.play(0) is W's move
  // Both stones are isolated: each gives 4 windows with 1 stone.
  // No 2×2 window contains both (they're far apart on 9×9, non-toroidal wrap notwithstanding).
  const f1 = extractFeatures(g, m1);
  const f2 = extractFeatures(g, m2);
  check(f1.count <= f2.count, 'limit=1 ≤ limit=2 feature count');
}
{
  // A board with two adjacent stones has 2-stone 2×2 windows; limit=1 excludes them.
  const m1 = createModel({2:1}, 2);
  const m2 = createModel({2:2}, 2);
  const g = new Game2(9, false);
  g.play(40); g.play(0); g.play(41);  // B@40, W@0, B@41 — 40 and 41 are adjacent
  const f1 = extractFeatures(g, m1);
  const f2 = extractFeatures(g, m2);
  check(f2.count > f1.count, 'adjacent stones: limit=2 extracts more features than limit=1');
}

// ── per-size maxStones ────────────────────────────────────────────────────────

section('per-size maxStones');
{
  // { 2: 4, 3: 1 } restricts 3×3 windows more than { 2: 4, 3: 4 }.
  const mLoose = createModel({ 2: 4, 3: 4 }, 3);
  const mTight = createModel({ 2: 4, 3: 1 }, 3);
  const g = new Game2(9, false);
  g.play(40); g.play(0); g.play(41); g.play(1); g.play(49);  // a few stones
  const fU = extractFeatures(g, mLoose);
  const fP = extractFeatures(g, mTight);
  check(fP.count <= fU.count,
    `{3:1} ≤ {3:4}: got fP=${fP.count} fU=${fU.count}`);
}
{
  // Absent sizes default to limit=0 (inactive).
  const m = createModel({ 2: 1 }, 2);  // only size 2, only 1-stone windows
  const g = new Game2(9, false);
  g.play(40);
  const f = extractFeatures(g, m);
  check(f.count === 4, `per-size {2:1}, single B stone → 4 features, got ${f.count}`);
}

// ── maxSize limit ─────────────────────────────────────────────────────────────

section('maxSize limit');
{
  const m2 = createModel({2:8}, 2);
  const m3 = createModel({2:8, 3:8}, 3);
  const g = new Game2(9, false);
  g.play(40); g.play(0); g.play(41);
  const f2 = extractFeatures(g, m2);
  const f3 = extractFeatures(g, m3);
  check(f3.count >= f2.count, 'maxSize=3 ≥ maxSize=2 feature count');
}

// ── D4 rotation equivalence ───────────────────────────────────────────────────

section('D4 rotation equivalence');
{
  // Two boards that are D4 rotations of each other should produce
  // the same set of canonical keys (with same polarities).
  // Use B at center (40) of 9×9 — center is rotation-invariant, so
  // all 4 orientations produce the same board; check key consistency.
  const m = createModel({2:1}, 2);
  const g = new Game2(9, false);
  g.play(40);
  const f = featureArr(extractFeatures(g, m));
  // All features should have the same canonKey (single-stone pattern, any rotation = same).
  const keys = new Set(f.map(x => x.key));
  check(keys.size === 1, `single stone: all windows have same canonKey (got ${keys.size} distinct)`);
}

// ── speculative placement ─────────────────────────────────────────────────────

section('speculative placement');
{
  // Features from speculative B@40 should match features from actually playing B@40.
  const g = new Game2(9, false);  // empty board, B to play
  const m1 = createModel({2:4, 3:4}, 3);
  const m2 = createModel({2:4, 3:4}, 3);

  const fSpec = featureArr(extractFeatures(g, m1, true, 40));

  g.play(40);  // actually play
  const fReal = featureArr(extractFeatures(g, m2, false));

  check(fSpec.length === fReal.length,
    `speculative B@40 count=${fSpec.length} matches actual count=${fReal.length}`);
}
{
  // Board is unchanged after speculative extraction.
  const g = new Game2(9, false);
  g.play(20);  // B@20
  const before = Array.from(g.cells);
  const m = createModel({2:4, 3:4}, 3);
  extractFeatures(g, m, true, 30);  // speculative B@30
  const after = Array.from(g.cells);
  const changed = before.some((v, i) => v !== after[i]);
  check(!changed, 'board cells unchanged after speculative extraction');
}
{
  // Speculative extraction with captures: captured stones temporarily disappear.
  // Set up: B group with one liberty at 40, W surrounds it.
  // Simpler: just verify that doSetNext=false and doSetNext=true give different results
  // when a move actually changes the board.
  const g = new Game2(9, false);
  const m = createModel({2:4, 3:4}, 3);
  const fBefore = featureArr(extractFeatures(g, m, false));
  const fSpec   = featureArr(extractFeatures(g, m, true, 40));
  check(fBefore.length !== fSpec.length || fSpec.length > 0,
    'speculative on empty board adds features (B@40)');
  check(fBefore.length === 0, 'empty board still has 0 features without doSetNext');
  check(fSpec.length > 0,   'speculative B@40 on empty board has >0 features');
}
{
  // Speculative extraction skips PASS: board unchanged, same features as non-speculative.
  const { PASS } = require('./game2.js');
  const g = new Game2(9, false);
  g.play(40);
  const m = createModel({2:4, 3:4}, 3);
  const fNormal = featureArr(extractFeatures(g, m, false));
  const fPass   = featureArr(extractFeatures(g, m, true, PASS));
  check(fNormal.length === fPass.length, 'doSetNext with PASS = no change');
}

// ── evaluate convenience ──────────────────────────────────────────────────────

section('evaluate');
{
  const g = new Game2(9, false);
  g.play(40);
  const m = createModel({2:4, 3:4}, 3);
  const f = evaluate(g, m);
  check(f.val === 0.5, `evaluate with zero weights → 0.5, got ${f.val}`);
  check(f.count > 0,   'evaluate returns non-zero feature count');
}
{
  const g = new Game2(9, false);
  g.play(40);
  const m = createModel({2:4, 3:4}, 3);
  const f = extractFeatures(g, m);
  // Set all active feature weights to a large positive value.
  for (let i = 0; i < f.count; i++) m.weights.set(f.keys[i], 10 * f.pols[i]);
  const f2 = evaluate(g, m);
  check(f2.val > 0.5, `evaluate with positive weights → >0.5, got ${f2.val}`);
}

// ── canonMap encoding ─────────────────────────────────────────────────────────
// encoded = canonKey * 4 + flags  where flags: 0=twin, 1=pol+1, 2=pol-1
// flags   = ((encoded % 4) + 4) % 4
// canonKey = (encoded - flags) / 4

section('canonMap encoding');
{
  function encode(canonKey, polarity, isTwin) {
    const flags = isTwin ? 0 : (polarity === 1 ? 1 : 2);
    return canonKey * 4 + flags;
  }
  function decodeFlags(enc) { return ((enc % 4) + 4) % 4; }
  function decodeKey(enc)   { const f = decodeFlags(enc); return (enc - f) / 4; }

  const cases = [
    // [canonKey, polarity, isTwin]
    [0,           1,  false],
    [0,          -1,  false],
    [0,           1,  true ],
    [1,           1,  false],
    [1,          -1,  false],
    [-1,          1,  false],
    [-1,         -1,  false],
    [-1,          1,  true ],
    [2147483647,  1,  false],  // max int32
    [2147483647, -1,  false],
    [-2147483648, 1,  false],  // min int32
    [-2147483648,-1,  false],
  ];

  for (const [key, pol, twin] of cases) {
    const enc   = encode(key, pol, twin);
    const flags = decodeFlags(enc);
    const decKey = decodeKey(enc);
    check(decKey === key,
      `round-trip canonKey: key=${key} pol=${pol} twin=${twin} → enc=${enc} → key=${decKey}`);
    check((flags === 0) === twin,
      `twin flag: key=${key} pol=${pol} twin=${twin} → flags=${flags}`);
    if (!twin) {
      const decPol = flags === 1 ? 1 : -1;
      check(decPol === pol,
        `polarity: key=${key} pol=${pol} → flags=${flags} → decPol=${decPol}`);
    }
  }
}
{
  // Verify via API: B and W single-stone patterns have opposite polarities and same canonKey.
  const { WHITE } = require('./game2.js');
  const m = createModel({2:1}, 2);
  const gB = new Game2(9, false);
  gB.play(40);
  const fB = featureArr(extractFeatures(gB, m));

  const gW = new Game2(9, false);
  gW.cells[40] = WHITE;
  const fW = featureArr(extractFeatures(gW, m));

  check(fB.length > 0, 'B stone produces features');
  check(fB.length === fW.length, 'B and W produce same feature count');
  const bKey = fB[0].key, wKey = fW[0].key;
  const bPol = fB[0].pol, wPol = fW[0].pol;
  check(bKey === wKey,   `same canonKey after encoding: B=${bKey} W=${wKey}`);
  check(bPol === -wPol,  `opposite polarities after encoding: B=${bPol} W=${wPol}`);
}
{
  // Encoding round-trip: keys from a fresh extraction must match a saved copy,
  // even when canonKeys are negative. Strategy: copy keys/pols before calling
  // evaluate (which overwrites the shared output buffer), then compare.
  const m = createModel({2:4, 3:4}, 3);
  const g = new Game2(9, false);
  g.play(20); g.play(30); g.play(40); g.play(50);
  const f = extractFeatures(g, m);
  // Copy feature data before evaluate() overwrites the model's output buffers.
  const savedKeys = f.keys.slice(0, f.count);
  const savedPols = f.pols.slice(0, f.count);
  check(f.count > 0, 'board with 4 stones has features');
  check(savedKeys.some(k => k < 0), 'some canonKeys are negative (exercises negative decoding)');
  // Assign distinct weights per unique key.
  for (let i = 0; i < f.count; i++) m.weights.set(savedKeys[i], (i + 1) * 0.1);
  // Manually compute expected value from saved (pre-overwrite) features.
  let z = 0;
  for (let i = 0; i < f.count; i++) z += savedPols[i] * m.weights.get(savedKeys[i]);
  const expected = 1 / (1 + Math.exp(-z));
  // evaluate re-extracts using the populated canonMap — keys must match.
  const result = evaluate(g, m);
  check(Math.abs(result.val - expected) < 1e-10,
    `encoding round-trip: result=${result.val.toFixed(6)} expected=${expected.toFixed(6)}`);
}

// ── Summary ───────────────────────────────────────────────────────────────────

console.log(`\n${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
