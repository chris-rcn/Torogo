#!/usr/bin/env node
'use strict';

// test-patterns.js — correctness tests for patterns.js

const assert = require('assert');
const { Game2, BLACK, WHITE, PASS } = require('./game2.js');
const { rawState, flipState, canonicalize, pattern1x1, pattern2x2, pattern3x3,
        MAX_LIBS, PERMS_2x2, PERMS_3x3 } = require('./patterns.js');

let pass = 0, fail = 0;

function check(cond, msg) {
  if (cond) { pass++; }
  else       { fail++; console.error('  FAIL:', msg); }
}

function section(name) { console.log(`\n── ${name} ──`); }

// ── rawState ──────────────────────────────────────────────────────────────────

section('rawState');
{
  // Test 1: empty cell returns 0.
  const g = new Game2(5, false);
  check(rawState(g, 0) === 0, 'rawState: empty cell → 0');
}

{
  // Test 2: BLACK stone → 1 (own stone, capped libs = MAX_LIBS = 1).
  const g = new Game2(5, false);
  g.play(12); // B@12
  check(rawState(g, 12) === 1, `rawState: B@12 → 1, got ${rawState(g, 12)}`);
}

{
  // Test 3: WHITE stone → MAX_LIBS + 1 = 2.
  const g = new Game2(5, false);
  g.play(12); // B@12
  g.play(7);  // W@7
  const expected = MAX_LIBS + 1;
  check(rawState(g, 7) === expected,
    `rawState: W@7 → ${expected}, got ${rawState(g, 7)}`);
}

// ── flipState ─────────────────────────────────────────────────────────────────

section('flipState');
{
  check(flipState(0) === 0, 'flipState(0) === 0');
  check(flipState(1) === MAX_LIBS + 1, `flipState(1) === ${MAX_LIBS + 1}`);
  check(flipState(MAX_LIBS + 1) === 1, `flipState(${MAX_LIBS + 1}) === 1`);
}

// ── pattern1x1 ────────────────────────────────────────────────────────────────

section('pattern1x1');
{
  // Empty cell → null.
  const g = new Game2(5, false);
  check(pattern1x1(g, 12) === null, 'pattern1x1: empty cell → null');
}

{
  // BLACK stone → key=1, polarity=+1.
  const g = new Game2(5, false);
  g.play(12);
  const r = pattern1x1(g, 12);
  check(r !== null && r.key === 1 && r.polarity === 1,
    `pattern1x1: B@12 → {key:1, polarity:1}, got ${JSON.stringify(r)}`);
}

{
  // WHITE stone → key=1, polarity=-1.
  const g = new Game2(5, false);
  g.play(12); // B@12
  g.play(7);  // W@7
  const r = pattern1x1(g, 7);
  check(r !== null && r.key === 1 && r.polarity === -1,
    `pattern1x1: W@7 → {key:1, polarity:-1}, got ${JSON.stringify(r)}`);
}

{
  // Color symmetry: B and W stones have same key, opposite polarity.
  const gB = new Game2(5, false);
  gB.play(12);
  const rB = pattern1x1(gB, 12);

  const gW = new Game2(5, false);
  gW.play(12);
  gW.play(0);
  const rW = pattern1x1(gW, 0);

  check(rB !== null && rW !== null && rB.key === rW.key && rB.polarity === 1 && rW.polarity === -1,
    `pattern1x1 color symmetry: keys ${rB && rB.key}/${rW && rW.key} polarities ${rB && rB.polarity}/${rW && rW.polarity}`);
}

// ── pattern2x2 ────────────────────────────────────────────────────────────────

section('pattern2x2');
{
  // All-empty → null.
  const g = new Game2(5, false);
  check(pattern2x2(g, 12) === null, 'pattern2x2: all-empty → null');
}

{
  // Single B stone at anchor TL: raw=[1,0,0,0].
  // Minimum encoding places the stone last: key=1, polarity=+1.
  const g = new Game2(5, false);
  g.play(12);
  const r = pattern2x2(g, 12);
  check(r !== null && r.key === 1 && r.polarity === 1,
    `pattern2x2 single B stone at TL: key=1, polarity=+1, got ${JSON.stringify(r)}`);
}

{
  // Rotation symmetry: same stone in BR of a different 2x2 window should give same key.
  // Stone at 12, anchor 7: TL=7(e), TR=8(e), BL=12(B), BR=13(e) → raw=[0,0,1,0] → key=1.
  const g = new Game2(5, false);
  g.play(12);
  const r7 = pattern2x2(g, 7);
  check(r7 !== null && r7.key === 1,
    `pattern2x2 rotation: anchor 7, B@12 in BL → key=1, got ${JSON.stringify(r7)}`);
}

{
  // Color symmetry: B and W single stone give same key, opposite polarity.
  const gA = new Game2(5, false);
  gA.play(12);
  const rA = pattern2x2(gA, 12);

  const gB = new Game2(5, false);
  gB.play(0);
  gB.play(12); // W@12
  const rB = pattern2x2(gB, 12);

  check(rA !== null && rB !== null && rA.key === rB.key && rA.polarity === 1 && rB.polarity === -1,
    `pattern2x2 color symmetry: keys ${rA && rA.key}/${rB && rB.key} polarities ${rA && rA.polarity}/${rB && rB.polarity}`);
}

{
  // Two adjacent B stones in top row and two in bottom row of the same 2×2 window
  // (FlipV symmetry) give the same key.
  // B@12, W@0, B@13: anchor 12 → raw=[1,1,0,0]; anchor 7 → raw=[0,0,1,1].
  // FlipV maps [s,s,0,0] → [0,0,s,s]. Both encodings: min = 0*9^3+0*9^2+1*9+1 = 10.
  const g = new Game2(5, false);
  g.play(12);
  g.play(0);
  g.play(13);
  const r12 = pattern2x2(g, 12);
  const r7  = pattern2x2(g, 7);
  check(r12 !== null && r7 !== null && r12.key === r7.key,
    `pattern2x2 FlipV symmetry: anchor 12 key=${r12 && r12.key} anchor 7 key=${r7 && r7.key}`);
}

// ── pattern3x3 (upper-left anchor) ───────────────────────────────────────────

section('pattern3x3');
{
  // All-empty → null.
  const g = new Game2(5, false);
  check(pattern3x3(g, 12) === null, 'pattern3x3: all-empty → null');
}

{
  // Single B stone at anchor (position 0,0): raw=[1,0,0,...,0].
  // Minimum encoding places stone at last cell: key=1, polarity=+1.
  const g = new Game2(5, false);
  g.play(12);
  const r = pattern3x3(g, 12);
  check(r !== null && r.key === 1 && r.polarity === 1,
    `pattern3x3 single stone at anchor: key=1, polarity=+1, got ${JSON.stringify(r)}`);
}

{
  // Rot180 symmetry: stone at (0,0) of window and stone at (2,2) of window give same key.
  // On 5×5, B@12. Anchor=12: stone at (0,0). Anchor=0: (2,2) is idx 12.
  const g = new Game2(5, false);
  g.play(12);
  const r12 = pattern3x3(g, 12);  // stone at (0,0)
  const r0  = pattern3x3(g, 0);   // stone at (2,2)
  check(r12 !== null && r0 !== null && r12.key === r0.key,
    `pattern3x3 rot180 symmetry: anchor 12 key=${r12 && r12.key} anchor 0 key=${r0 && r0.key}`);
}

{
  // Color symmetry: B and W stones at anchor give same key, opposite polarity.
  const gB = new Game2(5, false);
  gB.play(12);
  const rB = pattern3x3(gB, 12);

  const gW = new Game2(5, false);
  gW.play(0);
  gW.play(12); // W@12
  const rW = pattern3x3(gW, 12);

  check(rB !== null && rW !== null && rB.key === rW.key && rB.polarity === 1 && rW.polarity === -1,
    `pattern3x3 color symmetry: keys ${rB && rB.key}/${rW && rW.key} polarities ${rB && rB.polarity}/${rW && rW.polarity}`);
}

{
  // Two-stone symmetry: B stones at (0,0) and (0,1) of one window should match
  // B stones at (2,1) and (2,2) of another window (rot180 applied to row positions).
  // B@12, W@99 (far away, just to advance turn), B@13.
  // Anchor=12: raw=[1,1,0,...] (stones at c00 and c01).
  // Under rot180, [1,1,0,...,0] → [...,0,1,1]; key should be 0*..+1*9+1=10.
  // Anchor that sees B@12 at (2,1) and B@13 at (2,2):
  //   (2,1)=12 → anchor at (r-2,c-1) = (0,1) = idx 1.
  //   Check: from anchor=1, c20=below(below(1)), c21=right(c20), c22=right(c21).
  const g = new Game2(5, false);
  g.play(12);
  g.play(0);  // W@0 (outside both windows)
  g.play(13);

  const r12 = pattern3x3(g, 12);  // stones at (0,0) and (0,1)
  const r1  = pattern3x3(g, 1);   // stones at (2,1) and (2,2)
  check(r12 !== null && r1 !== null && r12.key === r1.key,
    `pattern3x3 two-stone rot180: anchor 12 key=${r12 && r12.key} anchor 1 key=${r1 && r1.key}`);
}

// ── Summary ───────────────────────────────────────────────────────────────────

console.log(`\n${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
