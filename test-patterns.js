#!/usr/bin/env node
'use strict';

// test-patterns.js — correctness tests for patterns.js

const assert = require('assert');
const { Game2, BLACK, WHITE, PASS } = require('./game2.js');
const { rawState, flipState, canonicalize, pattern1, pattern2, pattern3,
        PERMS_2x2, PERMS_3x3 } = require('./patterns.js');

// maxLibs used for all pattern calls; must match liberty-count expectations below.
const MAX_LIBS = 4;

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
  check(rawState(g, MAX_LIBS, 0) === 0, 'rawState: empty cell → 0');
}

{
  // Test 2: BLACK stone → positive value (own stone encoding).
  // B@12 = center of 5×5, 4 liberties; with MAX_LIBS=4, rawState = +4.
  const g = new Game2(5, false);
  g.play(12); // B@12
  const s = rawState(g, MAX_LIBS, 12);
  check(s === 4, `rawState: B@12 → 4, got ${s}`);
}

{
  // Test 3: WHITE stone → negative value (opponent encoding).
  // After B@12 and W@7: W@7=(1,2) has B@12 as its down-neighbour,
  // leaving 3 empty neighbours → rawState = -3.
  const g = new Game2(5, false);
  g.play(12); // B@12
  g.play(7);  // W@7
  const s = rawState(g, MAX_LIBS, 7);
  check(s === -3, `rawState: W@7 → -3, got ${s}`);
}

// ── flipState ─────────────────────────────────────────────────────────────────

section('flipState');
{
  // flipState negates: own ↔ opponent.
  check(flipState(0) === 0,   'flipState(0) === 0');
  check(flipState(1) === -1,  'flipState(1) === -1');
  check(flipState(-1) === 1,  'flipState(-1) === 1');
  check(flipState(3) === -3,  'flipState(3) === -3');
  check(flipState(-3) === 3,  'flipState(-3) === 3');
}

// ── pattern1 ──────────────────────────────────────────────────────────────────

section('pattern1');
{
  // Empty cell → null.
  const g = new Game2(5, false);
  check(pattern1(g, MAX_LIBS, 12) === null, 'pattern1: empty cell → null');
}

{
  // BLACK stone at center (4 liberties) → key=4, polarity=+1.
  const g = new Game2(5, false);
  g.play(12);
  const r = pattern1(g, MAX_LIBS, 12);
  check(r !== null && r.key === 4 && r.polarity === 1,
    `pattern1: B@12 → {key:4, polarity:1}, got ${JSON.stringify(r)}`);
}

{
  // WHITE stone → polarity=-1.
  const g = new Game2(5, false);
  g.play(12); // B@12
  g.play(7);  // W@7
  const r = pattern1(g, MAX_LIBS, 7);
  check(r !== null && r.polarity === -1,
    `pattern1: W@7 → polarity=-1, got ${JSON.stringify(r)}`);
}

{
  // Color symmetry: B and W stones at the same position (same board context)
  // give the same key but opposite polarity.
  // gB: B@12 alone.   gW: B@0, W@12 (B@0 is not adjacent to 12).
  const gB = new Game2(5, false);
  gB.play(12);
  const rB = pattern1(gB, MAX_LIBS, 12);

  const gW = new Game2(5, false);
  gW.play(0);   // B@0 (corner, not adjacent to 12)
  gW.play(12);  // W@12
  const rW = pattern1(gW, MAX_LIBS, 12);

  check(rB !== null && rW !== null && rB.key === rW.key && rB.polarity === 1 && rW.polarity === -1,
    `pattern1 color symmetry: keys ${rB && rB.key}/${rW && rW.key} polarities ${rB && rB.polarity}/${rW && rW.polarity}`);
}

// ── pattern2 ──────────────────────────────────────────────────────────────────

section('pattern2');
{
  // All-empty → null.
  const g = new Game2(5, false);
  check(pattern2(g, MAX_LIBS, 12) === null, 'pattern2: all-empty → null');
}

{
  // Single B stone at anchor TL: returns a non-null result.
  const g = new Game2(5, false);
  g.play(12);
  const r = pattern2(g, MAX_LIBS, 12);
  check(r !== null, `pattern2 single B stone at TL: non-null, got ${JSON.stringify(r)}`);
}

{
  // Rotation symmetry: a single stone anywhere in a 2×2 window should produce
  // the same canonical key regardless of which cell it occupies.
  // B@12, anchor 12: stone at TL. anchor 7: stone at BL (7→TL, 8→TR, 12→BL, 13→BR).
  const g = new Game2(5, false);
  g.play(12);
  const r12 = pattern2(g, MAX_LIBS, 12);
  const r7  = pattern2(g, MAX_LIBS, 7);
  check(r12 !== null && r7 !== null && r12.key === r7.key,
    `pattern2 rotation: anchor 12 key=${r12 && r12.key} anchor 7 key=${r7 && r7.key}`);
}

{
  // Color symmetry: B and W single stone give same key, opposite polarity.
  const gA = new Game2(5, false);
  gA.play(12);
  const rA = pattern2(gA, MAX_LIBS, 12);

  const gB = new Game2(5, false);
  gB.play(0);   // B@0 (not in the 2×2 window at anchor 12)
  gB.play(12);  // W@12
  const rB = pattern2(gB, MAX_LIBS, 12);

  // canonicalize guarantees same key and opposite polarity for color-swapped patterns.
  check(rA !== null && rB !== null && rA.key === rB.key && rA.polarity !== rB.polarity,
    `pattern2 color symmetry: keys ${rA && rA.key}/${rB && rB.key} polarities ${rA && rA.polarity}/${rB && rB.polarity}`);
}

{
  // FlipV symmetry: two B stones in top row vs bottom row of the same 2×2 window
  // should produce the same key.
  // B@12, W@0 (advances turn only), B@13.
  // anchor 12: raw=[B,B,e,e] (TL=12,TR=13).  anchor 7: raw=[e,e,B,B] (BL=12,BR=13).
  const g = new Game2(5, false);
  g.play(12);
  g.play(0);  // W@0 — not adjacent to 12 or 13
  g.play(13);
  const r12 = pattern2(g, MAX_LIBS, 12);
  const r7  = pattern2(g, MAX_LIBS, 7);
  check(r12 !== null && r7 !== null && r12.key === r7.key,
    `pattern2 FlipV symmetry: anchor 12 key=${r12 && r12.key} anchor 7 key=${r7 && r7.key}`);
}

// ── pattern3 (upper-left anchor) ─────────────────────────────────────────────

section('pattern3');
{
  // All-empty → null.
  const g = new Game2(5, false);
  check(pattern3(g, MAX_LIBS, 12) === null, 'pattern3: all-empty → null');
}

{
  // Single B stone at anchor (0,0) of window: returns a non-null result.
  const g = new Game2(5, false);
  g.play(12);
  const r = pattern3(g, MAX_LIBS, 12);
  check(r !== null, `pattern3 single stone at anchor: non-null, got ${JSON.stringify(r)}`);
}

{
  // Rot180 symmetry: stone at (0,0) of window and stone at (2,2) of window give same key.
  // On 5×5, B@12. anchor=12: stone at (0,0). anchor=0: (2,2) is idx 12.
  const g = new Game2(5, false);
  g.play(12);
  const r12 = pattern3(g, MAX_LIBS, 12);  // stone at (0,0) of window
  const r0  = pattern3(g, MAX_LIBS, 0);   // stone at (2,2) of window
  check(r12 !== null && r0 !== null && r12.key === r0.key,
    `pattern3 rot180 symmetry: anchor 12 key=${r12 && r12.key} anchor 0 key=${r0 && r0.key}`);
}

{
  // Color symmetry: B and W stones at anchor give same key, opposite polarity.
  const gB = new Game2(5, false);
  gB.play(12);
  const rB = pattern3(gB, MAX_LIBS, 12);

  const gW = new Game2(5, false);
  gW.play(0);   // B@0 — outside the window starting at anchor 12
  gW.play(12);  // W@12
  const rW = pattern3(gW, MAX_LIBS, 12);

  // canonicalize guarantees same key and opposite polarity for color-swapped patterns.
  check(rB !== null && rW !== null && rB.key === rW.key && rB.polarity !== rW.polarity,
    `pattern3 color symmetry: keys ${rB && rB.key}/${rW && rW.key} polarities ${rB && rB.polarity}/${rW && rW.polarity}`);
}

{
  // Two-stone rot180: B stones at (0,0)+(0,1) of one window match
  // B stones at (2,1)+(2,2) of another window (related by Rot180).
  // B@12, W@0 (turn-filler), B@13.
  // anchor=12: stones at c00=12 and c01=13.
  // anchor=1:  stones at c21=12 and c22=13.
  const g = new Game2(5, false);
  g.play(12);
  g.play(0);  // W@0 — outside both windows
  g.play(13);

  const r12 = pattern3(g, MAX_LIBS, 12);  // stones at (0,0) and (0,1)
  const r1  = pattern3(g, MAX_LIBS, 1);   // stones at (2,1) and (2,2)
  check(r12 !== null && r1 !== null && r12.key === r1.key,
    `pattern3 two-stone rot180: anchor 12 key=${r12 && r12.key} anchor 1 key=${r1 && r1.key}`);
}

// ── Summary ───────────────────────────────────────────────────────────────────

console.log(`\n${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
