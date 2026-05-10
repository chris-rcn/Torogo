#!/usr/bin/env node
'use strict';

const { Game2, BLACK, WHITE, PASS, parseBoard, parseMove } = require('./game2.js');
const {
  createLadderPat, softmax, normalizeKey,
  _patSalts, _ladderKeys, _captureKeys, _distKeys, _selfAtariKeys, _chainSizeKeys,
  MAX_LADDER_LEVELS, MAX_CAPTURE_FEAT, MAX_PREV_DIST, MAX_SELF_ATARI, MAX_CHAIN_SIZE,
  TYPE_KILL_URGENT, TYPE_SAVE_URGENT, TYPE_KILL_WASTED, TYPE_SAVE_WASTED,
} = require('./ladder-pat-lib.js');

let failures = 0;
function check(cond, msg) {
  if (!cond) { failures++; console.error('FAIL:', msg); }
}

// Helper: build key sets for identification.
const ladderKeySet    = new Set(_ladderKeys);
const captureKeySet   = new Set(_captureKeys);
const distKeySet      = new Set(_distKeys);
const selfAtariKeySet = new Set(_selfAtariKeys);
const chainSizeKeySet = new Set(_chainSizeKeys);

function classifyKeys(keys) {
  let pat = 0, ladder = 0, capture = 0, dist = 0, selfAtari = 0, chainSize = 0;
  for (const k of keys) {
    if (ladderKeySet.has(k)) ladder++;
    else if (captureKeySet.has(k)) capture++;
    else if (distKeySet.has(k)) dist++;
    else if (selfAtariKeySet.has(k)) selfAtari++;
    else if (chainSizeKeySet.has(k)) chainSize++;
    else pat++;
  }
  return { pat, ladder, capture, dist, selfAtari, chainSize };
}

function getCandidate(candidates, coord, N) {
  const idx = parseMove(coord, N);
  return candidates.find(c => c.move === idx);
}

// ── Pattern features ────────────────────────────────────────────────────────
{
  const lp = createLadderPat({ maxAdjLibs: 1, maxLadderStones: 0 });
  const g = new Game2(9, false);
  const c = lp.getFeatures(g);
  check(c.length > 0, 'pattern: empty board has candidates');
  // maxAdjLibs=1: each candidate gets 1 pattern key
  const cls = classifyKeys(c[0].keys);
  check(cls.pat === 1, 'pattern: maxAdjLibs=1 emits 1 pattern key');

  // All candidates on empty board (no prev move) should have same pattern
  const patKey = c[0].keys[0];
  let allSame = true;
  for (const cand of c) if (cand.keys[0] !== patKey) { allSame = false; break; }
  check(allSame, 'pattern: empty board all candidates have same pattern key');
}
{
  // maxAdjLibs=2: each candidate gets 2 pattern keys
  const lp = createLadderPat({ maxAdjLibs: 2, maxLadderStones: 0 });
  const g = new Game2(9, false);
  const c = lp.getFeatures(g);
  const cls = classifyKeys(c[0].keys);
  check(cls.pat === 2, 'pattern: maxAdjLibs=2 emits 2 pattern keys');
}
{
  // D4 symmetry: rotated positions produce same pattern key
  const lp = createLadderPat({ maxAdjLibs: 1, maxLadderStones: 0 });
  // Place one black stone; candidates at symmetric positions should share keys
  const g = parseBoard(`
    a b c d e
  5 · · · · ·
  4 · · · · ·
  3 · · ● · ·
  2 · · · · ·
  1 · · · · ·
  `, WHITE);
  g.lastMove = parseMove('c3', 5); // set prev so it doesn't affect symmetry test
  const c = lp.getFeatures(g);
  const N = g.N;
  // b3 and d3 are horizontal reflections of each other around c3
  const b3 = getCandidate(c, 'b3', N);
  const d3 = getCandidate(c, 'd3', N);
  check(b3 && d3, 'pattern D4: both b3 and d3 are candidates');
  if (b3 && d3) {
    check(b3.keys[0] === d3.keys[0], 'pattern D4: b3 and d3 have same pattern key');
  }
  // c2 and c4 are vertical reflections
  const c2 = getCandidate(c, 'c2', N);
  const c4 = getCandidate(c, 'c4', N);
  check(c2 && c4, 'pattern D4: both c2 and c4 are candidates');
  if (c2 && c4) {
    check(c2.keys[0] === c4.keys[0], 'pattern D4: c2 and c4 have same pattern key');
  }
}

// ── Capture features ────────────────────────────────────────────────────────
{
  // Black to play; d1 captures one white group in atari
  const lp = createLadderPat({ maxAdjLibs: 1, maxLadderStones: 0 });
  const g = parseBoard(`
    a b c d e
  5 · · · · ·
  4 · · · · ·
  3 · · · · ·
  2 · · ● ● ·
  1 · · ● ·(○)
  `, BLACK);
  // e1 is white with libs at d1 only (● at c1 and ● at e2... wait, no e2 is empty)
  // Actually let me set up a clearer capture: white at d1, black at c1,d2,e1
  const g2 = parseBoard(`
    a b c d e
  5 · · · · ·
  4 · · · · ·
  3 · · · · ·
  2 · · · ● ·
  1 · · ● ○ ●
  `, BLACK);
  // White d1 has libs: only... on toroidal board d1 neighbors are c1(●),e1(●),d2(●),d5(empty)
  // d5 is empty via wraparound, so white has 1 lib at d5. Not capturable from other cells.
  // Let me use a non-toroidal-confusing setup on bigger board.
  const g3 = parseBoard(`
    a b c d e f g
  7 · · · · · · ·
  6 · · · · · · ·
  5 · · · · · · ·
  4 · · · ● · · ·
  3 · · ● ○ ● · ·
  2 · · · ● · · ·
  1 · · · · · · ·
  `, BLACK);
  // d3 is white, neighbors: c3(●),e3(●),d4(●),d2(●) — all black. 0 libs? That's suicide for white.
  // Need to give white a liberty. Let me try:
  const g4 = parseBoard(`
    a b c d e f g
  7 · · · · · · ·
  6 · · · · · · ·
  5 · · · · · · ·
  4 · · · · · · ·
  3 · · ● ○ · · ·
  2 · · · ● · · ·
  1 · · · · · · ·
  `, BLACK);
  // White d3 has neighbors: c3(●), e3(empty), d4(empty), d2(●). Libs: e3, d4. Not in atari.
  // I need white in atari with exactly 1 lib. Let's surround more:
  const g5 = parseBoard(`
    a b c d e f g
  7 · · · · · · ·
  6 · · · · · · ·
  5 · · · · · · ·
  4 · · · ● · · ·
  3 · · ● ○ · · ·
  2 · · · ● · · ·
  1 · · · · · · ·
  `, BLACK);
  // d3 white: neighbors c3(●),e3(empty),d4(●),d2(●). 1 lib at e3.
  // Playing e3 captures d3. e3 should have capture=1.
  const c5 = lp.getFeatures(g5);
  const N5 = g5.N;
  const e3 = getCandidate(c5, 'e3', N5);
  check(e3 !== undefined, 'capture: e3 is a candidate');
  if (e3) {
    const cls = classifyKeys(e3.keys);
    check(cls.capture >= 1, 'capture: e3 captures 1 group (capture >= 1)');
  }
  // A non-capturing move should have 0 capture keys
  const a1 = getCandidate(c5, 'a1', N5);
  if (a1) {
    const cls = classifyKeys(a1.keys);
    check(cls.capture === 0, 'capture: a1 captures nothing (capture === 0)');
  }
}
{
  // Capture 2 distinct groups
  const lp = createLadderPat({ maxAdjLibs: 1, maxLadderStones: 0 });
  const g = parseBoard(`
    a b c d e f g
  7 · · · · · · ·
  6 · · · · · · ·
  5 · · · · · · ·
  4 · · · ● · · ·
  3 · · ● ○ ● · ·
  2 · · · · · · ·
  1 · · · · · · ·
  `, BLACK);
  // d3 white: neighbors c3(●),e3(●),d4(●),d2(empty). 1 lib at d2.
  // Now add another white group:
  const g2 = parseBoard(`
    a b c d e f g
  7 · · · · · · ·
  6 · · · · · · ·
  5 · · · ● · · ·
  4 · · ● ○ ● · ·
  3 · · ● · ● · ·
  2 · · · ● · · ·
  1 · · · · · · ·
  `, BLACK);
  // d4 white: c4(●),e4(●),d5(●),d3(empty). 1 lib at d3.
  // Playing d3 captures d4. But also need another group...
  // Two separate white groups both with 1 lib at d3:
  const g3 = parseBoard(`
    a b c d e f g
  7 · · · · · · ·
  6 · · · · · · ·
  5 · · · ● · · ·
  4 · · ● ○ ● · ·
  3 · · · · · · ·
  2 · · ● ○ ● · ·
  1 · · · ● · · ·
  `, BLACK);
  // d4: c4(●),e4(●),d5(●),d3(empty) → 1 lib at d3
  // d2: c2(●),e2(●),d1(●),d3(empty) → 1 lib at d3
  // Playing d3 captures both!
  const c3 = lp.getFeatures(g3);
  const N3 = g3.N;
  const d3 = getCandidate(c3, 'd3', N3);
  check(d3 !== undefined, 'capture2: d3 is a candidate');
  if (d3) {
    const cls = classifyKeys(d3.keys);
    check(cls.capture >= 2, 'capture2: d3 captures 2 groups (capture >= 2)');
  }
}

// ── Distance to previous move ───────────────────────────────────────────────
{
  const lp = createLadderPat({ maxAdjLibs: 1, maxLadderStones: 0 });
  const g = new Game2(9, false);
  g.play(40); // black plays center (e5)
  g.play(0);  // white plays a1
  // Now black to play, lastMove=0 (a1)
  const c = lp.getFeatures(g);
  const N = g.N;

  // b1 (idx=1): dist=1 from a1 → MAX_PREV_DIST dist keys
  const b1 = getCandidate(c, 'b1', N);
  check(b1 !== undefined, 'dist: b1 is a candidate');
  if (b1) {
    const cls = classifyKeys(b1.keys);
    check(cls.dist === MAX_PREV_DIST, 'dist: dist=1 emits MAX_PREV_DIST keys (' + MAX_PREV_DIST + '), got ' + cls.dist);
  }

  // e5 was played by black, not a candidate. Pick something far.
  // e1 (idx=4): dist=4 from a1
  const e1 = getCandidate(c, 'e1', N);
  if (e1) {
    const cls = classifyKeys(e1.keys);
    check(cls.dist === MAX_PREV_DIST - 3, 'dist: dist=4 emits ' + (MAX_PREV_DIST - 3) + ' keys, got ' + cls.dist);
  }
}
{
  // No previous move → no distance features
  const lp = createLadderPat({ maxAdjLibs: 1, maxLadderStones: 0 });
  const g = new Game2(9, false);
  const c = lp.getFeatures(g);
  const cls = classifyKeys(c[0].keys);
  check(cls.dist === 0, 'dist: no prev move → 0 dist keys');
}
{
  // Toroidal wrapping: on 9×9, a1 (idx=0) to i1 (idx=8) should be dist=1
  const lp = createLadderPat({ maxAdjLibs: 1, maxLadderStones: 0 });
  const g = new Game2(9, false);
  g.play(0);  // black a1
  g.play(40); // white e5
  // lastMove=40, black to play
  // Actually let's test wrapping specifically:
  const g2 = new Game2(9, false);
  g2.play(4);  // black e1
  g2.play(8);  // white i1, lastMove=8
  // Now black to play. a1 (idx=0): dx=|0-8|=8, min(8,9-8)=1. dy=0. dist=1.
  const c = lp.getFeatures(g2);
  const a1 = getCandidate(c, 'a1', 9);
  if (a1) {
    const cls = classifyKeys(a1.keys);
    check(cls.dist === MAX_PREV_DIST, 'dist toroidal: a1 to i1 wraps to dist=1, got ' + cls.dist + ' dist keys');
  }
}

// ── Self-atari features ─────────────────────────────────────────────────────
{
  // Simple self-atari: single stone, 1 empty neighbor, surrounded by foe
  const lp = createLadderPat({ maxAdjLibs: 1, maxLadderStones: 0 });
  const g = parseBoard(`
    a b c d e
  5 · · · · ·
  4 · · ○ · ·
  3 · ○ · ○ ·
  2 · · ○ · ·
  1 · · · · ·
  `, BLACK);
  // c3 empty, neighbors: b3(○),d3(○),c4(○),c2(○). 0 empty nbrs.
  // Actually that's suicide, not self-atari. It would be filtered as illegal.
  // Need 1 empty neighbor:
  const g2 = parseBoard(`
    a b c d e f g
  7 · · · · · · ·
  6 · · · · · · ·
  5 · · · · · · ·
  4 · · · ○ · · ·
  3 · · ○ · · · ·
  2 · · · ○ · · ·
  1 · · · · · · ·
  `, BLACK);
  // d3 empty: neighbors c3(○),e3(empty),d4(○),d2(○). emptyNbr=1, no friendly groups, no captures.
  // → cheap positive: self-atari size 1
  const c2 = lp.getFeatures(g2);
  const d3 = getCandidate(c2, 'd3', g2.N);
  check(d3 !== undefined, 'self-atari: d3 is a candidate');
  if (d3) {
    const cls = classifyKeys(d3.keys);
    check(cls.selfAtari === 1, 'self-atari: d3 is self-atari size 1, got ' + cls.selfAtari);
  }
}
{
  // Self-atari with friendly group: extending into atari
  const lp = createLadderPat({ maxAdjLibs: 1, maxLadderStones: 0 });
  const g = parseBoard(`
    a b c d e f g
  7 · · · · · · ·
  6 · · · · · · ·
  5 · · · · · · ·
  4 · · · ○ · · ·
  3 · · ○ ● · · ·
  2 · · · ○ · · ·
  1 · · · · · · ·
  `, BLACK);
  // d3=●(1 lib at e3). c3=○, d4=○, d2=○.
  // Playing e3: neighbors d3(●,1lib), f3(empty), e4(empty), e2(empty).
  // emptyNbr=3 → cheap negative, not self-atari. Correct (e3 has 3 empty nbrs).

  // Self-atari size 2: black at e3 with 1 lib (d3). Playing d3 merges with e3.
  // d3 has 1 empty neighbor (c3) so merged group {d3,e3} has 1 lib → self-atari size 2.
  const g2 = parseBoard(`
    a b c d e f g
  7 · · · · · · ·
  6 · · · · · · ·
  5 · · · · · · ·
  4 · · · ○ ○ · ·
  3 · · · · ● ○ ·
  2 · · · ○ ○ · ·
  1 · · · · · · ·
  `, BLACK);
  // e3=● neighbors: d3(empty),f3(○),e4(○),e2(○). 1 lib at d3.
  // d3 empty, neighbors: c3(empty),e3(●),d4(○),d2(○). emptyNbr=1 (c3), friend e3 has 1 lib.
  // → cheap positive self-atari. merged {d3,e3} has lib c3 only. Size=2.
  const c2 = lp.getFeatures(g2);
  const d3 = getCandidate(c2, 'd3', g2.N);
  check(d3 !== undefined, 'self-atari merge: d3 is a candidate');
  if (d3) {
    const cls = classifyKeys(d3.keys);
    check(cls.selfAtari === 2, 'self-atari merge: d3 is self-atari size 2, got ' + cls.selfAtari);
  }
}
{
  // Not self-atari: move has 2 empty neighbors
  const lp = createLadderPat({ maxAdjLibs: 1, maxLadderStones: 0 });
  const g = parseBoard(`
    a b c d e f g
  7 · · · · · · ·
  6 · · · · · · ·
  5 · · · · · · ·
  4 · · · · · · ·
  3 · · ○ · · · ·
  2 · · · ○ · · ·
  1 · · · · · · ·
  `, BLACK);
  // d3: neighbors c3(○),e3(empty),d4(empty),d2(○). emptyNbr=2 → not self-atari
  const c = lp.getFeatures(g);
  const d3 = getCandidate(c, 'd3', g.N);
  if (d3) {
    const cls = classifyKeys(d3.keys);
    check(cls.selfAtari === 0, 'not self-atari: 2 empty nbrs → 0 self-atari keys, got ' + cls.selfAtari);
  }
}
{
  // Not self-atari: adjacent friendly group has ≥3 libs
  const lp = createLadderPat({ maxAdjLibs: 1, maxLadderStones: 0 });
  const g = parseBoard(`
    a b c d e f g
  7 · · · · · · ·
  6 · · · · · · ·
  5 · · · · · · ·
  4 · · ○ ● · · ·
  3 · · ○ · ○ · ·
  2 · · · ○ · · ·
  1 · · · · · · ·
  `, BLACK);
  // d4=● has libs at d5,e4,d3(this cell). 3 libs.
  // d3: emptyNbr=0 (c3=○,e3=○,d4=●,d2=○), but maxFriendLibs=3 → not self-atari
  const c = lp.getFeatures(g);
  const d3 = getCandidate(c, 'd3', g.N);
  if (d3) {
    const cls = classifyKeys(d3.keys);
    check(cls.selfAtari === 0, 'not self-atari: friend has 3 libs → 0 self-atari keys, got ' + cls.selfAtari);
  }
}

// ── Ladder features ─────────────────────────────────────────────────────────
{
  // Basic ladder: white in atari, black can capture
  const lp = createLadderPat({ maxAdjLibs: 1, maxLadderStones: 2 });
  const g = parseBoard(`
    a b c d e f g
  7 · · · · · · ·
  6 · ● · · · · ·
  5 · · · · · · ·
  4 · · · · · ● ·
  3 · · ·(·)○ ○ ●
  2 · · · · ● ○ ●
  1 · · · · · ● ●
  `, BLACK);
  const c = lp.getFeatures(g);
  const N = g.N;
  // At least one candidate should have ladder keys
  let hasLadder = false;
  for (const cand of c) {
    const cls = classifyKeys(cand.keys);
    if (cls.ladder > 0) { hasLadder = true; break; }
  }
  check(hasLadder, 'ladder: at least one candidate has ladder keys');

  // maxLadderStones=0: no ladder features at all
  const lp0 = createLadderPat({ maxAdjLibs: 1, maxLadderStones: 0 });
  const c0 = lp0.getFeatures(g);
  let anyLadder = false;
  for (const cand of c0) {
    const cls = classifyKeys(cand.keys);
    if (cls.ladder > 0) { anyLadder = true; break; }
  }
  check(!anyLadder, 'ladder: maxLadderStones=0 → no ladder keys');
}

// ── softmax ─────────────────────────────────────────────────────────────────
{
  const candidates = [
    { move: 0, keys: [1, 2] },
    { move: 1, keys: [3] },
  ];
  const { vals, sum, max } = softmax(candidates, k => k === 1 ? 10 : 0);
  check(vals[0] / sum > 0.99, 'softmax: high-weight candidate gets >99% prob, got ' + (vals[0]/sum*100).toFixed(1) + '%');
  check(Math.abs(vals[0] + vals[1] - sum) < 1e-10, 'softmax: vals sum to reported sum');
  check(max === 10, 'softmax: max is correct');
}

// ── Resulting chain size ─────────────────────────────────────────────────────
{
  // Isolated stone: chain size = 1
  const lp = createLadderPat({ maxAdjLibs: 1, maxLadderStones: 0 });
  const g = new Game2(9, false);
  const c = lp.getFeatures(g);
  const cls = classifyKeys(c[0].keys);
  check(cls.chainSize === 1, 'chain size: isolated stone on empty board = 1, got ' + cls.chainSize);
}
{
  // Connect to 1 friendly stone: chain size = 2
  const lp = createLadderPat({ maxAdjLibs: 1, maxLadderStones: 0 });
  const g = parseBoard(`
    a b c d e f g
  7 · · · · · · ·
  6 · · · · · · ·
  5 · · · · · · ·
  4 · · · · · · ·
  3 · · · ● · · ·
  2 · · · · · · ·
  1 · · · · · · ·
  `, BLACK);
  const c = lp.getFeatures(g);
  const N = g.N;
  // d4: adjacent to d3(●). chain size = 1 + 1 = 2
  const d4 = getCandidate(c, 'd4', N);
  check(d4 !== undefined, 'chain size 2: d4 is a candidate');
  if (d4) {
    const cls = classifyKeys(d4.keys);
    check(cls.chainSize === 2, 'chain size 2: connecting to 1 stone = 2, got ' + cls.chainSize);
  }
  // e4: not adjacent to any friendly stone. chain size = 1
  const e4 = getCandidate(c, 'e4', N);
  if (e4) {
    const cls = classifyKeys(e4.keys);
    check(cls.chainSize === 1, 'chain size 1: no friendly neighbors = 1, got ' + cls.chainSize);
  }
}
{
  // Connect to 2 distinct friendly groups: chain size = 1 + 2 + 1 = 4
  const lp = createLadderPat({ maxAdjLibs: 1, maxLadderStones: 0 });
  const g = parseBoard(`
    a b c d e f g
  7 · · · · · · ·
  6 · · · · · · ·
  5 · · · · · · ·
  4 · · · ● · · ·
  3 · · · · · · ·
  2 · · ● ● · · ·
  1 · · · · · · ·
  `, BLACK);
  const c = lp.getFeatures(g);
  const N = g.N;
  // d3: neighbors d4(● size 1), d2(● part of {c2,d2} size 2), c3(empty), e3(empty).
  // chain size = 1 + 1 + 2 = 4
  const d3 = getCandidate(c, 'd3', N);
  check(d3 !== undefined, 'chain size 4: d3 is a candidate');
  if (d3) {
    const cls = classifyKeys(d3.keys);
    check(cls.chainSize === 4, 'chain size 4: connecting 2 groups = 4, got ' + cls.chainSize);
  }
}

// ── Board size independence ─────────────────────────────────────────────────
{
  // Same extractor works for different board sizes
  const lp = createLadderPat({ maxAdjLibs: 1, maxLadderStones: 0 });
  const g7 = new Game2(7, false);
  const g9 = new Game2(9, false);
  const c7 = lp.getFeatures(g7);
  const c9 = lp.getFeatures(g9);
  check(c7.length > 0 && c9.length > 0, 'board size: works for both 7x7 and 9x9');
  check(c7.length < c9.length, 'board size: 9x9 has more candidates than 7x7');
  // Same pattern on empty board regardless of size
  check(c7[0].keys[0] === c9[0].keys[0], 'board size: empty board pattern key is same for 7x7 and 9x9');
}

// ── Report ──────────────────────────────────────────────────────────────────
if (failures > 0) {
  console.error(`\n${failures} test(s) FAILED`);
  process.exit(1);
} else {
  console.log(`All ladder-pat-lib tests passed.`);
}
