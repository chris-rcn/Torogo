#!/usr/bin/env node
'use strict';

const { Game2, BLACK, WHITE, PASS, parseBoard, parseMove, coordStr } = require('./game2.js');

let failures = 0;
function check(cond, msg) {
  if (!cond) { failures++; console.error('FAIL:', msg); }
}

// ── toString / parseBoard round-trip ────────────────────────────────────────
{
  const g = parseBoard(`
    a b c d e
  5 · · · · ·
  4 · · ● · ·
  3 · ○ · · ·
  2 · · · · ·
  1 · · · · ·
  `, BLACK);

  // Round-trip without labels
  const s1 = g.toString(PASS);
  const g1 = parseBoard(s1, BLACK);
  let ok1 = true;
  for (let i = 0; i < 25; i++) if (g.cells[i] !== g1.cells[i]) { ok1 = false; break; }
  check(ok1, 'round-trip without labels preserves board');

  // Round-trip with labels
  const s2 = g.toString(PASS, { labels: true });
  const g2 = parseBoard(s2, BLACK);
  let ok2 = true;
  for (let i = 0; i < 25; i++) if (g.cells[i] !== g2.cells[i]) { ok2 = false; break; }
  check(ok2, 'round-trip with labels preserves board');
}

// ── toString row order ──────────────────────────────────────────────────────
{
  // Row N should be at the top, row 1 at the bottom
  const g = parseBoard(`
    a b c
  3 ● · ·
  2 · · ·
  1 · · ○
  `, BLACK);
  const s = g.toString(PASS, { labels: true });
  const lines = s.trim().split('\n');
  // First data line (after header) should be row 3
  check(lines[1].startsWith('3'), 'toString: top row is row N, got: ' + lines[1]);
  // Last data line should be row 1
  check(lines[3].startsWith('1'), 'toString: bottom row is row 1, got: ' + lines[3]);
}

// ── parseBoard with parenthesized cells ─────────────────────────────────────
{
  const g = parseBoard(`
    a b c d e
  5 · · · · ·
  4 · ·(·)· ·
  3 · · ● · ·
  2 · · · · ·
  1 · · · · ·
  `, BLACK);
  const N = g.N;
  // c4 should be empty (parens stripped)
  const c4 = parseMove('c4', N);
  check(g.cells[c4] === 0, 'parseBoard: parenthesized cell is empty');
  // Board should be 5x5 (parens don't break row length)
  check(N === 5, 'parseBoard: parens don\'t change board size, got N=' + N);
}

// ── parseBoard X/O notation ─────────────────────────────────────────────────
{
  const g = parseBoard(`
    a b c
  3 . X .
  2 . . .
  1 . . O
  `, BLACK);
  check(g.cells[parseMove('b3', 3)] === BLACK, 'parseBoard XO: X is BLACK');
  check(g.cells[parseMove('c1', 3)] === WHITE, 'parseBoard XO: O is WHITE');
  check(g.cells[parseMove('a1', 3)] === 0, 'parseBoard XO: . is empty');
}

// ── parseMove / coordStr round-trip ─────────────────────────────────────────
{
  check(parseMove('a1', 9) === 0, 'parseMove: a1 = 0');
  check(parseMove('b1', 9) === 1, 'parseMove: b1 = 1');
  check(parseMove('a2', 9) === 9, 'parseMove: a2 = 9');
  check(parseMove('pass', 9) === PASS, 'parseMove: pass = PASS');

  check(coordStr(0, 9) === 'a1', 'coordStr: 0 = a1');
  check(coordStr(1, 9) === 'b1', 'coordStr: 1 = b1');
  check(coordStr(9, 9) === 'a2', 'coordStr: 9 = a2');
  check(coordStr(PASS, 9) === 'pass', 'coordStr: PASS = pass');

  // Round-trip
  for (let idx = 0; idx < 81; idx++) {
    const s = coordStr(idx, 9);
    check(parseMove(s, 9) === idx, 'coordStr/parseMove round-trip: idx ' + idx);
  }
}

// ── Report ──────────────────────────────────────────────────────────────────
if (failures > 0) {
  console.error(`\n${failures} test(s) FAILED`);
  process.exit(1);
} else {
  console.log('All game2 tests passed.');
}
