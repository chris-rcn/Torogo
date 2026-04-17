#!/usr/bin/env node
'use strict';

// Test round-tripping between parseBoard and toString

const { Game3Precise, parseBoard, PASS, BLACK, WHITE, EMPTY } = require('./game3.js');

let passed = 0, failed = 0;

function assert(cond, msg) {
  if (cond) {
    passed++;
  } else {
    failed++;
    console.error(`✗ FAILED: ${msg}`);
  }
}

function normalizeBoard(boardStr) {
  return boardStr.trim().split('\n').map(l => l.trim()).filter(l => l.length > 0).join('\n');
}

console.log('Round-trip Test: parseBoard <-> toString');
console.log('='.repeat(60));

// Test 1: Empty board
console.log('\nTest 1: Empty board (5×5)');
const empty5x5 = `
  · · · · ·
  · · · · ·
  · · · · ·
  · · · · ·
  · · · · ·
`;
const g1 = parseBoard(empty5x5);
const str1 = g1.toString(PASS);
const g1_again = parseBoard(str1);
assert(g1_again.N === 5, 'Size matches');
assert(g1_again.emptyCount === 25, 'All empty');

// Test 2: Single stone
console.log('\nTest 2: Single stone');
const single = `
  · · ·
  · X ·
  · · ·
`;
const g2 = parseBoard(single);
const str2 = g2.toString(PASS);
const g2_again = parseBoard(str2);
assert(g2_again.cells[4] === BLACK, 'Stone preserved');
assert(g2_again.N === 3, 'Size preserved');

// Test 3: Multiple stones, both colors
console.log('\nTest 3: Multiple stones, both colors');
const multi = `
  X · · X
  · O O ·
  · O O ·
  X · · X
`;
const g3 = parseBoard(multi);
const str3 = g3.toString(PASS);
const g3_again = parseBoard(str3);
assert(g3_again.N === 4, 'Size preserved');
let match = true;
for (let i = 0; i < 16; i++) {
  if (g3.cells[i] !== g3_again.cells[i]) {
    match = false;
    console.log(`  Mismatch at ${i}: ${g3.cells[i]} vs ${g3_again.cells[i]}`);
  }
}
assert(match, 'All cells match');

// Test 4: Larger board (9×9) - use symmetric pattern for easier round-trip
console.log('\nTest 4: Larger board (9×9)');
const board9x9 = `
  X · O · X · O · X
  · · · · · · · · ·
  O · X · O · X · O
  · · · · · · · · ·
  X · O · X · O · X
  · · · · · · · · ·
  O · X · O · X · O
  · · · · · · · · ·
  X · O · X · O · X
`;
const g4 = parseBoard(board9x9);
const str4 = g4.toString(PASS);
const g4_again = parseBoard(str4);
assert(g4_again.N === 9, 'Size preserved');
match = true;
for (let i = 0; i < 81; i++) {
  if (g4.cells[i] !== g4_again.cells[i]) {
    match = false;
    console.log(`  Mismatch at ${i}: ${g4.cells[i]} vs ${g4_again.cells[i]}`);
  }
}
assert(match, 'All cells match');

// Test 5: Full density check (compare counts)
console.log('\nTest 5: Density preservation');
const g5 = parseBoard(board9x9);
let blackCount = 0, whiteCount = 0;
for (let i = 0; i < 81; i++) {
  if (g5.cells[i] === BLACK) blackCount++;
  if (g5.cells[i] === WHITE) whiteCount++;
}
const str5 = g5.toString(PASS);
const g5_again = parseBoard(str5);
let blackCount2 = 0, whiteCount2 = 0;
for (let i = 0; i < 81; i++) {
  if (g5_again.cells[i] === BLACK) blackCount2++;
  if (g5_again.cells[i] === WHITE) whiteCount2++;
}
assert(blackCount === blackCount2, `Black count matches (${blackCount})`);
assert(whiteCount === whiteCount2, `White count matches (${whiteCount})`);

// Test 6: Current player is independent of round-trip
console.log('\nTest 6: Current player setting');
const g6 = parseBoard(multi, BLACK);
assert(g6.current === BLACK, 'Current player set to BLACK');
const str6 = g6.toString(PASS);
const g6_white = parseBoard(str6, WHITE);
assert(g6_white.current === WHITE, 'Current player can be changed independently');

console.log('\n' + '='.repeat(60));
console.log(`Results: ${passed} passed, ${failed} failed`);

if (failed === 0) {
  console.log('✓ ALL TESTS PASSED - Round-trip is perfect');
  process.exit(0);
} else {
  console.log(`✗ ${failed} test(s) failed`);
  process.exit(1);
}
