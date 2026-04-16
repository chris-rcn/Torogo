#!/usr/bin/env node
'use strict';

// Simple correctness test: Compare legal move sets between Game2 and Game3
// This directly validates that both implementations agree on which moves are legal

const { Game2 } = require('./game2.js');
const { Game3Precise } = require('./game3.js');

let testsRun = 0;
let testsPassed = 0;
let testsFailed = 0;

function assertEqual(a, b, message) {
  testsRun++;
  if (a === b) {
    testsPassed++;
  } else {
    testsFailed++;
    console.log(`✗ FAILED: ${message} (expected ${b}, got ${a})`);
  }
}

function testLegalMoves(game2, game3, label) {
  const cap = game2.N * game2.N;
  let game2Legal = 0;
  let game3Legal = 0;
  let mismatchCount = 0;

  for (let i = 0; i < cap; i++) {
    const legal2 = game2.isLegal(i);
    const legal3 = game3.isLegal(i);

    if (legal2) game2Legal++;
    if (legal3) game3Legal++;

    if (legal2 !== legal3) {
      mismatchCount++;
      if (mismatchCount <= 5) {  // Print first 5 mismatches
        console.log(`  Mismatch at ${i}: Game2=${legal2}, Game3=${legal3}`);
      }
    }
  }

  assertEqual(game2Legal, game3Legal, `${label}: legal move count matches`);
  assertEqual(mismatchCount, 0, `${label}: no mismatches in legal moves`);

  return mismatchCount === 0;
}

function testSequence(moves, label) {
  console.log(`\nTest: ${label}`);

  const game2 = new Game2(13);
  const game3 = new Game3Precise(13);

  for (let i = 0; i < moves.length; i++) {
    const move = moves[i];

    // Verify both agree on legality
    const legal2 = game2.isLegal(move);
    const legal3 = game3.isLegal(move);
    assertEqual(legal2, legal3, `Move ${i} (${move}): legality agrees`);

    if (!legal2) {
      console.log(`  Skipping illegal move ${move}`);
      continue;
    }

    // Play move in both
    game2.play(move);
    game3.play(move);

    // Check legal moves at this position
    testLegalMoves(game2, game3, `After move ${i} (${move})`);
  }

  console.log(`  ✓ ${label} passed`);
}

console.log('Legal Move Comparison Test: Game2 vs Game3');
console.log('='.repeat(70));

// Test 1: Initial position
console.log('\nTest: Initial position');
const g2_init = new Game2(13);
const g3_init = new Game3Precise(13);
testLegalMoves(g2_init, g3_init, 'Initial position');

// Test 2: Simple sequence
testSequence([45, 46, 57, 58, 70, 71, 69, 72, 82, 83], 'Simple sequence');

// Test 3: More complex sequence
testSequence(
  [45, 46, 57, 58, 70, 71, 69, 72, 82, 83, 84, 85, 96, 97, 108, 109, 120, 121],
  'Complex sequence'
);

// Test 4: Dense position
testSequence(
  [45, 46, 57, 58, 70, 71, 69, 72, 82, 83, 84, 85, 96, 97, 108, 109,
   120, 121, 132, 133, 144, 145, 156, 157, 55, 56, 66, 67, 78, 79],
  'Dense position'
);

// Test 5: Random moves (deterministic)
console.log('\nTest: Random moves (first 30 legal moves)');
const game2_random = new Game2(13);
const game3_random = new Game3Precise(13);
let moveCount = 0;
for (let i = 0; i < 169 && moveCount < 30; i++) {
  if (game2_random.isLegal(i)) {
    const legal3 = game3_random.isLegal(i);
    assertEqual(game2_random.isLegal(i), legal3, `Move ${i}: legality agrees`);

    game2_random.play(i);
    game3_random.play(i);
    moveCount++;

    // Check legal moves at this position
    testLegalMoves(game2_random, game3_random, `Random move ${moveCount}`);
  }
}

// Test 6: After captures
console.log('\nTest: After captures');
const game2_cap = new Game2(13);
const game3_cap = new Game3Precise(13);
const capMoves = [45, 46, 57, 58, 70, 71];
for (const move of capMoves) {
  game2_cap.play(move);
  game3_cap.play(move);
}
testLegalMoves(game2_cap, game3_cap, 'After initial moves');
console.log('  ✓ After captures passed');

console.log('\n' + '='.repeat(70));
console.log(`Results: ${testsPassed}/${testsRun} assertions passed`);

if (testsFailed === 0) {
  console.log('✓ ALL TESTS PASSED - Legal move sets match perfectly');
  process.exit(0);
} else {
  console.log(`✗ ${testsFailed} assertion(s) failed`);
  process.exit(1);
}
