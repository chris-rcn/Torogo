#!/usr/bin/env node
'use strict';

const { Game2, BLACK, WHITE, parseBoard, PASS } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');

console.log('Testing round-trip board representation consistency\n');

let passed = 0;
let failed = 0;

for (let test = 0; test < 200; test++) {
  // Generate a random position
  const g2_original = new Game2(13);
  let consecutivePasses = 0;
  let moveCount = 0;

  // Play random moves until game ends
  while (consecutivePasses < 1 && !g2_original.gameOver && moveCount < 50) {
    const move = g2_original.randomLegalMove();
    if (move !== -1) {
      g2_original.play(move);
      consecutivePasses = 0;
      moveCount++;
    } else {
      g2_original.play(-1);
      consecutivePasses++;
      moveCount++;
    }
  }

  // Step 1: Get original toString from Game2 (no mark)
  const original = g2_original.toString(PASS);

  // Step 2: Convert to Game3
  const g3 = game3FromGame2(g2_original);

  // Step 3: Get converted toString from Game3 (no mark)
  const converted = g3.toString(PASS);

  // Step 4: Check converted == original
  if (converted !== original) {
    console.log(`✗ Test ${test + 1} FAILED: Game2.toString() != Game3.toString()`);
    console.log(`  Moves played: ${moveCount}`);
    failed++;
    continue;
  }

  // Step 5: Parse the converted string back to Game2
  const g2_parsed = parseBoard(converted, BLACK);

  // Step 6: Get parsed toString from Game2 (no mark)
  const parsed = g2_parsed.toString(PASS);

  // Step 7: Check parsed == original
  if (parsed !== original) {
    console.log(`✗ Test ${test + 1} FAILED: parseBoard() produced different board`);
    console.log(`  Moves played: ${moveCount}`);
    console.log(`\nOriginal:\n${original}\n`);
    console.log(`Parsed:\n${parsed}\n`);
    failed++;
    continue;
  }

  console.log(`✓ Test ${test + 1} passed (${moveCount} moves)`);
  passed++;
}

console.log(`\n${'='.repeat(50)}`);
console.log(`Results: ${passed} passed, ${failed} failed`);

if (failed > 0) {
  process.exit(1);
} else {
  console.log('All tests passed!');
  process.exit(0);
}
