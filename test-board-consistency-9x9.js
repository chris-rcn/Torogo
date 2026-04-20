#!/usr/bin/env node
'use strict';

const { Game2, BLACK, WHITE, parseBoard, PASS } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');

console.log('Testing round-trip board representation consistency (9x9)\n');

let passed = 0;
let failed = 0;

for (let test = 0; test < 500; test++) {
  // Generate a random position
  const g2_original = new Game2(9);
  let moveCount = 0;

  // Play random moves until game ends
  while (g2_original.consecutivePasses < 1 && !g2_original.gameOver) {
    const move = g2_original.randomLegalMove();
    g2_original.play(move);
  }

  // Step 1: Get original toString from Game2 (no mark)
  const original = g2_original.toString(PASS, { showAxes: false });

  // Step 2: Convert to Game3
  const g3 = game3FromGame2(g2_original);

  // Step 3: Get converted toString from Game3 (no mark)
  const converted = g3.toString(PASS, { showAxes: false });

  // Step 4: Check converted == original
  if (converted !== original) {
    console.log(`✗ Test ${test + 1} FAILED: Game2.toString() != Game3.toString()`);
    console.log(`  Moves played: ${moveCount}`);
    process.exit(1);
  }

  // Step 5: Parse the converted string back to Game2
  const g2_parsed = parseBoard(converted, g3.current);

  // Step 6: Get parsed toString from Game2 (no mark)
  const parsed = g2_parsed.toString(PASS, { showAxes: false });

  // Step 7: Check parsed == original
  if (parsed !== original) {
    console.log(`✗ Test ${test + 1} FAILED: parseBoard() produced different board`);
    console.log(`  Moves played: ${moveCount}`);
    console.log(`\nOriginal:\n${original}\n`);
    console.log(`Parsed:\n${parsed}\n`);
    process.exit(1);
  }

  console.log(`✓ Test ${test + 1} passed (${moveCount} moves)`);
  passed++;
}

console.log(`\n${'='.repeat(50)}`);
console.log(`Results: ${passed} passed, ${failed} failed`);
console.log('All tests passed!');
