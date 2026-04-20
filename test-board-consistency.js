#!/usr/bin/env node
'use strict';

const { Game2, BLACK, WHITE, parseBoard, PASS } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');

console.log('Testing board representation consistency (200 iterations)\n');

let passed = 0;
let failed = 0;

for (let test = 0; test < 200; test++) {
  // Generate a random 13x13 position using game2
  const game2 = new Game2(13);
  let consecutivePasses = 0;

  while (consecutivePasses < 1 && !game2.gameOver) {
    const move = game2.randomLegalMove();
    if (move !== -1) {
      game2.play(move);
      consecutivePasses = 0;
    } else {
      game2.play(-1);
      consecutivePasses++;
    }
  }

  // Call toString (original)
  const original = game2.toString(PASS);

  // Convert to game3
  const game3 = game3FromGame2(game2);

  // Call toString (converted)
  const converted = game3.toString(PASS);

  // Ensure converted == original
  if (converted !== original) {
    console.log(`✗ Test ${test + 1} FAILED: converted != original`);
    failed++;
    continue;
  }

  // Parse converted into a new game2
  const game2_parsed = parseBoard(converted, BLACK);

  // Call toString (parsed)
  const parsed = game2_parsed.toString(PASS);

  // Ensure parsed == original
  if (parsed !== original) {
    console.log(`✗ Test ${test + 1} FAILED: parsed != original`);
    failed++;
    continue;
  }

  console.log(`✓ Test ${test + 1} passed`);
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
