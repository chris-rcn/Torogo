#!/usr/bin/env node
'use strict';

const { Game2, BLACK, WHITE, PASS } = require('./game2.js');
const { Game3 } = require('./game3.js');

console.log('Testing parallel gameplay: game2 vs game3 (9x9)\n');

// Xorshift32 RNG for deterministic reproducible randomness
class XorShift32 {
  constructor(seed = 12345) {
    this.state = seed >>> 0;
  }
  random() {
    this.state ^= this.state << 13;
    this.state ^= this.state >> 17;
    this.state ^= this.state << 5;
    return (this.state >>> 0) / 0x100000000;
  }
}

let passed = 0;
let failed = 0;

for (let gameNum = 1; gameNum <= 1000; gameNum++) {
  const rng = new XorShift32(gameNum);

  // Create both game instances with no initial center stone
  const g2 = new Game2(9, false);
  const g3 = new Game3(9);

  let moveCount = 0;

  // Play identical moves on both games
  while (!g2.gameOver && !g3.gameOver) {
    // Get a legal move from game2
    const move = g2.randomLegalMove(rng);

    // Play on both games
    const g2Result = g2.play(move);
    const g3Result = g3.play(move);

    if (!g2Result || !g3Result) {
      console.log(`✗ Game ${gameNum} FAILED: Move ${moveCount} (${move === PASS ? 'PASS' : move}) failed`);
      console.log(`  Game2 result: ${g2Result}, Game3 result: ${g3Result}`);
      console.log(`\nGame2 board:`);
      console.log(g2.toString(PASS));
      console.log(`\nGame3 board:`);
      console.log(g3.toString(PASS));
      process.exit(1);
    }

    moveCount++;
  }

  // Compare final board states
  const g2Board = g2.toString(PASS);
  const g3Board = g3.toString(PASS);

  if (g2Board !== g3Board) {
    console.log(`✗ Game ${gameNum} FAILED: Final board states don't match`);
    console.log(`  Moves played: ${moveCount}`);
    console.log(`\nGame2 board:\n${g2Board}\n`);
    console.log(`Game3 board:\n${g3Board}\n`);
    process.exit(1);
  }

  console.log(`✓ Game ${gameNum} passed (${moveCount} moves)`);
  passed++;
}

console.log(`\n${'='.repeat(50)}`);
console.log(`Results: ${passed} passed, ${failed} failed`);
console.log('All games matched!');
