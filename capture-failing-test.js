#!/usr/bin/env node
'use strict';

const { Game2, BLACK, WHITE, parseBoard, PASS } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');

// Run until we find a failure
let attempts = 0;
while (attempts < 100) {
  for (let test = 0; test < 100; test++) {
    const g2_original = new Game2(13);
    let consecutivePasses = 0;
    let moveCount = 0;

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

    const original = g2_original.toString(PASS);
    const g3 = game3FromGame2(g2_original);
    const converted = g3.toString(PASS);

    if (converted !== original) {
      console.log(`Found failure in test ${test + 1}, attempt ${attempts}`);
      console.log(`Moves played: ${moveCount}\n`);

      const oLines = original.split('\n');
      const cLines = converted.split('\n');

      for (let i = 0; i < oLines.length; i++) {
        if (oLines[i] !== cLines[i]) {
          console.log(`Line ${i} differs:`);
          console.log(`  Game2: "${oLines[i]}"`);
          console.log(`  Game3: "${cLines[i]}"`);
        }
      }

      console.log('\nGame2 board:');
      console.log(original);
      console.log('\nGame3 board:');
      console.log(converted);

      process.exit(1);
    }
  }
  attempts++;
  console.log(`Completed attempt ${attempts}...`);
}

console.log('No failures found after 10000 tests');
