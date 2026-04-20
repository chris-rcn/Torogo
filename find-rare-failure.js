#!/usr/bin/env node
'use strict';

const { Game2, BLACK, WHITE, parseBoard, PASS } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');

// Find first failure
for (let run = 0; run < 50; run++) {
  for (let test = 0; test < 200; test++) {
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

    const original = game2.toString(PASS);
    const game3 = game3FromGame2(game2);
    const converted = game3.toString(PASS);

    if (converted !== original) {
      console.log(`FAILURE: Run ${run}, Test ${test + 1}\n`);

      const oLines = original.split('\n');
      const cLines = converted.split('\n');

      console.log('Line differences:');
      for (let i = 0; i < oLines.length; i++) {
        if (oLines[i] !== cLines[i]) {
          console.log(`Line ${i}:`);
          console.log(`  Game2: ${oLines[i]}`);
          console.log(`  Game3: ${cLines[i]}`);
        }
      }

      process.exit(1);
    }
  }
  console.log(`Run ${run} complete`);
}

console.log('All tests passed!');
