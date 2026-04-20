#!/usr/bin/env node
'use strict';

const { Game2, BLACK, WHITE, parseBoard } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');

// Reproduce test 37 - run tests until we find test 37
let testNum = 0;
for (let test = 0; test <= 100; test++) {
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

  testNum++;

  if (testNum === 37) {
    console.log('Found test 37');
    const original = g2_original.toString();
    console.log('\nGame2.toString():');
    console.log(original);
    console.log('\n' + '='.repeat(50));

    const g3 = game3FromGame2(g2_original);
    const converted = g3.toString();
    console.log('\nGame3.toString():');
    console.log(converted);
    console.log('\n' + '='.repeat(50));

    if (converted !== original) {
      console.log('\nMISMATCH FOUND');
      const origLines = original.split('\n');
      const convLines = converted.split('\n');

      for (let i = 0; i < Math.max(origLines.length, convLines.length); i++) {
        const o = origLines[i] || '(missing)';
        const c = convLines[i] || '(missing)';
        if (o !== c) {
          console.log(`Line ${i} differs:`);
          console.log(`  Original: ${o}`);
          console.log(`  Converted: ${c}`);
        }
      }
    } else {
      console.log('\nMatch!');
    }
    break;
  }
}
