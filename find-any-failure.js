#!/usr/bin/env node
'use strict';

const { Game2, BLACK, WHITE, parseBoard, PASS } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');

// Find first failure
for (let run = 0; run < 100; run++) {
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
      console.log(`FAILURE FOUND: Run ${run}, Test ${test + 1}\n`);

      // Find which cells differ
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

      console.log('\nCell differences:');
      let diffs = 0;
      for (let i = 0; i < 169; i++) {
        if (game2.cells[i] !== game3.cells[i]) {
          const x = i % 13;
          const y = (i / 13) | 0;
          const g2color = game2.cells[i] === 1 ? 'BLACK' : game2.cells[i] === -1 ? 'WHITE' : 'EMPTY';
          const g3color = game3.cells[i] === 1 ? 'BLACK' : game3.cells[i] === -1 ? 'WHITE' : 'EMPTY';
          console.log(`  Cell ${i} (${x},${y}): Game2=${g2color}, Game3=${g3color}`);
          diffs++;
          if (diffs > 10) {
            console.log(`  ... and ${diffs - 10} more`);
            break;
          }
        }
      }

      console.log('\nFailing position:');
      console.log(original);

      console.log('\nGame2 state:');
      console.log('  moveCount:', game2.moveCount);
      console.log('  current:', game2.current);
      console.log('  gameOver:', game2.gameOver);
      console.log('  consecutivePasses:', game2.consecutivePasses);

      process.exit(1);
    }
  }
  console.log(`Run ${run} complete - no failures`);
}

console.log('All 20000 tests passed!');
