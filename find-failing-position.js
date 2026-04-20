#!/usr/bin/env node
'use strict';

const { Game2, BLACK, WHITE, parseBoard } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');

for (let run = 0; run < 10; run++) {
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

    const original = g2_original.toString();
    const g3 = game3FromGame2(g2_original);
    const converted = g3.toString();

    if (converted !== original) {
      console.log(`\nFailing position found in run ${run}, test ${test + 1}`);
      console.log(`Moves played: ${moveCount}`);

      const origLines = original.split('\n');
      const convLines = converted.split('\n');

      let firstDiff = -1;
      for (let i = 0; i < Math.max(origLines.length, convLines.length); i++) {
        if ((origLines[i] || '') !== (convLines[i] || '')) {
          firstDiff = i;
          break;
        }
      }

      if (firstDiff >= 0) {
        console.log(`First difference at line ${firstDiff}:`);
        console.log(`  Original: ${origLines[firstDiff]}`);
        console.log(`  Converted: ${convLines[firstDiff]}`);
      }

      // Save the position
      const saveStr = original.split('\n').map((l, i) => {
        const y = 12 - i;  // Convert display row back to board row
        const m = l.match(/\((.)\)/);
        return { row: y, content: l, marked: m ? m[1] : null };
      });

      console.log('\nFull position:');
      console.log(original);

      process.exit(1);
    }
  }
}

console.log('No failures found after 1000 tests');
