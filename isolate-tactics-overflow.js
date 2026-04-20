#!/usr/bin/env node
'use strict';

// Isolate the exact board position that causes overflow

const { Game2 } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');
const { searchChains } = require('./tactics3.js');

console.log('Isolating overflow position...\n');

const fs = require('fs');

let foundOverflow = false;
let gameNum = 0;

while (!foundOverflow && gameNum < 20) {
  const g2 = new Game2(13);
  let moveNum = 0;

  while (moveNum < 50 && !g2.gameOver && !foundOverflow) {
    for (let i = 0; i < 169 && !foundOverflow; i++) {
      if (g2.isLegal(i)) {
        try {
          const g3 = game3FromGame2(g2);
          const startTime = Date.now();
          searchChains(g3);  // NO nodeLimit - let it overflow if it will
          const elapsed = Date.now() - startTime;

          if (elapsed > 100) {
            console.log(`Game ${gameNum} move ${moveNum}: searchChains took ${elapsed}ms (at risk)`);
          }

          g2.play(i);
          moveNum++;
        } catch (e) {
          if (e instanceof RangeError && e.message.includes('stack')) {
            console.log('\n✗ STACK OVERFLOW FOUND');
            console.log(`Game ${gameNum}, Move ${moveNum}`);

            // Reconstruct the exact state before the overflow
            const g2Clean = new Game2(13);
            let moveSequence = [];

            // Replay the game up to the current point
            const g2Test = new Game2(13);
            for (let j = 0; j < 169; j++) {
              if (g2Test.isLegal(j)) {
                if (g2Test.cells[j] === g2.cells[j]) {
                  g2Test.play(j);
                  moveSequence.push(j);
                  if (moveSequence.length === moveNum) break;
                }
              }
            }

            // Now at the failing state
            const g3 = game3FromGame2(g2);

            console.log('\nBoard visualization:');
            for (let y = 0; y < 13; y++) {
              let row = '';
              for (let x = 0; x < 13; x++) {
                const idx = y * 13 + x;
                const c = g3.cells[idx];
                if (c === 1) row += '●';
                else if (c === -1) row += '○';
                else row += '·';
              }
              console.log(row);
            }

            console.log('\nGame state:');
            console.log('- Board size: 13x13');
            console.log('- Move count:', g3.moveCount);
            console.log('- Current player:', g3.current === 1 ? 'BLACK' : 'WHITE');
            console.log('- Ko:', g3.ko);
            console.log('- Consecutive passes:', g3.consecutivePasses);

            // Save the exact board state
            const overflowData = {
              gameNum,
              moveNum,
              cells: Array.from(g3.cells),
              N: 13,
              current: g3.current,
              ko: g3.ko,
              moveCount: g3.moveCount,
              consecutivePasses: g3.consecutivePasses
            };

            fs.writeFileSync('overflow-board.json', JSON.stringify(overflowData, null, 2));
            console.log('\nBoard state saved to overflow-board.json');

            foundOverflow = true;
            process.exit(1);
          } else {
            throw e;
          }
        }
        break;
      }
    }

    if (!foundOverflow && moveNum < 50) {
      g2.play(-1);
      moveNum++;
    }
  }

  if (!foundOverflow) {
    console.log(`Game ${gameNum}: ${moveNum} moves - OK`);
  }
  gameNum++;
}

if (!foundOverflow) {
  console.log('\nNo overflow found');
  process.exit(0);
}
