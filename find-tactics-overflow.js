#!/usr/bin/env node
'use strict';

// Find the position that causes stack overflow in tactics3

const { Game2 } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');
const { searchChains } = require('./tactics3.js');

console.log('Searching for stack overflow position in tactics3...\n');

let foundOverflow = false;
let positionsChecked = 0;

// Try multiple random games
for (let gameNum = 0; gameNum < 20 && !foundOverflow; gameNum++) {
  const g2 = new Game2(13);
  let moveNum = 0;

  while (moveNum < 100 && !g2.gameOver && !foundOverflow) {
    // Try to find a legal move
    let foundMove = false;
    for (let i = 0; i < 169 && !foundMove; i++) {
      if (g2.isLegal(i)) {
        try {
          const g3 = game3FromGame2(g2);
          positionsChecked++;

          // Try searchChains with small limit first
          try {
            searchChains(g3, 1000);
          } catch (e) {
            if (e instanceof RangeError && e.message.includes('stack')) {
              console.log('✗ STACK OVERFLOW FOUND!');
              console.log(`Position: Game ${gameNum}, Move ${moveNum}, after playing at ${i}`);
              console.log(`Total positions checked: ${positionsChecked}`);
              console.log('\nBoard state (cells array):');

              // Show board in visual format
              const N = 13;
              for (let y = 0; y < N; y++) {
                let row = '';
                for (let x = 0; x < N; x++) {
                  const idx = y * N + x;
                  const c = g3.cells[idx];
                  if (c === 1) row += '●';
                  else if (c === -1) row += '○';
                  else row += '·';
                }
                console.log(row);
              }

              console.log('\nGame state:');
              console.log('Current player:', g3.current === 1 ? 'BLACK' : 'WHITE');
              console.log('Ko:', g3.ko);
              console.log('Move count:', g3.moveCount);

              // Save the board for further investigation
              console.log('\nSaving overflow board state to overflow-position.json');
              const fs = require('fs');
              fs.writeFileSync('overflow-position.json', JSON.stringify({
                gameNum,
                moveNum,
                lastMove: i,
                cells: Array.from(g3.cells),
                N: g3.N,
                current: g3.current,
                ko: g3.ko,
                moveCount: g3.moveCount
              }, null, 2));

              foundOverflow = true;
              process.exit(1);
            } else {
              throw e;
            }
          }

          g2.play(i);
          foundMove = true;
          moveNum++;
        } catch (e) {
          if (e instanceof RangeError && e.message.includes('stack')) {
            console.log('✗ STACK OVERFLOW during conversion or play!');
            console.log(`Game ${gameNum}, move ${moveNum}, trying to play at ${i}`);
            foundOverflow = true;
            process.exit(1);
          }
          throw e;
        }
      }
    }

    if (!foundMove) {
      // Try pass
      g2.play(-1);
      moveNum++;
    }
  }

  console.log(`Game ${gameNum}: Checked ${moveNum} moves successfully`);
}

if (!foundOverflow) {
  console.log(`\n✓ No stack overflow found after checking ${positionsChecked} positions`);
  process.exit(0);
}
