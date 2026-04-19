#!/usr/bin/env node
'use strict';

// Find the position that causes stack overflow in tactics3 using random moves

const { Game2 } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');
const { searchChains } = require('./tactics3.js');

console.log('Searching for stack overflow using random moves...\n');

let foundOverflow = false;
let gamesChecked = 0;

for (let gameNum = 0; gameNum < 1000 && !foundOverflow; gameNum++) {
  const g2 = new Game2(13);
  const moves = [];

  // Play random moves until game ends or we hit enough moves
  while (moves.length < 100 && !g2.gameOver && !foundOverflow) {
    const move = g2.randomLegalMove();

    try {
      const g3 = game3FromGame2(g2);

      // Try searchChains without nodeLimit
      try {
        searchChains(g3);
      } catch (e) {
        if (e instanceof RangeError && e.message.includes('stack')) {
          console.log('✓ STACK OVERFLOW FOUND!');
          console.log(`Game ${gameNum}, Move ${moves.length}`);
          console.log(`Move sequence: ${moves.join(', ')}`);

          // Save the board state
          const fs = require('fs');
          fs.writeFileSync('overflow-board.json', JSON.stringify({
            gameNum,
            moveCount: moves.length,
            moves,
            cells: Array.from(g3.cells),
            N: 13,
            current: g3.current,
            moveCount: g3.moveCount
          }, null, 2));

          console.log('\nBoard saved to overflow-board.json');

          // Display the board
          console.log('\nBoard visualization:');
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
          console.log('Move count:', g3.moveCount);

          foundOverflow = true;
        } else {
          throw e;
        }
      }

      if (g2.play(move)) {
        moves.push(move);
      }
    } catch (e) {
      if (e instanceof RangeError && e.message.includes('stack')) {
        console.log('Stack overflow during play');
        foundOverflow = true;
      } else {
        throw e;
      }
    }
  }

  if (!foundOverflow) {
    gamesChecked++;
    if (gamesChecked % 100 === 0) {
      console.log(`Checked ${gamesChecked} games...`);
    }
  }
}

if (!foundOverflow) {
  console.log(`\nNo overflow found after ${gamesChecked} random games`);
  process.exit(0);
}
