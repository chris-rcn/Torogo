#!/usr/bin/env node
'use strict';

const { Game2, BLACK, WHITE, EMPTY, PASS } = require('./game2.js');
const { Game3 } = require('./game3.js');

// Recreate the failing game
let game2 = null;
for (let test = 0; test < 1; test++) {
  game2 = new Game2(13);
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
  const game3 = new Game3(13);

  // Initialize Game3 like game3FromGame2 does
  const center = 84;
  game3.cells[center] = EMPTY;
  game3._gid[center] = -1;
  game3._gc[0] = EMPTY;
  game3._ss[0] = 0;
  game3._ls[0] = 0;
  game3.emptyCount = 169;
  game3._nextGid = 0;
  game3.moveCount = 0;

  for (let i = 0; i < game3._gc.length; i++) game3._gc[i] = EMPTY;
  for (let i = 0; i < game3._ss.length; i++) game3._ss[i] = 0;
  for (let i = 0; i < game3._ls.length; i++) game3._ls[i] = 0;
  for (let i = 0; i < game3._sw.length; i++) game3._sw[i] = 0;
  for (let i = 0; i < game3._lw.length; i++) game3._lw[i] = 0;

  // Try to place stones in order
  console.log('Analyzing stone placement:\n');

  for (let i = 0; i < 169; i++) {
    if (game2.cells[i] !== EMPTY) {
      game3.current = game2.cells[i];
      const legal = game3.isLegal(i);

      if (i === 67 || i === 77) {
        const x = i % 13;
        const y = (i / 13) | 0;
        console.log(`Cell ${i} (${x},${y}): color=${game2.cells[i]} (BLACK=1, WHITE=-1)`);
        console.log(`  isLegal: ${legal}`);

        if (!legal) {
          // Debug why it's not legal
          console.log(`  Checking neighbors and state:`);
          const nbr = game3._nbr;
          for (let j = 0; j < 4; j++) {
            const ni = nbr[i * 4 + j];
            console.log(`    Neighbor ${j}: cell ${ni} = ${game3.cells[ni]}`);
          }
          console.log(`  groupLibs at ${i}: ${game3.groupLibs(i).length}`);
        }
        console.log();
      }

      if (legal) {
        game3.play(i);
      }
    }
  }
}
