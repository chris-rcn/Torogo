#!/usr/bin/env node
'use strict';

const { Game2, EMPTY, parseBoard } = require('./game2.js');
const { Game3 } = require('./game3.js');

const boardStr = ` ● ● · · · ○ · · · · · ○ ·
 · · ○ · · · · · ● ○ ○ ● ●
 · · · · ○ · ● ○ ● · · ○ ○
 · · · ● ○ ● · ○ ○ ○ ○ ● ·
 · · ● · · ● · · · · · ○ ·
 ○ · · ○ · · · ● ● · ● · ●
 · ○ · · ○ ○ ● ● · ● ● ○ ○
 · ● ● ○ · ● · · · ○ · ● ●
 · ● ● ○ · ● · · ○ ○ · · ·
 · ○ · · ● · · · · ● ● ● ●
 ○ ○ · · ● ● ● · · · · ○ ●
 · ○ ● · · · · · · ○ ○ · ○
 · · · ● ● ○ · ○ ○ · · · ○`;

const g2 = parseBoard(boardStr, 1);
const N = 13;
const cap = 169;

// List all stones in index order
console.log('Stones to place (in index order):');
let count = 0;
for (let i = 0; i < cap; i++) {
  if (g2.cells[i] !== EMPTY) {
    const color = g2.cells[i] === 1 ? 'BLACK' : 'WHITE';
    console.log(`  ${count + 1}. Index ${i} (${i % 13}, ${(i / 13) | 0}): ${color}`);
    count++;
    if (i >= 51) {
      console.log('  ...');
      break;
    }
  }
}

// Now try to place them
console.log('\nAttempting to place stones...\n');

const game3 = new Game3(N);

// Initialize empty
for (let i = 0; i < cap; i++) {
  game3.cells[i] = EMPTY;
  game3._gid[i] = -1;
}
for (let i = 0; i < game3._gc.length; i++) game3._gc[i] = EMPTY;
for (let i = 0; i < game3._ss.length; i++) game3._ss[i] = 0;
for (let i = 0; i < game3._ls.length; i++) game3._ls[i] = 0;
for (let i = 0; i < game3._sw.length; i++) game3._sw[i] = 0;
for (let i = 0; i < game3._lw.length; i++) game3._lw[i] = 0;

game3._nextGid = 0;
game3.emptyCount = cap;

let placed = 0;
for (let i = 0; i < cap; i++) {
  if (g2.cells[i] !== EMPTY) {
    game3.current = g2.cells[i];
    const color = g2.cells[i] === 1 ? 'BLACK' : 'WHITE';

    let before = 0;
    for (let j = 0; j < cap; j++) {
      if (game3.cells[j] !== EMPTY) before++;
    }

    const legal = game3.isLegal(i);
    console.log(`${placed + 1}. Placing ${color} at index ${i} (${i % 13}, ${(i / 13) | 0}) - isLegal: ${legal}`);

    if (!legal) {
      console.log(`   SKIPPING (illegal)`);
      continue;
    }

    game3.play(i);

    let after = 0;
    for (let j = 0; j < cap; j++) {
      if (game3.cells[j] !== EMPTY) after++;
    }

    if (after < before + 1) {
      console.log(`   CAPTURE! Before: ${before}, After: ${after}`);
      break;
    } else {
      console.log(`   OK. Stones: ${before} -> ${after}`);
    }

    placed++;
  }
}
