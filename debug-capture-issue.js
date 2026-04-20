#!/usr/bin/env node
'use strict';

const { Game2, BLACK, WHITE, PASS, EMPTY, parseBoard } = require('./game2.js');
const { Game3, game3FromGame2 } = require('./game3.js');

// The position where capture was detected
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

const g2 = parseBoard(boardStr, BLACK);

console.log('Original Game2 board:');
console.log(g2.toString(PASS));
console.log();

console.log('Starting game3FromGame2 conversion...\n');

// Manually trace through the conversion
const N = 13;
const cap = 169;
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

// Collect stones to place
const toPlace = [];
for (let i = 0; i < cap; i++) {
  if (g2.cells[i] !== EMPTY) {
    toPlace.push(i);
  }
}

console.log(`Total stones to place: ${toPlace.length}\n`);

// Place stones
let placed = 0;
for (let idx of toPlace) {
  game3.current = g2.cells[idx];

  if (!game3.isLegal(idx)) {
    console.log(`ILLEGAL: ${idx} (${idx % 13}, ${(idx / 13) | 0}) color=${g2.cells[idx]}`);
    continue;
  }

  let before = 0;
  for (let i = 0; i < cap; i++) {
    if (game3.cells[i] !== EMPTY) before++;
  }

  game3.play(idx);

  let after = 0;
  for (let i = 0; i < cap; i++) {
    if (game3.cells[i] !== EMPTY) after++;
  }

  if (after < before + 1) {
    console.log(`\nCAPTURE at ${idx} (${idx % 13}, ${(idx / 13) | 0})`);
    console.log(`Before: ${before}, After: ${after}`);
    console.log('\nGame3 board after placement:');
    console.log(game3.toString(PASS));
    break;
  }

  placed++;
}

console.log(`\nPlaced ${placed} stones total`);
