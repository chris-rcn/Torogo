#!/usr/bin/env node
'use strict';

const { Game2, EMPTY, BLACK, WHITE, parseBoard, coordStr } = require('./game2.js');
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

const g2 = parseBoard(boardStr, BLACK);
const N = 13;
const cap = 169;

// Place stones 0-50 in Game3
const game3 = new Game3(N);

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

// Place stones 0-50
for (let i = 0; i < 51; i++) {
  if (g2.cells[i] !== EMPTY) {
    game3.current = g2.cells[i];
    game3.play(i);
  }
}

// Now examine cell 51 before placement
console.log('State before placing stone at cell 51:\n');
console.log('Game3 board:');
console.log(game3.toString(51));
console.log();

// Check cell 51 and neighbors
console.log('Cell 51 (m4) in Game3:');
console.log(`  cells[51] = ${game3.cells[51]} (should be 0=EMPTY)`);

const nbr = game3._nbr;
console.log(`\nNeighbors of cell 51:`);
for (let i = 0; i < 4; i++) {
  const ni = nbr[51 * 4 + i];
  if (ni >= 0) {
    const coord = coordStr(ni, 13);
    const cellVal = game3.cells[ni];
    const cellStr = cellVal === BLACK ? '●' : cellVal === WHITE ? '○' : '·';
    const gid = game3._gid[ni];
    const ls = cellVal === EMPTY ? '?' : game3._ls[gid];
    console.log(`  ${coord} (index ${ni}): ${cellStr} (gid=${gid}, liberties=${ls})`);
  }
}

// Try to place it and see what gets captured
console.log(`\nPlacing BLACK at cell 51...`);
game3.current = BLACK;

let before = 0;
for (let j = 0; j < cap; j++) {
  if (game3.cells[j] !== EMPTY) before++;
}

console.log(`Stones before play: ${before}`);
console.log(`isLegal(51): ${game3.isLegal(51)}`);

game3.play(51);

let after = 0;
for (let j = 0; j < cap; j++) {
  if (game3.cells[j] !== EMPTY) after++;
}

console.log(`Stones after play: ${after}`);
console.log(`Cell 51 after play: ${game3.cells[51]}`);

if (after !== before + 1) {
  console.log(`\nCAPTURE DETECTED! ${before - after} stones disappeared!`);
  console.log(`\nGame3 board after play:`);
  console.log(game3.toString(51));
}
