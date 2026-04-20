#!/usr/bin/env node
'use strict';

const { Game2, BLACK, WHITE, PASS, EMPTY, parseBoard, coordStr } = require('./game2.js');

// Failing position: capture detected at cell 51 during game3FromGame2
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

console.log('Position where capture is detected at cell 51 (m4 in Go notation):\n');
console.log(g2.toString(51));  // Mark cell 51
console.log();

// Show the area around cell 51
console.log('Cell 51 is at coordinates: ' + coordStr(51, 13));
console.log('Neighbors of cell 51:');

const nbr = g2._nbr;
const N = 13;
for (let i = 0; i < 4; i++) {
  const ni = nbr[51 * 4 + i];
  if (ni >= 0) {
    const coord = coordStr(ni, 13);
    const cellVal = g2.cells[ni];
    const cellStr = cellVal === 1 ? '●' : cellVal === -1 ? '○' : '·';
    console.log(`  ${coord}: ${cellStr}`);
  }
}
