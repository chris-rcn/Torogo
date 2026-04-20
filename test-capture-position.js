#!/usr/bin/env node
'use strict';

const { Game2, BLACK, WHITE, PASS, EMPTY, parseBoard, coordStr } = require('./game2.js');
const { Game3, game3FromGame2 } = require('./game3.js');

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

console.log('Testing board conversion that causes unintended capture\n');

const g2 = parseBoard(boardStr, BLACK);

console.log('Game2 board:');
console.log(g2.toString(PASS));
console.log();

console.log('Attempting game3FromGame2 conversion...\n');

try {
  const g3 = game3FromGame2(g2);
  console.log('Conversion successful!');
  console.log('\nGame3 board:');
  console.log(g3.toString(PASS));
} catch (err) {
  console.log('Error during conversion:', err.message);
  process.exit(1);
}
