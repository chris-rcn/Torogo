#!/usr/bin/env node
'use strict';

const { Game2, BLACK, WHITE, PASS, parseBoard, coordStr } = require('./game2.js');

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

console.log('Position where game3FromGame2 causes unintended capture:\n');
console.log('Group 34 stones at: m3, j4, k4, l4, m4');
console.log('Stones m3-l4 place successfully, m4 causes capture\n');

// Build board with axis labels
const lines = [];

// Column labels at top
let colLabel = '   ';
for (let x = 0; x < N; x++) {
  const letter = String.fromCharCode(97 + x); // a-m
  colLabel += letter + ' ';
}
lines.push(colLabel);

// Rows from top to bottom (display coordinates)
for (let displayY = N - 1; displayY >= 0; displayY--) {
  const boardY = displayY; // board y = display y
  const rowNum = (boardY + 1).toString().padStart(2, ' ');

  let row = rowNum + ' ';
  for (let boardX = 0; boardX < N; boardX++) {
    const idx = boardY * N + boardX;
    const cell = g2.cells[idx];
    const cellStr = cell === BLACK ? '●' : cell === WHITE ? '○' : '·';

    // Mark group 34 stones
    if ([38, 48, 49, 50, 51].includes(idx)) {
      row += '(' + cellStr + ')';
    } else {
      row += ' ' + cellStr + ' ';
    }
  }
  row += ' ' + (boardY + 1);
  lines.push(row);
}

// Bottom column labels
colLabel = '   ';
for (let x = 0; x < N; x++) {
  const letter = String.fromCharCode(97 + x);
  colLabel += letter + ' ';
}
lines.push(colLabel);

console.log(lines.join('\n'));

console.log('\nGroup 34 (marked with parentheses):');
console.log('  m3 (index 38) - already placed');
console.log('  j4 (index 48) - already placed');
console.log('  k4 (index 49) - already placed');
console.log('  l4 (index 50) - already placed');
console.log('  m4 (index 51) - CAUSES CAPTURE when placed');
