#!/usr/bin/env node
'use strict';

const { Game2, BLACK, WHITE, EMPTY, PASS } = require('./game2.js');
const { Game3 } = require('./game3.js');
const { parseBoard } = require('./game2.js');

const boardStr = ` ● ● · ● ○ ○ ○ ● ○ ○ · ○ ○
 ○ ● ● · ● ○ ● ● ○ · ○ ○ ○
 ○ ● ● ● ● ○ ○ ○ ○ ○ ○ ○ ·
 ○ ○ ○ ○ ○ ● ● ● ○ · ○ ○ ○
 ○ · ○ ○ ○ ● · ● ○ ○ · ○ ○
 ○ ○ ○ · ○ ○ ● ● ● ● ○ · ○
 · ○ ○ ○ ○ ● ● ● ● ○ ○ ○ ○
 ○ ○ ○ ● ○ ● ● ○ ○ · ○ ○ ·
 ● ● ● ● ● ● ● ○ ○ ○ ○ ○ ○
 ● · ● ● ● ● ● ● ○ ○ ● ○ ●
 ● ● ● ● · · ● ○ ○ ○ ● ○ ●
 ● · ● ● ● ● ● ○ ● ● ● ● ●
 · ● ● ● ● ● ● ● ● ○ ○ ○ ●`;

const game2 = parseBoard(boardStr, BLACK);
const game3 = new Game3(13);
const cap = 169;

// Track which cells we're interested in
const targetCells = [37, 50, 69, 149, 150, 160];

console.log('Tracing placement of critical cells:\n');

for (let i = 0; i < cap; i++) {
  if (game2.cells[i] !== EMPTY) {
    game3.current = game2.cells[i];

    const beforeCells = {};
    for (const idx of targetCells) {
      beforeCells[idx] = game3.cells[idx];
    }

    const result = game3.play(i);

    if (!result) {
      console.log(`Cell ${i}: play() returned false`);
    }

    // Check if any target cells changed
    for (const idx of targetCells) {
      if (beforeCells[idx] !== game3.cells[idx]) {
        const x = idx % 13;
        const y = (idx / 13) | 0;
        console.log(`After placing ${i}: Cell ${idx} (${x},${y}) changed from ${beforeCells[idx]} to ${game3.cells[idx]}`);
      }
    }
  }
}

console.log('\nFinal state of target cells:');
for (const idx of targetCells) {
  const x = idx % 13;
  const y = (idx / 13) | 0;
  const g2 = game2.cells[idx];
  const g3 = game3.cells[idx];
  const match = g2 === g3 ? '✓' : '✗';
  console.log(`${match} Cell ${idx} (${x},${y}): Game2=${g2}, Game3=${g3}`);
}
