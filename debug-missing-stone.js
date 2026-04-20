#!/usr/bin/env node
'use strict';

const { Game2, BLACK, WHITE, PASS } = require('./game2.js');
const { Game3, game3FromGame2 } = require('./game3.js');

// Create a game with specific seed/position
// We'll manually create the failing board
const boardStr = ` ○ · · · · · · · · ● · · ·
 · · · ● · · · · ○ · · · ·
 · · · · · · · · ○ · · · ·
 ● ○ · ○ ● · · · · · · · ○
 · · ○ · · ○ ○ ● ● ● · ● ·
 ○ ● · · ○ · · ○ ○ ● ○ · ·
 ● · · · ● · ● · · ○ · · ·
 · · · · · · · · · ● ● · ·
 · · · · · ○ · ● ● ● · ○ ·
 · · · · · ○ · ● · · · ● ○
 · · · · · · ● ○ ● · · · ·
 ● · ● · · · · · · · · · ●
 · · · · · ○ ○ ○ · ● ○ ○ ·`;

const { parseBoard } = require('./game2.js');
const g2 = parseBoard(boardStr, BLACK);

console.log('Stone at board cell 100 (row 7, col 9):');
console.log(`Game2: ${g2.cells[100]} (BLACK=1, WHITE=-1, EMPTY=0)`);

const g3 = game3FromGame2(g2);
console.log(`Game3: ${g3.cells[100]}`);

if (g2.cells[100] !== g3.cells[100]) {
  console.log('\nMissing stone! Debugging...');
  console.log(`Game2 gid: ${g2._gid[100]}`);
  console.log(`Game3 gid: ${g3._gid[100]}`);

  // Check neighbors
  const nbr = g2._nbr;
  const N = 13;
  const x = 100 % N;
  const y = (100 / N) | 0;
  console.log(`\nPosition: (${x}, ${y})`);
  console.log(`Neighbors in Game2:`);
  for (let i = 0; i < 4; i++) {
    const ni = nbr[100 * 4 + i];
    console.log(`  ${i}: cell ${ni} = ${g2.cells[ni]}`);
  }

  // Try to manually place it
  const N3 = g3.N;
  g3.current = BLACK;
  console.log(`\nTrying to place at ${100}:`);
  console.log(`  isLegal: ${g3.isLegal(100)}`);
  console.log(`  cells[100] before: ${g3.cells[100]}`);
  if (g3.isLegal(100)) {
    g3.play(100);
    console.log(`  cells[100] after play: ${g3.cells[100]}`);
  }
}
