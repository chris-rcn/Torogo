#!/usr/bin/env node
'use strict';

const { Game2, BLACK, WHITE, parseBoard } = require('./game2.js');
const { game3FromGame2, Game3 } = require('./game3.js');

const boardStr = ` · ○ ● · · · ○ · ○ · · ● ●
 ● ○ · · · · · · · · · · ·
 · · · · ○ ○ · ○ · · · · ·
 · · · · · · · · · · · · ●
 ○ · ○ ● · · ○ · · · ● · ·
 · · · · · · · · ○ · · ○ ●
 · ○ ● · ● · ● ● ● ○ · ○ ·
 · · · · · · · · · · ● · ●
 · · ●(●)· · · · · · · · ·
 · · ● · · ○ · · · · · · ○
 · · ○ ○ · ● ● ● ○ · · · ·
 · ● · ○ · ○ ○ · · · ○ · ·
 ● ○ ● · · ● · · · · · ● ·`;

const g2 = parseBoard(boardStr, BLACK);

console.log('Game2 cell 1:', g2.cells[1], '(WHITE=', WHITE, ')');
console.log('Game2 board at position 1:');
console.log(g2.toString());

const N = 13;
const cap = N * N;
const game3 = new Game3(N);

// Clear initial state like game3FromGame2 does
const center = ((N >> 1) * N) + (N >> 1);
game3.cells[center] = 0; // EMPTY
game3._gid[center] = -1;
game3._gc[0] = 0;
game3._ss[0] = 0;
game3._ls[0] = 0;
game3.emptyCount = cap;
game3._nextGid = 0;

// Clear bitsets
for (let i = 0; i < game3._gc.length; i++) game3._gc[i] = 0;
for (let i = 0; i < game3._ss.length; i++) game3._ss[i] = 0;
for (let i = 0; i < game3._ls.length; i++) game3._ls[i] = 0;
for (let i = 0; i < game3._sw.length; i++) game3._sw[i] = 0;
for (let i = 0; i < game3._lw.length; i++) game3._lw[i] = 0;

// Try placing stones 0 and 1
console.log('\n=== Attempting stone placements ===');
for (let i = 0; i < 2; i++) {
  if (g2.cells[i] !== 0) {
    game3.current = g2.cells[i];
    console.log(`\nPosition ${i}: color=${game3.current} (WHITE=2), isLegal=${game3.isLegal(i)}`);
    if (game3.isLegal(i)) {
      game3.play(i);
      console.log(`  → Placed`);
    } else {
      console.log(`  → NOT placed (isLegal returned false)`);
      console.log(`     Game3 cell ${i} is now: ${game3.cells[i]}`);
    }
  }
}

console.log('\n=== Result ===');
console.log('Game3 cell 1:', game3.cells[1]);
