#!/usr/bin/env node
'use strict';

const { Game2, BLACK, WHITE, PASS, EMPTY } = require('./game2.js');
const { Game3, game3FromGame2 } = require('./game3.js');
const { parseBoard } = require('./game2.js');

const boardStr = ` ○ · ○ · ○ · ● ○ · · · · ·
 · ● · · ○ · · · · · · · ·
 · ● · · · ● · · ○ · · · ·
 · ○ ○ ○ ○ · · · ● · · · ○
 · · · ○ · · ● · · · · ● ○
 ○ · ● · · ○ ○ · · · · · ○
 · ● · · · · ● · ● · · · ·
 · ● ○ · ● · · · · · ● · ○
 · · · ● · ● · · · · · · ○
 · · · ● · · · · · · · · ·
 · · ● · · · · · · · · · ●
 · ● ● · · · · ○ · ○ · · ·
 ● ○ · ○ · · ● ● ○ · · ● ●`;

const g2 = parseBoard(boardStr, BLACK);
const N = 13;
const cap = 169;

console.log('Attempting conversion...\n');

const game3 = new Game3(N);
const center = ((N >> 1) * N) + (N >> 1);
game3.cells[center] = EMPTY;
game3._gid[center] = -1;
game3._gc[0] = EMPTY;
game3._ss[0] = 0;
game3._ls[0] = 0;
game3.emptyCount = cap;
game3._nextGid = 0;

for (let i = 0; i < game3._gc.length; i++) game3._gc[i] = EMPTY;
for (let i = 0; i < game3._ss.length; i++) game3._ss[i] = 0;
for (let i = 0; i < game3._ls.length; i++) game3._ls[i] = 0;
for (let i = 0; i < game3._sw.length; i++) game3._sw[i] = 0;
for (let i = 0; i < game3._lw.length; i++) game3._lw[i] = 0;

const toPlace = [];
for (let i = 0; i < cap; i++) {
  if (g2.cells[i] !== EMPTY) {
    toPlace.push(i);
  }
}

console.log(`Total stones to place: ${toPlace.length}`);

let passNum = 0;
while (toPlace.length > 0 && passNum < 20) {
  let placed = false;
  console.log(`Pass ${passNum}: ${toPlace.length} stones remaining`);

  for (let i = toPlace.length - 1; i >= 0; i--) {
    const idx = toPlace[i];
    game3.current = g2.cells[idx];
    if (game3.isLegal(idx)) {
      game3.play(idx);
      toPlace.splice(i, 1);
      placed = true;
    }
  }

  if (!placed) {
    console.log(`\nNo stones placed in pass ${passNum}. Stuck with ${toPlace.length} unplaced stones:`);
    for (const idx of toPlace) {
      const x = idx % 13;
      const y = (idx / 13) | 0;
      console.log(`  Cell ${idx} (${x},${y}): color=${g2.cells[idx]}`);
    }
    break;
  }

  passNum++;
}

console.log(`\nFinal result: ${toPlace.length} stones unplaced after ${passNum} passes`);
