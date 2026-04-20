#!/usr/bin/env node
'use strict';

const { Game2, BLACK, parseBoard, coordStr } = require('./game2.js');

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

// Mark all stones in group 34
console.log('Position showing group 34 (will be merged when placing m4):\n');

// Create a board with marks at each group stone
const groupStones = [38, 48, 49, 50, 51];
const markIdx = 38; // Mark first stone in the group

console.log(g2.toString(markIdx));
console.log();

console.log('Group 34 stones:');
for (const idx of groupStones) {
  console.log(`  ${coordStr(idx, 13)} (index ${idx})`);
}

console.log('\nWhen stones 0-50 are placed:');
for (const idx of groupStones) {
  if (idx <= 50) {
    console.log(`  ${coordStr(idx, 13)} - PLACED`);
  } else {
    console.log(`  ${coordStr(idx, 13)} - NOT YET PLACED (causes capture when play() is called)`);
  }
}
