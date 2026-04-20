#!/usr/bin/env node
'use strict';

const { Game2, BLACK, WHITE, parseBoard } = require('./game2.js');

// Create a simple test board
const g = new Game2(5);
g.play(0);   // a5
g.play(1);   // b5
g.play(5);   // a4
g.play(6);   // b4

console.log('Original board:');
console.log(g.toString());
console.log('\nLastMove:', g.lastMove);

const str = g.toString();
console.log('\nParsing this string back:');
console.log(str);

const g2 = parseBoard(str, BLACK);
console.log('\nParsed board:');
console.log(g2.toString());
console.log('\nParsed lastMove:', g2.lastMove);

console.log('\nAre they equal?', str === g2.toString());

// Check individual cells
console.log('\nCell comparison:');
for (let i = 0; i < 25; i++) {
  if (g.cells[i] !== g2.cells[i]) {
    console.log(`Cell ${i}: original=${g.cells[i]}, parsed=${g2.cells[i]}`);
  }
}
