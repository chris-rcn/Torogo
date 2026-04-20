#!/usr/bin/env node
'use strict';

const { Game2, BLACK, WHITE, PASS, coordStr } = require('./game2.js');
const { Game3 } = require('./game3.js');

console.log('Debugging move 79: f9 capture at f8-f7\n');

class XorShift32 {
  constructor(seed = 12345) {
    this.state = seed >>> 0;
  }
  random() {
    this.state ^= this.state << 13;
    this.state ^= this.state >> 17;
    this.state ^= this.state << 5;
    return (this.state >>> 0) / 0x100000000;
  }
}

const rng = new XorShift32(12345);
const N = 9;

// Play through to move 78
const g2 = new Game2(N, false);
const g3 = new Game3(N);

for (let i = 0; i < 79; i++) {
  const move = g2.randomLegalMove(rng);
  g2.play(move);
  g3.play(move);
}

console.log('Before move 79 (f9):\n');
console.log('Game2 board:');
console.log(g2.toString(PASS));
console.log('\nGame3 board:');
console.log(g3.toString(PASS));

// Check liberties before move 79
const f8_idx = 68;  // f8
const f7_idx = 59;  // f7
const f9_idx = 77;  // f9

console.log('\nChecking f8 (idx 68) liberties in Game2:');
const f8_libs_g2 = g2.groupLibs2(f8_idx);
console.log('f8 liberties:', f8_libs_g2);

console.log('\nChecking f8 (idx 68) liberties in Game3:');
const f8_libs_g3 = g3.groupLibs2 ? g3.groupLibs2(f8_idx) : 'N/A';
console.log('f8 liberties:', f8_libs_g3);

console.log('\n--- Playing move 79: f9 ---\n');

const move79 = g2.randomLegalMove(rng);
console.log('Move 79 is at:', coordStr(move79, N), '(index', move79 + ')');

console.log('\nGame2 playing move 79...');
const g2_result = g2.play(move79);
console.log('Game2 result:', g2_result);

console.log('\nGame3 playing move 79...');
const g3_result = g3.play(move79);
console.log('Game3 result:', g3_result);

console.log('\nAfter move 79:\n');
console.log('Game2 board:');
console.log(g2.toString(PASS));
console.log('\nGame3 board:');
console.log(g3.toString(PASS));

// Check if stones were captured
console.log('\nStone at f8 (68) - Game2:', g2.cells[68] === BLACK ? 'BLACK' : g2.cells[68] === WHITE ? 'WHITE' : 'EMPTY');
console.log('Stone at f8 (68) - Game3:', g3.cells[68] === BLACK ? 'BLACK' : g3.cells[68] === WHITE ? 'WHITE' : 'EMPTY');

console.log('\nStone at f7 (59) - Game2:', g2.cells[59] === BLACK ? 'BLACK' : g2.cells[59] === WHITE ? 'WHITE' : 'EMPTY');
console.log('Stone at f7 (59) - Game3:', g3.cells[59] === BLACK ? 'BLACK' : g3.cells[59] === WHITE ? 'WHITE' : 'EMPTY');
