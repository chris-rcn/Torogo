#!/usr/bin/env node
'use strict';

const { Game2, BLACK, WHITE, PASS, coordStr } = require('./game2.js');
const { Game3 } = require('./game3.js');

console.log('Debugging moves 88-89 for game 12\n');

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

const rng = new XorShift32(12);
const N = 9;

const g2 = new Game2(N, false);
const g3 = new Game3(N);

// Play through to move 88
for (let i = 0; i < 88; i++) {
  const move = g2.randomLegalMove(rng);
  g2.play(move);
  g3.play(move);
}

console.log('Before move 88:\n');
console.log('Game2 board:');
console.log(g2.toString(PASS));
console.log('\nGame3 board:');
console.log(g3.toString(PASS));

const move88 = g2.randomLegalMove(rng);
console.log('\n--- Playing move 88: ' + coordStr(move88, N) + ' ---\n');
console.log('Game2.play:', g2.play(move88));
console.log('Game3.play:', g3.play(move88));
console.log('Game2.ko:', g2.ko === PASS ? 'PASS' : coordStr(g2.ko, N));
console.log('Game3.ko:', g3.ko === PASS ? 'PASS' : coordStr(g3.ko, N));

console.log('\nAfter move 88:\n');
console.log('Game2 board:');
console.log(g2.toString(PASS));
console.log('\nGame3 board:');
console.log(g3.toString(PASS));

const move89 = g2.randomLegalMove(rng);
console.log('\n--- Playing move 89: ' + coordStr(move89, N) + ' ---\n');
console.log('Game2.play:', g2.play(move89));
console.log('Game3.play:', g3.play(move89));
console.log('Game2.ko:', g2.ko === PASS ? 'PASS' : coordStr(g2.ko, N));
console.log('Game3.ko:', g3.ko === PASS ? 'PASS' : coordStr(g3.ko, N));

console.log('\nAfter move 89:\n');
console.log('Game2 board:');
console.log(g2.toString(PASS));
console.log('\nGame3 board:');
console.log(g3.toString(PASS));
