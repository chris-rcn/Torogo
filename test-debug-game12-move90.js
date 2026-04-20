#!/usr/bin/env node
'use strict';

const { Game2, BLACK, WHITE, PASS, coordStr } = require('./game2.js');
const { Game3 } = require('./game3.js');

console.log('Debugging Game 12, Move 90\n');

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

// Play through to move 89
for (let i = 0; i < 90; i++) {
  const move = g2.randomLegalMove(rng);
  g2.play(move);
  g3.play(move);
}

console.log('Before move 90:\n');
console.log('Game2 board:');
console.log(g2.toString(PASS));
console.log('\nGame3 board:');
console.log(g3.toString(PASS));

const move90 = g2.randomLegalMove(rng);
console.log('\n--- Playing move 90: ' + coordStr(move90, N) + ' (index ' + move90 + ') ---\n');

console.log('Game2.isLegal:', g2.isLegal(move90));
console.log('Game3.isLegal:', g3.isLegal(move90));

console.log('\nGame2.ko:', g2.ko, '(' + (g2.ko === PASS ? 'PASS' : coordStr(g2.ko, N)) + ')');
console.log('Game3.ko:', g3.ko, '(' + (g3.ko === PASS ? 'PASS' : coordStr(g3.ko, N)) + ')');

console.log('\nGame2 playing move 90...');
const g2_result = g2.play(move90);
console.log('Game2 result:', g2_result);

console.log('\nGame3 playing move 90...');
const g3_result = g3.play(move90);
console.log('Game3 result:', g3_result);

if (g2_result !== g3_result) {
  console.log('\nMISMATCH: Game2 result !== Game3 result');
}

console.log('\nAfter move 90:\n');
console.log('Game2 board:');
console.log(g2.toString(PASS));
console.log('\nGame3 board:');
console.log(g3.toString(PASS));
