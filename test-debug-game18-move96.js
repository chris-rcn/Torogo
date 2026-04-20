#!/usr/bin/env node
'use strict';

const { Game2, BLACK, WHITE, PASS, coordStr } = require('./game2.js');
const { Game3 } = require('./game3.js');

console.log('Debugging Game 18, Move 96\n');

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

const rng = new XorShift32(12345 + 17); // Game 18 = seed offset 17
const N = 9;

const g2 = new Game2(N, false);
const g3 = new Game3(N);

// Play 95 moves
for (let i = 0; i < 95; i++) {
  const move = g2.randomLegalMove(rng);
  g2.play(move);
  g3.play(move);
}

console.log('Before move 96:\n');
console.log('Game2 board:');
console.log(g2.toString(PASS));
console.log('\nGame3 board:');
console.log(g3.toString(PASS));

// Get move 96
const move96 = g2.randomLegalMove(rng);
console.log('\n--- Playing move 96: ' + coordStr(move96, N) + ' (index ' + move96 + ') ---\n');

console.log('Game2 isLegal:', g2.isLegal(move96));
console.log('Game3 isLegal:', g3.isLegal(move96));

console.log('\nGame2 playing move 96...');
const g2_result = g2.play(move96);
console.log('Game2 result:', g2_result);

console.log('\nGame3 playing move 96...');
const g3_result = g3.play(move96);
console.log('Game3 result:', g3_result);

if (g2_result !== g3_result) {
  console.log('\nMISMATCH: Game2 result !== Game3 result');
  console.log('Game2:', g2_result, 'Game3:', g3_result);
}

console.log('\nAfter move 96:\n');
console.log('Game2 board:');
console.log(g2.toString(PASS));
console.log('\nGame3 board:');
console.log(g3.toString(PASS));
