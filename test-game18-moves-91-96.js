#!/usr/bin/env node
'use strict';

const { Game2, BLACK, WHITE, PASS, coordStr } = require('./game2.js');
const { Game3 } = require('./game3.js');

console.log('Game 18: Showing board state for moves 91-96\n');

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

const rng = new XorShift32(12345 + 17);
const N = 9;

const g2 = new Game2(N, false);
const g3 = new Game3(N);

let moveCount = 0;

while (!g2.gameOver && !g3.gameOver && moveCount < 96) {
  const move = g2.randomLegalMove(rng);

  if (moveCount >= 90) {
    console.log(`\nBefore move ${moveCount} (${coordStr(move, N)}):`);
    console.log('d3 in Game2:', g2.cells[21] === BLACK ? 'BLACK' : g2.cells[21] === WHITE ? 'WHITE' : 'EMPTY');
    console.log('d3 in Game3:', g3.cells[21] === BLACK ? 'BLACK' : g3.cells[21] === WHITE ? 'WHITE' : 'EMPTY');
  }

  g2.play(move);
  g3.play(move);

  if (moveCount >= 90) {
    console.log(`After move ${moveCount} (${coordStr(move, N)}):`);
    console.log('d3 in Game2:', g2.cells[21] === BLACK ? 'BLACK' : g2.cells[21] === WHITE ? 'WHITE' : 'EMPTY');
    console.log('d3 in Game3:', g3.cells[21] === BLACK ? 'BLACK' : g3.cells[21] === WHITE ? 'WHITE' : 'EMPTY');
  }

  moveCount++;
}

console.log(`\nFinal state before move 96:`);
console.log('d3 in Game2:', g2.cells[21] === BLACK ? 'BLACK' : g2.cells[21] === WHITE ? 'WHITE' : 'EMPTY');
console.log('d3 in Game3:', g3.cells[21] === BLACK ? 'BLACK' : g3.cells[21] === WHITE ? 'WHITE' : 'EMPTY');

const move96 = g2.randomLegalMove(rng);
console.log(`\nMove 96: ${coordStr(move96, N)}`);
console.log('Game2.isLegal:', g2.isLegal(move96));
console.log('Game3.isLegal:', g3.isLegal(move96));
console.log('Game2.play:', g2.play(move96));
console.log('Game3.play:', g3.play(move96));
