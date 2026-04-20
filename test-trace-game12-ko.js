#!/usr/bin/env node
'use strict';

const { Game2, BLACK, WHITE, PASS, coordStr } = require('./game2.js');
const { Game3 } = require('./game3.js');

console.log('Tracing ko state for game 12\n');

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

let moveCount = 0;

while (!g2.gameOver && !g3.gameOver && moveCount < 92) {
  const move = g2.randomLegalMove(rng);

  const g2Result = g2.play(move);
  const g3Result = g3.play(move);

  if (!g2Result || !g3Result) {
    console.log(`Move ${moveCount} (${coordStr(move, N)}) failed`);
    process.exit(1);
  }

  // Show ko state when it changes or near end
  if (moveCount >= 85) {
    console.log(`Move ${moveCount}: ${coordStr(move, N)} - Game2.ko=${g2.ko === PASS ? 'PASS' : coordStr(g2.ko, N)}, Game3.ko=${g3.ko === PASS ? 'PASS' : coordStr(g3.ko, N)}`);
  }

  moveCount++;
}
