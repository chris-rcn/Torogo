#!/usr/bin/env node
'use strict';

const { Game2, BLACK, WHITE, PASS, coordStr } = require('./game2.js');
const { Game3 } = require('./game3.js');

console.log('Finding where game 18 boards diverge\n');

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

while (!g2.gameOver && !g3.gameOver) {
  const move = g2.randomLegalMove(rng);

  const g2Result = g2.play(move);
  const g3Result = g3.play(move);

  if (!g2Result || !g3Result) {
    console.log(`✗ Move ${moveCount} (${coordStr(move, N)}) failed`);
    console.log(`  Game2 result: ${g2Result}, Game3 result: ${g3Result}`);
    process.exit(1);
  }

  // Compare boards after each move
  const g2Board = g2.toString(PASS);
  const g3Board = g3.toString(PASS);

  if (g2Board !== g3Board) {
    console.log(`✗ Move ${moveCount} (${coordStr(move, N)}) caused divergence\n`);

    // Find which cells differ
    const g2Lines = g2Board.split('\n');
    const g3Lines = g3Board.split('\n');
    console.log('Differences:');
    for (let i = 0; i < g2Lines.length; i++) {
      if (g2Lines[i] !== g3Lines[i]) {
        console.log(g2Lines[i]);
        console.log(g3Lines[i]);
        console.log('');
      }
    }

    process.exit(1);
  }

  console.log(`✓ Move ${moveCount}: ${coordStr(move, N)} (boards match)`);
  moveCount++;
}

console.log(`\nCompleted ${moveCount} moves with identical boards`);
