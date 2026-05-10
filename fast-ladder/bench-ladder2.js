'use strict';

const { Game2, PASS } = require('./game2.js');
const { getAllLadderStatuses } = require('./ladder2.js');

const SIZE = 13;
const ITERS = 5000;
let totalTime = 0;
let totalPositions = 0;

for (let i = 0; i < ITERS; i++) {
  const g = new Game2(SIZE);
  // Play a random number of moves to get a mid-game position
  const nMoves = 20 + (Math.random() * 80 | 0);
  for (let m = 0; m < nMoves; m++) {
    const move = g.randomLegalMove();
    if (move === PASS) break;
    g.play(move);
  }
  const t0 = performance.now();
  getAllLadderStatuses(g);
  totalTime += performance.now() - t0;
  totalPositions++;
}

console.log(`${totalPositions} positions, ${totalTime.toFixed(0)} ms elapsed, avg ${(totalTime / totalPositions).toFixed(3)} ms/position`);
