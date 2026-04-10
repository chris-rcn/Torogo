'use strict';

const { createState, extractFeatures } = require('./ppat-lib.js');
const { Game2 } = require('./game2.js');

function xorshift32(seed) {
  let s = seed >>> 0 || 1;
  return () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 4294967296; };
}

const N = 9, POSITIONS = 200, MIN_MS = 10000;
const positions = [];
const savedRandom = Math.random;
Math.random = xorshift32(0xdeadbeef);
for (let p = 0; p < POSITIONS; p++) {
  const g = new Game2(N);
  const moves = 15 + ((p * 7) % 40);  // 15–54 moves, varied
  for (let i = 0; i < moves && !g.gameOver; i++) {
    const m = g.randomLegalMove();
    if (m >= 0) g.play(m); else g.play(-1);
  }
  positions.push(g);
}
Math.random = savedRandom;

const st = createState(N);
let totalMoves = 0, calls = 0;
const t0 = Date.now();
do {
  for (const g of positions) {
    extractFeatures(g, st);
    totalMoves += st.count;
    calls++;
  }
} while (Date.now() - t0 < MIN_MS);
const ms = Date.now() - t0;
const usPerCall = (ms / calls * 1000).toFixed(1);
const avgMoves = (totalMoves / calls).toFixed(1);
console.log(`${usPerCall} µs/call avg (${avgMoves} moves/pos avg, ${calls} calls)`);
