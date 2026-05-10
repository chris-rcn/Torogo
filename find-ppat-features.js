#!/usr/bin/env node
'use strict';

// Count ppat feature bit frequencies over random games.
// Usage: node find-ppat-features.js [games] [moves_per_game]

const { Game2 } = require('./game2.js');
const { createState, extractFeatures, NUM_PATTERNS, PHASE_COUNT } = require('./ppat-lib.js');

const games = parseInt(process.argv[2], 10) || 50;
const moves = parseInt(process.argv[3], 10) || 60;

const bits = new Array(7).fill(0);
let total = 0;
const st = createState(9);
const prevBase = PHASE_COUNT * NUM_PATTERNS;

for (let t = 0; t < games; t++) {
  const g = new Game2(9);
  for (let m = 0; m < moves && !g.gameOver; m++) {
    extractFeatures(g, st);
    for (let i = 0; i < st.count; i++) {
      for (let fi = st.featStart[i]; fi < st.featStart[i + 1]; fi++) {
        const key = st.feat[fi];
        if (key >= prevBase) bits[(key - prevBase) % 7]++;
      }
      total++;
    }
    g.play(g.randomLegalMove());
  }
}

console.log(`games: ${games}  moves/game: ${moves}  candidates: ${total}`);
console.log(`  bit 0 (contiguous):       ${String(bits[0]).padStart(6)}  (${(100 * bits[0] / total).toFixed(2)}%)`);
console.log(`  bit 1 (save by capture):  ${String(bits[1]).padStart(6)}  (${(100 * bits[1] / total).toFixed(2)}%)`);
console.log(`  bit 2 (capture+sa):       ${String(bits[2]).padStart(6)}  (${(100 * bits[2] / total).toFixed(2)}%)`);
console.log(`  bit 3 (extend):           ${String(bits[3]).padStart(6)}  (${(100 * bits[3] / total).toFixed(2)}%)`);
console.log(`  bit 4 (extend+sa):        ${String(bits[4]).padStart(6)}  (${(100 * bits[4] / total).toFixed(2)}%)`);
console.log(`  bit 5 (ko-solve):         ${String(bits[5]).padStart(6)}  (${(100 * bits[5] / total).toFixed(2)}%)`);
console.log(`  bit 6 (2pt semeai):       ${String(bits[6]).padStart(6)}  (${(100 * bits[6] / total).toFixed(2)}%)`);
