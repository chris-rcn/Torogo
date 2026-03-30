#!/usr/bin/env node
'use strict';

// count-patterns.js — count unique 1x1, 2x2, and 3x3 patterns across random games.
//
// Usage: node count-patterns.js [--games N] [--size N]
//   --games  number of games to play  (default: 200)
//   --size   board size               (default: 13)
//
// Scans every board position after each move and extracts all three pattern
// sizes from every cell.  Accumulates unique canonical keys in three Sets.

const { Game2, PASS } = require('./game2.js');
const { pattern1, pattern2, pattern3 } = require('./patterns.js');

const args = process.argv.slice(2);
const get  = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

const size     = parseInt(get('--size', '13'),  10);
const maxLibs  = parseInt(get('--libs', '1'),  10);

const unique = new Set();
const unique1 = new Set();
const unique2 = new Set();
const unique3 = new Set();

const cap = size * size;

let totalPrev = 0;
let totalPrevPrev = 0;
let nextReport = 1;
const col = 7
console.log(`${'elapsed'.padStart(col)} ${'games'.padStart(col)} ${'p1'.padStart(col)} ${'p2'.padStart(col)} ${'p3'.padStart(col)} ${'total'.padStart(col)}`);
const startTime = Date.now();
for (let g = 0; ; g++) {
  if (g >= nextReport) {
    const total = unique1.size + unique2.size + unique3.size;
    if (total < unique.size) {
      console.log("Collision!");
    }
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`${String(elapsed).padStart(col-1)}s ${String(g).padStart(col)} ${String(unique1.size).padStart(col)} ${String(unique2.size).padStart(col)} ${String(unique3.size).padStart(col)} ${String(total).padStart(col)}`);
    if (total == totalPrevPrev) break;
    totalPrevPrev = totalPrev;
    totalPrev = total;
    nextReport += g / 2;
  }

  const game = new Game2(size, false);

  while (!game.gameOver) {
    const move = game.randomLegalMove();
    game.play(move);

    // Scan all cells and extract patterns at this board position.
    for (let idx = 0; idx < cap; idx++) {
      {
        const r = pattern1(game, maxLibs, idx);
        if (r !== null) {unique1.add(r.key); unique.add(r.key)};
      }
      {
        const r = pattern2(game, maxLibs, idx);
        if (r !== null) {unique2.add(r.key); unique.add(r.key)};
      }
      {
        const r = pattern3(game, maxLibs, idx);
        if (r !== null) {unique3.add(r.key); unique.add(r.key)};
      }
    }
  }
}

