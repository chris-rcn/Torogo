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

const { Game2 } = require('./game2.js');
const { extractFeatures } = require('./vpatterns.js');

const args = process.argv.slice(2);
const get  = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

const size    = parseInt(get('--size', '13'), 10);
const maxLibs = parseInt(get('--libs', '1'),  10);

const specs = [
  { size: 1, maxLibs },
  { size: 2, maxLibs },
  { size: 3, maxLibs },
];

const unique = new Set();

let totalPrev = 0;
let totalPrevPrev = 0;
let nextReport = 1;
let totalPositions = 0;
let totalScanMs = 0;
const col = 7;
console.log(`${'elapsed'.padStart(col)} ${'games'.padStart(col)} ${'total'.padStart(col)}  ${'us/pos'.padStart(8)}`);
const startTime = Date.now();
for (let g = 0; ; g++) {
  if (g >= nextReport) {
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    const usPerPos = totalPositions > 0 ? ((totalScanMs / totalPositions) * 1000).toFixed(1) : '-';
    console.log(`${String(elapsed).padStart(col-1)}s ${String(g).padStart(col)} ${String(unique.size).padStart(col)}  ${usPerPos.padStart(8)}`);
    if (unique.size === totalPrevPrev) break;
    totalPrevPrev = totalPrev;
    totalPrev = unique.size;
    nextReport += g / 2;
  }

  const game = new Game2(size, false);

  while (!game.gameOver) {
    const move = game.randomLegalMove();
    game.play(move);

    const t0 = Date.now();
    for (const { key } of extractFeatures(game, specs)) unique.add(key);
    totalScanMs += Date.now() - t0;
    totalPositions++;
  }
}

