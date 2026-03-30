#!/usr/bin/env node
'use strict';

// move-proximity.js — distribution of distances between consecutive moves.
//
// Usage:
//   node move-proximity.js <games-file>

const fs = require('fs');
const { Game2, PASS } = require('./game2.js');

const gamesFile = process.argv[2];
if (!gamesFile) {
  process.stderr.write('Usage: node move-proximity.js <games-file>\n');
  process.exit(1);
}

const counts = new Map();  // distance string → weighted count
let total = 0;

for (const line of fs.readFileSync(gamesFile, 'utf8').trim().split('\n')) {
  const fields    = line.split(',');
  const N         = parseInt(fields[0], 10);
  const cap       = N * N;
  const lastField = fields[fields.length - 1];
  const hasWinner = lastField === 'b' || lastField === 'w';
  const moves     = hasWinner ? fields.slice(1, -1) : fields.slice(1);
  const game      = new Game2(N, false);

  let prevIdx = -1;
  for (const token of moves) {
    const isPass = !token || token === '..';
    if (!isPass) {
      const idx = (token.charCodeAt(1) - 97) * N + (token.charCodeAt(0) - 97);
      if (prevIdx !== -1) {
        // Count legal non-eye moves available before this move was played.
        let M = 0;
        for (let i = 0; i < cap; i++) {
          if (game.cells[i] === 0 && !game.isTrueEye(i) && game.isLegal(i)) M++;
        }
        if (M > 0) {
          const d = game.distance(prevIdx, idx).toFixed(4);
          const w = 1 / M;
          counts.set(d, (counts.get(d) ?? 0) + w);
          total += w;
        }
      }
      game.play(idx);
      prevIdx = idx;
    } else {
      game.play(PASS);
      prevIdx = -1;  // reset — next move has no valid predecessor
    }
  }
}

const sorted = [...counts.entries()].sort((a, b) => parseFloat(a[0]) - parseFloat(b[0]));
process.stdout.write(`total weight: ${total.toFixed(2)}\n\n`);
process.stdout.write(`distance    weight    ratio\n`);
for (const [d, n] of sorted) {
  process.stdout.write(`${parseFloat(d).toFixed(2)}    ${n.toFixed(2).padStart(9)}    ${(n / total * 100).toFixed(2).padStart(6)}%\n`);
}
