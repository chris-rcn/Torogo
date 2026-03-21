#!/usr/bin/env node
'use strict';

// minepatterns2.js — mine 3×3 pattern statistics from recorded games.
// Like minepatterns.js but uses Game2 and patternHash2 for speed.
//
// Usage: node minepatterns2.js --file <path>
//   --file    path to game records produced by recordgames.js (required)
//
// Output: one line per observed pattern hash:
//   <hash>,<selection_ratio>,<seen_count>

const fs = require('fs');
const { Game2, WHITE } = require('./game2.js');
const { patternHash2 } = require('./patterns2.js');

const args = process.argv.slice(2);
const get  = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

const file = get('--file', null);

if (!file) {
  console.error('Usage: node minepatterns2.js --file <path>');
  process.exit(1);
}

const lines = fs.readFileSync(file, 'utf8').trim().split('\n').filter(l => l.trim());

// Map from patternHash2 → { seen: number, selected: number }
const stats = new Map();

function bump(hash, selected) {
  let s = stats.get(hash);
  if (!s) { s = { seen: 0, selected: 0 }; stats.set(hash, s); }
  s.seen++;
  if (selected) s.selected++;
}

for (let gi = 0; gi < lines.length; gi++) {
  const fields = lines[gi].split(',');
  const size   = parseInt(fields[0], 10);
  const moves  = fields.slice(1);
  const N      = size;
  const cap    = N * N;

  const g = new Game2(size);

  // Collect per-move data; defer bumping until the winner is known.
  const gameMoves = [];

  // moves[0] is already placed by the constructor; process from moves[1] onward.
  for (let mi = 1; mi < moves.length; mi++) {
    const token = moves[mi];
    if (token === '..') { g.play(-1); continue; }

    const color = g.current;
    const mx    = token.charCodeAt(0) - 97;
    const my    = token.charCodeAt(1) - 97;
    const midx  = my * N + mx;

    // Enumerate all legal non-true-eye candidates (excluding the selected move).
    const others = [];
    for (let i = 0; i < cap; i++) {
      if (i === midx) continue;
      if (g.cells[i] !== 0) continue;
      if (g.isTrueEye(i)) continue;
      if (!g.isLegal(i)) continue;
      others.push(patternHash2(g, i, color));
    }

    if (g.cells[midx] !== 0)
      process.stderr.write(`WARNING: game ${gi + 1} move ${mi + 1}: selected move ${token} is occupied\n`);
    else if (g.isTrueEye(midx))
      process.stderr.write(`WARNING: game ${gi + 1} move ${mi + 1}: selected move ${token} is a true eye\n`);
    else if (!g.isLegal(midx))
      process.stderr.write(`WARNING: game ${gi + 1} move ${mi + 1}: selected move ${token} is illegal\n`);

    const selHash = patternHash2(g, midx, color);
    gameMoves.push({ color, selHash, others });

    g.play(midx);
  }

  // Determine the winner (komi gives white any tie).
  const winner = g.calcWinner() ?? WHITE;

  // Only record patterns for moves made by the winning player.
  for (const { color, selHash, others } of gameMoves) {
    if (color !== winner) continue;
    for (const h of others) bump(h, false);
    bump(selHash, true);
  }
}

// Output: hash,ratio,seen
for (const [hash, { seen, selected }] of stats) {
  console.log(`${hash},${(selected / seen).toFixed(6)},${seen}`);
}
