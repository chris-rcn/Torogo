#!/usr/bin/env node
'use strict';

// minepatterns2.js — mine 3×3 pattern statistics from recorded games.
//
// Usage: node minepatterns2.js --file <path>
//   --file    path to game records produced by recordgames.js (required)
//
// Output: one line per observed pattern hash:
//   <hash>,<selection_ratio>,<seen_count>

const fs = require('fs');
const { Game2, BLACK, WHITE } = require('./game2.js');
const { patternHashes2 } = require('./pattern12.js');

const args = process.argv.slice(2);
const get  = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

const file = get('--file', null);

if (!file) {
  console.error('Usage: node minepatterns2.js --file <path>');
  process.exit(1);
}

const lines = fs.readFileSync(file, 'utf8').trim().split('\n').filter(l => l.trim());

// Map from pHash → { seen: number, selected: number }
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
  const N      = size;
  const cap    = N * N;

  const lastField = fields[fields.length - 1];
  const hasWinner = lastField === 'b' || lastField === 'w';
  const moves     = hasWinner ? fields.slice(1, -1) : fields.slice(1);

  // Use the pre-computed winner if present; otherwise replay pass 1 to compute it.
  let winner;
  if (hasWinner) {
    winner = lastField === 'b' ? BLACK : WHITE;
  } else {
    const g = new Game2(size);
    for (let mi = 1; mi < moves.length; mi++) {
      const token = moves[mi];
      if (token === '..') { g.play(-1); continue; }
      const mx = token.charCodeAt(0) - 97;
      const my = token.charCodeAt(1) - 97;
      g.play(my * N + mx);
    }
    winner = g.calcWinner();
  }

  // Pass 2: replay and extract patterns for winner's moves only.
  const g = new Game2(size);
  for (let mi = 1; mi < moves.length; mi++) {
    const token = moves[mi];
    if (token === '..') { g.play(-1); continue; }

    const color = g.current;
    const mx    = token.charCodeAt(0) - 97;
    const my    = token.charCodeAt(1) - 97;
    const selected = my * N + mx;

    if (color === winner) {
      if (g.cells[selected] !== 0)
        process.stderr.write(`WARNING: game ${gi + 1} move ${mi + 1}: selected move ${token} is occupied\n`);
      else if (g.isTrueEye(selected))
        process.stderr.write(`WARNING: game ${gi + 1} move ${mi + 1}: selected move ${token} is a true eye\n`);
      else if (!g.isLegal(selected))
        process.stderr.write(`WARNING: game ${gi + 1} move ${mi + 1}: selected move ${token} is illegal\n`);

      const nonSelected = [];
      for (let i = 0; i < cap; i++) {
        if (i === selected) continue;
        if (g.cells[i] !== 0) continue;
        if (g.isTrueEye(i)) continue;
        if (!g.isLegal(i)) continue;
        nonSelected.push(i);
      }

      const hashes = patternHashes2(g, [selected, ...nonSelected]);
      bump(hashes[0].pHash, true);
      for (let i = 1; i < hashes.length; i++) bump(hashes[i].pHash, false);
    }

    g.play(selected);
  }
}

// Output: hash,ratio,seen
for (const [hash, { seen, selected }] of stats) {
  console.log(`${hash},${(selected / seen).toFixed(6)},${seen}`);
}
