#!/usr/bin/env node
'use strict';

// minepatterns.js — mine 3×3 pattern statistics from recorded games.
//
// Usage: node minepatterns.js --file <path>
//   --file   path to game records produced by recordgames.js (required)
//
// For every move in each game (excluding passes) the script:
//   1. Enumerates all legal non-true-eye placements as candidates.
//   2. Records a "seen" event for the pattern hash at each candidate.
//   3. Records a "selected" event for the pattern hash of the actual move.
//
// Output: one line per observed pattern hash:
//   <hash>,<selection_ratio>,<seen_count>

const fs = require('fs');
const { Game, DEFAULT_KOMI } = require('./game.js');
const { patternHash } = require('./patterns.JS');

const args = process.argv.slice(2);
const get  = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

const file = get('--file', null);
if (!file) {
  console.error('Usage: node minepatterns.js --file <path>');
  process.exit(1);
}

const lines = fs.readFileSync(file, 'utf8').trim().split('\n').filter(l => l.trim());

// Map from patternHash → { seen: number, selected: number }
const stats = new Map();

function bump(hash, selected) {
  let s = stats.get(hash);
  if (!s) { s = { seen: 0, selected: 0 }; stats.set(hash, s); }
  s.seen++;
  if (selected) s.selected++;
}

for (const line of lines) {
  const fields = line.split(',');
  const size   = parseInt(fields[0], 10);
  const moves  = fields.slice(1);

  const g = new Game(size, DEFAULT_KOMI);

  // moves[0] is already placed by the constructor; process from moves[1] onward.
  for (let mi = 1; mi < moves.length; mi++) {
    const token = moves[mi];
    if (token === '..') { g.pass(); continue; }

    const color = g.current;
    const board = g.board;
    const N     = size;

    // Decode the actual move.
    const mx = token.charCodeAt(0) - 97;
    const my = token.charCodeAt(1) - 97;

    // Enumerate all legal non-true-eye candidates.
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        if (board.get(x, y) !== null) continue;
        if (board.isTrueEye(x, y, color)) continue;
        if (board.isSuicide(x, y, color)) continue;
        if (board.isKo(x, y, color, g.koFlag)) continue;
        const hash = patternHash(g, x, y);
        bump(hash, x === mx && y === my);
      }
    }

    g.placeStone(mx, my);
  }
}

// Output: hash,ratio,seen
for (const [hash, { seen, selected }] of stats) {
  console.log(`${hash},${(selected / seen).toFixed(6)},${seen}`);
}
