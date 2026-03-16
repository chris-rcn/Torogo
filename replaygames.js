#!/usr/bin/env node
'use strict';

// replaygames.js — replay games recorded by recordgames.js
//
// Usage: node replaygames.js --file <path> [--verbose]
//   --file     path to game records file (required)
//   --verbose  print the board after each move

const fs = require('fs');
const { Game, DEFAULT_KOMI } = require('./game.js');

const args = process.argv.slice(2);
const get  = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };
const has  = flag => args.includes(flag);

const file    = get('--file', null);
const verbose = has('--verbose');

if (!file) {
  console.error('Usage: node replaygames.js --file <path> [--verbose]');
  process.exit(1);
}

const lines = fs.readFileSync(file, 'utf8').trim().split('\n').filter(l => l.trim());

for (let gi = 0; gi < lines.length; gi++) {
  const fields = lines[gi].split(',');
  const size   = parseInt(fields[0], 10);
  const moves  = fields.slice(1); // moves[0] is the auto-placed center stone

  // Game constructor places black at the board center as move 1.
  const g = new Game(size, DEFAULT_KOMI);

  if (verbose) {
    console.log(`\n=== Game ${gi + 1} (${size}x${size}) ===`);
    console.log(`Move 1 (black): ${moves[0]}`);
    console.log(g.board.toAscii(g.lastMove));
  }

  // moves[0] is already placed by the constructor; replay from moves[1] onward.
  for (let mi = 1; mi < moves.length; mi++) {
    const token = moves[mi];
    const color = g.current; // capture before the move flips current

    if (token === '..') {
      g.pass();
      if (verbose) {
        console.log(`\nMove ${mi + 1} (${color}): pass`);
        console.log(g.board.toAscii(null));
      }
    } else {
      const x = token.charCodeAt(0) - 97;
      const y = token.charCodeAt(1) - 97;
      g.placeStone(x, y);
      if (verbose) {
        console.log(`\nMove ${mi + 1} (${color}): ${token}`);
        console.log(g.board.toAscii(g.lastMove));
      }
    }
  }

  if (g.scores) {
    const { black, white } = g.scores;
    const winner = black.total > white.total ? 'Black' : 'White';
    console.log(`Game ${gi + 1}: Black ${black.total} – White ${white.total}  →  ${winner} wins`);
  } else {
    console.log(`Game ${gi + 1}: incomplete (${moves.length} moves)`);
  }
}
