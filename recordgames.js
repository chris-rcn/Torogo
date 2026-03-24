#!/usr/bin/env node
'use strict';

// recordgames.js — play an agent against itself and print a game record per line.
//
// Usage: node recordgames.js --size <n> [--agent rave] [--budget 500] [--max 0]
//   --agent   ai agent name  (default: rave)
//   --size    board size     (required)
//   --budget  ms per move    (default: 500)
//   --max     games to generate, 0 = unlimited (default: 0)
//
// Output: one game per line, comma-separated.
//   First field : board size.
//   Each move   : two characters — column letter (a=0) then row letter (a=0).
//   Pass moves  : ..
//
// Example line for a 7x7 game:
//   7,dd,ac,cd,...,..,..,

const { Game2, PASS } = require('./game2.js');

const args = process.argv.slice(2);
const get = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

const agentName = get('--agent',  'rave');
if (!get('--size')) { console.error('--size is required'); process.exit(1); }
const size      = parseInt(get('--size'),   10);
const budget    = parseInt(get('--budget', '500'), 10);
const max       = parseInt(get('--max',    '0'),   10);

const agent = require(`./ai/${agentName}.js`);

const col = x => String.fromCharCode(97 + x);
const row = y => String.fromCharCode(97 + y);

let count = 0;
while (max === 0 || count < max) {
  count++;
  const g = new Game2(size);
  const parts = [size];

  // The constructor always places black at the board centre as move 1.
  const c = (size / 2) | 0;
  parts.push(col(c) + row(c));

  while (!g.gameOver) {
    const move = agent(g, budget);
    if (move.type === 'place') {
      g.play(move.y * size + move.x);
      parts.push(col(move.x) + row(move.y));
    } else {
      g.play(PASS);
      parts.push('..');
    }
  }

  console.log(parts.join(','));
}
