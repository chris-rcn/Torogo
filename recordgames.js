#!/usr/bin/env node
'use strict';

// playgames.js — play an agent against itself and print a game record per line.
//
// Usage: node playgames.js [agent] [size] [budget]
//   agent   ai agent name (default: rave)
//   size    board size    (default: 7)
//   budget  ms per move   (default: 500)
//
// Output: one game per line, comma-separated.
//   First field : board size.
//   Each move   : two characters — column letter (a=0) then row letter (a=0).
//   Pass moves  : ..
//
// Example line for a 7x7 game:
//   7,dd,ac,cd,...,..,..,

const { Game, DEFAULT_KOMI } = require('./game.js');

const [,, agentName = 'rave', sizeArg = '7', budgetArg = '500'] = process.argv;
const size   = parseInt(sizeArg,   10);
const budget = parseInt(budgetArg, 10);

const agent = require(`./ai/${agentName}.js`);

const col = x => String.fromCharCode(97 + x);
const row = y => String.fromCharCode(97 + y);

while (true) {
  const g = new Game(size, DEFAULT_KOMI);
  const parts = [size];

  // The constructor always places black at the board centre as move 1.
  const c = (size / 2) | 0;
  parts.push(col(c) + row(c));

  while (!g.gameOver) {
    const move = agent(g, budget);
    if (move.type === 'place') {
      g.placeStone(move.x, move.y);
      parts.push(col(move.x) + row(move.y));
    } else {
      g.pass();
      parts.push('..');
    }
  }

  console.log(parts.join(','));
}
