#!/usr/bin/env node
'use strict';

// findblunders.js — generate a blunder dataset by self-play.
//
// Usage: node findblunders.js [--agent <name>] [--budget <ms>] [--size <n>]
//   --agent   AI policy name  (default: ravepat)
//   --budget  short-budget ms (default: 50); long budget is 10×
//   --size    board size      (default: 9)
//
// For each position in a self-play game, two short-budget and two long-budget
// genMove calls are made.  If both short-budget calls agree on the same place
// move but that move disagrees with both long-budget calls, the position is
// emitted as a blunder: the short-budget move is added to "prohibited".
// Once a blunder is found the current game is abandoned and a new one starts.
//
// Output: one JSON object per line (NDJSON), shaped like an evalladders position:
//   { "toPlay": "●", "board": "...", "prohibited": ["j8"], "comment": "..." }

const path = require('path');
const { Game, DEFAULT_KOMI } = require('./game.js');

const args = process.argv.slice(2);
const get  = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

if (args.includes('--help') || args.includes('-h')) {
  console.error('Usage: node findblunders.js [--agent <name>] [--budget <ms>] [--size <n>]');
  process.exit(0);
}

const agentName   = get('--agent',  'ravepat');
const shortBudget = parseInt(get('--budget', '50'), 10);
const boardSize   = parseInt(get('--size',   '9'),  10);

if (isNaN(shortBudget) || shortBudget < 1) {
  console.error('--budget must be a positive integer'); process.exit(1);
}
if (isNaN(boardSize) || boardSize < 2) {
  console.error('--size must be >= 2'); process.exit(1);
}

const LONG_FACTOR = 10;
const longBudget  = shortBudget * LONG_FACTOR;

const agent = require(path.join(__dirname, 'ai', agentName + '.js'));

function coordStr(move) {
  return String.fromCharCode(97 + move.x) + (move.y + 1);
}

function sameMove(a, b) {
  if (a.type !== b.type) return false;
  return a.type === 'pass' || (a.x === b.x && a.y === b.y);
}

let gameCount   = 0;
let blunderCount = 0;

// Run forever; pipe output to a file or Ctrl-C to stop.
while (true) {
  gameCount++;
  const game = new Game(boardSize, DEFAULT_KOMI);

  while (!game.gameOver) {
    // Two short-budget calls on the current position.
    const s1 = agent(game.clone(), shortBudget);
    const s2 = agent(game.clone(), shortBudget);

    // Only flag place moves; passes are not useful blunder entries.
    if (s1.type === 'place' && sameMove(s1, s2)) {
      const l1 = agent(game.clone(), longBudget);
      const l2 = agent(game.clone(), longBudget);

      if (!sameMove(s1, l1) && !sameMove(s1, l2)) {
        blunderCount++;
        const toPlay     = game.current === 'black' ? '●' : '○';
        const board      = game.board.toAscii();
        const prohibited = [coordStr(s1)];
        const comment    = `game ${gameCount} blunder ${blunderCount}: ` +
                           `short-budget plays ${coordStr(s1)}, ` +
                           `long-budget plays ${coordStr(l1)} / ${coordStr(l2)}`;
        process.stdout.write(JSON.stringify({ toPlay, board, prohibited, comment }) + '\n');
        break;  // abandon this game; outer loop starts a new one
      }
    }

    // Advance the game with the first short-budget move.
    if (s1.type === 'place') game.placeStone(s1.x, s1.y);
    else game.pass();
  }
}
