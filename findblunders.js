#!/usr/bin/env node
'use strict';

// findblunders.js — generate a blunder dataset by self-play.
//
// Usage: node findblunders.js [--agent <name>] [--budget <ms>] [--long-budget <ms>] [--size <n>]
//   --agent        AI policy name           (default: rave)
//   --budget       short-budget ms          (default: 100)
//   --long-budget  long-budget ms           (default: 4× --budget)
//   --size         board size               (default: 11)
//
// For each position in a self-play game, two short-budget and two long-budget
// genMove calls are made.  If both short-budget calls agree on the same move
// but that move disagrees with both long-budget calls, the position is
// emitted as a blunder: the short-budget move is added to "prohibited".
// Once a blunder is found the current game is abandoned and a new one starts.
// Positions already emitted are tracked by Zobrist hash and skipped if
// encountered again in a later game.
//
// Output: one JSON object per line (NDJSON), shaped like an evalladders position:
//   { "toPlay": "●", "board": "...", "prohibited": ["j8"], "comment": "..." }

const path = require('path');
const { Game, DEFAULT_KOMI } = require('./game.js');

const args = process.argv.slice(2);
const get  = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

if (args.includes('--help') || args.includes('-h')) {
  console.error('Usage: node findblunders.js [--agent <name>] [--budget <ms>] [--longbudget <ms>] [--size <n>]');
  process.exit(0);
}

const agentName   = get('--agent',       'rave');
const shortBudget = parseInt(get('--budget',      '100'), 10);
const boardSize   = parseInt(get('--size',         '11'), 10);

if (isNaN(shortBudget) || shortBudget < 1) {
  console.error('--budget must be a positive integer'); process.exit(1);
}
if (isNaN(boardSize) || boardSize < 2) {
  console.error('--size must be >= 2'); process.exit(1);
}

const longBudget = parseInt(get('--longbudget', String(shortBudget * 4)), 10);
if (isNaN(longBudget) || longBudget <= shortBudget) {
  console.error('--longbudget must be an integer greater than --budget'); process.exit(1);
}

const agent = require(path.join(__dirname, 'ai', agentName + '.js'));

function coordStr(move) {
  if (move.type === 'pass') return 'pass';
  return String.fromCharCode(97 + move.x) + (move.y + 1);
}

function sameMove(a, b) {
  if (a.type !== b.type) return false;
  return a.type === 'pass' || (a.x === b.x && a.y === b.y);
}

// Positions already emitted, keyed by "<zobrist>|<color>" to distinguish
// identical boards where different players are to move.
const seen = new Set();

let gameCount    = 0;
let blunderCount = 0;

// Run forever; pipe output to a file or Ctrl-C to stop.
while (true) {
  gameCount++;
  const game = new Game(boardSize, DEFAULT_KOMI);

  while (!game.gameOver) {
    const posKey = `${game.hash}|${game.current}`;

    // Two short-budget calls on the current position.
    const s1 = agent(game.clone(), shortBudget);
    const s2 = agent(game.clone(), shortBudget);

    // Skip positions already emitted to avoid duplicates across games.
    if (sameMove(s1, s2) && !seen.has(posKey)) {
      const l1 = agent(game.clone(), longBudget);
      const l2 = agent(game.clone(), longBudget);

      if (!sameMove(s1, l1) && !sameMove(s1, l2)) {
        seen.add(posKey);
        blunderCount++;
        const toPlay     = game.current === 'black' ? '●' : '○';
        const board      = game.board.toAscii();
        const prohibit = [coordStr(s1)];
        const comment    = `game ${gameCount} blunder ${blunderCount}: ` +
                           `short-budget plays ${coordStr(s1)}, ` +
                           `long-budget plays ${coordStr(l1)} / ${coordStr(l2)}`;
        process.stdout.write(JSON.stringify({ toPlay, board, prohibit, comment }) + '\n');
        break;  // abandon this game; outer loop starts a new one
      }
    }

    // Advance the game with the first short-budget move.
    if (s1.type === 'place') game.placeStone(s1.x, s1.y);
    else game.pass();
  }
}
