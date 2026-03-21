#!/usr/bin/env node
'use strict';

// findblunders.js — generate a blunder dataset by self-play.
//
// Usage: node findblunders.js [--agent <name>] [--budget <ms>] [--longbudget <ms>] [--size <n>]
//   --agent        AI policy name           (default: rave)
//   --budget       short-budget ms          (default: 100)
//   --longbudget   long-budget ms           (default: 2× --budget)
//   --size         board size               (default: 11)
//
// For each position in a self-play game, three short-budget and two long-budget
// genMove calls are made.  If all three short-budget calls agree on the same
// move, both long-budget calls agree on a different move, AND the average
// long-budget winRatio differs from the average short-budget winRatio by at least
// WIN_RATIO_DIFF_THRESH, the position is emitted as a blunder: the
// short-budget move is added to "prohibited".
// Once a blunder is found the current game is abandoned and a new one starts.
// Positions already emitted are tracked by Zobrist hash and skipped if
// encountered again in a later game.
//
// Output: JS object literals formatted like evalladders positions, one per blunder.
//   Paste the output into the POSITIONS array in evalladders.js.

const path = require('path');
const { Game } = require('./game.js');

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

const longBudget = parseInt(get('--longbudget', String(2 * shortBudget)), 10);
if (isNaN(longBudget) || longBudget <= shortBudget) {
  console.error('--longbudget must be an integer greater than --budget'); process.exit(1);
}

const agent = require(path.join(__dirname, 'ai', agentName + '.js'));

const WIN_RATIO_DIFF_THRESH = 0.05;

// ─── Header ───────────────────────────────────────────────────────────────────

console.log('// Auto-generated blunder positions');
console.log(`// Generated using: agent=${agentName} size=${boardSize} short=${shortBudget}ms long=${longBudget}ms`);
console.log('');

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
  const game = new Game(boardSize);

  while (!game.gameOver) {
    const posKey = `${game.board.toAscii()}|${game.current}`;

    // Three short-budget calls on the current position.
    const s1 = agent(game.clone(), shortBudget);
    const s2 = agent(game.clone(), shortBudget);
    const s3 = agent(game.clone(), shortBudget);

    for (const r of [s1, s2, s3]) {
      if (r.winRatio === undefined) { console.error('agent did not return winRatio'); process.exit(1); }
    }

    // Skip positions already emitted to avoid duplicates across games.
    if (sameMove(s1, s2) && sameMove(s1, s3) && !seen.has(posKey)) {
      const l1 = agent(game.clone(), longBudget);
      const l2 = agent(game.clone(), longBudget);

      for (const r of [l1, l2]) {
        if (r.winRatio === undefined) { console.error('agent did not return winRatio'); process.exit(1); }
      }

      const shortAvg = (s1.winRatio + s2.winRatio + s3.winRatio) / 3;
      const longAvg  = (l1.winRatio + l2.winRatio) / 2;

      if (sameMove(l1, l2) && !sameMove(s1, l1) && Math.abs(longAvg - shortAvg) > WIN_RATIO_DIFF_THRESH) {
        seen.add(posKey);
        blunderCount++;
        const toPlayChar = game.current === 'black' ? '●' : '○';
        const comment    = `game ${gameCount} blunder ${blunderCount}: ` +
                           `short-budget plays ${coordStr(s1)} (winRatio ${shortAvg.toFixed(3)}), ` +
                           `long-budget plays ${coordStr(l1)} (winRatio ${longAvg.toFixed(3)})`;
        const indented   = game.board.toAscii(s1).split('\n').map(r => '      ' + r).join('\n');
        console.log(`  {`);
        console.log(`    toPlay: '${toPlayChar}',`);
        console.log(`    comment: '${comment}',`);
        console.log(`    prohibit: ['${coordStr(s1)}'],`);
        console.log(`    board: \``);
        console.log(indented);
        console.log(`    \`,`);
        console.log(`  },`);
        break;  // abandon this game; outer loop starts a new one
      }
    }

    // Advance the game with the first short-budget move.
    if (s1.type === 'place') game.placeStone(s1.x, s1.y);
    else game.pass();
  }
}
