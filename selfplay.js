'use strict';
/**
 * Self-play script — play multiple games of one AI policy against another.
 *
 * Usage:
 *   node selfplay.js [policy1] [policy2] [numGames] [boardSize]
 *
 * Arguments (all optional, positional):
 *   policy1   - AI policy for Black (default: random-move)
 *   policy2   - AI policy for White (default: random-move)
 *   numGames  - Number of games to play  (default: 100)
 *   boardSize - Board size: 9, 13, or 19 (default: 9)
 *
 * Policy names are filenames without the .js extension inside the ai/ folder.
 *
 * Examples:
 *   node selfplay.js random-move always-pass 20 9
 *   node selfplay.js always-pass always-pass 5
 *   node selfplay.js random-move random-move 100 13
 */

const path = require('path');
const { Game } = require('./game.js');

const args = process.argv.slice(2);
const p1Name    = args[0] || 'random-move';
const p2Name    = args[1] || 'random-move';
const numGames  = parseInt(args[2] || '100', 10);
const boardSize = parseInt(args[3] || '9',   10);

if (isNaN(numGames) || numGames < 1) {
  console.error('numGames must be a positive integer');
  process.exit(1);
}
if (![9, 13, 19].includes(boardSize)) {
  console.error('boardSize must be 9, 13, or 19');
  process.exit(1);
}

const p1 = require(path.join(__dirname, 'ai', p1Name + '.js'));
const p2 = require(path.join(__dirname, 'ai', p2Name + '.js'));

console.log(`Black: ${p1Name}  |  White: ${p2Name}  |  Games: ${numGames}  |  Board: ${boardSize}x${boardSize}\n`);

const tally = { black: 0, white: 0, draw: 0 };
const verbose = numGames <= 20;

for (let g = 0; g < numGames; g++) {
  const game = new Game(boardSize);

  while (!game.gameOver) {
    const policy = game.current === 'black' ? p1 : p2;
    const move = policy(game);
    if (move.type === 'place') {
      game.placeStone(move.x, move.y);
    } else {
      game.pass();
    }
  }

  const { black, white } = game.scores;
  let winner;
  if (black.total > white.total)       { tally.black++; winner = 'BLACK'; }
  else if (white.total > black.total)  { tally.white++; winner = 'WHITE'; }
  else                                 { tally.draw++;  winner = 'DRAW';  }

  if (verbose) {
    console.log(
      `Game ${String(g + 1).padStart(2)}: ` +
      `B ${black.total} (${black.territory}t+${black.captures}c)  ` +
      `W ${white.total} (${white.territory}t+${white.captures}c)  → ${winner}`
    );
  }
}

const pct = (n) => ((100 * n) / numGames).toFixed(1) + '%';
console.log(`\n${'='.repeat(50)}`);
console.log(`Results after ${numGames} game${numGames === 1 ? '' : 's'} on ${boardSize}x${boardSize}:`);
console.log(`  Black (${p1Name}): ${tally.black} wins (${pct(tally.black)})`);
console.log(`  White (${p2Name}): ${tally.white} wins (${pct(tally.white)})`);
console.log(`  Draws:             ${tally.draw}  (${pct(tally.draw)})`);
