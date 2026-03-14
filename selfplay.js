'use strict';
/**
 * Self-play script — play multiple games of one AI policy against another.
 *
 * Usage:
 *   node selfplay.js [options]
 *
 * Options:
 *   --black <policy>   AI policy for Black        (default: random-move)
 *   --white <policy>   AI policy for White        (default: random-move)
 *   --games <n>        Number of games to play    (default: 100)
 *   --size  <n>        Board size: 9, 13, or 19   (default: 9)
 *   --help             Show this help message
 *
 * Policy names are filenames without the .js extension inside the ai/ folder.
 *
 * Examples:
 *   node selfplay.js --black random-move --white always-pass --games 20
 *   node selfplay.js --size 13 --games 50
 *   node selfplay.js --black always-pass --white always-pass --games 5 --size 9
 */

const path = require('path');
const { Game } = require('./game.js');

// Parse --key value switches from argv.
function parseArgs(argv) {
  const opts = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') { opts.help = true; continue; }
    if (arg.startsWith('--')) {
      const key = arg.slice(2);
      const val = argv[i + 1];
      if (val === undefined || val.startsWith('--')) {
        console.error(`Missing value for ${arg}`);
        process.exit(1);
      }
      opts[key] = val;
      i++;
    } else {
      console.error(`Unknown argument: ${arg}`);
      process.exit(1);
    }
  }
  return opts;
}

const opts = parseArgs(process.argv.slice(2));

if (opts.help) {
  console.log(`Usage: node selfplay.js [--black <policy>] [--white <policy>] [--games <n>] [--size <n>]`);
  process.exit(0);
}

const p1Name    = opts.black || 'random-move';
const p2Name    = opts.white || 'random-move';
const numGames  = parseInt(opts.games || '100', 10);
const boardSize = parseInt(opts.size  || '9',   10);

if (isNaN(numGames) || numGames < 1) {
  console.error('--games must be a positive integer');
  process.exit(1);
}
if (![9, 13, 19].includes(boardSize)) {
  console.error('--size must be 9, 13, or 19');
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
