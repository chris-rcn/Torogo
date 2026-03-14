'use strict';
const { performance } = require('perf_hooks');
/**
 * Self-play script — play multiple games of one AI policy against another.
 *
 * Usage:
 *   node selfplay.js [options]
 *
 * Options:
 *   --p1    <policy>   AI policy for player 1      (default: random-move)
 *   --p2    <policy>   AI policy for player 2      (default: random-move)
 *   --games <n>        Number of games to play     (default: 100)
 *   --size  <n>        Board size: 9, 13, or 19    (default: 9)
 *   --help             Show this help message
 *
 * Colors alternate each game: p1 is black in odd games, white in even games.
 * Policy names are filenames without the .js extension inside the ai/ folder.
 *
 * Examples:
 *   node selfplay.js --p1 random-move --p2 always-pass --games 20
 *   node selfplay.js --size 13 --games 50
 *   node selfplay.js --p1 always-pass --p2 always-pass --games 5 --size 9
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
  console.log(`Usage: node selfplay.js [--p1 <policy>] [--p2 <policy>] [--games <n>] [--size <n>]`);
  process.exit(0);
}

const p1Name    = opts.p1    || 'random-move';
const p2Name    = opts.p2    || 'random-move';
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

console.log(`P1: ${p1Name}  |  P2: ${p2Name}  |  Games: ${numGames}  |  Board: ${boardSize}x${boardSize}`);
console.log(`Colors alternate each game (P1=black in game 1, P1=white in game 2, …)\n`);

const tally = { p1: 0, p2: 0, draw: 0 };
const stats = { p1: { ms: 0, moves: 0 }, p2: { ms: 0, moves: 0 } };
const verbose = numGames <= 20;

for (let g = 0; g < numGames; g++) {
  // Alternate colors: even g → p1=black, odd g → p1=white
  const p1IsBlack = g % 2 === 0;
  const black = p1IsBlack ? p1 : p2;
  const white = p1IsBlack ? p2 : p1;

  const game = new Game(boardSize);

  while (!game.gameOver) {
    const isBlackTurn = game.current === 'black';
    const policy = isBlackTurn ? black : white;
    const mover  = (isBlackTurn === p1IsBlack) ? 'p1' : 'p2';
    const t0 = performance.now();
    const move = policy(game);
    stats[mover].ms    += performance.now() - t0;
    stats[mover].moves += 1;
    if (move.type === 'place') {
      game.placeStone(move.x, move.y);
    } else {
      game.pass();
    }
  }

  const scores = game.scores;
  const blackScore = scores.black.total;
  const whiteScore = scores.white.total;

  let winner;
  if (blackScore > whiteScore) {
    const winningPlayer = p1IsBlack ? 'p1' : 'p2';
    tally[winningPlayer]++;
    winner = `BLACK (${winningPlayer === 'p1' ? p1Name : p2Name})`;
  } else if (whiteScore > blackScore) {
    const winningPlayer = p1IsBlack ? 'p2' : 'p1';
    tally[winningPlayer]++;
    winner = `WHITE (${winningPlayer === 'p1' ? p1Name : p2Name})`;
  } else {
    tally.draw++;
    winner = 'DRAW';
  }

  if (verbose) {
    const p1Color = p1IsBlack ? 'B' : 'W';
    const p2Color = p1IsBlack ? 'W' : 'B';
    console.log(
      `Game ${String(g + 1).padStart(2)} [p1=${p1Color} p2=${p2Color}]: ` +
      `B ${blackScore} (${scores.black.territory}t+${scores.black.captures}c)  ` +
      `W ${whiteScore} (${scores.white.territory}t+${scores.white.captures}c)  → ${winner}`
    );
  }
}

const wW = Math.max(4, String(numGames).length); // "wins" column width
const pct   = (n) => ((100 * n) / numGames).toFixed(1).padStart(6) + '%';
const avgMs = (s) => s.moves ? (s.ms / s.moves).toFixed(2).padStart(7) : '      —';
const wCol  = (n) => String(n).padStart(wW);
console.log(`\n${'='.repeat(50)}`);
console.log(`Results after ${numGames} game${numGames === 1 ? '' : 's'} on ${boardSize}x${boardSize}:`);
console.log(`  ${''.padEnd(6)}  ${'wins'.padStart(wW)}  ${'%'.padStart(7)}  ${'ms/move'.padStart(7)}  policy`);
console.log(`  P1:     ${wCol(tally.p1)}  ${pct(tally.p1)}  ${avgMs(stats.p1)}  ${p1Name}`);
console.log(`  P2:     ${wCol(tally.p2)}  ${pct(tally.p2)}  ${avgMs(stats.p2)}  ${p2Name}`);
console.log(`  Draws:  ${wCol(tally.draw)}  ${pct(tally.draw)}`);
