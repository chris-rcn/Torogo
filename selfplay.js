'use strict';
const { performance } = require('perf_hooks');
/**
 * Self-play script — play multiple games of one AI policy against another.
 *
 * Usage:
 *   node selfplay.js [options]
 *
 * Options:
 *   --p1      <policy>   AI policy for player 1      (default: random)
 *   --p2      <policy>   AI policy for player 2      (default: random)
 *   --games   <n>        Number of games to play     (default: 100)
 *   --size    <n>        Board size: 9, 13, or 19    (default: 9)
 *   --komi    <n>        Komi (white's bonus points) (default: 3.5)
 *   --verbose            Print the board after every move
 *   --help               Show this help message
 *
 * Colors alternate each game: p1 is black in odd games, white in even games.
 * Policy names are filenames without the .js extension inside the ai/ folder.
 *
 * Examples:
 *   node selfplay.js --p1 random --p2 always-pass --games 20
 *   node selfplay.js --size 13 --games 50
 *   node selfplay.js --p1 always-pass --p2 always-pass --games 5 --size 9
 */

const path = require('path');
const { Game, DEFAULT_KOMI } = require('./game.js');

// Boolean flags that take no value.
const BOOL_FLAGS = new Set(['help', 'verbose']);

// Parse --key value switches from argv.
function parseArgs(argv) {
  const opts = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') { opts.help = true; continue; }
    if (arg.startsWith('--')) {
      const key = arg.slice(2);
      if (BOOL_FLAGS.has(key)) { opts[key] = true; continue; }
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
  console.log(`Usage: node selfplay.js [--p1 <policy>] [--p2 <policy>] [--games <n>] [--size <n>] [--komi <n>] [--verbose]`);
  process.exit(0);
}

const p1Name    = opts.p1    || 'random';
const p2Name    = opts.p2    || 'random';
const numGames  = parseInt(opts.games || '100', 10);
const boardSize = parseInt(opts.size  || '9',   10);
const komi      = opts.komi !== undefined ? parseFloat(opts.komi) : DEFAULT_KOMI;

if (isNaN(numGames) || numGames < 1) {
  console.error('--games must be a positive integer');
  process.exit(1);
}
if (!Number.isInteger(boardSize) || boardSize < 7 || boardSize > 19 || boardSize % 2 === 0) {
  console.error('--size must be an odd integer between 7 and 19');
  process.exit(1);
}
if (!Number.isFinite(komi)) {
  console.error('--komi must be a number');
  process.exit(1);
}

const p1 = require(path.join(__dirname, 'ai', p1Name + '.js'));
const p2 = require(path.join(__dirname, 'ai', p2Name + '.js'));

function printBoard(game) {
  const size = game.boardSize;
  const cols = 'ABCDEFGHJKLMNOPQRST'.slice(0, size); // skip 'I' like real Go
  console.log('   ' + cols.split('').map(c => ' ' + c + ' ').join(''));
  const last = game.lastMove; // { x, y } or null
  for (let y = 0; y < size; y++) {
    const row = String(size - y).padStart(2) + ' ';
    const cells = [];
    for (let x = 0; x < size; x++) {
      const v = game.board.get(x, y);
      const ch = v === 'black' ? '●' : v === 'white' ? '○' : '·';
      if (last && x === last.x && y === last.y) {
        cells.push('(' + ch + ')');
      } else {
        cells.push(' ' + ch + ' ');
      }
    }
    console.log(row + cells.join(''));
  }
  console.log();
}


const tally = { p1: 0, p2: 0, draw: 0, black: 0, white: 0 };
const stats = { p1: { ms: 0, moves: 0 }, p2: { ms: 0, moves: 0 } };
const verbose = numGames <= 20;
const verboseBoard = !!opts.verbose;

// Column widths for per-game score output, derived from board size.
const scoreW = String(boardSize * boardSize * 2).length + 2; // +2 for '.5' from komi
const statW  = String(boardSize * boardSize).length;         // territory / captures

for (let g = 0; g < numGames; g++) {
  // Alternate colors: even g → p1=black, odd g → p1=white
  const p1IsBlack = g % 2 === 0;
  const black = p1IsBlack ? p1 : p2;
  const white = p1IsBlack ? p2 : p1;

  const game = new Game(boardSize, komi);

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
    if (verboseBoard) printBoard(game);
  }

  const scores = game.scores;
  const blackScore = scores.black.total;
  const whiteScore = scores.white.total;

  let winner;
  if (blackScore > whiteScore) {
    tally.black++;
    const winningPlayer = p1IsBlack ? 'p1' : 'p2';
    tally[winningPlayer]++;
    winner = `BLACK (${winningPlayer === 'p1' ? p1Name : p2Name})`;
  } else if (whiteScore > blackScore) {
    tally.white++;
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
    const fmtScore = (n) => String(n).padStart(scoreW);
    const fmtStat  = (n) => String(n).padStart(statW);
    console.log(
      `Game ${String(g + 1).padStart(2)} [p1=${p1Color} p2=${p2Color}]:` +
      `  B ${fmtScore(blackScore)} (${fmtStat(scores.black.territory)}t+${fmtStat(scores.black.captures)}c)` +
      `  W ${fmtScore(whiteScore)} (${fmtStat(scores.white.territory)}t+${fmtStat(scores.white.captures)}c)` +
      `  → ${winner}`
    );
  }
}

const labelW = 'Black:'.length;
const wW     = Math.max(4, String(numGames).length);
const label  = (s) => s.padEnd(labelW);
const wCol   = (n) => String(n).padStart(wW);
const pct    = (n) => ((100 * n) / numGames).toFixed(1).padStart(6) + '%';
const avgMs  = (s) => s.moves ? (s.ms / s.moves).toFixed(2).padStart(7) : '      —';
console.log(`  ${label('')}  ${'wins'.padStart(wW)}  ${'%'.padStart(7)}  ${'ms/move'.padStart(7)}  policy`);
console.log(`  ${label('P1:')}  ${wCol(tally.p1)}  ${pct(tally.p1)}  ${avgMs(stats.p1)}  ${p1Name}`);
console.log(`  ${label('P2:')}  ${wCol(tally.p2)}  ${pct(tally.p2)}  ${avgMs(stats.p2)}  ${p2Name}`);
if (p1Name === p2Name)
  console.log(`  ${label('Black:')}  ${wCol(tally.black)}  ${pct(tally.black)}  ${''.padStart(7)}  (color win rate, komi=${komi})`);
