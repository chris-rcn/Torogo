'use strict';
const { performance } = require('perf_hooks');
/**
 * Self-play script — play games of one AI policy against another indefinitely.
 *
 * Usage:
 *   node selfplay.js [options]
 *
 * Options:
 *   --p1      <policy>   AI policy for player 1      (default: random)
 *   --p2      <policy>   AI policy for player 2      (default: random)
 *   --size    <n>        Board size: 9, 13, or 19    (default: 9)
 *   --budget  <ms>       Time budget per move in ms  (default: 500)
 *   --verbose            Print the board after every move
 *   --help               Show this help message
 *
 * Colors alternate each game: p1 is black in odd games, white in even games.
 * Policy names are filenames without the .js extension inside the ai/ folder.
 *
 * Examples:
 *   node selfplay.js --p1 random --p2 always-pass
 *   node selfplay.js --size 13
 *   node selfplay.js --p1 always-pass --p2 always-pass --size 9
 */

const path = require('path');
const { Game, KOMI } = require('./game.js');

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
  console.log(`Usage: node selfplay.js [--p1 <policy>] [--p2 <policy>] [--size <n>] [--budget <ms>] [--verbose]`);
  process.exit(0);
}

if (opts.games !== undefined) {
  console.error('--games is not supported; this script runs indefinitely');
  process.exit(1);
}

const p1Name    = opts.p1   || 'random';
const p2Name    = opts.p2   || 'random';
const boardSize = parseInt(opts.size   || '9',   10);
const budgetMs  = parseInt(opts.budget || '500', 10);

if (!Number.isInteger(boardSize) || boardSize < 7 || boardSize > 19 || boardSize % 2 === 0) {
  console.error('--size must be an odd integer between 7 and 19');
  process.exit(1);
}

const p1 = require(path.join(__dirname, 'ai', p1Name + '.js'));
const p2 = require(path.join(__dirname, 'ai', p2Name + '.js'));

function printBoard(game) {
  console.log(game.board.toAscii(game.lastMove));
  if (!game.lastMove) {
    const passer = game.current === 'black' ? 'White' : 'Black';
    console.log(passer + ' passed');
  }
  console.log();
}

const tally = { p1: 0, p2: 0 };
const stats  = { p1: { ms: 0, moves: 0 }, p2: { ms: 0, moves: 0 } };
const startTime = performance.now();
const verboseBoard = !!opts.verbose;

// Column widths for the summary table.
const GW = 6;   // games
const PW = 6;   // percentage  "66.7%"
const MW = 7;   // ms/move     "123.45"
const EW = 8;   // elapsed     "1234.5s"

const hdr =
  `${'games'.padStart(GW)}` +
  `  ${'elapsed'.padStart(EW)}` +
  `  ${'p2%'.padStart(PW)}` +
  `  ${'p1ms'.padStart(MW)}  ${'p2ms'.padStart(MW)}`;
console.log(hdr);

let printPeriodMs  = 1000;
let lastPrintTime  = startTime;
let lastPrintGames = 0;

function maybePrint(gamesPlayed) {
  const now = performance.now();
  if (now - lastPrintTime < printPeriodMs) return;
  if (gamesPlayed === lastPrintGames) return;

  lastPrintTime  = now;
  lastPrintGames = gamesPlayed;

  const n       = gamesPlayed;
  const pct     = (w) => ((100 * w / n).toFixed(1) + '%').padStart(PW);
  const avgMs   = (s) => (s.moves ? (s.ms / s.moves).toFixed(2) : '—').padStart(MW);
  const elapsed = (((now - startTime) / 1000).toFixed(1) + 's').padStart(EW);

  console.log(
    `${String(n).padStart(GW)}` +
    `  ${elapsed}` +
    `  ${pct(tally.p2)}` +
    `  ${avgMs(stats.p1)}  ${avgMs(stats.p2)}`
  );

  printPeriodMs = Math.round(printPeriodMs * 1.5);
}

// Run games forever.
for (let g = 0; ; g++) {
  const p1IsBlack = g % 2 === 0;
  const black = p1IsBlack ? p1 : p2;
  const white = p1IsBlack ? p2 : p1;

  const game = new Game(boardSize);

  while (!game.gameOver) {
    const isBlackTurn = game.current === 'black';
    const policy = isBlackTurn ? black : white;
    const mover  = (isBlackTurn === p1IsBlack) ? 'p1' : 'p2';
    const t0 = performance.now();
    const move = policy(game, budgetMs);
    stats[mover].ms    += performance.now() - t0;
    stats[mover].moves += 1;
    if (move.type === 'place') {
      game.placeStone(move.x, move.y);
    } else {
      game.pass();
    }
    if (verboseBoard) printBoard(game);
  }

  const t = game.calcTerritory();
  if (t.black > t.white + KOMI) {
    tally[p1IsBlack ? 'p1' : 'p2']++;
  } else if (t.white + KOMI > t.black) {
    tally[p1IsBlack ? 'p2' : 'p1']++;
  }

  maybePrint(g + 1);
}
