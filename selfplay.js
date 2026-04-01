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
 *   --p2      <policy>   AI policy for player 2      (default: p1)
 *   --size    <n>        Board size: 9, 13, or 19    (required)
 *   --budget  <ms>       Time budget per move in ms  (required)
 *   --limit   <n>        Stop after this many games and print final stats
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
const { Game2, BLACK, WHITE, PASS } = require('./game2.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help', 'verbose']);

if (opts.help) {
  console.log(`Usage: node selfplay.js [--p1 <policy>] [--p2 <policy>] [--size <n>] [--budget <ms>] [--limit <n>] [--verbose]`);
  process.exit(0);
}

const gameLimit = opts.limit !== undefined ? parseInt(opts.limit, 10) : Infinity;
if (isNaN(gameLimit) || gameLimit < 1) {
  console.error('--limit must be a positive integer');
  process.exit(1);
}

const p1Name    = opts.p1   || 'random';
const p2Name    = opts.p2   || p1Name;
if (!opts.size) { console.error('--size is required'); process.exit(1); }
const boardSize = parseInt(opts.size, 10);
const budgetMs  = parseInt(opts.budget || '1', 10);

if (!Number.isInteger(boardSize)) {
  console.error('--size must be an odd integer between 7 and 19');
  process.exit(1);
}

const { getMove: p1 } = require(path.join(__dirname, 'ai', p1Name + '.js'));
const { getMove: p2 } = require(path.join(__dirname, 'ai', p2Name + '.js'));

function printBoard(game) {
  console.log(game.toString());
  if (game.lastMove === PASS) {
    const passer = game.current === BLACK ? 'White' : 'Black';
    console.log(passer + ' passed');
  }
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
  `  ${'p1ms'.padStart(MW)}  ${'p2ms'.padStart(MW)}  maxLen`;
console.log(hdr);

let printPeriodMs  = 1000;
let lastPrintTime  = startTime;
let lastPrintGames = 0;
let maxGameLen = 0;

function printStats(gamesPlayed) {
  const now     = performance.now();
  const pct     = (w) => ((100 * w / gamesPlayed).toFixed(1) + '%').padStart(PW);
  const avgMs   = (s) => (s.moves ? (s.ms / s.moves).toFixed(2) : '—').padStart(MW);
  const elapsed = (((now - startTime) / 1000).toFixed(1) + 's').padStart(EW);
  console.log(
    `${String(gamesPlayed).padStart(GW)}` +
    `  ${elapsed}` +
    `  ${pct(tally.p2)}` +
    `  ${avgMs(stats.p1)}  ${avgMs(stats.p2)}  ${String(maxGameLen).padStart(6)}`
  );
}

function maybePrint(gamesPlayed) {
  const now = performance.now();
  if (now - lastPrintTime < printPeriodMs) return;
  if (gamesPlayed === lastPrintGames) return;

  lastPrintTime  = now;
  lastPrintGames = gamesPlayed;
  printStats(gamesPlayed);
  printPeriodMs = Math.round(printPeriodMs * 1.5);
}

// Run games until the limit (or forever if no limit).
for (let g = 0; g < gameLimit; g++) {
  const p1IsBlack = g % 2 === 0;
  const names = [ p1IsBlack ? p1Name : p2Name, p1IsBlack ? p2Name : p1Name]
  const black = p1IsBlack ? p1 : p2;
  const white = p1IsBlack ? p2 : p1;

  if (verboseBoard) {
    console.log(`${names[0]} ● vs ${names[1]} ○`);
  }

  const game = new Game2(boardSize);

  while (!game.gameOver) {
    const isBlackTurn = game.current === BLACK;
    const policy = isBlackTurn ? black : white;
    const mover  = (isBlackTurn === p1IsBlack) ? 'p1' : 'p2';
    const t0 = performance.now();
    const move = policy(game, budgetMs);
    stats[mover].ms    += performance.now() - t0;
    stats[mover].moves += 1;
    const idx = move.move !== undefined ? move.move : (move.type === 'place' ? move.y * boardSize + move.x : PASS);
    if (!game.play(idx)) {
      console.error(`Illegal move from ${mover} (${p1IsBlack ? p1Name : p2Name}): ${JSON.stringify(move)}`);
      process.exit(1);
    }
    if (verboseBoard) {
      console.log(`${names[isBlackTurn?0:1]}:`);
      printBoard(game);
      console.log(`Agent info: ${move.info}`);
      console.log();
    };
  }
  maxGameLen = Math.max(maxGameLen, game.moveCount);
  const winner = game.calcWinner();
  if (winner === BLACK) {
    tally[p1IsBlack ? 'p1' : 'p2']++;
  } else if (winner === WHITE) {
    tally[p1IsBlack ? 'p2' : 'p1']++;
  }

  maybePrint(g + 1);
}

// Final stats row (always printed, even if maybePrint already fired).
printStats(tally.p1 + tally.p2);

