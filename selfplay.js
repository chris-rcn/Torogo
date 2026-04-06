'use strict';
const { performance } = require('perf_hooks');
/**
 * Self-play script — play games of one AI policy against another indefinitely.
 *
 * Usage:
 *   node selfplay.js [options]
 *
 * Options:
 *   --p1      <policy>   AI policy for player 1      (default: prod)
 *   --p2      <policy>   AI policy for player 2      (default: p1)
 *   --size    <n>        Board size: 9, 13, or 19    (required)
 *   --budget  <ms>       Time budget per move in ms  (required)
 *   --limit   <n>        Stop after this many games and print final stats
 *   --help               Show this help message
 *
 * Env variables:
 *   VERBOSE=1            Print the board after every move
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

const VERBOSE = Util.envInt('VERBOSE', 0);

const opts = Util.parseArgs(process.argv.slice(2), ['help']);

if (opts.help) {
  console.log(`Usage: node selfplay.js [--p1 <policy>] [--p2 <policy>] [--size <n>] [--budget <ms>] [--limit <n>]`);
  process.exit(0);
}

const gameLimit = opts.limit !== undefined ? parseInt(opts.limit, 10) : Infinity;
if (isNaN(gameLimit) || gameLimit < 1) {
  console.error('--limit must be a positive integer');
  process.exit(1);
}

const p1Name    = opts.p1   || 'prod';
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

// Column widths for the summary table.
const GW = 6;   // games
const PW = 6;   // percentage  "66.7%"
const MW = 7;   // ms/move     "123.45"
const EW = 8;   // elapsed     "1234.5s"

console.log(
  `${'games'.padStart(6)}  ` +
  `${'elapsed'.padStart(8)}  ` +
  `${'black%'.padStart(6)}  ` +
  `${'maxLen'.padStart(6)}  ` +
  `${'p1ms'.padStart(7)}  ` +
  `${'p2ms'.padStart(7)}  ` +
  `${'p2%'.padStart(6)}  ` +
  `${'p2Better%'.padStart(9)}  ` +
  ``);

let printPeriodMs  = 1000;
let lastPrintTime  = startTime;
let lastPrintGames = 0;
let blackWinCount = 0;
let maxGameLen = 0;

function printStats(gamesPlayed) {
  const now = performance.now();
  const strGames = String(gamesPlayed);
  const strElapsed = (((now - startTime) / 1000).toFixed(1) + 's');
  const strBlackWin = (100 * blackWinCount / gamesPlayed).toFixed(1) + '%';
  const strGameLen = String(maxGameLen);
  const strAvgMs = (s) => (s.moves ? (s.ms / s.moves).toFixed(2) : '—');
  const strWinRatio = (w) => ((100 * w / gamesPlayed).toFixed(1) + '%');
  const strP2Better = (100 * probPlayerBetter(tally.p2, gamesPlayed)).toFixed(1) + '%';
  console.log(
    `${strGames.padStart(6)}  ` +
    `${strElapsed.padStart(8)}  ` +
    `${strBlackWin.padStart(6)}  ` +
    `${strGameLen.padStart(6)}  ` +
    `${strAvgMs(stats.p1).padStart(7)}  ` +
    `${strAvgMs(stats.p2).padStart(7)}  ` +
    `${strWinRatio(tally.p2).padStart(6)}  ` +
    `${strP2Better.padStart(9)}  ` +
    ``);
}

function maybePrint(gamesPlayed) {
  if (VERBOSE) {
    printStats(gamesPlayed);
    return;
  }
  const now = performance.now();
  if (now - lastPrintTime < printPeriodMs) return;
  if (gamesPlayed === lastPrintGames) return;

  lastPrintTime  = now;
  lastPrintGames = gamesPlayed;
  printStats(gamesPlayed);
  printPeriodMs = Math.round(printPeriodMs * 1.5);
}

// Probability that true win rate p > 0.5 given w wins out of n games
// Uses a Beta(w+1, n-w+1) posterior with uniform prior

function probPlayerBetter(w, n) {
  if (w < 0 || n <= 0 || w > n) {
    throw new Error("Invalid inputs");
  }

  const a = w + 1;
  const b = n - w + 1;

  return 1 - regularizedIncompleteBeta(0.5, a, b);
}

/*
 * Regularized incomplete beta function Ix(a,b)
 * Implementation via continued fraction (Numerical Recipes style)
 */

function regularizedIncompleteBeta(x, a, b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;

  const bt =
    Math.exp(
      logGamma(a + b) -
      logGamma(a) -
      logGamma(b) +
      a * Math.log(x) +
      b * Math.log(1 - x)
    );

  if (x < (a + 1) / (a + b + 2)) {
    return (bt * betaCF(x, a, b)) / a;
  } else {
    return 1 - (bt * betaCF(1 - x, b, a)) / b;
  }
}

function betaCF(x, a, b) {
  const MAX_ITER = 100;
  const EPS = 1e-12;

  let am = 1;
  let bm = 1;
  let az = 1;
  let qab = a + b;
  let qap = a + 1;
  let qam = a - 1;
  let bz = 1 - qab * x / qap;

  for (let m = 1; m <= MAX_ITER; m++) {
    let em = m;
    let tem = em + em;

    let d = em * (b - em) * x / ((qam + tem) * (a + tem));
    let ap = az + d * am;
    let bp = bz + d * bm;

    d = -(a + em) * (qab + em) * x / ((a + tem) * (qap + tem));
    let app = ap + d * az;
    let bpp = bp + d * bz;

    let aold = az;
    am = ap / bpp;
    bm = bp / bpp;
    az = app / bpp;
    bz = 1;

    if (Math.abs(az - aold) < EPS * Math.abs(az)) {
      return az;
    }
  }

  return az;
}

// Lanczos approximation for log gamma
function logGamma(z) {
  const cof = [
    76.18009172947146, -86.50532032941677,
    24.01409824083091, -1.231739572450155,
    0.001208650973866179, -0.000005395239384953
  ];

  let x = z;
  let y = z;
  let tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);

  let ser = 1.000000000190015;
  for (let j = 0; j < cof.length; j++) {
    y += 1;
    ser += cof[j] / y;
  }

  return -tmp + Math.log(2.5066282746310005 * ser / x);
}

// Run games until the limit (or forever if no limit).
for (let g = 0; g < gameLimit; g++) {
  const p1IsBlack = g % 2 === 0;
  const names = [ p1IsBlack ? p1Name : p2Name, p1IsBlack ? p2Name : p1Name]
  const black = p1IsBlack ? p1 : p2;
  const white = p1IsBlack ? p2 : p1;

  if (VERBOSE) {
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
    if (VERBOSE) {
      console.log(`${names[isBlackTurn?0:1]}:`);
      printBoard(game);
      console.log(`Agent info: ${move.info}`);
      console.log();
    };
  }
  maxGameLen = Math.max(maxGameLen, game.moveCount);
  const winner = game.calcWinner();
  if (winner === BLACK) {
    blackWinCount++;
    tally[p1IsBlack ? 'p1' : 'p2']++;
  } else if (winner === WHITE) {
    tally[p1IsBlack ? 'p2' : 'p1']++;
  }

  maybePrint(g + 1);
}

// Final stats row (always printed, even if maybePrint already fired).
printStats(tally.p1 + tally.p2);


