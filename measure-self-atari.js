'use strict';

// Measure how often an agent plays a non-capture self-atari (a move that
// leaves the resulting friendly group with exactly 1 liberty AND captures
// nothing).  Plays full games against itself forever, printing stats at an
// exponentially-growing wall-clock interval (×1.5 each time).
//
// Usage: node measure-self-atari.js --agent <name> --size <N>
//   --agent   name of the agent module under ai/ (e.g. vlibpat-ref-3x3)
//   --size    board size (default 13)

const path = require('path');
const { Game2, PASS } = require('./game2.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help']);

if (opts.help) {
  console.error('Usage: node measure-self-atari.js --agent <name> --size <N>');
  process.exit(0);
}

const AGENT_NAME = opts.agent;
if (!AGENT_NAME) { console.error('--agent is required'); process.exit(1); }
const SIZE = parseInt(opts.size || '13', 10);

const { getMove } = require(path.join(__dirname, 'ai', AGENT_NAME + '.js'));

console.log(`agent=${AGENT_NAME}  size=${SIZE}`);
console.log([
  'elapsed' .padStart(7),
  'games'   .padStart(6),
  'moves'   .padStart(7),
  'selfAt'  .padStart(7),
  'rate'    .padStart(6),
  'tMove'   .padStart(7),
].join('  '));

let totalGames    = 0;
let totalMoves    = 0;
let saMoves       = 0;
let moveTimeMsSum = 0;

const t0 = Date.now();
let printIntervalMs = 1000;
let nextPrintAtMs   = printIntervalMs;

function maybePrint() {
  const elapsed = Date.now() - t0;
  if (elapsed < nextPrintAtMs) return;
  const rate   = totalMoves > 0 ? saMoves / totalMoves : 0;
  const tMove  = totalMoves > 0 ? moveTimeMsSum / totalMoves : 0;
  console.log([
    Util.fmtMs(elapsed).padStart(7),
    Util.fmt4(totalGames).padStart(6),
    Util.fmt4(totalMoves).padStart(7),
    Util.fmt4(saMoves).padStart(7),
    Util.fmtRatio4(rate).padStart(6),
    Util.fmtMs(tMove).padStart(7),
  ].join('  '));
  printIntervalMs *= 1.5;
  nextPrintAtMs    = elapsed + printIntervalMs;
}

const RANDOM_RATE = 0.1;   // fraction of moves played uniformly at random
const maxMoves = 4 * SIZE * SIZE;
while (true) {
  const game = new Game2(SIZE, false);
  while (!game.gameOver && game.moveCount < maxMoves) {
    if (Math.random() < RANDOM_RATE) {
      // Off-policy noise — keeps the move stream out of repeated openings so
      // the agent visits a wider distribution of positions.  Does NOT count
      // toward self-atari stats.
      game.play(game.randomLegalMove());
      continue;
    }
    const tStart = Date.now();
    const result = getMove(game);
    const tEnd   = Date.now();
    const mv = (result && result.move !== undefined) ? result.move : PASS;
    if (mv === PASS) {
      game.play(PASS);
      continue;
    }
    game.play(mv);
    // After play: check if the just-placed stone's group has 1 liberty and
    // the move captured nothing.
    const captured = game._lastCaptureCount > 0;
    const libs     = game.groupLibs2(mv).count;
    if (libs === 1 && !captured) saMoves++;
    totalMoves++;
    moveTimeMsSum += tEnd - tStart;
    maybePrint();
  }
  totalGames++;
  maybePrint();
}
