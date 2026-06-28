#!/usr/bin/env node
'use strict';

// Generate per-position move-value data by self-play.
//
// Each self-play game is played to completion, recording the agent's
// rootWinRatio at every position.  One random eligible position — win ratio
// within [0.3, 0.7] and board fullness (phase) below 0.7 — is then revisited
// for deep analysis: every legal move
// is enumerated, the game is cloned, the move is made, then the agent's
// getMove is called with the full budget.  The rootWinRatio is flipped to the
// original player's perspective and recorded as kwr (× 1000).  One position
// per game keeps the samples independent.  Games with no eligible position
// emit nothing.
//
// Output is newline-delimited JSON, one object per position, appended to the
// save file as it is produced.  The first line is a '#' comment recording the
// generation parameters.  Status is printed at an exponentially
// increasing interval (× 1.5 each time).  Runs indefinitely (Ctrl-C to stop).
//
// Usage:
//   node createmovedetails.js [--agent prod] [--budget 2000] [--size 13]
//                             [--save <path.ndjson>]
//
//   --save    output path  (default: out/movedetails-<random>.ndjson)

const fs   = require('fs');
const path = require('path');
const { performance } = require('perf_hooks');
const { Game2, PASS, coordStr } = require('./game2.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['agent', 'budget', 'save', 'size']);

if (opts.help) {
  console.log('Usage: node createmovedetails.js [--agent <name>] [--budget <ms>] [--size <n>]');
  process.exit(0);
}

const agentName = opts.agent || 'prod';
const budget    = parseInt(opts.budget || '1', 10);
const boardSize = parseInt(opts.size   || '13',   10);
const SAVE_PATH = opts.save || `out/movedetails-${Math.random().toString(36).slice(2, 10)}.ndjson`;
const PLAYOUTS  = process.env.PLAYOUTS || '';   // agent's fixed playout count, if set (overrides budget)

if (isNaN(budget) || budget < 1)       { console.error('--budget must be a positive integer'); process.exit(1); }
if (isNaN(boardSize) || boardSize < 2) { console.error('--size must be >= 2'); process.exit(1); }

const { getMove: agent } = require(path.join(__dirname, 'ai', agentName + '.js'));

// A position is eligible for analysis only when |win ratio − 0.5| is within
// this, keeping only contested positions in the dataset.
const WR_DEV = 0.2;

// ...and only in the opening/middle game: board fullness (phase = 1 −
// empty/area) must be below this, skipping crowded late-game positions.
const PHASE_MAX = 0.7;

function legalMoves(game2) {
  const N   = game2.N;
  const cap = N * N;
  const moves = [];
  for (let i = 0; i < cap; i++) {
    if (game2.cells[i] === 0 && game2.isLegal(i)) moves.push(i);
  }
  moves.push(PASS);
  return moves;
}

fs.writeFileSync(SAVE_PATH, `# createmovedetails.js  agent=${agentName}  budget=${budget}ms  PLAYOUTS=${PLAYOUTS || '(budget)'}  size=${boardSize}  wr-dev=${WR_DEV}  phase-max=${PHASE_MAX}\n`);

console.log(`agent=${agentName}  size=${boardSize}  budget=${budget}ms`);
console.log(`Out: ${SAVE_PATH}`);
console.log();
console.log([
  'pos'    .padStart(5),
  'elapsed'.padStart(7),
  'tPos'   .padStart(5),
].join('  '));

const startTime = performance.now();
let printPeriodMs = 1000;
let lastPrintTime = startTime;
let posCount = 0;

function printStats() {
  const elapsedMs = performance.now() - startTime;
  console.log([
    Util.fmt4i(posCount)            .padStart(5),
    Util.fmtMs(elapsedMs)           .padStart(7),
    Util.fmtMs(elapsedMs / posCount).padStart(5),
  ].join('  '));
}

while (true) {
  const N = boardSize;

  // Play a full self-play game, recording each move and which positions
  // (identified by the number of moves played before them) were eligible.
  const game     = new Game2(boardSize);
  const moves    = [];
  const eligible = [];

  while (!game.gameOver) {
    const advancingMove = agent(game, budget);
    if (advancingMove.rootWinRatio === undefined) {
      console.error('agent did not return rootWinRatio');
      process.exit(1);
    }
    const phase = 1 - game.emptyCount / (N * N);
    if (Math.abs(advancingMove.rootWinRatio - 0.5) <= WR_DEV && phase < PHASE_MAX) eligible.push(moves.length);
    moves.push(advancingMove.move);
    game.play(advancingMove.move);
  }

  if (eligible.length === 0) continue;

  // Return to one random eligible position for the deep analysis.
  const k = eligible[Math.floor(Math.random() * eligible.length)];
  const position = new Game2(boardSize);
  for (let i = 0; i < k; i++) position.play(moves[i]);
  const history = moves.slice(0, k).map(m => coordStr(m, N));

  const moveInfos = [];
  for (const move of legalMoves(position)) {
    const clone = position.clone();
    clone.play(move);

    if (clone.gameOver) {
      moveInfos.push({ m: coordStr(move, N), kwr: null });
      continue;
    }

    const oppResponseMove = agent(clone, budget);
    if (oppResponseMove.rootWinRatio === undefined) {
      console.error('agent did not return rootWinRatio');
      process.exit(1);
    }
    const wr = 1 - oppResponseMove.rootWinRatio;
    moveInfos.push({ m: coordStr(move, N), kwr: Math.round(1000 * wr) });
  }

  moveInfos.sort((a, b) => (b.kwr ?? -Infinity) - (a.kwr ?? -Infinity));

  fs.appendFileSync(SAVE_PATH, JSON.stringify({
    boardSize,
    history,
    candidates: moveInfos,
  }) + '\n');
  posCount++;

  const now = performance.now();
  if (now - lastPrintTime >= printPeriodMs) {
    lastPrintTime = now;
    printPeriodMs = Math.round(printPeriodMs * 1.5);
    printStats();
  }
}
