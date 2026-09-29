#!/usr/bin/env node
'use strict';

// Generate per-position move-value data by self-play.
//
// Each self-play game is played to completion by the POSITION agent, which
// only has to return moves.  One position with board fullness (phase) below
// 0.8 is then drawn for deep analysis and checked for contestedness by the
// VALUE agent: its win ratio must be within [0.3, 0.7], else the position is
// discarded and another drawn (a few tries per game).  Phase bands
// (--phase-buckets N equal-width bands of [0,1]) are kept level: the game's
// positions are grouped by band, the band with the fewest samples so far in
// this run is taken (ties at random), and one of its positions is drawn at
// random.  Then every legal move
// is enumerated, the game is cloned, the move is made, then the VALUE
// agent's getMove is called with the full budget.  The rootWinRatio is flipped to the
// original player's perspective and recorded as kwr (× 1000).  One position
// per game keeps the samples independent.  Games with no eligible position
// emit nothing.
//
// Output goes to stdout, one row per position as it is produced, after a
// '#' comment recording the generation parameters (redirect it to the file).
// Status goes to stderr at an exponentially increasing interval (× 1.5 each
// time).  Runs indefinitely (Ctrl-C to stop).  Several processes can run at
// once, each to its own file; the files concatenate.
//
// Usage:
//   node createmovedetails.js --position-agent <name> --value-agent <name>
//                             [--budget 2000] [--size 13]
//                             [--phase-buckets 10] > out/movedetails-<name>.md
//
//   --phase-buckets  equal-width phase bands kept level (default 10)

const path = require('path');
const { performance } = require('perf_hooks');
const { Game2, PASS, coordStr } = require('./game2.js');
const Util = require('./util.js');
const MD = require('./movedetails-format.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['position-agent', 'value-agent', 'budget', 'size', 'phase-buckets']);

if (opts.help || !opts['position-agent'] || !opts['value-agent']) {
  console.log(`Usage: node createmovedetails.js --position-agent <name> --value-agent <name> [--budget <ms>] [--size <n>] [--phase-buckets <n>] > out.md
  --position-agent  ai/<name>.js that plays the self-play games the positions come from (required)
  --value-agent     ai/<name>.js that judges a drawn position's contestedness and labels
                    every legal move of it; must return rootWinRatio (required)
A fixed-playout agent ignores --budget; PLAYOUTS in the env applies to both.`);
  process.exit(opts.help ? 0 : 1);
}

const posAgentName = opts['position-agent'], valAgentName = opts['value-agent'];
const budget    = parseInt(opts.budget || '1', 10);
const boardSize = parseInt(opts.size   || '13',   10);
const PLAYOUTS  = process.env.PLAYOUTS || '';   // agent's fixed playout count, if set (overrides budget)
const phaseBuckets = parseInt(opts['phase-buckets'] || '10', 10);

if (isNaN(budget) || budget < 1)       { console.error('--budget must be a positive integer'); process.exit(1); }
if (isNaN(phaseBuckets) || phaseBuckets < 1) { console.error('--phase-buckets must be a positive integer'); process.exit(1); }
if (isNaN(boardSize) || boardSize < 2) { console.error('--size must be >= 2'); process.exit(1); }

// stdout is the data stream.  Agents print their load banners with
// console.log, so from here on console.log goes to stderr with the status.
console.log = console.error;
// create(cfg)-style agents (phase-mux, the puct family) instantiate with a
// plain env reader; bare { getMove } modules are used directly.
function loadAgent(name) {
  const mod = require(path.join(__dirname, 'ai', name + '.js'));
  return (typeof mod.create === 'function' ? mod.create(Util.makeCfg(null)) : mod).getMove;
}
const posAgent = loadAgent(posAgentName);
const valAgent = posAgentName === valAgentName ? posAgent : loadAgent(valAgentName);

// A position is eligible for analysis only when |win ratio − 0.5| is within
// this, keeping only contested positions in the dataset.
const WR_DEV = 0.2;

// ...and only in the opening/middle game: board fullness (phase = 1 −
// empty/area) must be below this, skipping crowded late-game positions.
const PHASE_MAX = 0.8;

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

process.stdout.write(`# createmovedetails.js  position-agent=${posAgentName}  value-agent=${valAgentName}  budget=${budget}ms  PLAYOUTS=${PLAYOUTS || '(budget)'}  size=${boardSize}  wr-dev=${WR_DEV}  phase-max=${PHASE_MAX}  phase-buckets=${phaseBuckets}\n`);

console.error(`position-agent=${posAgentName}  value-agent=${valAgentName}  size=${boardSize}  budget=${budget}ms`);
console.error();
console.error([
  'pos'    .padStart(5),
  'elapsed'.padStart(7),
  'tPos'   .padStart(5),
  'bands',
].join('  '));

const startTime = performance.now();
let printPeriodMs = 1000;
let lastPrintTime = startTime;
let posCount = 0;

function printStats() {
  const elapsedMs = performance.now() - startTime;
  console.error([
    Util.fmt4i(posCount)            .padStart(5),
    Util.fmtMs(elapsedMs)           .padStart(7),
    Util.fmtMs(elapsedMs / posCount).padStart(5),
    Array.from(bandCount).join(','),   // samples per phase band so far
  ].join('  '));
}
const bandCount = new Int32Array(phaseBuckets);   // samples written per phase band, this run

while (true) {
  const N = boardSize;

  // Play a full self-play game, recording each move and which positions
  // (identified by the number of moves played before them) fall in each band.
  const game     = new Game2(boardSize, true);
  const moves    = [];
  const inBand = Array.from({ length: phaseBuckets }, () => []);   // per band: move counts of its positions

  while (!game.gameOver) {
    const advancingMove = posAgent(game, budget);
    const phase = 1 - game.emptyCount / (N * N);
    if (phase < PHASE_MAX) inBand[Math.min(phaseBuckets - 1, Math.floor(phase * phaseBuckets))].push(moves.length);
    moves.push(advancingMove.move);
    game.play(advancingMove.move);
  }

  // Draw: the band with the fewest samples so far among those this game
  // still offers (ties at random), then one of its positions at random, and
  // keep it only if the value agent finds it contested; otherwise drop it
  // from the pool and draw again, up to DRAW_TRIES per game.
  const DRAW_TRIES = 5;
  let k = -1, band = -1;
  for (let attempt = 0; attempt < DRAW_TRIES && k < 0; attempt++) {
    let fewest = Infinity, ties = 0; band = -1;
    for (let b = 0; b < phaseBuckets; b++) {
      if (inBand[b].length === 0) continue;
      if (bandCount[b] < fewest) { fewest = bandCount[b]; band = b; ties = 1; }
      else if (bandCount[b] === fewest && Math.random() * ++ties < 1) band = b;
    }
    if (band < 0) break;
    const i = Math.floor(Math.random() * inBand[band].length);
    const cand = inBand[band][i];
    inBand[band].splice(i, 1);
    const probe = new Game2(boardSize, true);
    for (let j = 0; j < cand; j++) probe.play(moves[j]);
    const v = valAgent(probe, budget);
    if (v.rootWinRatio === undefined) { console.error('value agent did not return rootWinRatio'); process.exit(1); }
    if (Math.abs(v.rootWinRatio - 0.5) <= WR_DEV) k = cand;
  }
  if (k < 0) continue;
  bandCount[band]++;
  const position = new Game2(boardSize, true);
  for (let i = 0; i < k; i++) position.play(moves[i]);
  const history = moves.slice(0, k).map(m => coordStr(m, N));

  const moveInfos = [];
  for (const move of legalMoves(position)) {
    const clone = position.clone();
    clone.play(move);

    if (clone.gameOver) {
      moveInfos.push({ m: coordStr(move, N), winRatio: null });
      continue;
    }

    const oppResponseMove = valAgent(clone, budget);
    if (oppResponseMove.rootWinRatio === undefined) {
      console.error('value agent did not return rootWinRatio');
      process.exit(1);
    }
    const wr = 1 - oppResponseMove.rootWinRatio;
    moveInfos.push({ m: coordStr(move, N), winRatio: wr });
  }

  moveInfos.sort((a, b) => (b.winRatio ?? -Infinity) - (a.winRatio ?? -Infinity));

  // The new format holds EVERY stone: prepend the free centre stone so the
  // position replays from an empty board.  Phase is stored (board fullness).
  const phase = 1 - position.emptyCount / (N * N);
  process.stdout.write(
    MD.formatRow({ boardSize, phase, history: [MD.centerMove(N), ...history], candidates: moveInfos }) + '\n');
  posCount++;

  const now = performance.now();
  if (now - lastPrintTime >= printPeriodMs) {
    lastPrintTime = now;
    printPeriodMs = Math.round(printPeriodMs * 1.5);
    printStats();
  }
}
