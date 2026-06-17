#!/usr/bin/env node
'use strict';

// Evaluate an agent against pre-computed move details.
//
// Reads a newline-delimited JSON file produced by createmovedetails.js.
// For each position, reconstructs the game from the history, asks the agent
// to choose a move, then compares its win ratio to the top-rated move's.
// Reports the RMS of the win-ratio gap.
//
// Status is printed at an exponentially increasing interval (× 1.5 each time).
//
// Usage:
//   node evalmovedetails.js --agent <name> --file <path> [--budget <ms>]
//                           [--limit <n>] [--oversample <n>] [--verbose]
//
//   --agent       ai agent name under ai/                   (required)
//   --file        positions file from createmovedetails.js  (required)
//   --budget      ms per move                               (default: 2000)
//   --limit       evaluate only the first n positions       (default: all)
//   --oversample  evaluate each position this many times    (default: 1)
//   --verbose     print a per-position comparison table

const fs   = require('fs');
const path = require('path');
const { performance } = require('perf_hooks');
const { Game2, coordStr, parseMove } = require('./game2.js');
const { makeRng } = require('./xorshift.js');
const Util = require('./util.js');

// Hardcoded seeds so position sampling and agent search are reproducible
// across runs.  Each agent invocation gets a fresh rng seeded from a counter:
// a time-budgeted search then replays the same playout sequence every run,
// so a wall-clock difference only perturbs the marginal playouts of that one
// invocation instead of shifting a shared stream and diverging every
// invocation after it.  Distinct seeds keep --oversample repeats distinct.
const rng = makeRng(1);   // position sampling
let agentSeed = 1;        // per-invocation agent rng

function loadPositions(filePath) {
  return fs.readFileSync(filePath, 'utf8').split('\n')
    .filter(l => l.trim() && !l.startsWith('#'))   // '#' lines hold generation parameters
    .map(l => JSON.parse(l));
}

// Evaluate the agent on a single position.  Returns the agent's move (string
// form), the top and matching candidates, and the win-ratio gap to the top
// move.  Terminal candidates carry a null rating; when the agent's move has
// no usable rating the worst rated candidate is charged instead.
function evalPosition(agent, position, budgetMs) {
  const { boardSize, history, candidates } = position;
  const game = new Game2(boardSize);
  for (const h of history) game.play(parseMove(h, boardSize));

  const agentMove = agent(game, budgetMs, { rng: makeRng(agentSeed++) });
  const agentStr  = coordStr(agentMove.move, boardSize);

  const topCand   = candidates[0];
  const found     = candidates.find(c => c.m === agentStr);
  const agentCand = (found?.kwr != null) ? found : candidates.findLast(c => c.kwr != null);

  return { agentMove, agentStr, topCand, agentCand, gap: (topCand.kwr - agentCand.kwr) / 1000 };
}

// Evaluate agent on positions; returns { rmsErr, count }.
function evalPositions(agent, positions, budgetMs) {
  let gapSqSum = 0;
  for (const position of positions) {
    const { gap } = evalPosition(agent, position, budgetMs);
    gapSqSum += gap * gap;
  }
  return { rmsErr: Math.sqrt(gapSqSum / positions.length), count: positions.length };
}

// Evaluate agent on a random sample of n positions from the pool.
// If n >= pool.length, uses the full pool.  Returns { rmsErr, count }.
function evalPositionsSample(agent, pool, n, budgetMs) {
  let positions = pool;
  if (n < pool.length) {
    const sample = pool.slice();
    for (let i = 0; i < n; i++) {
      const j = i + Math.floor(rng.random() * (sample.length - i));
      [sample[i], sample[j]] = [sample[j], sample[i]];
    }
    positions = sample.slice(0, n);
  }
  return evalPositions(agent, positions, budgetMs);
}

if (require.main === module) {
  const opts = Util.parseArgs(process.argv.slice(2), ['help', 'verbose']);

  if (opts.help || !opts.file || !opts.agent) {
    console.log('Usage: node evalmovedetails.js --agent <name> --file <path> [--budget <ms>] [--limit <n>] [--oversample <n>] [--verbose]');
    process.exit(opts.help ? 0 : 1);
  }

  const agentName  = opts.agent;
  const budgetMs   = parseInt(opts.budget     || '2000', 10);
  const limit      = opts.limit !== undefined ? parseInt(opts.limit, 10) : Infinity;
  const oversample = parseInt(opts.oversample || '1',    10);
  const verbose    = !!opts.verbose;

  if (isNaN(budgetMs) || budgetMs < 1)     { console.error('--budget must be a positive integer'); process.exit(1); }
  if (isNaN(limit) || limit < 1)           { console.error('--limit must be a positive integer'); process.exit(1); }
  if (isNaN(oversample) || oversample < 1) { console.error('--oversample must be a positive integer'); process.exit(1); }

  const { getMove: agent } = require(path.join(__dirname, 'ai', agentName + '.js'));
  const pool      = loadPositions(opts.file);
  const positions = pool.slice(0, limit);

  console.log(`agent=${agentName}  budget=${budgetMs}ms  oversample=${oversample}  positions=${positions.length}/${pool.length}`);
  console.log();
  console.log([
    'pos'    .padStart(5),
    'elapsed'.padStart(7),
    'tMv'    .padStart(5),
    'rms'    .padStart(5),
  ].join('  '));

  const startTime = performance.now();
  let printPeriodMs = 1000;
  let lastPrintTime = startTime;
  let gapSqSum = 0;

  function printStats(count) {
    const elapsedMs = performance.now() - startTime;
    console.log([
      Util.fmt4(count)                           .padStart(5),
      Util.fmtMs(elapsedMs)                      .padStart(7),
      Util.fmtMs(elapsedMs / count)              .padStart(5),
      Util.fmtRatio4(Math.sqrt(gapSqSum / count)).padStart(5),
    ].join('  '));
  }

  // Column widths for the verbose table.
  const wMove = 5;
  const wWR   = 5;

  if (verbose) {
    console.log(
      `${'hist'.padStart(4)}  ` +
      `${'top'.padEnd(wMove)} ${'WR'.padStart(wWR)}  ` +
      `${'agent'.padEnd(wMove)} ${'WR'.padStart(wWR)}  gap`
    );
    console.log('-'.repeat(4 + wMove + wWR + wMove + wWR + 20));
  }

  // Oversample is the outer loop: each pass sweeps all positions, so stats
  // at any point cover the whole position set rather than a prefix of it.
  let evals = 0;
  for (let j = 0; j < oversample; j++) {
    for (let i = 0; i < positions.length; i++) {
      const { agentMove, agentStr, topCand, agentCand, gap } = evalPosition(agent, positions[i], budgetMs);
      gapSqSum += gap * gap;
      evals++;

      if (verbose) console.log(
        `${String(positions[i].history.length).padStart(4)}  ` +
        `${topCand.m.padEnd(wMove)} ${(topCand.kwr / 1000).toFixed(3).padStart(wWR)}  ` +
        `${agentStr.padEnd(wMove)} ${(agentCand.kwr / 1000).toFixed(3).padStart(wWR)}  ` +
        `${gap.toFixed(3)}  ` + agentMove.info
      );

      const now = performance.now();
      if (now - lastPrintTime >= printPeriodMs) {
        lastPrintTime = now;
        printPeriodMs = Math.round(printPeriodMs * 1.5);
        printStats(evals);
      }
    }
  }

  printStats(evals);
}

module.exports = { loadPositions, evalPositions, evalPositionsSample };
