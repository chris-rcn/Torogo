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
//                           [--limit <n>] [--index <n>] [--oversample <n>] [--verbose]
//
//   --agent       ai agent name under ai/                   (required)
//   --file        positions file from createmovedetails.js  (required)
//   --budget      ms per move                               (default: 2000)
//   --limit       evaluate only the first n positions       (default: all)
//   --index       evaluate only the position at 0-based index n, with the
//                 seed it had in a full sweep
//   --seed        starting agent rng seed (overrides default/per-index seed)
//   --oversample  evaluate each position this many times    (default: 1)
//   --show-phases P  at the end, print a P-row table of phase-band → RMS,
//                 binning every eval by game phase (board fullness, in [0,1])
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

  // Game phase ∈ [0,1]: board fullness = 1 − emptyCount/area (the codebase's
  // canonical phase, e.g. puct-hybrid / compare-moves).  Not move count —
  // captures make the two diverge.
  const phase = 1 - game.emptyCount / (boardSize * boardSize);

  const agentMove = agent(game, budgetMs, { rng: makeRng(agentSeed++) });
  const agentStr  = coordStr(agentMove.move, boardSize);

  const topCand   = candidates[0];
  const found     = candidates.find(c => c.m === agentStr);
  const agentCand = (found?.kwr != null) ? found : candidates.findLast(c => c.kwr != null);

  return { agentMove, agentStr, topCand, agentCand, phase, gap: (topCand.kwr - agentCand.kwr) / 1000 };
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
    console.log('Usage: node evalmovedetails.js --agent <name> --file <path> [--budget <ms>] [--limit <n>] [--index <n>] [--seed <n>] [--oversample <n>] [--show-phases <P>] [--verbose]');
    process.exit(opts.help ? 0 : 1);
  }

  const agentName  = opts.agent;
  const budgetMs   = parseInt(opts.budget     || '1000', 10);
  const limit      = opts.limit !== undefined ? parseInt(opts.limit, 10) : Infinity;
  const index      = opts.index !== undefined ? parseInt(opts.index, 10) : null;   // 0-based file/array index
  const seed       = opts.seed !== undefined ? parseInt(opts.seed, 10) : null;     // starting agent rng seed
  const oversample = parseInt(opts.oversample || '1',    10);
  const showPhases = opts['show-phases'] !== undefined ? parseInt(opts['show-phases'], 10) : null;
  const verbose    = !!opts.verbose;

  if (isNaN(budgetMs) || budgetMs < 1)     { console.error('--budget must be a positive integer'); process.exit(1); }
  if (isNaN(limit) || limit < 1)           { console.error('--limit must be a positive integer'); process.exit(1); }
  if (isNaN(oversample) || oversample < 1) { console.error('--oversample must be a positive integer'); process.exit(1); }
  if (seed !== null && isNaN(seed))        { console.error('--seed must be an integer'); process.exit(1); }
  if (showPhases !== null && (isNaN(showPhases) || showPhases < 1)) { console.error('--show-phases must be a positive integer'); process.exit(1); }

  const { getMove: agent } = require(path.join(__dirname, 'ai', agentName + '.js'));
  const pool      = loadPositions(opts.file);
  // --index N evaluates only the position at 0-based index N, restoring the
  // agent rng seed it would have had in a full sweep so the result reproduces
  // that position exactly.
  let positions;
  if (index !== null) {
    if (isNaN(index) || index < 0 || index >= pool.length) {
      console.error(`--index must be in 0..${pool.length - 1}`); process.exit(1);
    }
    positions = [pool[index]];
    agentSeed = index + 1;   // position i (0-based) uses seed i+1 in a full sweep
  } else {
    positions = pool.slice(0, limit);
  }
  // --seed sets the starting agent rng seed explicitly, overriding the default
  // (1) and the per-index seed restored above.
  if (seed !== null) agentSeed = seed;

  console.log(`agent=${agentName}  budget=${budgetMs}ms  oversample=${oversample}  positions=${positions.length}/${pool.length}`);
  console.log();
  console.log([
    'pos'    .padStart(5),
    'elapsed'.padStart(7),
    'tMv'    .padStart(5),
    'rms'    .padStart(5),
  ].join('  '));

  // Phase bands: partition phase ∈ [0,1] (board fullness) into P equal-width
  // bands and accumulate squared gap per band, so the end-of-run table shows
  // where in the game the agent loses the most.
  let phaseBandSq = null, phaseBandN = null;
  if (showPhases !== null) {
    phaseBandSq = new Float64Array(showPhases);
    phaseBandN  = new Int32Array(showPhases);
  }
  function phaseBandOf(phase) {
    let b = Math.floor(phase * showPhases);
    if (b >= showPhases) b = showPhases - 1;   // phase === 1 lands in the last band
    if (b < 0) b = 0;
    return b;
  }

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
  const wIdx  = Math.max(3, String(pool.length - 1).length);   // 0-based file index
  const wMove = 5;
  const wWR   = 5;
  // 0-based file index of positions[i]: --index pins a single position, else
  // the slice starts at the file head so the array index is the file index.
  const indexBase = index !== null ? index : 0;

  if (verbose) {
    console.log(
      `${'idx'.padStart(wIdx)}  ` +
      `${'hist'.padStart(4)}  ` +
      `${'top'.padEnd(wMove)} ${'WR'.padStart(wWR)}  ` +
      `${'agent'.padEnd(wMove)} ${'WR'.padStart(wWR)}  gap`
    );
    console.log('-'.repeat(wIdx + 2 + 4 + wMove + wWR + wMove + wWR + 20));
  }

  // Oversample is the outer loop: each pass sweeps all positions, so stats
  // at any point cover the whole position set rather than a prefix of it.
  let evals = 0;
  const WORST_N = 3;
  const worst = [];   // up to WORST_N highest-gap samples (sorted by gap desc), reported at the end
  for (let j = 0; j < oversample; j++) {
    for (let i = 0; i < positions.length; i++) {
      const { agentMove, agentStr, topCand, agentCand, phase, gap } = evalPosition(agent, positions[i], budgetMs);
      gapSqSum += gap * gap;
      evals++;

      if (showPhases !== null) {
        const b = phaseBandOf(phase);
        phaseBandSq[b] += gap * gap;
        phaseBandN[b]++;
      }

      if (worst.length < WORST_N || gap > worst[worst.length - 1].gap) {
        const rec = { index: indexBase + i, gap, hist: positions[i].history.length,
                      top: topCand.m, topKwr: topCand.kwr,
                      agent: agentStr, agentKwr: agentCand.kwr, info: agentMove.info };
        let pos = worst.length;
        while (pos > 0 && worst[pos - 1].gap < gap) pos--;
        worst.splice(pos, 0, rec);
        if (worst.length > WORST_N) worst.length = WORST_N;
      }

      if (verbose) console.log(
        `${String(indexBase + i).padStart(wIdx)}  ` +
        `${String(positions[i].history.length).padStart(4)}  ` +
        `${topCand.m.padEnd(wMove)} ${(topCand.kwr / 1000).toFixed(3).padStart(wWR)}  ` +
        `${agentStr.padEnd(wMove)} ${(agentCand.kwr / 1000).toFixed(3).padStart(wWR)}  ` +
        `${gap.toFixed(3)}` + (agentMove.info ? `  ${agentMove.info}` : '')
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

  if (worst.length) {
    console.log(`\nworst ${worst.length} samples:`);
    for (const w of worst) {
      console.log(
        `  idx=${w.index} gap=${w.gap.toFixed(3)} hist=${w.hist}  ` +
        `top=${w.top} (${(w.topKwr / 1000).toFixed(3)})  ` +
        `agent=${w.agent} (${(w.agentKwr / 1000).toFixed(3)})` + (w.info ? `  ${w.info}` : '')
      );
    }
  }

  if (showPhases !== null) {
    console.log(`\nphase bands (phase = board fullness 1−empty/area, in [0,1]):`);
    console.log([
      'phase'.padStart(9),
      'n'    .padStart(5),
      'rms'  .padStart(5),
    ].join('  '));
    for (let b = 0; b < showPhases; b++) {
      const lo = b / showPhases;
      const hi = (b + 1) / showPhases;
      const n  = phaseBandN[b];
      const rms = n > 0 ? Util.fmtRatio4(Math.sqrt(phaseBandSq[b] / n)) : '-';
      console.log([
        `${lo.toFixed(2)}-${hi.toFixed(2)}`.padStart(9),
        Util.fmt4(n) .padStart(5),
        rms          .padStart(5),
      ].join('  '));
    }
  }
}

module.exports = { loadPositions, evalPositions, evalPositionsSample };
