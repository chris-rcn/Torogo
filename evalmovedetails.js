#!/usr/bin/env node
'use strict';

// Evaluate an agent against pre-computed move details.
//
// Reads a newline-delimited JSON file produced by createmovedetails.js.
// For each position, reconstructs the game from the history, asks the agent
// to choose a move, then compares its win ratio to the top-rated move's.
// Reports the mean win-ratio gap (mae — the expected strength cost).
//
// Status is printed at an exponentially increasing interval (× 1.5 each time).
//
// Usage:
//   node evalmovedetails.js --agent <name> --file <path> [--budget <ms>]
//                           [--limit <n>] [--index <n>] [--oversample <n>] [--verbose]
//
//   --agent       ai agent name under ai/                   (required)
//   --file        positions file from createmovedetails.js  (required)
//   --budget      ms per move                               (default: 1000)
//   --limit       evaluate only the first n positions       (default: all)
//   --index       evaluate only the position at 0-based index n, with the
//                 seed it had in a full sweep
//   --seed        starting agent rng seed (default: random, logged at startup)
//   --oversample  evaluate each position this many times    (default: 1)
//   --show-phases P  at the end, print a P-row table of phase-band → MAE,
//                 binning every eval by game phase (board fullness, in [0,1])
//   --verbose     print a per-position comparison table

const fs   = require('fs');
const path = require('path');
const { performance } = require('perf_hooks');
const { Game2, coordStr, parseMove } = require('./game2.js');
const { makeRng } = require('./xorshift.js');
const Util = require('./util.js');
const MD = require('./movedetails-format.js');

// Fixed default seeds so the exported helpers (evalPositions/…, used by
// train-vpatterns) are reproducible.  The CLI overrides agentSeed with a random
// base by default (see --seed below).  Each agent invocation gets a fresh rng
// seeded from a counter: a time-budgeted search then replays the same playout
// sequence every run, so a wall-clock difference only perturbs the marginal
// playouts of that one invocation instead of shifting a shared stream and
// diverging every invocation after it.  Distinct seeds keep --oversample
// repeats distinct.
const rng = makeRng(1);   // position sampling (library default; CLI does not use it)
let agentSeed = 1;        // per-invocation agent rng (CLI reseeds from a random base)

function loadPositions(filePath) {
  const out = [];
  for (const line of fs.readFileSync(filePath, 'utf8').split('\n')) {
    const p = MD.parseRow(line);   // null for a comment/blank line
    if (p) out.push(p);
  }
  return out;
}

// Board fullness (1 − empty/area) of a position, from replaying its history —
// the same phase evalPosition reports, computed up front for band filtering.
function positionPhase(position) {
  return position.phase;   // stored in the file (board fullness 1 - empty/area)
}

// Evaluate the agent on a single position.  Returns the agent's move (string
// form), the top and matching candidates, and the win-ratio gap to the top
// move.  Terminal candidates carry a null rating; when the agent's move has
// no usable rating the worst rated candidate is charged instead.
function evalPosition(agent, position, budgetMs) {
  const { boardSize, candidates } = position;
  const game = MD.buildGame(position);   // empty board + full history (incl. the centre stone)
  const phase = position.phase;          // board fullness, stored in the file

  const agentMove = agent(game, budgetMs, { rng: makeRng(agentSeed++) });
  const agentStr  = coordStr(agentMove.move, boardSize);

  const topCand   = candidates[0];
  const found     = candidates.find(c => c.m === agentStr);
  const agentCand = (found?.winRatio != null) ? found : candidates.findLast(c => c.winRatio != null);

  return { agentMove, agentStr, topCand, agentCand, phase, gap: topCand.winRatio - agentCand.winRatio };
}

// Evaluate agent on positions; returns { maeErr, rmsErr, count }.  maeErr (mean
// win-prob gap) is the expected-strength-cost headline; rmsErr keeps the
// blunder-risk view (it over-weights big gaps).
function evalPositions(agent, positions, budgetMs) {
  let gapSum = 0, gapSqSum = 0;
  for (const position of positions) {
    const { gap } = evalPosition(agent, position, budgetMs);
    gapSum += gap;
    gapSqSum += gap * gap;
  }
  return { maeErr: gapSum / positions.length,
           rmsErr: Math.sqrt(gapSqSum / positions.length),
           count: positions.length };
}

// Evaluate agent on a random sample of n positions from the pool.
// If n >= pool.length, uses the full pool.  Returns { maeErr, rmsErr, count }.
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
  const opts = Util.parseArgs(process.argv.slice(2), ['help', 'verbose'], ['agent', 'budget', 'file', 'index', 'limit', 'oversample', 'seed', 'show-phases', 'min-phase', 'max-phase', 'verbose']);

  if (opts.help || !opts.file || !opts.agent) {
    console.log(`Usage: node evalmovedetails.js --agent <name> --file <path> [options]

Evaluate an agent against pre-computed move details (createmovedetails.js
output): replay each position, ask the agent for a move, and charge it the
win-ratio gap to the file's top-rated move.  Reports the mean gap (mae).

  --agent NAME      ai/<name>.js (required)
  --file PATH       positions file from createmovedetails.js (required)
  --budget MS       per-move budget (default 1000)
  --limit N         evaluate only the first N positions (default: all)
  --min-phase F     evaluate only positions at board fullness >= F (default 0)
  --max-phase F     evaluate only positions at board fullness <= F (default 1)
  --index N         evaluate only the position at 0-based index N, with the
                    agent seed it had in a full sweep (not with --min/max-phase)
  --seed N          starting agent rng seed (default: random, logged at startup)
  --oversample N    evaluate each position N times, distinct seeds (default 1)
  --show-phases P   at the end, print a P-row phase-band -> MAE table
                    (phase = board fullness in [0,1])
  --verbose         per-position comparison table
  --help            show this message`);
    process.exit(opts.help ? 0 : 1);
  }

  const agentName  = opts.agent;
  const budgetMs   = parseInt(opts.budget     || '1000', 10);
  const limit      = opts.limit !== undefined ? parseInt(opts.limit, 10) : Infinity;
  const index      = opts.index !== undefined ? parseInt(opts.index, 10) : null;   // 0-based file/array index
  const seed       = opts.seed !== undefined ? parseInt(opts.seed, 10) : null;     // starting agent rng seed
  const oversample = parseInt(opts.oversample || '1',    10);
  const showPhases = opts['show-phases'] !== undefined ? parseInt(opts['show-phases'], 10) : null;
  const minPhase   = opts['min-phase'] !== undefined ? parseFloat(opts['min-phase']) : 0;
  const maxPhase   = opts['max-phase'] !== undefined ? parseFloat(opts['max-phase']) : 1;
  const verbose    = !!opts.verbose;

  if (isNaN(budgetMs) || budgetMs < 1)     { console.error('--budget must be a positive integer'); process.exit(1); }
  if (isNaN(limit) || limit < 1)           { console.error('--limit must be a positive integer'); process.exit(1); }
  if (isNaN(oversample) || oversample < 1) { console.error('--oversample must be a positive integer'); process.exit(1); }
  if (seed !== null && isNaN(seed))        { console.error('--seed must be an integer'); process.exit(1); }
  if (showPhases !== null && (isNaN(showPhases) || showPhases < 1)) { console.error('--show-phases must be a positive integer'); process.exit(1); }
  if (isNaN(minPhase) || isNaN(maxPhase) || minPhase < 0 || maxPhase > 1 || minPhase > maxPhase) {
    console.error('--min-phase/--max-phase must satisfy 0 <= min <= max <= 1'); process.exit(1);
  }
  const bandActive = minPhase > 0 || maxPhase < 1;
  if (bandActive && index !== null) {
    console.error('--min-phase/--max-phase cannot combine with --index (its seed is tied to the full-sweep position order)');
    process.exit(1);
  }

  // Default agent rng seed is random (logged at startup) so runs differ; a fixed
  // --seed pins the starting seed and reproduces a run exactly.
  const baseSeed = seed !== null ? seed : Util.randomSeed();

  const _agentMod = require(path.join(__dirname, 'ai', agentName + '.js'));
// create(cfg)-style agents (phase-mux, the puct family) instantiate with a
// plain env reader; bare { getMove } modules are used directly.
const agent = (typeof _agentMod.create === 'function'
    ? _agentMod.create(Util.makeCfg(null)) : _agentMod).getMove;
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
    agentSeed = baseSeed + index;   // the seed position `index` gets in a baseSeed full sweep
  } else {
    // --min-phase/--max-phase: keep only positions whose board fullness is in
    // the band, then apply --limit to what survives.
    const banded = bandActive ? pool.filter(p => {
      const ph = positionPhase(p);
      return ph >= minPhase && ph <= maxPhase;
    }) : pool;
    positions = banded.slice(0, limit);
    agentSeed = baseSeed;
  }
  // --seed pins the starting agent rng seed, overriding the random default and
  // the per-index seed set above (so --seed with --index starts exactly at N).
  if (seed !== null) agentSeed = seed;

  console.log(`agent=${agentName}  budget=${budgetMs}ms  oversample=${oversample}  positions=${positions.length}/${pool.length}  seed=${agentSeed}` +
    (bandActive ? `  band=[${minPhase}, ${maxPhase}]` : ''));
  console.log();
  console.log([
    'pos'    .padStart(5),
    'elapsed'.padStart(7),
    'tMv'    .padStart(5),
    'mae'    .padStart(5),   // headline: mean win-prob gap (expected strength cost)
  ].join('  '));

  // Phase bands: partition phase ∈ [0,1] (board fullness) into P equal-width
  // bands and accumulate gap per band, so the end-of-run table shows where in
  // the game the agent loses the most (mae per band).
  let phaseBandSum = null, phaseBandN = null;
  if (showPhases !== null) {
    phaseBandSum = new Float64Array(showPhases);
    phaseBandN   = new Int32Array(showPhases);
  }
  function phaseBandOf(phase) {
    let b = Math.floor(phase * showPhases);
    if (b >= showPhases) b = showPhases - 1;   // phase === 1 lands in the last band
    if (b < 0) b = 0;
    return b;
  }

  const startTime = performance.now();
  let nextPrintPos = 1, printedAt = -1;   // geometric row schedule by positions
  let gapSum = 0;

  function printStats(count) {
    const elapsedMs = performance.now() - startTime;
    console.log([
      Util.fmt4i(count)                          .padStart(5),
      Util.fmtMs(elapsedMs)                      .padStart(7),
      Util.fmtMs(elapsedMs / count)              .padStart(5),
      Util.fmtRatio4(gapSum / count)             .padStart(5),
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
      gapSum += gap;
      evals++;

      if (showPhases !== null) {
        const b = phaseBandOf(phase);
        phaseBandSum[b] += gap;
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

      if (evals >= nextPrintPos) {
        printStats(evals);
        printedAt = evals;
        nextPrintPos = Math.max(Math.ceil(nextPrintPos * 1.5), nextPrintPos + 1);
      }
    }
  }

  if (evals !== printedAt) printStats(evals);   // final total, unless the loop just printed it

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
    console.log(`\nPhase bands:`);
    console.log([
      'phase'.padStart(9),
      'n'    .padStart(5),
      'mae'  .padStart(5),
    ].join('  '));
    for (let b = 0; b < showPhases; b++) {
      const lo = b / showPhases;
      const hi = (b + 1) / showPhases;
      const n  = phaseBandN[b];
      const mae = n > 0 ? Util.fmtRatio4(phaseBandSum[b] / n) : '-';
      console.log([
        `${lo.toFixed(2)}-${hi.toFixed(2)}`.padStart(9),
        Util.fmt4i(n).padStart(5),
        mae          .padStart(5),
      ].join('  '));
    }
  }

  // Single greppable summary line (grep for "SUMMARY").
  const elapsedMs = performance.now() - startTime;
  console.log(`SUMMARY agent=${agentName} file=${path.basename(opts.file)} ` +
    `pos=${positions.length} evals=${evals} mae=${(gapSum / evals).toFixed(4)} ` +
    `budget=${budgetMs}ms tMv=${Util.fmtMs(elapsedMs / evals).trim()} elapsed=${Util.fmtMs(elapsedMs).trim()}` +
    (bandActive ? ` band=${minPhase}-${maxPhase}` : ''));
}

module.exports = { loadPositions, evalPositions, evalPositionsSample };
