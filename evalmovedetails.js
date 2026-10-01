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
//   node evalmovedetails.js --agent <name> [--file <path>] [--budget <ms>]
//                           [--limit <n>] [--index <n>] [--oversample <n>] [--verbose]
//
//   --agent       ai agent name under ai/                   (required)
//   --file        positions file from createmovedetails.js  (default out/md-trunc30k-827.md)
//   --budget      ms per move                               (default: 1000)
//   --limit       evaluate only the first n positions       (default: all)
//   --index       evaluate only the position at 0-based index n, with the
//                 seed it had in a full sweep
//   --seed        starting agent rng seed (default: random, logged at startup)
//   --oversample  evaluate each position this many times    (default: 1)
//   --show-phases    at the end, print a table of phase-band → MAE and MSE over
//                    --phase-buckets N equal-width bands (default 10),
//   --elo-map PATH   map the per-band mae to a CGOS Elo estimate with the
//                    curves md-phase-fit.js --save wrote (default
//                    out/elo-map-trunc30k-827.json); elo= joins SUMMARY,
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

// Identity of a movedetails file's position SET, independent of row order (so a
// shuffled copy matches): sha256 of its sorted position rows, first 16 hex.
// The elo map records the fingerprint of the file it was fitted on.
function mdFingerprint(filePath) {
  const rows = fs.readFileSync(filePath, 'utf8').split('\n').filter(l => MD.parseRow(l)).sort();
  return require('crypto').createHash('sha256').update(rows.join('\n')).digest('hex').slice(0, 16);
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
  const opts = Util.parseArgs(process.argv.slice(2), ['help', 'verbose', 'show-phases'], ['agent', 'budget', 'file', 'index', 'limit', 'oversample', 'seed', 'show-phases', 'phase-buckets', 'elo-map', 'min-phase', 'max-phase', 'verbose']);

  if (opts.help || !opts.agent) {
    console.log(`Usage: node evalmovedetails.js --agent <name> [--file <path>] [options]

Evaluate an agent against pre-computed move details (createmovedetails.js
output): replay each position, ask the agent for a move, and charge it the
win-ratio gap to the file's top-rated move.  Reports the mean gap (mae).

  --agent NAME      ai/<name>.js (required)
  --file PATH       positions file from createmovedetails.js
                    (default out/md-trunc30k-827.md)
  --budget MS       per-move budget (default 1000)
  --limit N         evaluate only the first N positions (default: all)
  --min-phase F     evaluate only positions at board fullness >= F (default 0)
  --max-phase F     evaluate only positions at board fullness <= F (default 1)
  --index N         evaluate only the position at 0-based index N, with the
                    agent seed it had in a full sweep (not with --min/max-phase)
  --seed N          starting agent rng seed (default: random, logged at startup)
  --oversample N    evaluate each position N times, distinct seeds (default 1)
  --show-phases     at the end, print a phase-band -> MAE and MSE table
                    (phase = board fullness in [0,1])
  --phase-buckets N equal-width phase bands for the table, the same knob
                    filter-movedetails uses (default 10)
  --elo-map PATH    estimate CGOS Elo from the per-band mae with the mapping
                    md-phase-fit.js --save wrote (its bucket count applies);
                    elo= joins SUMMARY (default out/elo-map-trunc30k-827.json,
                    fitted on the default --file).  Only for the map's own MD
                    file (any row order), unfiltered: with another file,
                    --min/--max-phase, --limit or --index, elo=- and stderr
                    says why
  --verbose         per-position comparison table
  --help            show this message`);
    process.exit(opts.help ? 0 : 1);
  }

  const agentName  = opts.agent;
  // The default positions file, shown relative to the cwd as a user would type it.
  const mdFile     = opts.file || path.relative(process.cwd(), path.join(__dirname, 'out', 'md-trunc30k-827.md'));
  const budgetMs   = parseInt(opts.budget     || '1000', 10);
  const limit      = opts.limit !== undefined ? parseInt(opts.limit, 10) : Infinity;
  const index      = opts.index !== undefined ? parseInt(opts.index, 10) : null;   // 0-based file/array index
  const seed       = opts.seed !== undefined ? parseInt(opts.seed, 10) : null;     // starting agent rng seed
  const oversample = parseInt(opts.oversample || '1',    10);
  const eloMapPath   = opts['elo-map'] || path.join(__dirname, 'out', 'elo-map-trunc30k-827.json');
  if (!fs.existsSync(eloMapPath)) { console.error(`--elo-map ${eloMapPath} not found (write one with md-phase-fit.js --save)`); process.exit(1); }
  const eloMap       = JSON.parse(fs.readFileSync(eloMapPath, 'utf8'));
  if (opts['phase-buckets'] !== undefined && parseInt(opts['phase-buckets'], 10) !== eloMap.phaseBuckets) {
    console.error(`--phase-buckets ${opts['phase-buckets']} differs from the elo map's ${eloMap.phaseBuckets}`); process.exit(1);
  }
  // Only mae curves are computed; a zero-weight curve of another kind adds
  // nothing and is ignored, any other is an older map that needs a refit.
  const badCurves = Object.entries(eloMap.curves).filter(([kind, { A }]) => kind !== 'mae' && A !== 0).map(([kind]) => kind);
  if (badCurves.length) { console.error(`--elo-map ${eloMapPath} weights ${badCurves.join(', ')}, which evalmovedetails no longer computes; refit with md-phase-fit.js`); process.exit(1); }
  const phaseBuckets = eloMap.phaseBuckets;   // band count: the map's, always accumulated
  const printPhases  = !!opts['show-phases'];
  const minPhase   = opts['min-phase'] !== undefined ? parseFloat(opts['min-phase']) : 0;
  const maxPhase   = opts['max-phase'] !== undefined ? parseFloat(opts['max-phase']) : 1;
  const verbose    = !!opts.verbose;

  if (isNaN(budgetMs) || budgetMs < 1)     { console.error('--budget must be a positive integer'); process.exit(1); }
  if (isNaN(limit) || limit < 1)           { console.error('--limit must be a positive integer'); process.exit(1); }
  if (isNaN(oversample) || oversample < 1) { console.error('--oversample must be a positive integer'); process.exit(1); }
  if (seed !== null && isNaN(seed))        { console.error('--seed must be an integer'); process.exit(1); }
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
  const pool      = loadPositions(mdFile);
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

  console.log(`agent=${agentName}  file=${mdFile}  budget=${budgetMs}ms  oversample=${oversample}  positions=${positions.length}/${pool.length}  seed=${agentSeed}` +
    (bandActive ? `  band=[${minPhase}, ${maxPhase}]` : ''));
  console.log();
  console.log([
    'pos'    .padStart(5),
    'elapsed'.padStart(7),
    'tMv'    .padStart(5),
    'mae'    .padStart(5),   // headline: mean win-prob gap (expected strength cost)
  ].join('  '));

  // Phase bands: partition phase ∈ [0,1] (board fullness) into the map's
  // equal-width bands and accumulate gap and gap² per band, for the Elo map
  // and the --show-phases table (mae per band).
  const phaseBandSum   = new Float64Array(phaseBuckets);
  const phaseBandN     = new Int32Array(phaseBuckets);
  function phaseBandOf(phase) {
    let b = Math.floor(phase * phaseBuckets);
    if (b >= phaseBuckets) b = phaseBuckets - 1;   // phase === 1 lands in the last band
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
      gapSum   += gap;
      evals++;

      const b = phaseBandOf(phase);
      phaseBandSum[b]   += gap;
      phaseBandN[b]++;

      if (worst.length < WORST_N || gap > worst[worst.length - 1].gap) {
        const rec = { index: indexBase + i, gap, hist: positions[i].history.length,
                      top: topCand.m, topWR: topCand.winRatio,
                      agent: agentStr, agentWR: agentCand.winRatio, info: agentMove.info };
        let pos = worst.length;
        while (pos > 0 && worst[pos - 1].gap < gap) pos--;
        worst.splice(pos, 0, rec);
        if (worst.length > WORST_N) worst.length = WORST_N;
      }

      if (verbose) console.log(
        `${String(indexBase + i).padStart(wIdx)}  ` +
        `${String(positions[i].history.length).padStart(4)}  ` +
        `${topCand.m.padEnd(wMove)} ${topCand.winRatio.toFixed(3).padStart(wWR)}  ` +
        `${agentStr.padEnd(wMove)} ${agentCand.winRatio.toFixed(3).padStart(wWR)}  ` +
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
        `top=${w.top} (${w.topWR.toFixed(3)})  ` +
        `agent=${w.agent} (${w.agentWR.toFixed(3)})` + (w.info ? `  ${w.info}` : '')
      );
    }
  }

  if (printPhases) {
    console.log(`\nPhase bands:`);
    console.log([
      'phase'.padStart(9),
      'n'    .padStart(5),
      'mae'  .padStart(5),
    ].join('  '));
    for (let b = 0; b < phaseBuckets; b++) {
      const lo = b / phaseBuckets;
      const hi = (b + 1) / phaseBuckets;
      const n  = phaseBandN[b];
      const mae = n > 0 ? Util.fmtRatio4(phaseBandSum[b] / n) : '-';
      console.log([
        `${lo.toFixed(2)}-${hi.toFixed(2)}`.padStart(9),
        Util.fmt4i(n).padStart(5),
        mae          .padStart(5),
      ].join('  '));
    }
  }

  // --elo-map: elo = intercept - A * sum_b exp(k * mid_b) * mae_b over the
  // map's bands.
  // A mapped band with no results leaves the estimate undefined: elo=- and
  // a note on stderr, the rest of the summary as usual.
  // The map is only valid on the positions it was fitted on: its own file,
  // every position of it.
  const eloBlock =
      !eloMap.mdFingerprint                         ? `the map records no MD file (refit it with md-phase-fit.js --save)`
    : mdFingerprint(mdFile) !== eloMap.mdFingerprint ? `${mdFile} is not the map's MD file (${eloMap.mdFile})`
    : bandActive                                    ? `phase-filtered (--min-phase/--max-phase)`
    : positions.length < pool.length                ? `not every position (${index !== null ? '--index' : '--limit'})`
    : null;
  const empty = eloMap.bands.filter(({ lo, hi }) => phaseBandN[phaseBandOf((lo + hi) / 2)] === 0);
  let elo = eloMap.intercept;
  if (eloBlock) {
    console.error(`--elo-map: elo not estimated: ${eloBlock}`);
    elo = NaN;
  } else if (empty.length) {
    console.error(`--elo-map: no results in band${empty.length > 1 ? 's' : ''} ${empty.map(({ lo, hi }) => `${lo.toFixed(2)}-${hi.toFixed(2)}`).join(', ')}; elo not estimated`);
    elo = NaN;
  } else {
    const { A, k } = eloMap.curves.mae;
    for (const { lo, hi } of eloMap.bands) {
      const b = phaseBandOf((lo + hi) / 2), n = phaseBandN[b];
      elo -= A * Math.exp(k * (lo + hi) / 2) * (phaseBandSum[b] / n);
    }
  }

  // Single greppable summary line (grep for "SUMMARY").
  const elapsedMs = performance.now() - startTime;
  // Fixed-width fields, the headline (mae) last but for elo; the band always
  // prints as two decimals (0.00-1.00 when unbanded) so the columns line up.
  console.log(`SUMMARY band=${minPhase.toFixed(2)}-${maxPhase.toFixed(2)} ` +
    `evals=${String(evals).padStart(4)} tMv=${Util.fmtMs(elapsedMs / evals)} elapsed=${Util.fmtMs(elapsedMs)} ` +
    `mae=${(gapSum / evals).toFixed(4)} elo=${(isNaN(elo) ? '-' : elo.toFixed(0)).padStart(5)}`);
}

module.exports = { loadPositions, evalPositions, evalPositionsSample, mdFingerprint };
