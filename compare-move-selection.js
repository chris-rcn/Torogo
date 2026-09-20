#!/usr/bin/env node
'use strict';

// compare-move-selection.js — compare the move selection of two agents, scored
// against an oracle.  Two position/oracle sources:
//
//   --file <movedetails>  fixed, pre-rated positions (createmovedetails.js):
//                         each chosen move is charged the win-prob gap to the
//                         file's top-rated candidate.  Controlled, reproducible.
//   --referee <agent>     LIVE: play games where each move is taken from a
//                         random one of the two agents (a 50/50 mixture, so
//                         both see the same neutral distribution), sample
//                         decisions, and score each side's move by the
//                         referee's valueB on the resulting position (mapped to
//                         the mover's frame).  Match-realistic distribution.
//
// Headline metric, identical in both modes: the paired mean win-prob
//   Δ = mean over decisions of [ value(p2's move) - value(p1's move) ]
// (the oracle's "best" cancels in the difference).  Positive => p1 gives up
// more, i.e. p2 is the better mover by that margin.  Same-move decisions are
// exact zeros: kept as-is in --file mode; in --referee mode the referee is
// skipped for them (they're still counted for the agreement rate).
//
// Agents load like selfplay.js: --p1 / --p2 through slot-scoped config readers,
// so P1_ / P2_ env prefixes differentiate the two sides (P1_/P2_BUDGET too).
// The referee loads on slot 'R' (PR_* config); its budget is --referee-budget
// (default 4x the agent budget, so it out-searches them).  p1/p2 must export
// getMove(); the referee must export valueB(game, opts) -> P(BLACK wins).
//
// Oracle-agreement instrument, NOT a game-strength verdict — settle deployment
// calls with a selfplay match.
//
// Usage:
//   node compare-move-selection.js --p1 <name> --p2 <name>
//        (--file <path> | --referee <name>) [--budget MS] [--min-phase F]
//        [--max-phase F] [--limit N] [--seed N]
//        (--file only:) [--oversample N]
//        (--referee only:) [--size N] [--rand-moves N] [--sample F]

const fs   = require('fs');
const path = require('path');
const { performance } = require('perf_hooks');
const { Game2, BLACK, coordStr, parseMove } = require('./game2.js');
const { makeRng } = require('./xorshift.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'],
  ['p1', 'p2', 'file', 'referee', 'budget', 'referee-budget', 'limit', 'oversample', 'seed',
   'min-phase', 'max-phase', 'size', 'rand-moves', 'sample']);

if (opts.help || !opts.p1 || !opts.p2 || (!opts.file && !opts.referee)) {
  console.error(`Usage: node compare-move-selection.js --p1 <name> --p2 <name> (--file <path> | --referee <name>) [options]

Compare two agents' move selection, scored against an oracle.  Headline is the
paired mean win-prob difference Δ(p2-p1): positive => p2 (the challenger) is the
better mover by that margin.  Agents are interleaved per decision (shared load).

  --p1 NAME         ai/<name>.js for slot 1 (needs getMove()); P1_* env config
  --p2 NAME         ai/<name>.js for slot 2; P2_* env config
  --file PATH       fixed pre-rated positions (createmovedetails.js).  One of
                    --file / --referee is required (mutually exclusive)
  --referee NAME    LIVE mode: generate positions by mixed p1/p2 play (each
                    move from a random agent) and score each move by this
                    referee's valueB() (P(BLACK wins)).  Slot 'R' config (PR_*)
  --budget MS       per-move agent budget (default 1000); P1_/P2_BUDGET override
  --referee-budget MS  (--referee only) referee search budget (default: 4x
                    --budget, so the oracle out-searches the agents)
  --min-phase F     restrict to board fullness >= F (default 0)
  --max-phase F     restrict to board fullness <= F (default 1)
  --limit N         --file: first N positions; --referee: stop after N labelled
                    (disagreeing) decisions (default: all / run until killed)
  --seed N          starting rng seed (default: random, logged at startup)
  --oversample N    (--file only) evaluate each position N times (default 1)
  --size N          (--referee only) board size for generated games (default 13)
  --rand-moves N    (--referee only) random opening moves per game (default 4)
  --sample F        (--referee only) fraction of decisions sampled (default 1 = all)
  --help            show this message`);
  process.exit(opts.help ? 0 : 1);
}

const p1Name   = opts.p1;
const p2Name   = opts.p2;
const FILE_MODE = !!opts.file;
const REF_MODE  = !!opts.referee;
if (FILE_MODE && REF_MODE) { console.error('--file and --referee are mutually exclusive'); process.exit(1); }

const budgetMs = opts.budget !== undefined ? parseInt(opts.budget, 10) : 1000;
const limit    = opts.limit !== undefined ? parseInt(opts.limit, 10) : Infinity;
// Default seed is non-deterministic (time/pid); it's logged at startup so a run
// can be reproduced with --seed.
const SEED     = opts.seed !== undefined ? parseInt(opts.seed, 10) : Util.randomSeed();
const minPhase = opts['min-phase'] !== undefined ? parseFloat(opts['min-phase']) : 0;
const maxPhase = opts['max-phase'] !== undefined ? parseFloat(opts['max-phase']) : 1;

if (isNaN(budgetMs) || budgetMs < 1) { console.error('--budget must be a positive integer'); process.exit(1); }
if (isNaN(limit) || limit < 1)       { console.error('--limit must be a positive integer'); process.exit(1); }
if (isNaN(SEED))                     { console.error('--seed must be an integer'); process.exit(1); }
if (isNaN(minPhase) || isNaN(maxPhase) || minPhase < 0 || maxPhase > 1 || minPhase > maxPhase) {
  console.error('--min-phase/--max-phase must satisfy 0 <= min <= max <= 1'); process.exit(1);
}
if (FILE_MODE && (opts.size !== undefined || opts['rand-moves'] !== undefined ||
                  opts.sample !== undefined || opts['referee-budget'] !== undefined)) {
  console.error('--size/--rand-moves/--sample/--referee-budget apply only with --referee'); process.exit(1);
}
if (REF_MODE && opts.oversample !== undefined) {
  console.error('--oversample applies only with --file'); process.exit(1);
}
const bandActive = minPhase > 0 || maxPhase < 1;

// Slot-aware agent load (P1_/P2_/PR_ env prefixes), as in selfplay.js.
function loadAgent(name, slot) {
  const mod = require(path.join(__dirname, 'ai', name + '.js'));
  return (typeof mod.create === 'function') ? mod.create(Util.makeCfg(slot)) : mod;
}
function need(inst, method, name) {
  if (typeof inst[method] !== 'function') {
    console.error(`agent '${name}' does not export ${method}()`); process.exit(1);
  }
  return inst[method];
}
function slotBudget(slot) {
  const v = process.env['P' + slot + '_BUDGET'];
  return v !== undefined ? parseInt(v, 10) : budgetMs;
}
const p1 = need(loadAgent(p1Name, 1), 'getMove', p1Name);
const p2 = need(loadAgent(p2Name, 2), 'getMove', p2Name);
const budget1 = slotBudget(1);
const budget2 = slotBudget(2);

// Per-side reproducible agent rng seeds, decoupled from move order.
let p1Seed = SEED, p2Seed = SEED;

const r4  = Util.fmtRatio4;
const sgn = v => (v >= 0 ? '+' : '-') + r4(Math.abs(v));

if (FILE_MODE) runFileMode();
else           runRefereeMode();

// ── --file: fixed pre-rated positions ──────────────────────────────────────────
function runFileMode() {
  const oversample = opts.oversample !== undefined ? parseInt(opts.oversample, 10) : 1;
  if (isNaN(oversample) || oversample < 1) { console.error('--oversample must be a positive integer'); process.exit(1); }

  const loadPositions = fp => fs.readFileSync(fp, 'utf8').split('\n')
    .filter(l => l.trim() && !l.startsWith('#')).map(l => JSON.parse(l));
  const positionPhase = position => {
    const { boardSize, history } = position;
    const g = new Game2(boardSize, true);
    for (const h of history) g.play(parseMove(h, boardSize));
    return 1 - g.emptyCount / (boardSize * boardSize);
  };

  const pool = loadPositions(opts.file);
  const positions = (bandActive
    ? pool.filter(p => { const ph = positionPhase(p); return ph >= minPhase && ph <= maxPhase; })
    : pool).slice(0, limit);
  if (positions.length === 0) {
    console.error(`no positions${bandActive ? ` in band [${minPhase}, ${maxPhase}]` : ''} in ${opts.file}`); process.exit(1);
  }

  // One agent's chosen move + its win-prob gap to the file's top-rated move.
  const moveGap = (agent, boardSize, history, candidates, top, budget, isP1) => {
    const game = new Game2(boardSize, true);
    for (const h of history) game.play(parseMove(h, boardSize));
    const mv  = agent(game, budget, { rng: makeRng(isP1 ? p1Seed++ : p2Seed++) });
    const str = coordStr(mv.move, boardSize);
    const found = candidates.find(c => c.m === str);
    const cand  = (found?.kwr != null) ? found : candidates.findLast(c => c.kwr != null);
    return { str, gap: (top.kwr - cand.kwr) / 1000 };
  };

  console.log(`p1=${p1Name}  p2=${p2Name}  budget=${budgetMs}ms  seed=${SEED}`);
  console.log(`${positions.length} positions` +
    (bandActive ? ` in phase [${minPhase}, ${maxPhase}]` : '') +
    (oversample > 1 ? ` (oversampled ${oversample}x to ${positions.length * oversample})` : ''));
  console.log();
  console.log(['pos'.padStart(5), 'elapsed'.padStart(7),
               'gapP1'.padStart(6), 'gapP2'.padStart(6), 'Δp2-p1'.padStart(7), 'agree'.padStart(6)].join('  '));

  const startTime = performance.now();
  let evals = 0, sum1 = 0, sum2 = 0, agreeN = 0, nextPrintPos = 1, printedAt = -1;   // geometric row schedule by positions
  const printStats = () => {
    console.log([
      Util.fmt4i(evals).padStart(5), Util.fmtMs(performance.now() - startTime).padStart(7),
      r4(sum1 / evals).padStart(6), r4(sum2 / evals).padStart(6),
      sgn((sum1 - sum2) / evals).padStart(7), (agreeN / evals).toFixed(3).padStart(6),
    ].join('  '));
  };

  for (let j = 0; j < oversample; j++) {
    for (let i = 0; i < positions.length; i++) {
      const { boardSize, history, candidates } = positions[i];
      const top = candidates[0];
      let r1, r2;
      if (evals % 2 === 0) {   // alternate the first mover
        r1 = moveGap(p1, boardSize, history, candidates, top, budget1, true);
        r2 = moveGap(p2, boardSize, history, candidates, top, budget2, false);
      } else {
        r2 = moveGap(p2, boardSize, history, candidates, top, budget2, false);
        r1 = moveGap(p1, boardSize, history, candidates, top, budget1, true);
      }
      sum1 += r1.gap; sum2 += r2.gap;
      if (r1.str === r2.str) agreeN++;
      evals++;
      if (evals >= nextPrintPos) { printStats(); printedAt = evals; nextPrintPos = Math.max(Math.ceil(nextPrintPos * 1.5), nextPrintPos + 1); }
    }
  }
  if (evals !== printedAt) printStats();   // final total, unless the loop just printed it
}

// ── --referee: live mixed-p1/p2 play, referee-labelled ─────────────────────────
function runRefereeMode() {
  const size      = opts.size !== undefined ? parseInt(opts.size, 10) : 13;
  const randMoves = opts['rand-moves'] !== undefined ? parseInt(opts['rand-moves'], 10) : 4;
  const sample    = opts.sample !== undefined ? parseFloat(opts.sample) : 1;
  if (isNaN(size) || size < 3)                { console.error('--size must be an integer >= 3'); process.exit(1); }
  if (isNaN(randMoves) || randMoves < 0)      { console.error('--rand-moves must be a non-negative integer'); process.exit(1); }
  if (isNaN(sample) || !(sample > 0 && sample <= 1)) { console.error('--sample must be in (0, 1]'); process.exit(1); }

  const refValueB = need(loadAgent(opts.referee, 'R'), 'valueB', opts.referee);
  // The referee should out-search the agents to be a useful oracle: default its
  // budget to 4x the agent budget (passed to valueB; ignored by fixed-depth
  // referees that don't honour a time budget).
  const refBudget = opts['referee-budget'] !== undefined ? parseInt(opts['referee-budget'], 10) : 4 * budgetMs;
  if (isNaN(refBudget) || refBudget < 1) { console.error('--referee-budget must be a positive integer'); process.exit(1); }
  const area      = size * size;
  const gameRng   = makeRng(SEED);   // openings + sampling
  let refSeed     = SEED;            // referee rng

  // Win prob (mover's perspective) of playing `move`: play it, then ask the
  // referee's valueB for the resulting position (P(BLACK wins)) and map it to
  // the mover's frame.
  const labelMove = (game, move) => {
    const mover = game.current;
    const clone = game.clone();
    clone.play(move);
    if (clone.gameOver) return clone.calcWinner() === mover ? 1 : 0;
    const vB = refValueB(clone, { rng: makeRng(refSeed++), budgetMs: refBudget });   // P(BLACK wins)
    return mover === BLACK ? vB : 1 - vB;                                            // P(mover wins)
  };

  console.log(`p1=${p1Name}  p2=${p2Name}  referee=${opts.referee}@${refBudget}ms  budget=${budgetMs}ms  ` +
    `sample=${sample}  rand-moves=${randMoves}  size=${size}  seed=${SEED}` +
    (bandActive ? `  band=[${minPhase}, ${maxPhase}]` : '') + (limit !== Infinity ? `  limit=${limit}` : ''));
  console.log();
  console.log(['sampled'.padStart(7), 'elapsed'.padStart(7),
               'Δp2-p1'.padStart(7), 'agree'.padStart(6)].join('  '));

  const startTime = performance.now();
  let sampled = 0, agreed = 0, disag = 0, sumD = 0, nextPrintPos = 1, printedAt = -1;   // geometric row schedule by sampled positions
  const printStats = () => {
    const el = performance.now() - startTime;
    console.log([
      Util.fmt4i(sampled).padStart(7), Util.fmtMs(el).padStart(7),
      (sampled ? sgn(sumD / sampled) : '-').padStart(7), (sampled ? (agreed / sampled).toFixed(3) : '-').padStart(6),
    ].join('  '));
  };

  outer:
  while (disag < limit) {
    const game = new Game2(size, true);
    for (let i = 0; i < randMoves && !game.gameOver; i++) game.play(game.randomLegalMove(gameRng));
    const maxMoves = area * 4;
    while (!game.gameOver && game.moveCount < maxMoves) {
      const phase = 1 - game.emptyCount / area;
      if (phase > maxPhase) break;   // past the band — nothing more eligible; new game
      // Each move is taken from a randomly chosen agent, so the game is a 50/50
      // MIXTURE of the two: both face the same neutral position distribution
      // (no colour asymmetry, and it won't resolve toward the stronger agent).
      const useP1   = gameRng.random() < 0.5;
      const advMove = (useP1 ? p1 : p2)(game, useP1 ? budget1 : budget2,
                                        { rng: makeRng(useP1 ? p1Seed++ : p2Seed++) }).move;
      if (phase >= minPhase && gameRng.random() < sample) {
        sampled++;
        const otherMove = (useP1 ? p2 : p1)(game, useP1 ? budget2 : budget1,
                                            { rng: makeRng(useP1 ? p2Seed++ : p1Seed++) }).move;
        const m1 = useP1 ? advMove : otherMove;   // p1's choice here
        const m2 = useP1 ? otherMove : advMove;   // p2's choice here
        if (m1 === m2) {
          agreed++;                               // zero difference — no referee
        } else {
          const wr1 = labelMove(game, m1);        // same mover for both, so
          const wr2 = labelMove(game, m2);        // directly comparable
          sumD += (wr2 - wr1); disag++;
          if (disag >= limit) break outer;
        }
      }
      game.play(advMove);
      if (sampled >= nextPrintPos) { printStats(); printedAt = sampled; nextPrintPos = Math.max(Math.ceil(nextPrintPos * 1.5), nextPrintPos + 1); }
    }
  }
  if (sampled === 0) { console.error('no decisions sampled'); process.exit(1); }
  if (sampled !== printedAt) printStats();   // final total, unless the loop just printed it
}
