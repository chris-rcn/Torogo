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
 *   --size    <n>        Board size: 9, 13, or 19    (default 13)
 *   --budget  <ms>       Time budget per move in ms  (required)
 *   --limit   <n>        Stop after this many games and print final stats
 *   --rand-moves <n>     Play n random moves at the start of each position (default 3)
 *   --min-phase <f>      p1/p2 only play once board phase (1−empty/area) ≥ f; the
 *                        fallback plays the opening up to it       (default 0)
 *   --max-phase <f>      p1/p2 stop once phase > f; the fallback completes the
 *                        game                                      (default 1)
 *   --fallback <policy>  Agent that plays outside the phase window (default ref-npat)
 *   --help               Show this help message
 *
 * Env variables:
 *   VERBOSE=1            Print the board after every move
 *   P1_ / P2_ prefix     Per-slot agent config (e.g. P1_PPAT_DATA, P2_RAVE_K)
 *   P1_BUDGET, P2_BUDGET Per-slot time budget override (asymmetric-time matches)
 *   PF_ prefix, PF_BUDGET Fallback-agent config and budget
 *
 * Colors alternate each game: p1 is black in odd games, white in even games.
 * Policy names are filenames without the .js extension inside the ai/ folder.
 * A phase window ([min-phase, max-phase] ≠ [0,1]) isolates the agents' play to a
 * phase band — the opening (rand-moves + fallback to min-phase) is built once per
 * colour-swapped pair so the comparison stays fair.  Use it to profile
 * strength by game phase.
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

const opts = Util.parseArgs(process.argv.slice(2), ['help'],
  ['p1', 'p2', 'size', 'budget', 'limit', 'rand-moves', 'stop-tol', 'stop-min',
   'min-phase', 'max-phase', 'fallback']);

if (opts.help) {
  console.log(`Usage: node selfplay.js [--p1 <policy>] [--p2 <policy>] [--size <n>] [--budget <ms>] [--limit <n>]\n` +
              `       [--stop-tol <a>] [--stop-min <n>]  (early-stop when P(p2 better) reaches 1-a or a)`);
  process.exit(0);
}

const gameLimit = opts.limit !== undefined ? parseInt(opts.limit, 10) : Infinity;
if (isNaN(gameLimit) || gameLimit < 1) {
  console.error('--limit must be a positive integer');
  process.exit(1);
}

const p1Name    = opts.p1   || 'prod';
const p2Name    = opts.p2   || p1Name;
const boardSize = parseInt(opts.size || '13', 10);
const budgetMs  = parseInt(opts.budget || '1', 10);
const randMoves = parseInt(opts['rand-moves'] || '3', 10);

// Phase window: p1/p2 only play moves with phase (= 1 − empty/area) in
// [min-phase, max-phase]; the fallback agent plays both sides outside it.  The
// opening (rand-moves + fallback up to min-phase) is built ONCE per match pair
// and shared by both colour assignments, so the windowed comparison stays fair
// under colour swap.  Past max-phase the fallback completes the game (per-move
// gate).  This isolates the agents' contribution to a phase band — a
// game-played strength-by-phase profile.  Defaults (0,1) = whole game, no fallback.
const minPhase     = opts['min-phase'] !== undefined ? parseFloat(opts['min-phase']) : 0;
const maxPhase     = opts['max-phase'] !== undefined ? parseFloat(opts['max-phase']) : 1;
const fallbackName = opts.fallback || 'ref-npat';
if (minPhase < 0 || maxPhase > 1 || minPhase > maxPhase) {
  console.error('--min-phase/--max-phase must satisfy 0 <= min <= max <= 1');
  process.exit(1);
}

// Early-stop (SPRT-style): stop once the decision is confident either way.
// --stop-tol a → stop when P(p2 truly better than 50%) reaches 1-a (p2/candidate
//                confidently better) or falls to a (confidently worse).  Symmetric.
// --stop-min n → minimum games before the bound can trigger (avoids tiny-sample stops).
const stopTol = opts['stop-tol'] !== undefined ? parseFloat(opts['stop-tol']) : null;
const stopMin = opts['stop-min'] !== undefined ? parseInt(opts['stop-min'], 10) : 0;

if (!Number.isInteger(boardSize)) {
  console.error('--size must be an odd integer between 7 and 19');
  process.exit(1);
}

// Load an agent for slot 1 or 2.  Factory agents (exporting create) get their
// own instance built from a slot-scoped config reader, so the same agent file
// can run on both sides with differentiated config (P1_*/P2_* env).  Legacy
// agents (no create) are used directly — they read plain env at load, so they
// can't be differentiated, but otherwise behave as before.
function loadAgent(name, slot) {
  const mod  = require(path.join(__dirname, 'ai', name + '.js'));
  const inst = (typeof mod.create === 'function') ? mod.create(Util.makeCfg(slot)) : mod;
  return inst.getMove;
}
// Per-slot time budget: P<slot>_BUDGET overrides the shared --budget for that
// side (enables asymmetric-time matches), else falls back to --budget.
function slotBudget(slot) {
  const v = typeof process !== 'undefined' ? process.env['P' + slot + '_BUDGET'] : undefined;
  return v !== undefined ? parseInt(v, 10) : budgetMs;
}
const p1 = loadAgent(p1Name, 1);
const p2 = loadAgent(p2Name, 2);
const p1Budget = slotBudget(1);
const p2Budget = slotBudget(2);

// Fallback agent (slot 'F': PF_* config / PF_BUDGET) — only loaded when a phase
// window is requested.  It plays the opening up to min-phase and completes the
// game past max-phase, for both sides.
const usePhaseWindow = minPhase > 0 || maxPhase < 1;
const fallback       = usePhaseWindow ? loadAgent(fallbackName, 'F') : null;
const fallbackBudget = slotBudget('F');
if (usePhaseWindow)
  console.log(`phase window: [${minPhase}, ${maxPhase}]  fallback: ${fallbackName} (budget ${fallbackBudget}ms)`);

// Board fullness used to gate the window: 1 − empty/area, per the canonical
// "phase" definition.  Not strictly monotonic (captures lower it), which is fine
// here — only the max-phase cutoff is a per-move gate; min-phase is baked into
// the shared opening.
function phaseOf(g) { return g.phase(); }

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

console.log([
  'games'   .padStart(5),
  'elapsed' .padStart(7),
  'blkWR'   .padStart(5),
  'avgLen'  .padStart(6),
  'maxLen'  .padStart(6),
  'tP1mv'   .padStart(5),
  'tP2Mv'   .padStart(5),
  'P2WR'    .padStart(4),
  'P2Better'.padStart(8),
].join('  '));

let printPeriodMs  = 1000;
let lastPrintTime  = startTime;
let lastPrintGames = 0;
let blackWinCount = 0;
let totalGameLen = 0;
let maxGameLen = 0;

function printStats(gamesPlayed) {
  const now = performance.now();
  const avgMs = (s) => (s.moves ? Util.fmtMs(s.ms / s.moves) : '    -').padStart(5);
  console.log([
    Util.fmt4i(gamesPlayed)                                .padStart(5),
    Util.fmtMs(now - startTime)                            .padStart(7),
    Util.fmtRatio4(blackWinCount / gamesPlayed)            .padStart(5),
    Util.fmt4(totalGameLen / gamesPlayed)                  .padStart(6),
    Util.fmt4i(maxGameLen)                                 .padStart(6),
    avgMs(stats.p1),
    avgMs(stats.p2),
    Util.fmtRatio4(tally.p2 / gamesPlayed)                 .padStart(4),
    Util.fmtRatio4(probPlayerBetter(tally.p2, gamesPlayed)).padStart(8),
  ].join('  '));
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

// Play one game from a given starting position with assigned colors.
function playGame(startGame, p1IsBlack) {
  const names = [ p1IsBlack ? p1Name : p2Name, p1IsBlack ? p2Name : p1Name];
  const black = p1IsBlack ? p1 : p2;
  const white = p1IsBlack ? p2 : p1;

  if (VERBOSE) console.log(`${names[0]} ● vs ${names[1]} ○`);

  const game = startGame.clone();

  while (!game.gameOver) {
    const isBlackTurn = game.current === BLACK;

    // Past max-phase: the fallback completes the game for both sides, not
    // attributed to p1/p2 timing.  (min-phase is already satisfied by the shared
    // opening; we don't re-gate on it mid-game — see phaseOf note above.)
    if (fallback && phaseOf(game) > maxPhase) {
      const fm = fallback(game, fallbackBudget);
      if (!game.play(fm.move)) {
        console.error(`Illegal fallback move: ${JSON.stringify(fm)}`);
        process.exit(1);
      }
      continue;
    }

    const policy = isBlackTurn ? black : white;
    const mover  = (isBlackTurn === p1IsBlack) ? 'p1' : 'p2';
    const budget = mover === 'p1' ? p1Budget : p2Budget;
    const t0 = performance.now();
    const move = policy(game, budget);
    stats[mover].ms    += performance.now() - t0;
    stats[mover].moves += 1;
    const idx = move.move;
    if (!game.play(idx)) {
      console.error(`Illegal move from ${mover} (${p1IsBlack ? p1Name : p2Name}): ${JSON.stringify(move)}`);
      process.exit(1);
    }
    if (VERBOSE) {
      console.log(`${names[isBlackTurn?0:1]}:`);
      printBoard(game);
      console.log(`Agent info: ${move.info}`);
      console.log();
    }
  }
  totalGameLen += game.moveCount;
  maxGameLen = Math.max(maxGameLen, game.moveCount);
  const winner = game.calcWinner();
  if (winner === BLACK) {
    blackWinCount++;
    tally[p1IsBlack ? 'p1' : 'p2']++;
  } else if (winner === WHITE) {
    tally[p1IsBlack ? 'p2' : 'p1']++;
  }
}

// Run games until the limit (or forever if no limit).
// Each opening is played twice with swapped colors.
let gamesPlayed = 0;
let decided = false;
while (gamesPlayed < gameLimit && !decided) {
  // Generate a random opening position.
  const opening = new Game2(boardSize);
  for (let i = 0; i < randMoves && !opening.gameOver; i++)
    opening.play(opening.randomLegalMove());

  // Advance with the fallback (both sides) up to min-phase, as part of the
  // shared opening — so both colour assignments start from the identical
  // position and the windowed comparison is fair under colour swap.
  if (fallback) {
    while (!opening.gameOver && phaseOf(opening) < minPhase) {
      const fm = fallback(opening, fallbackBudget);
      if (!opening.play(fm.move)) {
        console.error(`Illegal fallback setup move: ${JSON.stringify(fm)}`);
        process.exit(1);
      }
    }
  }

  // Play from this opening with both color assignments.
  for (let swap = 0; swap < 2 && gamesPlayed < gameLimit; swap++) {
    playGame(opening, swap === 0);
    gamesPlayed++;
    maybePrint(gamesPlayed);

    // SPRT-style early stop: bail once the result is confident either way.
    if (stopTol !== null && gamesPlayed >= stopMin) {
      const pb = probPlayerBetter(tally.p2, gamesPlayed);
      if (pb >= 1 - stopTol || pb <= stopTol) { decided = true; break; }
    }
  }
}

// Final stats row (always printed, even if maybePrint already fired).
printStats(gamesPlayed);


