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
 *   --rand-moves <n>     Play n random moves at the start of each position (default 0).
 *                        A randomised opening (this or --rand-mirror-pairs) starts
 *                        from an EMPTY board — the free centre stone is a
 *                        first-move time-saver, and the random opening supplies
 *                        that move itself.  --rand-moves 0 keeps the free stone.
 *                        Defaults to 0 so the default balanced opening stays
 *                        balanced; pass it explicitly to add symmetry-breaking
 *                        moves on top.
 *                        The default of 4 reproduces the old opening exactly: the
 *                        free centre stone plus 3 random moves was 4 stones with
 *                        black to move, and 4 random moves is the same position
 *                        distribution up to a translation (the torus is
 *                        vertex-transitive, so black's first move is arbitrary)
 *   --rand-mirror-pairs <n>  Before --rand-moves, play n random MIRROR PAIRS: each
 *                        random move is answered by its 180-degree rotation.  On a
 *                        toroidal board that rotation is an exact automorphism, so
 *                        the result is invariant under (rotate 180 + swap colours)
 *                        — an exactly balanced position, no side favoured.  Plain
 *                        random openings are fair (the pair shares one opening) but
 *                        a lopsided one is decided before either agent moves, and
 *                        the colour-swapped pair then splits 1-1 and carries no
 *                        information.  Balancing converts those wasted pairs into
 *                        informative ones.  Starts from an empty board (see
 *                        --rand-moves): the free centre stone is unpaired and
 *                        would break the antisymmetry outright        (default 0)
 *   --min-phase <f>      p1/p2 only play once board phase (1−empty/area) ≥ f; the
 *                        fallback plays the opening up to it       (default 0)
 *   --max-phase <f>      p1/p2 stop once phase > f; the fallback completes the
 *                        game                                      (default 1)
 *   --fallback <policy>  Agent that plays outside the phase window (default ref-ab2-fp4-vpat)
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
const { makeRng } = require('./xorshift.js');
const Util = require('./util.js');

const VERBOSE = Util.envInt('VERBOSE', 0);

const opts = Util.parseArgs(process.argv.slice(2), ['help'],
  ['p1', 'p2', 'size', 'budget', 'limit', 'rand-moves', 'rand-mirror-pairs',
   'min-phase', 'max-phase', 'fallback', 'adjudication-playouts']);

if (opts.help) {
  console.log(`Usage: node selfplay.js [options]

Play matches between two agents and report the win statistics.  Colours
alternate between games; per-agent env config uses the P1_/P2_ prefixes
(e.g. P2_PPAT_DATA=... for --p2's ppat weights).

  --p1 AGENT        ai/<name>.js for player 1 (default prod)
  --p2 AGENT        ai/<name>.js for player 2 (default: same as --p1)
  --size N          board size (default 13)
  --budget MS       per-move time budget in ms (default 1)
  --limit N         stop after N games (default: run indefinitely)

  --rand-moves N    random opening moves per game for diversity (default 0)
  --rand-mirror-pairs N
                    balanced opening: N random black/white stone pairs placed
                    at mirrored positions (antisymmetric start) (default 3)

  --min-phase F     p1/p2 play only at board fullness >= F; the fallback agent
                    plays both sides before that.  The opening is built once
                    per match pair and shared across the colour swap
  --max-phase F     past this phase the fallback completes the game (per-move
                    gate).  Defaults (0,1) = whole game, no fallback
  --fallback AGENT  agent for moves outside the phase window
                    (default ref-featurepol-softmax)
  --adjudication-playouts N
                    adjudicate instead of playing out past max-phase: run N
                    standard playouts (mc-ppat) from the handoff position and
                    score the game to the side whose win-ratio exceeds 0.5
                    (exact ties flip a coin).  Far lower variance than a
                    played-out fallback.  Votes are PURE-UNIFORM playouts
                    by default (no learned components in the judge); judge
                    env config uses the PJ_ prefix, e.g.
                    PJ_PPAT_MIN_PHASE=0.6 for standard ppat-flavoured
                    votes.  The pre-min-phase opening still uses the
                    --fallback agent.  (default 0 = play out)

  --help            show this message

  env VERBOSE=1     print the board and agent info after every move`);
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
const randMirrorPairs = parseInt(opts['rand-mirror-pairs'] ?? '3', 10);
// --rand-moves defaults to 0: the default opening is the balanced one, and extra
// unpaired stones would undo the antisymmetry the mirror pairs establish (and
// silently deepen the opening, which is not what "balanced opening" should mean).
// Pass --rand-moves explicitly to compose the two deliberately.
const randMoves = parseInt(opts['rand-moves'] ?? '0', 10);

// Phase window: p1/p2 only play moves with phase (= 1 − empty/area) in
// [min-phase, max-phase]; the fallback agent plays both sides outside it.  The
// opening (rand-moves + fallback up to min-phase) is built ONCE per match pair
// and shared by both colour assignments, so the windowed comparison stays fair
// under colour swap.  Past max-phase the fallback completes the game (per-move
// gate).  This isolates the agents' contribution to a phase band — a
// game-played strength-by-phase profile.  Defaults (0,1) = whole game, no fallback.
const minPhase     = opts['min-phase'] !== undefined ? parseFloat(opts['min-phase']) : 0;
const maxPhase     = opts['max-phase'] !== undefined ? parseFloat(opts['max-phase']) : 1;
const fallbackName = opts.fallback || 'ref-ab2-fp4-vpat';
const adjudicationPlayouts = parseInt(opts['adjudication-playouts'] || '0', 10);
if (!(adjudicationPlayouts >= 0)) {
  console.error('--adjudication-playouts must be a non-negative integer');
  process.exit(1);
}
if (minPhase < 0 || maxPhase > 1 || minPhase > maxPhase) {
  console.error('--min-phase/--max-phase must satisfy 0 <= min <= max <= 1');
  process.exit(1);
}

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
// Post-max-phase adjudicator (--adjudication-playouts): an mc-ppat instance whose
// valueB runs the votes.  Judge config uses slot 'J' (PJ_* overrides, plain
// env fallback), so judge-only knobs cannot leak into the contestants'
// plain-env reads.  Votes default to PURE-UNIFORM playouts
// (PPAT_MIN_PHASE 1): a judge with no learned components shares no training
// lineage with the contestants, so its errors cannot correlate with the
// knob under test (measured cost vs ppat-flavoured votes: ~3x the games —
// 1k-vs-2k calibration, 2026-09-06).  PJ_PPAT_MIN_PHASE opts back into
// ppat-flavoured votes; the PLAYOUTS override pins the count.
const adjudicator = usePhaseWindow && adjudicationPlayouts > 0
  ? require('./ai/mc-ppat.js').create(Util.makeCfg('J', {
      PLAYOUTS: String(adjudicationPlayouts),
      PPAT_MIN_PHASE: process.env.PJ_PPAT_MIN_PHASE !== undefined ? process.env.PJ_PPAT_MIN_PHASE : '1',
    }))
  : null;
let adjCount = 0, adjCloseCount = 0, adjAbsSum = 0;   // adjudication margin stats
if (usePhaseWindow)
  console.log(`phase window: [${minPhase}, ${maxPhase}]  fallback: ${fallbackName} (budget ${fallbackBudget}ms)` +
    (adjudicator ? `  adjudication: ${adjudicationPlayouts} standard playouts past max-phase` : ''));

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
  ].join('  '));
  // Adjudication margin stats: diagnostic only (close-call fraction flags a
  // margin-compressed matchup); VERBOSE to keep routine output to the table.
  if (VERBOSE && adjCount > 0) {
    console.log(`adj: ${adjCount} games  close(|P-0.5|<0.05): ${adjCloseCount}  avg|margin|: ${(adjAbsSum / adjCount).toFixed(3)}`);
  }
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

// Play one game from a given starting position with assigned colors.
//
// seatSeeds [blackSeed, whiteSeed]: per opening pair, each SEAT gets its own
// seeded RNG stream, recreated identically for the colour-swapped replay — so
// both agents experience the same randomness in the same seat (common random
// numbers).  The seat stream is passed as the move's options.rng (the agents'
// own generator, which otherwise seeds from the clock every move) AND
// installed as Math.random for its duration, so all agent-internal
// randomness (dither, playout sampling, tie-breaks) draws from it.  Agents
// that build one generator per instance ignore both, and stay unpaired.
// Identical agents therefore mirror exactly
// across a pair when moves are deterministic given the stream (e.g. fixed
// playouts); time budgets reintroduce divergence via playout-count jitter.
function playGame(startGame, p1IsBlack, seatSeeds) {
  const seatRng = [makeRng(seatSeeds[0]), makeRng(seatSeeds[1])];   // [black, white]
  const origRandom = Math.random;
  const names = [ p1IsBlack ? p1Name : p2Name, p1IsBlack ? p2Name : p1Name];
  const black = p1IsBlack ? p1 : p2;
  const white = p1IsBlack ? p2 : p1;

  if (VERBOSE) console.log(`${names[0]} ● vs ${names[1]} ○`);

  const game = startGame.clone();
  let adjP = null;   // adjudicated P(BLACK wins), when --adjudication-playouts ends the game

  while (!game.gameOver) {
    const isBlackTurn = game.current === BLACK;

    // Past max-phase: the fallback completes the game for both sides, not
    // attributed to p1/p2 timing.  (min-phase is already satisfied by the shared
    // opening; we don't re-gate on it mid-game — see phaseOf note above.)
    if (fallback && phaseOf(game) > maxPhase) {
      if (adjudicator) {
        // Adjudicate instead of playing out: N standard playouts from the
        // handoff position; the game scores to the playout-majority side.
        adjP = adjudicator.valueB(game, { rng: isBlackTurn ? seatRng[0] : seatRng[1] });
        adjCount++;
        const m = Math.abs(adjP - 0.5);
        adjAbsSum += m;
        if (m < 0.05) adjCloseCount++;
        if (VERBOSE) console.log(`adjudicated at phase ${phaseOf(game).toFixed(3)}: P(BLACK) = ${adjP.toFixed(3)}`);
        break;
      }
      const rng = isBlackTurn ? seatRng[0] : seatRng[1];
      Math.random = rng.random;
      const fm = fallback(game, fallbackBudget, { rng });
      Math.random = origRandom;
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
    const rng = isBlackTurn ? seatRng[0] : seatRng[1];
    Math.random = rng.random;
    const move = policy(game, budget, { rng });
    Math.random = origRandom;
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
  const winner = adjP !== null
    ? (adjP > 0.5 ? BLACK : adjP < 0.5 ? WHITE
       : (seatRng[0].random() < 0.5 ? BLACK : WHITE))
    : game.calcWinner();
  if (winner === BLACK) {
    blackWinCount++;
    tally[p1IsBlack ? 'p1' : 'p2']++;
  } else if (winner === WHITE) {
    tally[p1IsBlack ? 'p2' : 'p1']++;
  }
}

// Run games until the limit (or forever if no limit).
// 180-degree rotation of a board index.  The torus has no edges, so the point
// reflection (x,y) -> (-x, -y) mod N is an exact automorphism of the board.
function mirror180(idx, N) {
  const x = idx % N, y = (idx - x) / N;
  return ((N - y) % N) * N + ((N - x) % N);
}

// True when the position is invariant under (rotate 180 + swap colours), i.e.
// exactly balanced: every stone has an opposite-coloured counterpart at its
// rotation, so neither side can hold an advantage.
function isAntisymmetric(game) {
  const N = game.N, area = N * N;
  for (let i = 0; i < area; i++)
    if (game.cells[i] !== -game.cells[mirror180(i, N)]) return false;
  return true;
}

// Play a random move and its 180-degree rotation, leaving the position invariant
// under (rotate 180 + swap colours) — exactly balanced, neither side favoured.
// Returns false when no pair can be placed.  Two ways a draw fails: the move is a
// fixed point of the involution (one such point on odd N, four on even N), or the
// mirror is illegal once the first stone is down (ko).  Validated on a clone
// because Game2 has no undo.
function playMirrorPair(game) {
  for (let tries = 0; tries < 32; tries++) {
    const idx = game.randomLegalMove();
    if (idx === PASS) return false;
    const m = mirror180(idx, game.N);
    if (m === idx) continue;
    const trial = game.clone();
    if (!trial.play(idx) || !trial.play(m)) continue;
    // Captures can interact when the two moves land close together on the torus:
    // the first stone's capture changes the board before its mirror is played, so
    // the mirrored capture need not match.  Rare (~1 in 400 openings), and cheap
    // to exclude outright by checking the invariant rather than assuming it.
    if (!isAntisymmetric(trial)) continue;
    game.play(idx);
    game.play(m);
    return true;
  }
  return false;
}

// Each opening is played twice with swapped colors.
let gamesPlayed = 0;
while (gamesPlayed < gameLimit) {
  // Generate a random opening position.
  // Any randomised opening starts from an EMPTY board.  Game2's free centre stone
  // is only a time-saver — the torus is vertex-transitive, so black's first move
  // is arbitrary and may as well be placed for free — but once the opening is
  // randomised the first random move already IS that move, and keeping the centre
  // stone would just prepend a fixed stone to a random opening.  With mirror pairs
  // it also breaks the antisymmetry outright, being unpaired (and on even N it
  // sits on a fixed point of the involution, where it cannot be mirrored at all).
  const opening = new Game2(boardSize, randMirrorPairs === 0 && randMoves === 0);
  // Mirror pairs first (balanced), then plain random moves (which break the
  // symmetry), then the fallback up to --min-phase.
  for (let i = 0; i < randMirrorPairs && !opening.gameOver; i++)
    if (!playMirrorPair(opening)) break;
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

  // Seat seeds for this pair: both colour assignments replay the same
  // black-stream and white-stream (see playGame).
  const seatSeeds = [(Math.random() * 0x100000000) | 0,
                     (Math.random() * 0x100000000) | 0];

  // Play from this opening with both color assignments.
  for (let swap = 0; swap < 2 && gamesPlayed < gameLimit; swap++) {
    playGame(opening, swap === 0, seatSeeds);
    gamesPlayed++;
    maybePrint(gamesPlayed);
  }
}

// Final stats row, unless the last periodic row was already this game.
if (gamesPlayed !== lastPrintGames) printStats(gamesPlayed);


