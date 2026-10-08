#!/usr/bin/env node
'use strict';

// Generate per-position move-value data by self-play.
//
// Each self-play game starts from an empty board with MIRROR_PAIRS random
// mirror pairs (a random move and its 180-degree rotation in the other colour,
// so the opening is exactly balanced), then is played by the POSITION agent,
// which only has to return moves, until it ends or its board fullness (phase)
// first reaches 0.8;
// later positions are never drawn.  One position with phase below 0.8 is then
// drawn for deep analysis and checked for contestedness by the
// VALUE agent: its win ratio must be within [0.3, 0.7], else the game is
// discarded.  Phase bands
// (--phase-buckets N equal-width bands of [0,1]) are kept level: the game's
// positions are grouped by band, the band with the fewest samples so far in
// this run is taken (ties at random), and one of its positions is drawn at
// random.  Then every legal move
// is enumerated, the game is cloned, the move is made, then the VALUE
// agent's getMove is called with the full budget.  The rootWinRatio is flipped to the
// original player's perspective and recorded as kwr (× 1000).  A FLAT position
// — more than MAX_NEAR_BEST other moves rated within NEAR_BEST_DEV of the best
// — is discarded and does not count toward its band, so the band keeps
// being drawn: flat positions barely separate agents of different strength
// (verbose fleet sweep, 2026-10-07), and most opening positions are flat.
// Moves are rated in ref-featurepol's prior order (ORDER_MODEL) and rating
// stops as soon as the moves rated so far are flat against the best so far;
// a later move could still have beaten that best, so a few positions that
// are not flat are discarded too (~4% on md-trunc30k, for ~24% fewer
// ratings, mostly in the opening).  One
// position per game keeps the samples independent.  Games with no eligible
// position emit nothing.
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
const FeaturePol = require('./featurepol-lib.js');
const { game3FromGame2 } = require('./game3.js');

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

// Random mirror pairs opening each self-play game, before the position agent.
const MIRROR_PAIRS = 2;

// ...and not flat: at most MAX_NEAR_BEST other moves rated within
// NEAR_BEST_DEV of the best one.
const NEAR_BEST_DEV = 0.01;
const MAX_NEAR_BEST = 10;

// Rating order: ref-featurepol's prior, most probable first.
const ORDER_MODEL = path.join(__dirname, 'featurepol-cbk7wa32.js');
const { weights: orderWeights, modelName: orderModelName } = FeaturePol.loadModel({ name: 'featurepol', path: ORDER_MODEL });
const orderState = FeaturePol.createState(boardSize, orderWeights.spec);

// Legal moves by descending prior; moves the policy does not list (PASS) last.
function legalMovesByPrior(game2) {
  const N   = game2.N;
  const cap = N * N;
  FeaturePol.extractFeatures(game2, orderState, orderWeights, orderWeights.spec.needsLadder ? game3FromGame2(game2) : undefined);
  FeaturePol.computeSoftmax(orderState, orderWeights);
  const prior = new Float64Array(cap + 1).fill(-1);   // index cap = PASS
  for (let i = 0; i < orderState.count; i++) {
    const m = orderState.moves[i];
    prior[m === PASS ? cap : m] = orderState.probs[i];
  }
  const moves = [];
  for (let i = 0; i < cap; i++) {
    if (game2.cells[i] === 0 && game2.isLegal(i)) moves.push(i);
  }
  moves.sort((a, b) => prior[b] - prior[a]);
  moves.push(PASS);
  return moves;
}

// ── Mirror-pair openings (ported from selfplay.js) ────────────────────────────
function mirror180(idx, N) {
  const x = idx % N, y = (idx - x) / N;
  return ((N - y) % N) * N + ((N - x) % N);
}
function isAntisymmetric(game) {
  const N = game.N, area = N * N;
  for (let i = 0; i < area; i++)
    if (game.cells[i] !== -game.cells[mirror180(i, N)]) return false;
  return true;
}
// Play a random move and its 180-degree rotation, appending both to `record`;
// validated on a clone (no undo), rejecting fixed points and pairs whose
// captures interact.  False when no pair can be placed.
function playMirrorPair(game, record) {
  for (let tries = 0; tries < 32; tries++) {
    const idx = game.randomLegalMove();
    if (idx === PASS) return false;
    const m = mirror180(idx, game.N);
    if (m === idx) continue;
    const trial = game.clone();
    if (!trial.play(idx) || !trial.play(m)) continue;
    if (!isAntisymmetric(trial)) continue;
    game.play(idx);
    game.play(m);
    record.push(idx, m);
    return true;
  }
  return false;
}

process.stdout.write(`# createmovedetails.js  position-agent=${posAgentName}  value-agent=${valAgentName}  budget=${budget}ms  PLAYOUTS=${PLAYOUTS || '(budget)'}  size=${boardSize}  wr-dev=${WR_DEV}  phase-max=${PHASE_MAX}  mirror-pairs=${MIRROR_PAIRS}  max-near-best=${MAX_NEAR_BEST}@${NEAR_BEST_DEV}  order-model=${orderModelName}  phase-buckets=${phaseBuckets}\n`);

console.error(`position-agent=${posAgentName}  value-agent=${valAgentName}  size=${boardSize}  budget=${budget}ms`);
console.error();
console.error([
  'pos'    .padStart(5),
  'elapsed'.padStart(7),
  'tPos'   .padStart(5),
  'flat'   .padStart(5),
  'bands',
].join('  '));

const startTime = performance.now();
let printPeriodMs = 1000;
let lastPrintTime = startTime;
let posCount = 0;
let flatCount = 0;   // labeled positions discarded as flat

function printStats() {
  const elapsedMs = performance.now() - startTime;
  console.error([
    Util.fmt4i(posCount)            .padStart(5),
    Util.fmtMs(elapsedMs)           .padStart(7),
    Util.fmtMs(elapsedMs / posCount).padStart(5),
    Util.fmt4i(flatCount)           .padStart(5),
    Array.from(bandCount).join(','),   // samples per phase band so far
  ].join('  '));
}
const bandCount = new Int32Array(phaseBuckets);   // samples written per phase band, this run

while (true) {
  const N = boardSize;

  // Play the self-play game up to PHASE_MAX (nothing later is eligible),
  // recording each move and which positions (identified by the number of
  // moves played before them) fall in each band.  The mirror pairs come
  // first; positions inside the opening are not eligible.
  const game     = new Game2(boardSize, false);
  const moves    = [];
  const inBand = Array.from({ length: phaseBuckets }, () => []);   // per band: move counts of its positions
  for (let i = 0; i < MIRROR_PAIRS; i++) if (!playMirrorPair(game, moves)) break;

  while (!game.gameOver) {
    const phase = 1 - game.emptyCount / (N * N);
    if (phase >= PHASE_MAX) break;
    inBand[Math.min(phaseBuckets - 1, Math.floor(phase * phaseBuckets))].push(moves.length);
    const advancingMove = posAgent(game, budget);
    moves.push(advancingMove.move);
    game.play(advancingMove.move);
  }

  // Draw: the band with the fewest samples so far among those this game
  // offers (ties at random), then one of its positions at random.  The value
  // agent judges it; a position that is not contested discards the game.
  let band = -1, fewest = Infinity, ties = 0;
  for (let b = 0; b < phaseBuckets; b++) {
    if (inBand[b].length === 0) continue;
    if (bandCount[b] < fewest) { fewest = bandCount[b]; band = b; ties = 1; }
    else if (bandCount[b] === fewest && Math.random() * ++ties < 1) band = b;
  }
  if (band < 0) continue;
  const k = inBand[band][Math.floor(Math.random() * inBand[band].length)];
  const probe = new Game2(boardSize, false);
  for (let j = 0; j < k; j++) probe.play(moves[j]);
  const v = valAgent(probe, budget);
  if (v.rootWinRatio === undefined) { console.error('value agent did not return rootWinRatio'); process.exit(1); }
  if (Math.abs(v.rootWinRatio - 0.5) > WR_DEV) continue;
  const position = new Game2(boardSize, false);
  for (let i = 0; i < k; i++) position.play(moves[i]);
  const history = moves.slice(0, k).map(m => coordStr(m, N));

  // Flat: more than MAX_NEAR_BEST rated moves within NEAR_BEST_DEV of the
  // best so far (compared as stored, kwr thousandths) — stop rating and
  // discard without counting toward the band.
  const kwr = w => Math.round(w * 1000), nearDev = Math.round(NEAR_BEST_DEV * 1000);
  const ratedKwr = [];
  let bestKwr = -Infinity, flat = false;
  const moveInfos = [];
  for (const move of legalMovesByPrior(position)) {
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

    const wk = kwr(wr);
    ratedKwr.push(wk);
    if (wk > bestKwr) bestKwr = wk;
    let nearBest = -1;   // the best itself is not counted
    for (const r of ratedKwr) if (bestKwr - r <= nearDev) nearBest++;
    if (nearBest > MAX_NEAR_BEST) { flat = true; break; }
  }
  if (flat) { flatCount++; continue; }
  bandCount[band]++;

  moveInfos.sort((a, b) => (b.winRatio ?? -Infinity) - (a.winRatio ?? -Infinity));

  // The history holds EVERY stone (mirror pairs included), so the position
  // replays from an empty board.  Phase is stored (board fullness).
  const phase = 1 - position.emptyCount / (N * N);
  process.stdout.write(
    MD.formatRow({ boardSize, phase, history, candidates: moveInfos }) + '\n');
  posCount++;

  const now = performance.now();
  if (now - lastPrintTime >= printPeriodMs) {
    lastPrintTime = now;
    printPeriodMs = Math.round(printPeriodMs * 1.5);
    printStats();
  }
}
