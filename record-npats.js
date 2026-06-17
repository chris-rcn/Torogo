#!/usr/bin/env node
'use strict';

// record-npats.js — generate policy-training data from search self-play.
//
// Plays an agent against itself at a fixed playout count and records, for
// every move decision, the root visit distribution plus the search's value
// estimate.  Output is newline-delimited JSON, one record per decision,
// flushed a whole game at a time once the winner is known; the first line is
// a '#' comment recording the generation parameters.
//
//   {"boardSize":13,"history":["gg","dc"],"visits":[["dd",812],["cc",401]],
//    "q":0.512,"z":1}
//
//   history — replay path (coordStr tokens, e.g. "k4", 'pass' for PASS;
//             reconstruct with new Game2(boardSize) then parseMove+play each
//             token — same convention as the movedetails files)
//   visits  — root visit counts, sparse (moves with ≥ 1 rounded visit only)
//   q       — the search's rootWinRatio, mover's perspective
//   z       — final game outcome, mover's perspective (1 = mover won)
//
// Move selection: for the first --temp-moves decisions of each game the move
// is sampled ∝ visits^(1/temp) for opening diversity; afterwards (or with
// --temp 0) the agent's own choice (max visits) is played.  Independently,
// with probability --epsilon a uniform random legal move is played instead —
// full-game exploration that drags self-play off the policy's own
// distribution (the recorded targets are unaffected: every position keeps
// the search's visit distribution; only the continuation is randomised).
// Note ε-moves contaminate z, so record with --epsilon 0 if z is ever
// revisited as a training signal.
//
// The agent must run a fixed playout count per decision (options.playoutLimit)
// and report flat-index `children` visit stats and `rootWinRatio` — ai/puct.js
// is the intended teacher.
//
// In-run training: each time a status line prints, --train-epochs
// cross-entropy passes over all recorded games (train-npats.js machinery)
// update the agent's live npats weights in-process — the next games are
// recorded by the improved teacher.  One status line = one generation (`gen`
// column counts them; `ce` is the pass's cross-entropy); the model is saved
// each round.  Dogfooding requires the agent module to export its npatsModel
// (ai/puct.js does); an agent without one is frozen by nature, so it always
// trains a fresh trainee model.
//
// --frozen-teacher decouples the loop for npatsModel agents too: a SEPARATE
// model (seeded from the agent's loaded weights) is trained while the agent's
// own play stays untouched — no dogfooding; the data distribution is the
// fixed teacher's.  The wts/ce/rms columns and the saved model all track the
// trainee, not the player.
//
// With --eval-file, each status line also runs an evalmovedetails pass over
// the whole file using the RAW policy argmax (the live weights being
// trained, no search on top) and reports the win-ratio-gap rms.  One
// extraction per position, so the pass is near-instant and every round
// measures the identical full position set — directly comparable across
// generations (and against ai/npat.js's rms on the same file).
//
// Status is printed at an exponentially increasing interval (× 1.5 each time).
// Runs indefinitely unless --limit is given (Ctrl-C to stop; only the
// in-progress game is lost).
//
// Usage:
//   node record-npats.js [--agent puct] [--size 13] [--playouts 4000]
//                         [--opening-playouts <n>] [--opening-phase 0.25]
//                         [--temp-moves 20] [--temp 1] [--epsilon 0.1]
//                         [--seed <n>] [--limit <games>] [--save <path.ndjson>]
//                         [--train-epochs 1] [--lr 0.05] [--target-temp 1]
//                         [--z-weight 0] [--model-save <path.js>]
//                         [--frozen-teacher] [--eval-file <md.ndjson>]

const fs   = require('fs');
const path = require('path');
const { performance } = require('perf_hooks');
const { Game2, PASS, coordStr } = require('./game2.js');
const { makeRng } = require('./xorshift.js');
const Util    = require('./util.js');
const NPats  = require('./npats-lib.js');
const Trainer = require('./train-npats.js');
const EvalMD  = require('./evalmovedetails.js');
const Ladders = require('./evalladders.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help', 'frozen-teacher']);

if (opts.help) {
  console.log('Usage: node record-npats.js [--agent <name>] [--size <n>] [--playouts <n>] ' +
              '[--opening-playouts <n>] [--opening-phase <f>] ' +
              '[--temp-moves <n>] [--temp <t>] [--epsilon <f>] [--seed <n>] [--limit <games>] [--save <path>] ' +
              '[--train-epochs <n>] [--lr <f>] [--target-temp <t>] ' +
              '[--model-save <path>] [--frozen-teacher] [--eval-file <md.ndjson>]');
  process.exit(0);
}

const agentName = opts.agent || 'puct';
const size      = parseInt(opts.size       || '13',   10);
const playouts  = parseInt(opts.playouts   || '4000', 10);

// Phase-varying teacher: when --opening-playouts is set, decisions made
// before the board reaches --opening-phase fullness (1 - opens/area) use
// that playout count instead of --playouts.  Cheap clean labels in the
// opening (where deep-search choices distill poorly), full search after.
const openingPlayouts = parseInt(opts['opening-playouts'] || '0', 10);
const openingPhase    = parseFloat(opts['opening-phase'] || '0.25');
const tempMoves = parseInt(opts['temp-moves'] || '20', 10);
const temp      = parseFloat(opts.temp     || '1');
const epsilon   = parseFloat(opts.epsilon  || '0.1');
// Cap the in-memory training set (the on-disk NDJSON always keeps every
// game).  0 = unbounded.  Indefinite streams need this — the dataset grows
// without limit otherwise and eventually OOMs the recorder.
const maxTrainGames = parseInt(opts['max-train-games'] || '0', 10);
const seed      = opts.seed !== undefined ? parseInt(opts.seed, 10) : undefined;
const gameLimit = opts.limit !== undefined ? parseInt(opts.limit, 10) : Infinity;
const SAVE_PATH = opts.save || `out/npats-${Math.random().toString(36).slice(2, 10)}.ndjson`;

const TRAIN_EPOCHS = parseInt(opts['train-epochs'] || '1', 10);
const LR           = parseFloat(opts.lr || '0.05');
const TARGET_T     = parseFloat(opts['target-temp'] || '1');
const Z_WEIGHT     = parseFloat(opts['z-weight'] || '0');
const EVAL_FILE    = opts['eval-file'] || null;

if (isNaN(playouts) || playouts < 1) { console.error('--playouts must be a positive integer'); process.exit(1); }

const agentMod = require(path.join(__dirname, 'ai', agentName + '.js'));
const agent = agentMod.getMove;
const FROZEN = !!opts['frozen-teacher'];

// Training is on by default.  Normally it updates the agent's live policy
// model (dogfooding); with --frozen-teacher a separate trainee model is used
// and the agent's play stays fixed.  An agent without a npatsModel export
// is frozen by nature, so it always gets a fresh trainee.
let trainModel;
if (FROZEN || !agentMod.npatsModel) {
  trainModel = NPats.freshModel();
  if (agentMod.npatsModel) {
    agentMod.npatsModel.weights.forEach((k, v) => trainModel.weights.set(k, v));
    trainModel.cfg = agentMod.npatsModel.cfg;
  }
} else {
  trainModel = agentMod.npatsModel;
}
const MODEL_PATH = opts['model-save'] || SAVE_PATH.replace(/\.ndjson$/, '') + '-model.js';
const trainer = Trainer.createTrainer(trainModel, { lr: LR, targetTemp: TARGET_T, zWeight: Z_WEIGHT });

// Optional per-round evalmovedetails pass using the RAW policy argmax (the
// live weights, no search) — near-instant, full pool every round.
const evalPool = EVAL_FILE ? EvalMD.loadPositions(EVAL_FILE) : null;
if (evalPool) console.log(`eval: ${evalPool.length} positions from ${EVAL_FILE} (raw npats argmax)`);
let lastRms = NaN;

// Raw-policy argmax agent over the live model (same selection as ai/npats.js).
const evalStateByN = new Map();
function npatsArgmax(game) {
  if (game.gameOver) return { move: PASS };
  let state = evalStateByN.get(game.N);
  if (!state) { state = NPats.createState(game.N); evalStateByN.set(game.N, state); }
  const n = NPats.computeProbs(game, state, undefined, trainModel, 0);
  if (n === 0) return { move: PASS };
  let best = 0;
  for (let i = 1; i < n; i++) if (state.probs[i] > state.probs[best]) best = i;
  return { move: state.moves[best] };
}

function evalRound() {
  if (!evalPool) return;
  lastRms = EvalMD.evalPositions(npatsArgmax, evalPool, 0).rmsErr;
}

// Standing per-checkpoint gate: the ladder test suite against the raw policy
// argmax (deterministic, so one trial per position; near-instant).
let lastLadders = NaN;
function ladderRound() {
  const { passed, total } = Ladders.evalLadders(npatsArgmax, { trials: 1 });
  lastLadders = passed / total;
}

// One rng stream drives the agent and the temperature sampling, so a given
// --seed reproduces the run exactly; unseeded runs are time-seeded.
const rng = makeRng(seed);

// Sample an index from children ∝ visits^(1/temp).
function sampleByVisits(children) {
  const invT = 1 / temp;
  let sum = 0;
  const w = children.map(c => { const x = Math.pow(c.visits, invT); sum += x; return x; });
  let r = rng.random() * sum;
  for (let i = 0; i < children.length; i++) {
    r -= w[i];
    if (r <= 0) return i;
  }
  return children.length - 1;
}

fs.writeFileSync(SAVE_PATH,
  `# record-npats.js  agent=${agentName}  size=${size}  playouts=${playouts}` +
  (openingPlayouts > 0 ? `  opening-playouts=${openingPlayouts}  opening-phase=${openingPhase}` : '') +
  `  temp-moves=${tempMoves}  temp=${temp}  epsilon=${epsilon}${seed !== undefined ? `  seed=${seed}` : ''}` +
  `  train-epochs=${TRAIN_EPOCHS}  lr=${LR}  target-temp=${TARGET_T}  z-weight=${Z_WEIGHT}` +
  (FROZEN ? '  frozen-teacher=1' : '') + '\n');

console.log(`agent=${agentName}  size=${size}  playouts=${playouts}` +
  (openingPlayouts > 0 ? ` (opening: ${openingPlayouts} below phase ${openingPhase})` : '') +
  `  temp-moves=${tempMoves}  temp=${temp}  lr=${LR}`);
console.log(`Out: ${SAVE_PATH}  model: ${MODEL_PATH}`);
console.log();
console.log([
  'games'  .padStart(5),
  'pos'    .padStart(6),
  'elapsed'.padStart(7),
  'tMv'    .padStart(5),
  'gen' .padStart(4),
  'wts' .padStart(5),
  'avgW'.padStart(5),
  'maxW'.padStart(5),
  'ce'  .padStart(5),
  'ladr'.padStart(5),
  ...(evalPool ? ['rms'.padStart(5)] : []),
].join('  '));

const startTime = performance.now();
let printPeriodMs = 1000;
let lastPrintTime = startTime;
let gamesDone = 0;
let posCount  = 0;
let moveMs    = 0;

// In-run training state.
const dataset = [];   // finished games in train-npats format
let gen = 0;
let lastCe = NaN;
const trainRng = makeRng(seed !== undefined ? seed + 1 : undefined);

// Train one generation on all games recorded so far and save the model.
// Updates the agent's live weights.  `ce` is the cross-entropy observed
// during the pass (each position measured against the pre-update weights).
function trainGeneration() {
  if (dataset.length === 0) return;
  let ce = 0, count = 0;
  for (let e = 0; e < TRAIN_EPOCHS; e++) {
    Util.shuffle(dataset, trainRng);
    for (const g of dataset) {
      const r = trainer.trainGame(g);
      ce += r.ce;
      count += r.count;
    }
  }
  gen++;
  lastCe = ce / count;
  NPats.saveModel(MODEL_PATH, trainModel.weights, trainModel.cfg,
    `record-npats.js  source=${SAVE_PATH}  gen=${gen}  games=${dataset.length}  lr=${LR}  target-temp=${TARGET_T}`);
}

function printStats() {
  const elapsedMs = performance.now() - startTime;
  console.log([
    Util.fmt4(gamesDone)          .padStart(5),
    Util.fmt4(posCount)           .padStart(6),
    Util.fmtMs(elapsedMs)         .padStart(7),
    Util.fmtMs(moveMs / posCount) .padStart(5),
    ...(() => {
      let sumAbs = 0, maxAbs = 0;
      trainModel.weights.forEach((k, v) => {
        const a = Math.abs(v);
        sumAbs += a;
        if (a > maxAbs) maxAbs = a;
      });
      const n = trainModel.weights.size;
      return [
        Util.fmt4(gen)                  .padStart(4),
        Util.fmt4(n)                    .padStart(5),
        Util.fmt4(n ? sumAbs / n : 0)   .padStart(5),
        Util.fmt4(maxAbs)               .padStart(5),
        Util.fmt4(lastCe)               .padStart(5),
        Util.fmtRatio4(lastLadders)     .padStart(5),
      ];
    })(),
    ...(evalPool ? [Util.fmtRatio4(lastRms).padStart(5)] : []),
  ].join('  '));
}

while (gamesDone < gameLimit) {
  const game     = new Game2(size);
  const maxMoves = size * size * 4;
  const tokens   = [];     // full game history, two-letter tokens
  const records  = [];     // { histLen, visits, q, mover } per decision
  let decision   = 0;
  let discarded  = false;

  while (!game.gameOver) {
    if (game.moveCount >= maxMoves) {
      console.error(`Game exceeded ${maxMoves} moves — discarded`);
      discarded = true;
      break;
    }

    const fullness = (size * size - game.emptyCount) / (size * size);
    const movePlayouts = (openingPlayouts > 0 && fullness < openingPhase) ? openingPlayouts : playouts;
    const t0 = performance.now();
    const r = agent(game, 0, { playoutLimit: movePlayouts, rng });
    moveMs += performance.now() - t0;

    if (r.rootWinRatio === undefined) { console.error('agent did not return rootWinRatio'); process.exit(1); }
    if (r.children && r.children.length && typeof r.children[0].move !== 'number') {
      console.error('agent children[].move is not a flat index'); process.exit(1);
    }

    // Sparse visit distribution: moves with at least one rounded visit.
    // Token form for the NDJSON record, flat form for in-run training.
    const visits = [];
    const visitsFlat = [];
    for (const c of r.children || []) {
      const v = Math.round(c.visits);
      if (v >= 1) {
        visits.push([coordStr(c.move, size), v]);
        visitsFlat.push([c.move, v]);
      }
    }

    if (visits.length > 0) {
      records.push({ histLen: tokens.length, visits, visitsFlat, q: r.rootWinRatio, mover: game.current });
      posCount++;
      decision++;
    }

    let move;
    if (epsilon > 0 && rng.random() < epsilon) {
      move = game.randomLegalMove(rng);
    } else if (temp > 0 && decision <= tempMoves && visits.length > 0) {
      move = r.children[sampleByVisits(r.children)].move;
    } else {
      move = r.move;
    }
    if (!game.play(move)) {
      console.error(`Illegal move from ${agentName}: ${move}`);
      process.exit(1);
    }
    tokens.push(coordStr(move, size));

    const now = performance.now();
    if (now - lastPrintTime >= printPeriodMs) {
      trainGeneration();
      evalRound();
      ladderRound();
      printStats();
      lastPrintTime = performance.now();   // exclude training/eval time from the interval
      printPeriodMs = Math.round(printPeriodMs * 1.5);
    }
  }

  if (discarded) { posCount -= records.length; continue; }
  gamesDone++;

  // Winner known — back-fill z and flush the game's records.
  const winner = game.calcWinner();
  const lines = records.map(rec => JSON.stringify({
    boardSize: size,
    history:   tokens.slice(0, rec.histLen),
    visits:    rec.visits,
    q:         rec.q,
    z:         rec.mover === winner ? 1 : 0,
  }) + '\n').join('');
  fs.appendFileSync(SAVE_PATH, lines);

  dataset.push({
    boardSize: size,
    tokens,
    recs: records.map(rec => ({
      histLen: rec.histLen,
      visits:  rec.visitsFlat,
      q:       rec.q,
      z:       rec.mover === winner ? 1 : 0,
    })),
  });
  // Sliding window: drop oldest games beyond the cap (disk file keeps all).
  if (maxTrainGames > 0 && dataset.length > maxTrainGames) {
    dataset.splice(0, dataset.length - maxTrainGames);
  }
}

trainGeneration();
evalRound();
ladderRound();
printStats();
console.log(`Reached --limit ${gameLimit} games — saved ${SAVE_PATH}  (model gen ${gen}: ${MODEL_PATH})`);
