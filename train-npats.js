#!/usr/bin/env node
'use strict';

// train-npats.js — supervised policy training from record-npats.js data.
//
// Distills search root-visit distributions into the linear softmax policy
// (npats-lib) by cross-entropy SGD:
//   Δw[k] = lr · Σ_a (p_target(a) − π(a)) · count(k, f_a)
// where p_target is the record's normalised visit distribution (optionally
// temperature-shaped) over the policy's candidates.  PASS visits are dropped
// (the policy has no PASS action) and the rest renormalised; records whose
// visits were all on PASS are skipped.  The objective is convex, so plain
// SGD with a modest learning rate converges.
//
// Data is streamed game by game (records of one game are adjacent and share
// history prefixes), each game replayed forward once with a lockstep Game3
// mirror.  A deterministic slice of games is held out for validation; status
// rows report train and holdout cross-entropy plus holdout top-1 agreement
// with the teacher's argmax.
//
// Status is printed at an exponentially increasing interval (× 1.5 each
// time); the model is saved at every print.  Runs until --epochs (or
// indefinitely when 0; Ctrl-C to stop).
//
// Also used as a library (record-npats.js trains in-process between games):
//   loadGames(filePath)            → games array
//   createTrainer(model, opts)     → { trainGame, evalGame, missedVisits }
//
// Usage:
//   node train-npats.js --file <data.ndjson> [--save <model.js>]
//                        [--load <model.js>] [--lr 0.05] [--epochs 0]
//                        [--holdout 0.2] [--target-temp 1] [--seed 1]
//
//   --file         recorder output                      (required)
//   --save         output model path  (default: out/npats-<random>.js)
//   --load         resume from an existing policy model (default: fresh)
//   --lr           SGD learning rate                    (default: 0.05)
//   --epochs       passes over the data, 0 = unlimited  (default: 0)
//   --holdout      fraction of games held out           (default: 0.2)
//   --target-temp  visit-count temperature (v^(1/t))    (default: 1)
//   --z-weight     outcome-channel strength (see createTrainer)  (default: 0)
//   --rank-template <path>  reshape targets to this mass-by-rank curve while
//                  keeping the search's move ordering (see createTrainer);
//                  overrides --target-temp.  Emit a curve with
//                  compare-dists.js --emit-template.
//   --merge-z      statistical tie width for rank-template blocks: ranks
//                  within z·√head visits of the block head share mass
//                  (see createTrainer)                  (default: 0)
//   --seed         shuffle rng seed                     (default: 1)

const fs = require('fs');
const { performance } = require('perf_hooks');
const { Game2, PASS, parseMove } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');
const NPat   = require('./npat-lib.js');
const NPats = require('./npats-lib.js');
const Util   = require('./util.js');
const Ladders = require('./evalladders.js');
const { makeRng } = require('./xorshift.js');

// ── Data ──────────────────────────────────────────────────────────────────────

// Parse recorder NDJSON into games: { boardSize, tokens, recs }, where tokens
// is the game's longest recorded history and each rec is
// { histLen, visits: [[flatIdx, count], ...], q, z }.  Game boundaries are
// detected by non-increasing history length (within a game it strictly grows).
// Reads in chunks with byte-level newline splitting, so files beyond Node's
// maximum string length (~512MB — concatenated recordings get there) load
// fine.
function loadGames(filePath) {
  const games = [];
  let cur = null;
  let lastHistLen = Infinity;
  // A parse failure is only acceptable on the file's final line (torn write
  // by a still-running recorder); it becomes fatal if any line follows it.
  let tornError = null;

  function processLine(line) {
    line = line.trim();
    if (!line || line.startsWith('#')) return;
    if (tornError) throw tornError;
    let rec;
    try {
      rec = JSON.parse(line);
    } catch (e) {
      tornError = e;
      return;
    }
    if (cur === null || rec.history.length <= lastHistLen) {
      cur = { boardSize: rec.boardSize, tokens: [], recs: [] };
      games.push(cur);
    }
    lastHistLen = rec.history.length;
    if (rec.history.length >= cur.tokens.length) cur.tokens = rec.history;
    cur.recs.push({
      histLen: rec.history.length,
      visits:  rec.visits.map(([m, v]) => [parseMove(m, rec.boardSize), v]),
      q: rec.q,
      z: rec.z,
    });
  }

  const fd = fs.openSync(filePath, 'r');
  try {
    const chunk = Buffer.alloc(1 << 22);
    let leftover = Buffer.alloc(0);
    let bytes;
    while ((bytes = fs.readSync(fd, chunk, 0, chunk.length, null)) > 0) {
      let buf = Buffer.concat([leftover, chunk.subarray(0, bytes)]);
      let start = 0, nl;
      while ((nl = buf.indexOf(10, start)) !== -1) {
        processLine(buf.toString('utf8', start, nl));
        start = nl + 1;
      }
      leftover = Buffer.from(buf.subarray(start));
    }
    if (leftover.length) processLine(leftover.toString('utf8'));
  } finally {
    fs.closeSync(fd);
  }
  if (tornError) console.warn('train-npats: dropped torn final line');
  return games;
}

// Deterministic train/holdout split: the LAST ceil(holdoutFrac · N) games are
// held out (capped so at least one game trains).  On a growing dataset this
// stays contamination-free: a game is only ever trained on after it leaves
// the holdout window, never before.  Training requires at least 2 games
// (1 train + 1 holdout); callers should check first.
function splitGames(games, holdoutFrac) {
  if (!(holdoutFrac > 0)) return { trainGames: games.slice(), holdoutGames: [] };
  const hc = Math.min(games.length - 1, Math.ceil(holdoutFrac * games.length));
  return {
    trainGames:   games.slice(0, games.length - hc),
    holdoutGames: games.slice(games.length - hc),
  };
}

// ── Training core ─────────────────────────────────────────────────────────────

// Returns a trainer bound to `model` ({ weights, cfg }):
//   trainGame(game) — replay + cross-entropy updates; returns { ce, top1, count }
//   evalGame(game)  — replay only (no updates);        returns { ce, top1, count }
//   missedVisits    — recorded visits with no matching npats candidate so far
//
// zWeight > 0 adds an outcome channel (the z-blend): alongside the CE step
// toward the visit targets, each position gets a REINFORCE step on the move
// actually played, with advantage A = z − q — the game outcome's surprise
// relative to the teacher's own value estimate.  Where the teacher's
// distribution was right, A ≈ 0 and the term vanishes; where the game
// contradicted the teacher (misread ladders, ε-blunders), the played move's
// features are credited/blamed directly, REINFORCE-style — signal the visit
// targets cannot carry.  Implemented as a second crossEntropyUpdate with a
// one-hot target and signed rate lr·zWeight·A (identical gradient form).
//
// rankTemplate (array of mean probability mass by rank, e.g. npat's measured
// curve from compare-dists --emit-template) replaces the temperature shaping
// entirely: the target keeps the search's move ORDERING (by visit count) but
// reassigns the MASSES to the template's rank profile, so the trained policy
// inherits the template's calibration with the search's choices.  Blocks of
// equal visit counts (ties — including the zero-visit tail) share the mean
// template mass over their rank span, so arbitrary tie order fabricates no
// distinctions.  The result is renormalised over the position's candidates.
function createTrainer(model, { lr = 0.05, targetTemp = 1, zWeight = 0, rankTemplate = null, mergeZ = 0 } = {}) {
  // Rewrite target[0..n) in place: search ordering, template masses.  Ranks
  // whose visit counts are statistically indistinguishable share the mean
  // template mass over their span — a block extends while the gap to its
  // head count is within mergeZ·√head (mergeZ 0 = exact ties only).  Low-
  // evidence distinctions (3 vs 2 visits) thus collapse into shared-mass
  // blocks instead of fabricating template-sized differences.  A block never
  // crosses the visited/unvisited boundary: visited-over-unvisited is an
  // ordering the search asserted deterministically (selection chose those
  // moves), not a noisy sample — at playouts=1 the binomial model would
  // otherwise merge the teacher's choice into the tail and yield uniform
  // targets.  Returns the new sum (caller normalises).
  const rankScratch = new Int32Array(512);
  function applyRankTemplate(target, n, template, mergeZ) {
    const tLast = template.length - 1;
    const tAt = r => template[Math.min(r, tLast)];
    for (let i = 0; i < n; i++) rankScratch[i] = i;
    const order = rankScratch.subarray(0, n);
    order.sort((a, b) => target[b] - target[a]);
    let sum = 0;
    for (let a = 0; a < n; ) {
      const head = target[order[a]];
      const slack = mergeZ * Math.sqrt(head);
      let b = a + 1;
      // head > 0: merge statistically-tied visited ranks (never across the
      // visited/unvisited boundary).  head === 0: everything left is the
      // unvisited tail — one shared block.
      while (b < n && (head > 0
        ? (target[order[b]] > 0 && head - target[order[b]] <= slack)
        : true)) b++;
      let mass = 0;
      for (let r = a; r < b; r++) mass += tAt(r);
      mass /= (b - a);
      for (let r = a; r < b; r++) { target[order[r]] = mass; sum += mass; }
      a = b;
    }
    return sum;
  }

  const stateByN   = new Map();
  const scratchIdx = new Map();   // boardSize → Int32Array(area) move → candidate index
  const target     = new Float64Array(512);
  const zTarget    = new Float64Array(512);
  let missedVisits = 0;

  function processGame(game, train) {
    const N = game.boardSize;
    let state = stateByN.get(N);
    if (!state) { state = NPat.createState(N); stateByN.set(N, state); }
    let idxOf = scratchIdx.get(N);
    if (!idxOf) { idxOf = new Int32Array(N * N); scratchIdx.set(N, idxOf); }

    const g2 = new Game2(N);
    const g3 = game3FromGame2(g2);
    let played = 0;
    let ce = 0, top1 = 0, count = 0;

    for (const rec of game.recs) {
      while (played < rec.histLen) {
        const idx = parseMove(game.tokens[played], N);
        g2.play(idx);
        g3.play(idx);
        played++;
      }

      const n = NPats.computeProbs(g2, state, g3, model);
      if (n === 0) continue;

      idxOf.fill(-1);
      for (let i = 0; i < n; i++) idxOf[state.moves[i]] = i;

      target.fill(0, 0, n);
      const invT = 1 / targetTemp;
      let sum = 0;
      for (const [mIdx, v] of rec.visits) {
        if (mIdx === PASS) continue;
        const i = idxOf[mIdx];
        if (i < 0) { missedVisits++; continue; }
        const w = (targetTemp === 1 || rankTemplate) ? v : Math.pow(v, invT);
        target[i] += w;
        sum += w;
      }
      if (sum === 0) continue;   // all visit mass was on PASS
      if (rankTemplate) sum = applyRankTemplate(target, n, rankTemplate, mergeZ);
      const inv = 1 / sum;
      let tBest = 0;
      for (let i = 0; i < n; i++) {
        target[i] *= inv;
        if (target[i] > target[tBest]) tBest = i;
      }

      let pBest = 0;
      for (let i = 0; i < n; i++) {
        if (target[i] > 0) ce -= target[i] * Math.log(Math.max(state.probs[i], 1e-12));
        if (state.probs[i] > state.probs[pBest]) pBest = i;
      }
      if (pBest === tBest) top1++;
      count++;

      if (train) {
        NPats.crossEntropyUpdate(state, model, target, lr);

        // z-blend: REINFORCE step on the played move (the game's next token;
        // absent only for a game's final record when loaded from file).
        if (zWeight > 0) {
          const playedTok = game.tokens[rec.histLen];
          if (playedTok !== undefined) {
            const played = parseMove(playedTok, N);
            const pi = played === PASS ? -1 : idxOf[played];
            const A = rec.z - rec.q;
            if (pi >= 0 && A !== 0) {
              zTarget.fill(0, 0, n);
              zTarget[pi] = 1;
              NPats.crossEntropyUpdate(state, model, zTarget, lr * zWeight * A);
            }
          }
        }
      }
    }
    return { ce, top1, count };
  }

  return {
    trainGame: game => processGame(game, true),
    evalGame:  game => processGame(game, false),
    get missedVisits() { return missedVisits; },
  };
}

module.exports = { loadGames, splitGames, createTrainer };

// ── CLI ───────────────────────────────────────────────────────────────────────

if (require.main === module) {

const opts = Util.parseArgs(process.argv.slice(2), ['help']);

if (opts.help || !opts.file) {
  console.log('Usage: node train-npats.js --file <data.ndjson> [--save <model.js>] [--load <model.js>] ' +
              '[--lr 0.05] [--epochs 0] [--holdout 0.2] [--target-temp 1] [--seed 1]');
  process.exit(opts.help ? 0 : 1);
}

const FILE      = opts.file;
const SAVE_PATH = opts.save || `out/npats-${Math.random().toString(36).slice(2, 10)}.js`;
const LOAD_PATH = opts.load || null;
const LR        = parseFloat(opts.lr || '0.05');
const EPOCHS    = parseInt(opts.epochs || '0', 10);
const HOLDOUT   = parseFloat(opts.holdout || '0.2');
const TARGET_T  = parseFloat(opts['target-temp'] || '1');
const Z_WEIGHT  = parseFloat(opts['z-weight'] || '0');
const TEMPLATE  = opts['rank-template'] ? require(require('path').resolve(opts['rank-template'])) : null;
const MERGE_Z   = parseFloat(opts['merge-z'] || '0');
const SEED      = parseInt(opts.seed || '1', 10);

let model;
if (LOAD_PATH) {
  const { weights, cfg, modelName } = NPats.loadModel({ name: 'train-npats', path: LOAD_PATH });
  model = { weights, cfg };
  console.log(`Loaded ${weights.size} weights from ${modelName}`);
} else {
  model = NPats.freshModel();
}

const trainer = createTrainer(model, { lr: LR, targetTemp: TARGET_T, zWeight: Z_WEIGHT, rankTemplate: TEMPLATE, mergeZ: MERGE_Z });

const games = loadGames(FILE);
if (games.length < 2) {
  console.error(`train-npats: need at least 2 games (got ${games.length})`);
  process.exit(1);
}
const { trainGames, holdoutGames } = splitGames(games, HOLDOUT);
const trainPos = trainGames.reduce((s, g) => s + g.recs.length, 0);

console.log(`file=${FILE}  games=${games.length} (train ${trainGames.length} / holdout ${holdoutGames.length})  ` +
            `positions=${trainPos}  lr=${LR}  ` +
            (TEMPLATE ? `rank-template=${opts['rank-template']} (${TEMPLATE.length} ranks)  merge-z=${MERGE_Z}` : `target-temp=${TARGET_T}`) +
            (LOAD_PATH ? `  (resumed from ${LOAD_PATH})` : ''));
console.log(`Out: ${SAVE_PATH}`);
console.log();
console.log([
  'epoch'  .padStart(5),
  'pos'    .padStart(6),
  'elapsed'.padStart(7),
  'tPos'   .padStart(5),
  'ce'     .padStart(5),
  'hCE'    .padStart(5),
  'hTop1'  .padStart(5),
  'ladr'   .padStart(5),
].join('  '));

// Standing per-print gate: the ladder suite against the raw argmax of the
// model being trained (deterministic, so one trial per position).
const ladderStateByN = new Map();
function ladderArgmax(game) {
  if (game.gameOver) return { move: PASS };
  let state = ladderStateByN.get(game.N);
  if (!state) { state = NPats.createState(game.N); ladderStateByN.set(game.N, state); }
  const n = NPats.computeProbs(game, state, undefined, model, 0);
  if (n === 0) return { move: PASS };
  let best = 0;
  for (let i = 1; i < n; i++) if (state.probs[i] > state.probs[best]) best = i;
  return { move: state.moves[best] };
}

const rng = makeRng(SEED);
const startTime = performance.now();
let printPeriodMs = 1000;
let lastPrintTime = startTime;
let totalPos = 0;
let intervalCe = 0, intervalPos = 0;

function maybePrint(epoch, force) {
  const now = performance.now();
  if (!force && now - lastPrintTime < printPeriodMs) return;
  lastPrintTime = now;
  printPeriodMs = Math.round(printPeriodMs * 1.5);

  let hCe = NaN, hTop1 = NaN;
  if (holdoutGames.length > 0) {
    let ce = 0, top1 = 0, count = 0;
    for (const g of holdoutGames) {
      const r = trainer.evalGame(g);
      ce += r.ce; top1 += r.top1; count += r.count;
    }
    hCe = ce / count;
    hTop1 = top1 / count;
  }

  const { passed, total } = Ladders.evalLadders(ladderArgmax, { oversample: 1 });

  const elapsedMs = performance.now() - startTime;
  console.log([
    Util.fmt4(epoch)                  .padStart(5),
    Util.fmt4(totalPos)               .padStart(6),
    Util.fmtMs(elapsedMs)             .padStart(7),
    Util.fmtMs(elapsedMs / totalPos)  .padStart(5),
    Util.fmt4(intervalCe / intervalPos).padStart(5),
    Util.fmt4(hCe)                    .padStart(5),
    Util.fmtRatio4(hTop1)             .padStart(5),
    Util.fmtRatio4(passed / total)    .padStart(5),
  ].join('  '));
  intervalCe = 0;
  intervalPos = 0;

  NPats.saveModel(SAVE_PATH, model.weights, model.cfg,
    `train-npats.js  source=${FILE}  lr=${LR}  target-temp=${TARGET_T}  epoch=${epoch}  positions=${totalPos}`);
}

let epoch = 0;
while (EPOCHS === 0 || epoch < EPOCHS) {
  epoch++;
  Util.shuffle(trainGames, rng);
  for (const game of trainGames) {
    const r = trainer.trainGame(game);
    totalPos    += r.count;
    intervalCe  += r.ce;
    intervalPos += r.count;
    maybePrint(epoch, false);
  }
}
maybePrint(epoch, true);
if (trainer.missedVisits > 0) console.warn(`train-npats: ${trainer.missedVisits} recorded visits had no matching npats candidate`);
console.log(`Reached --epochs ${EPOCHS} — saved ${SAVE_PATH}`);

}
