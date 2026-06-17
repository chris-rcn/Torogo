#!/usr/bin/env node
'use strict';

// train-laddergate.js — ladder-gated TD(2) trainer.
//
// ref-npat-softmax plays self-play games (free initial stone).  The trainee
// vlibpat value observes the trajectory and, at each position, performs a
// two-step (same-player-to-move) logistic TD update — but each update is GATED
// by the ladder suite: snapshot the touched weights, apply the update, measure
// single-trial ladder performance (deterministic, no-dither full-width argmax),
// and KEEP the update only if the ladder pass count did not regress; otherwise
// revert.  This fights the dilution that normally erases ladder skill as
// positional features accumulate magnitude.
//
//   bootstrap:  V(s_{t-2}) ← V(s_t)         (gated)
//   terminal:   trailing two positions ← outcome z   (gated)
//
// Usage:
//   node train-laddergate.js [--size 11] [--lr 0.3] [--specs 2,3] [--ext ref-npat-softmax]
//        [--games N] [--load model.js] [--save out.js]

const path = require('path');
const { performance } = require('perf_hooks');
const { Game2, BLACK, PASS, parseBoard, parseMove } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');
const { extractFeatures, evaluateFeatures, prepareSpecs, loadWeights, saveWeights } = require('./vlibpat.js');
const { loadCases } = require('./evalladders2.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help', 'no-gate']);
if (opts.help) {
  console.log('Usage: node train-laddergate.js [--size 11] [--lr 0.3] [--specs 2,3] ' +
              '[--ext ref-npat-softmax] [--games N] [--load model.js] [--save out.js]');
  process.exit(0);
}

const SIZE      = parseInt(opts.size || '11', 10);
const LR        = parseFloat(opts.lr || '0.3');
const GAMES     = parseInt(opts.games || '0', 10);   // 0 = unlimited
const EXT_NAME  = opts.ext || 'ref-npat-softmax';
const GATE      = opts['no-gate'] === undefined;   // --no-gate → plain TD(2) (fast pre-train), no ladder gating
const EPSILON   = parseFloat(opts.epsilon !== undefined ? opts.epsilon : '0.1');   // exploration: random move fraction
const LOAD_PATH = opts.load || null;
const SAVE_PATH = opts.save || `out/laddergate-${Math.random().toString(36).slice(2, 10)}.js`;
const LADDER_FILE = opts['ladder-file'] || 'out/ladders.txt';   // gate suite (evalladders2 format)
const SUBSAMPLE   = parseInt(opts.subsample || '0', 10);        // random cases per type×size cell per gate check

let specs = opts.specs
  ? opts.specs.split(',').map(t => t === '1' ? { size: 1 } : t === '2' ? { size: 2 } : t === '3' ? { size: 3 }
      : t === '1p' ? { size: 1, ladder: false } : t === '3p' ? { size: 3, ladder: false }
      : (console.error(`bad spec ${t}`), process.exit(1)))
  : [{ size: 2 }, { size: 3 }];
const tacticsOpts = { tactics: opts.tactics || 'ladder2' };
let prepSpecs = prepareSpecs(specs, tacticsOpts);

let weights = new Map();
if (LOAD_PATH) {
  const m = loadWeights(LOAD_PATH);
  weights = m.weights;
  if (m.specs) { specs = m.specs; prepSpecs = m.preparedSpecs || prepareSpecs(specs, tacticsOpts); }
  console.log(`Loaded ${weights.size} weights from ${LOAD_PATH}`);
}

const extGetMove = require(path.join(__dirname, 'ai', EXT_NAME + '.js')).getMove;

// ── Value / TD ────────────────────────────────────────────────────────────────
function tdUpdate(features, target, lr) {
  const n = features.count;
  if (n === 0) return;
  const step = lr * (target - features.val) / n;
  const { keys, pols } = features;
  for (let i = 0; i < n; i++) { const k = keys[i]; weights.set(k, (weights.get(k) ?? 0) + pols[i] * step); }
}

// Full-width 1-ply argmax (deterministic; the ladder gate / play policy).
function search1ply(game, game3) {
  const area = game.N * game.N;
  const isBlack = game.current === BLACK;
  let bestMove = PASS, bestScore = isBlack ? -Infinity : Infinity;
  for (let coord = 0; coord < area; coord++) {
    if (!game.isLegal(coord) || game.isTrueEye(coord)) continue;
    const f = extractFeatures(game, prepSpecs, true, coord, game3);
    evaluateFeatures(f, weights);
    if (isBlack === (f.val > bestScore)) { bestScore = f.val; bestMove = coord; }
  }
  if (bestMove !== PASS && (game.consecutivePasses > 0 || game.emptyCount < area / 2)) {
    const pf = extractFeatures(game, prepSpecs, false, undefined, game3);
    evaluateFeatures(pf, weights);
    if (isBlack === (pf.val > bestScore)) bestMove = PASS;
  }
  return bestMove;
}
// Gate suite: pre-parse the evalladders2 cases once; the gate scores the
// trainee's own argmax (search1ply) against them (require → must play it,
// prohibit → must avoid all).
const ladderCases = loadCases(LADDER_FILE).map(c => {
  const game = parseBoard(c.board, c.toPlay);
  const N = game.N;
  return {
    game,
    req: c.require  ? c.require.map(s => parseMove(s, N))  : null,
    pro: c.prohibit ? c.prohibit.map(s => parseMove(s, N)) : null,
    cell: `${c.type}:${c.chainSize}`,
  };
});
// group by (type, chainSize) cell, for subsampled gating
const cellGroups = [...ladderCases.reduce((m, c) => (m.get(c.cell)?.push(c) || m.set(c.cell, [c]), m), new Map()).values()];

function scoreCase(c) {
  const mv = c.game.gameOver ? PASS : search1ply(c.game, game3FromGame2(c.game));
  if (c.req) return c.req.includes(mv) ? 1 : 0;
  if (c.pro) return c.pro.includes(mv) ? 0 : 1;
  return 1;
}
function scoreList(list) { let p = 0; for (const c of list) p += scoreCase(c); return p; }
function ladderScore() { return scoreList(ladderCases); }                 // full suite (status line)
// k distinct random cases from arr (all of them if k >= arr.length)
function pickK(arr, k) {
  if (k <= 0 || k >= arr.length) return arr;
  const idx = arr.map((_, i) => i);
  for (let i = 0; i < k; i++) { const j = i + ((Math.random() * (idx.length - i)) | 0); [idx[i], idx[j]] = [idx[j], idx[i]]; }
  return idx.slice(0, k).map(i => arr[i]);
}
function sampleCells() { return cellGroups.flatMap(g => pickK(g, SUBSAMPLE)); }   // SUBSAMPLE random cases / cell

// ── Gated update: snapshot touched keys, apply, test ladder, keep iff no regress
let baseLadder = 0;          // ladder pass count of the current (accepted) model
let nUpdates = 0, nAccepted = 0;
function gatedUpdate(features, target) {
  const { keys, count } = features;
  if (count === 0) return;
  if (!GATE) { tdUpdate(features, target, LR); nUpdates++; nAccepted++; return; }   // plain TD(2)
  // Gate on a fresh subsample (one random case per type×size cell).  The sample
  // rotates each check, so score it both before and after this update.
  const sample = sampleCells();
  const before = scoreList(sample);
  const snap = new Array(count);
  for (let i = 0; i < count; i++) snap[i] = weights.get(keys[i]);   // may be undefined
  tdUpdate(features, target, LR);
  nUpdates++;
  const after = scoreList(sample);
  if (after < before) {                        // regressed on the sample → revert
    for (let i = 0; i < count; i++) {
      if (snap[i] === undefined) weights.delete(keys[i]); else weights.set(keys[i], snap[i]);
    }
  } else { nAccepted++; }                       // keep
}

// ── Play + train ────────────────────────────────────────────────────────────
const LADDER_TOTAL = ladderCases.length;
baseLadder = ladderScore();
console.log(`size=${SIZE}  lr=${LR}  ext=${EXT_NAME}  eps=${EPSILON}  gate=${GATE ? 'on' : 'OFF'}  specs=${JSON.stringify(specs)}`);
console.log(`Out: ${SAVE_PATH}   gate suite: ${LADDER_FILE} (${LADDER_TOTAL} cases, subsample ${SUBSAMPLE}/cell)   initial ladder ${baseLadder}/${LADDER_TOTAL}`);
console.log();
console.log(['games'.padStart(5), 'pos'.padStart(8), 'elapsed'.padStart(7), 'acc%'.padStart(5),
             'ladr'.padStart(5), 'avgW'.padStart(6), 'maxW'.padStart(6)].join('  '));

const t0 = performance.now();
let gamesDone = 0, posCount = 0;
let printPeriodMs = 1000, nextPrintAt = t0 + printPeriodMs;   // exponential backoff (status line + save)

function printStats() {
  baseLadder = ladderScore();   // full-suite score for the status line (gate uses a subsample)
  let wAbs = 0, wMax = 0;
  for (const w of weights.values()) { const a = Math.abs(w); wAbs += a; if (a > wMax) wMax = a; }
  console.log([
    Util.fmt4(gamesDone).padStart(5),
    Util.fmt4(posCount).padStart(8),
    Util.fmtMs(performance.now() - t0).padStart(7),
    Util.fmtRatio4(nUpdates ? nAccepted / nUpdates : 0).padStart(5),
    Util.fmtRatio4(baseLadder / LADDER_TOTAL).padStart(5),
    (weights.size ? wAbs / weights.size : 0).toFixed(4).padStart(6),
    wMax.toFixed(3).padStart(6),
  ].join('  '));
  saveWeights(SAVE_PATH, { weights, specs, opts: tacticsOpts, preparedSpecs: prepSpecs });
}

function playGame() {
  const game = new Game2(SIZE);              // free initial stone
  const game3 = game3FromGame2(game);
  const maxMoves = SIZE * SIZE * 4;
  let prev2 = null, prev1 = null, moves = 0;
  while (!game.gameOver && moves < maxMoves) {
    const f = extractFeatures(game, prepSpecs, false, undefined, game3);
    evaluateFeatures(f, weights);
    if (prev2 !== null) gatedUpdate(prev2, f.val);   // TD(2) bootstrap, gated
    prev2 = prev1; prev1 = f;
    const move = (Math.random() < EPSILON) ? game.randomLegalMove() : extGetMove(game).move;
    game.play(move); game3.play(move);
    moves++; posCount++;
    if (performance.now() >= nextPrintAt) { printStats(); nextPrintAt = performance.now() + printPeriodMs; printPeriodMs = Math.round(printPeriodMs * 1.4); }
  }
  const outcome = game.calcWinner() === BLACK ? 1 : 0;   // absolute P(BLACK wins)
  for (const p of [prev2, prev1]) if (p) gatedUpdate(p, outcome);   // terminal anchor, gated
}

while (GAMES === 0 || gamesDone < GAMES) { playGame(); gamesDone++; }
printStats();
console.log(`Done ${gamesDone} games — saved ${SAVE_PATH}`);
