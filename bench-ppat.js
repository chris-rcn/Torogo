'use strict';

// bench-ppat.js — ppat playout cost on a fixed, reproducible workload.
//
// Start positions: every position with --min-phase <= phase <= --max-phase of
// --games full 13x13 games, each opened with 2 mirror pairs and played by
// featurepol (featurepol-cbk7wa32.js, softmax temperature 1 — the rfs
// policy) from a seeded rng, so the set depends only on --seed and not on the
// ppat code under test.  From each start position --playouts ppat playouts
// run to the end (or puct-trunc's move cap, 3·empty + 20), every move by the
// ppat policy (ppatMinPhase 0: no uniform early moves), from one seeded rng.
//
// Only the playouts are timed, after an untimed warm-up on separately seeded
// playouts.  The checksum (FNV-1a over every playout move) is identical for
// two implementations of the same policy given the same arguments — the
// exactness check for an optimisation.
//
//   node bench-ppat.js [--model ppat-data-1623136.js] [--games 20] [--playouts 4]
//                      [--min-phase 0.55] [--max-phase 1] [--seed 1] [--size 13]

const path = require('path');
const { performance } = require('perf_hooks');
const Util = require('./util.js');
const { Game2, PASS } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');
const { makeRng } = require('./xorshift.js');
const FeaturePol = require('./featurepol-lib.js');
const PPat = require('./ppat-lib.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['model', 'games', 'playouts', 'min-phase', 'max-phase', 'seed', 'size']);
if (opts.help) {
  console.log('Usage: node bench-ppat.js [--model ppat-data-1623136.js] [--games 20] [--playouts 4] [--min-phase 0.55] [--max-phase 1] [--seed 1] [--size 13]');
  process.exit(0);
}
const MODEL    = opts.model || path.join(__dirname, 'ppat-data-1623136.js');
const GAMES    = parseInt(opts.games || '20', 10);
const PLAYOUTS = parseInt(opts.playouts || '4', 10);
const MIN_PHASE = parseFloat(opts['min-phase'] ?? '0.55');
const MAX_PHASE = parseFloat(opts['max-phase'] ?? '1');
const SEED     = parseInt(opts.seed || '1', 10);
const SIZE     = parseInt(opts.size || '13', 10);
if (!(GAMES >= 1) || !(PLAYOUTS >= 1)) { console.error('--games and --playouts must be >= 1'); process.exit(1); }
if (!(MIN_PHASE >= 0 && MIN_PHASE <= MAX_PHASE && MAX_PHASE <= 1)) {
  console.error('--min-phase and --max-phase must satisfy 0 <= min <= max <= 1'); process.exit(1);
}

const model = PPat.loadWeights(MODEL);
if (!model) { console.error(`cannot load ppat weights from ${MODEL}`); process.exit(1); }
model.ppatMinPhase = 0;   // every playout move by the ppat policy

// ── Start positions ──────────────────────────────────────────────────────────
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
// A random move and its 180-degree rotation (selfplay.js's opening).
function playMirrorPair(game, rng) {
  for (let tries = 0; tries < 32; tries++) {
    const idx = game.randomLegalMove(rng);
    if (idx === PASS) return false;
    const m = mirror180(idx, game.N);
    if (m === idx) continue;
    const trial = game.clone();
    if (!trial.play(idx) || !trial.play(m)) continue;
    if (!isAntisymmetric(trial)) continue;
    game.play(idx);
    game.play(m);
    return true;
  }
  return false;
}

const genRng = makeRng(SEED);
const { weights: fpWeights } = FeaturePol.loadModel({ name: 'bench-ppat', path: path.join(__dirname, 'featurepol-cbk7wa32.js') });
const fpState = FeaturePol.createState(SIZE, fpWeights.spec);
const starts = [];
for (let gi = 0; gi < GAMES; gi++) {
  const g = new Game2(SIZE, false);
  for (let p = 0; p < 2; p++) playMirrorPair(g, genRng);
  const g3 = game3FromGame2(g);
  for (let ply = 0; !g.gameOver && ply < 4 * SIZE * SIZE; ply++) {
    const phase = g.phase();
    if (phase > MAX_PHASE) break;
    if (phase >= MIN_PHASE) starts.push(g.clone());
    const m = FeaturePol.policyMove(g, fpState, fpWeights, genRng, fpWeights.spec.needsLadder ? g3 : undefined, 1).move;
    g.play(m);
    g3.play(m);
  }
}
if (starts.length === 0) { console.error('no start positions in the phase window'); process.exit(1); }

// ── Playouts ─────────────────────────────────────────────────────────────────
const state = PPat.createState(SIZE);
// One playout from `start`; returns its move count, folding every move into h.
let h = 0x811c9dc5;
function playout(start, rng, hash) {
  const g = start.clone();
  const limit = 3 * g.emptyCount + 20;
  let moves = 0;
  while (!g.gameOver && moves < limit) {
    const m = PPat.ppatMove(g, state, model, rng);
    g.play(m);
    moves++;
    if (hash) h = Math.imul(h ^ (m + 1), 0x01000193) >>> 0;
  }
  return moves;
}

const warmRng = makeRng(SEED + 7919);
for (let i = 0; i < Math.min(starts.length, 50); i++) playout(starts[i], warmRng, false);

console.log(`model: ${path.basename(MODEL)}, games: ${GAMES}, start positions: ${starts.length} (phase ${MIN_PHASE}-${MAX_PHASE}), ` +
            `playouts per position: ${PLAYOUTS}, seed: ${SEED}`);
console.log(['pos'.padStart(5), 'plays'.padStart(6), 'moves'.padStart(6), 'tElp'.padStart(6), 'tMv'.padStart(6), 'checksum'].join('  '));
const rng = makeRng(SEED + 1);
let nMoves = 0, nPlayouts = 0, ms = 0, nextRow = 1;
function row(pos) {
  console.log([Util.fmt4i(pos).padStart(5), Util.fmt4i(nPlayouts).padStart(6), Util.fmt4i(nMoves).padStart(6),
               Util.fmtMs(ms).padStart(6), Util.fmtMs(ms / nMoves).padStart(6), h.toString(16).padStart(8, '0')].join('  '));
}
for (let i = 0; i < starts.length; i++) {
  const t = performance.now();
  for (let k = 0; k < PLAYOUTS; k++) { nMoves += playout(starts[i], rng, true); nPlayouts++; }
  ms += performance.now() - t;
  if (i + 1 >= nextRow) { row(i + 1); nextRow = Math.max(nextRow + 1, Math.ceil(nextRow * 1.4)); }
}
if (starts.length !== nextRow - 1) row(starts.length);
console.log(`us/move: ${(1000 * ms / nMoves).toFixed(3)}, moves/s: ${Math.round(nMoves / (ms / 1000))}, checksum: ${h.toString(16).padStart(8, '0')}`);
