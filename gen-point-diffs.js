'use strict';

// gen-point-diffs.js — generate offline training samples for the featurepol POINT
// predictor (train-featurepol-points.js).  A sample's label for a move is the
// expected TERRITORY gained by playing it over passing:
//
//     diff(move) = E[score | play move] − E[score | pass]
//
// both scores from the MOVER's perspective at the chosen position, estimated by
// full-random Monte-Carlo playouts.  Subtracting the pass baseline strips the
// position's absolute "who's winning" offset, leaving only the move's marginal effect.
//
// Per sample:
//   1. The supplied agent self-plays a full game with epsilon-greedy exploration
//      (epsilon fraction of moves are uniform-random non-eye moves) for diversity.
//   2. Pick one random position from that game.
//   3. Run N full-random playouts after each legal non-eye stone placement, and 3N
//      after PASS (the shared baseline, so estimated more precisely).
//   4. Emit a line: the position as a move CSV (replayed exactly → preserves ko),
//      then each candidate's point differential.
//
// N is a COST knob, not a noise knob: label noise averages out across the many
// positions that share a feature, so total playouts (N × positions) sets accuracy.
// Tune N up until the playouts dominate per-sample overhead (game gen + I/O).
//
// Usage:
//   node gen-point-diffs.js --out FILE [--agent prod] [--size 9] [--playouts 30]
//                           [--samples N] [--epsilon 0.1] [--seed 1]
// (--samples defaults to unlimited — runs until killed, appending as it goes.)
//
// Output line:  <coord,coord,...> ; <coord>=<diff> <coord>=<diff> ...
// (lines starting with '#' are header/metadata comments.)

const fs   = require('fs');
const { Game2, PASS, BLACK, coordStr } = require('./game2.js');
const { makeRng } = require('./xorshift.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'],
  ['out', 'agent', 'size', 'playouts', 'samples', 'epsilon', 'seed']);
if (opts.help || !opts.out) {
  console.log('Usage: node gen-point-diffs.js --out FILE [--agent prod] [--size 9] [--playouts 30] [--samples N] [--epsilon 0.1] [--seed N]   (--samples default: unlimited; --seed default: random)');
  process.exit(opts.help ? 0 : 1);
}
const AGENT    = opts.agent || 'prod';
const SIZE     = parseInt(opts.size || '9', 10);
const N        = parseInt(opts.playouts || '30', 10);   // playouts per move (3N for pass)
const SAMPLES  = opts.samples !== undefined ? parseInt(opts.samples, 10) : Infinity;   // default: run until killed
const EPSILON  = opts.epsilon !== undefined ? parseFloat(opts.epsilon) : 0.1;
// Default seed is non-deterministic (time ^ pid ^ random), so parallel/repeat runs
// appending to the same file produce DIFFERENT data.  The chosen seed is printed and
// written to the file header, so any run stays reproducible after the fact via --seed.
const SEED     = opts.seed !== undefined ? parseInt(opts.seed, 10)
                                         : ((Date.now() ^ (process.pid << 16) ^ (Math.random() * 0x7fffffff | 0)) >>> 0);
const OUT      = opts.out;

const agent = require('./ai/' + AGENT + '.js');
const rng   = makeRng(SEED);
const MAXMV = SIZE * SIZE * 4;   // matches Game2's own game-over cap

// One full-random playout from `base` after playing `move`; returns the final score
// from `mover`'s perspective (own area − opponent area, komi included).
function playout(base, move, mover) {
  const g = base.clone();
  g.play(move);
  let steps = 0;
  while (!g.gameOver && steps < MAXMV) { g.play(g.randomLegalMove(rng)); steps++; }
  const s = g.estimateScore();
  return mover === BLACK ? s.black - s.white : s.white - s.black;
}

// Mean playout score after `move`, over `count` playouts.
function meanPlayout(base, move, mover, count) {
  let sum = 0;
  for (let p = 0; p < count; p++) sum += playout(base, move, mover);
  return sum / count;
}

// Agent epsilon-greedy self-play; returns the move sequence (flat indices / PASS).
function selfPlayGame() {
  const game = new Game2(SIZE);
  const moves = [];
  const options = { rng };
  while (!game.gameOver && moves.length < MAXMV) {
    const mv = rng.random() < EPSILON ? game.randomLegalMove(rng)
                                      : agent.getMove(game, 0, options).move;
    game.play(mv);
    moves.push(mv);
  }
  return moves;
}

// Evaluate the position reached by replaying `prefix`: pass baseline (3N) and each
// candidate's diff (N).  Returns { diffs: [[move, diff], ...] } or null if no candidates.
function evaluatePosition(prefix) {
  const base = new Game2(SIZE);
  for (const m of prefix) base.play(m);
  const mover = base.current;
  const cands = [];
  const ec = base.emptyCount, eCells = base._emptyCells;
  for (let i = 0; i < ec; i++) {
    const idx = eCells[i];
    if (base.isLegal(idx) && !base.isTrueEye(idx)) cands.push(idx);
  }
  if (cands.length === 0) return null;
  const passScore = meanPlayout(base, PASS, mover, 3 * N);
  const diffs = cands.map(m => [m, meanPlayout(base, m, mover, N) - passScore]);
  return { diffs };
}

// ── Generate ──────────────────────────────────────────────────────────────────

// Append (never truncate): runs accumulate into the same file.  Each run writes its own
// metadata header comment (lines starting with '#' are ignored by the trainer).
fs.appendFileSync(OUT, `# point-diffs size=${SIZE} agent=${AGENT} playouts=${N} pass=${3 * N} epsilon=${EPSILON} seed=${SEED}\n`);

process.stderr.write(`gen-point-diffs: agent: ${AGENT}  size: ${SIZE}  playouts: ${N}  pass: ${3 * N}  samples: ${SAMPLES === Infinity ? 'unlimited' : SAMPLES}  epsilon: ${EPSILON}  seed: ${SEED}  out: ${OUT}\n`);

// Progress table (stderr).  Columns: wall time, positions & candidate samples written,
// cumulative playouts and throughput, and running differential stats (mean, mean-abs, std) —
// the last three are what reveal how the move-value scale shifts with board size.
const COLS = ['elapsed', 'posns', 'cands', 'playout', 'p/s', 'meanDiff', 'absDiff', 'sdDiff'];
const COLW = [7, 6, 7, 8, 6, 8, 8, 7];
const printRow = cells => process.stderr.write(cells.map((c, i) => String(c).padStart(COLW[i])).join('  ') + '\n');
printRow(COLS);

const t0 = Date.now();
let written = 0, playoutTotal = 0, nextReport = 1;       // progress on an exponential cadence
let diffCount = 0, diffSum = 0, diffAbsSum = 0, diffSqSum = 0;   // running differential stats

function report() {
  const secs = (Date.now() - t0) / 1000;
  const pps  = secs > 0 ? (playoutTotal / secs) | 0 : 0;
  const mean = diffCount ? diffSum / diffCount : 0;
  const mAbs = diffCount ? diffAbsSum / diffCount : 0;
  const std  = diffCount ? Math.sqrt(Math.max(0, diffSqSum / diffCount - mean * mean)) : 0;
  printRow([Util.fmtMs(Date.now() - t0), Util.fmt4i(written), Util.fmt4i(diffCount),
            Util.fmt4i(playoutTotal), Util.fmt4i(pps), mean.toFixed(2), mAbs.toFixed(2), std.toFixed(2)]);
}

for (let s = 0; s < SAMPLES; s++) {
  const moves = selfPlayGame();
  if (moves.length < 2) { s--; continue; }                 // degenerate game, retry

  // pick a random position (before some move) that has candidates; a few tries
  let res = null, prefixLen = 0;
  for (let tries = 0; tries < 8 && !res; tries++) {
    prefixLen = (rng.random() * moves.length) | 0;
    res = evaluatePosition(moves.slice(0, prefixLen));
  }
  if (!res) { s--; continue; }                             // no candidate position found

  const prefixCsv = moves.slice(0, prefixLen).map(m => coordStr(m, SIZE)).join(',');
  const diffCsv   = res.diffs.map(([m, d]) => `${coordStr(m, SIZE)}=${d.toFixed(2)}`).join(' ');
  fs.appendFileSync(OUT, `${prefixCsv};${diffCsv}\n`);
  written++;
  playoutTotal += (res.diffs.length + 3) * N;              // ~ candidates*N + pass(3N)
  for (const [, d] of res.diffs) { diffCount++; diffSum += d; diffAbsSum += Math.abs(d); diffSqSum += d * d; }

  if (written >= nextReport) {
    report();
    nextReport = Math.max(nextReport + 1, Math.ceil(nextReport * 1.5));
  }
}
report();
process.stderr.write(`done: ${written} positions -> ${OUT}\n`);
