'use strict';

// gen-deltas.js — generate point-predictor training data, recording BOTH a point target and a
// win-ratio target from the same full-random playouts (win = mover's final score > 0, free to
// record).  Output is NDJSON, one record per position, storing the raw per-candidate playout
// stats and the shared pass baseline so either target — or a blend — is computable at train time:
//
//   {"size":13,"pos":"<csv moves>","moves":"<csv candidate moves>","p":[..],"w":[..],"pp":-6.93,"pw":0.418}
//     size : board size
//     pos  : the position as a CSV of moves replayed exactly (preserves ko)
//     moves: candidate moves, CSV (index basis for p/w)
//     p[i] : candidate i's mean playout POINT score  (quantized, 2 decimals)
//     w[i] : candidate i's playout WIN ratio in [0,1] (quantized, 3 decimals)
//     pp   : PASS baseline mean point score (shared, 3N playouts)   (2 decimals)
//     pw   : PASS baseline win ratio (shared)                       (3 decimals)
//   → point-delta_i   = p[i] − pp ;   winratio-delta_i = w[i] − pw.
//
// Format is NDJSON for extensibility (add fields without breaking readers) and conciseness
// (short keys, quantized arrays).  Win ratios are plain decimals here; the ×1000 (kwr) encoding
// is only for the base64/binary data files, not text.
//
// Usage (NDJSON records go to STDOUT — redirect to a file; progress/settings go to stderr):
//   node gen-deltas.js [--agent prod] [--size 13] [--playouts 30]
//                      [--samples N] [--epsilon 0.1] [--seed N]  > out/deltas.ndjson   (--samples default: unlimited)

const { Game2, PASS, BLACK, coordStr } = require('./game2.js');
const { makeRng } = require('./xorshift.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'],
  ['agent', 'size', 'playouts', 'samples', 'epsilon', 'seed']);
if (opts.help) {
  console.log('Usage: node gen-deltas.js [--agent prod] [--size 13] [--playouts 30] [--samples N] [--epsilon 0.1] [--seed N] > FILE   (NDJSON to stdout)');
  process.exit(0);
}
const AGENT   = opts.agent || 'prod';
const SIZE    = parseInt(opts.size || '13', 10);
const N       = parseInt(opts.playouts || '30', 10);   // playouts per candidate (3N for pass)
const SAMPLES = opts.samples !== undefined ? parseInt(opts.samples, 10) : Infinity;
const EPSILON = opts.epsilon !== undefined ? parseFloat(opts.epsilon) : 0.1;
const SEED    = opts.seed !== undefined ? parseInt(opts.seed, 10)
                                        : ((Date.now() ^ (process.pid << 16) ^ (Math.random() * 0x7fffffff | 0)) >>> 0);

const agent = require('./ai/' + AGENT + '.js');
const rng   = makeRng(SEED);
const MAXMV = SIZE * SIZE * 4;
const q2    = x => Math.round(x * 100) / 100;          // points: 2 decimals
const q3    = x => Math.round(x * 1000) / 1000;        // win ratio: 3 decimals (decimal, not kwr)

// One full-random playout from `base` after `move`; score from `mover`'s perspective (incl. komi).
function playout(base, move, mover) {
  const g = base.clone();
  g.play(move);
  let steps = 0;
  while (!g.gameOver && steps < MAXMV) { g.play(g.randomLegalMove(rng)); steps++; }
  const s = g.estimateScore();
  return mover === BLACK ? s.black - s.white : s.white - s.black;
}

// Mean point score and win ratio over `count` playouts after `move`.
function stats(base, move, mover, count) {
  let sum = 0, wins = 0;
  for (let p = 0; p < count; p++) { const s = playout(base, move, mover); sum += s; if (s > 0) wins++; }
  return { pts: sum / count, win: wins / count };
}

// Agent epsilon-greedy self-play; returns the move sequence (flat indices / PASS).
function selfPlayGame() {
  const game = new Game2(SIZE), moves = [], options = { rng };
  while (!game.gameOver && moves.length < MAXMV) {
    const mv = rng.random() < EPSILON ? game.randomLegalMove(rng) : agent.getMove(game, 0, options).move;
    game.play(mv); moves.push(mv);
  }
  return moves;
}

// Evaluate the position reached by replaying `prefix`: pass baseline (3N) and each candidate (N).
// Returns { moves, p, w, pp, pw } or null if no candidates.
function evaluatePosition(prefix) {
  const base = new Game2(SIZE);
  for (const m of prefix) base.play(m);
  const mover = base.current;
  const cands = [];
  const ec = base.emptyCount, eCells = base._emptyCells;
  for (let i = 0; i < ec; i++) { const idx = eCells[i]; if (base.isLegal(idx) && !base.isTrueEye(idx)) cands.push(idx); }
  if (cands.length === 0) return null;
  const pass = stats(base, PASS, mover, 3 * N);
  const p = new Array(cands.length), w = new Array(cands.length);
  for (let c = 0; c < cands.length; c++) {
    const st = stats(base, cands[c], mover, N);
    p[c] = q2(st.pts); w[c] = q3(st.win);
  }
  return { moves: cands, p, w, pp: q2(pass.pts), pw: q3(pass.win) };
}

// ── Generate (append; one NDJSON record per position) ─────────────────────────────
process.stderr.write(`gen-deltas: agent: ${AGENT}  size: ${SIZE}  playouts: ${N}  pass: ${3 * N}  samples: ${SAMPLES === Infinity ? 'unlimited' : SAMPLES}  epsilon: ${EPSILON}  seed: ${SEED}\n`);

const COLS = ['elapsed', 'posns', 'cands', 'playout', 'p/s'];
const COLW = [7, 7, 8, 9, 6];
const printRow = c => process.stderr.write(c.map((v, i) => String(v).padStart(COLW[i])).join('  ') + '\n');
printRow(COLS);

const t0 = Date.now();
let written = 0, candTotal = 0, playoutTotal = 0, nextReport = 1;
function report() {
  const secs = (Date.now() - t0) / 1000;
  printRow([Util.fmtMs(Date.now() - t0), Util.fmt4i(written), Util.fmt4i(candTotal),
            Util.fmt4i(playoutTotal), Util.fmt4i(secs > 0 ? (playoutTotal / secs) | 0 : 0)]);
}

for (let s = 0; s < SAMPLES; s++) {
  const moves = selfPlayGame();
  if (moves.length < 2) { s--; continue; }
  let res = null, prefixLen = 0;
  for (let tries = 0; tries < 8 && !res; tries++) { prefixLen = (rng.random() * moves.length) | 0; res = evaluatePosition(moves.slice(0, prefixLen)); }
  if (!res) { s--; continue; }

  const rec = {
    size: SIZE,
    pos: moves.slice(0, prefixLen).map(m => coordStr(m, SIZE)).join(','),
    moves: res.moves.map(m => coordStr(m, SIZE)).join(','),
    p: res.p, w: res.w, pp: res.pp, pw: res.pw,
  };
  process.stdout.write(JSON.stringify(rec) + '\n');
  written++; candTotal += res.moves.length; playoutTotal += (res.moves.length + 3) * N;
  if (written >= nextReport) { report(); nextReport = Math.max(nextReport + 1, Math.ceil(nextReport * 1.5)); }
}
report();
process.stderr.write(`done: ${written} positions\n`);
