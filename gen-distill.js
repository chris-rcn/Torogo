'use strict';

// gen-distill.js — generate DISTILLATION targets by running a search teacher (puct-distill) at each
// sampled position and recording every root child's Q (win rate) and N (visit count).  NDJSON to
// stdout, one record per position:
//
//   {"size":13,"pos":"..","moves":"..","q":[..],"n":[..],"qp":0.41,"np":12,"p":[..],"pp":-1.5,"pn":[..],"ppn":12}
//     q[i] : root child i's win rate = wins/visits (mover's perspective)  (3 decimals)
//     n[i] : root child i's visit count                                   (integer)
//     qp/np: the PASS child's win rate / visit count
//     p[i] : root child i's mean point margin (mover's perspective)       (2 decimals)
//     pn[i]: # scored playouts feeding p[i]                               (integer; = n[i], all playouts are full)
//     pp/ppn: the PASS child's mean point margin / scored-playout count
//   → train-time: win target = q[i] − qp (weight f(n[i])); point target = p[i] − pp (weight f(pn[i])).
//
// Positions come from --agent epsilon-greedy self-play (fast, for diversity); the per-move targets
// come from --teacher run once at --per-cand playouts per root candidate (total budget scales with the
// root width, so every candidate gets ~equal visits under uniform root visiting).  Distillation ceiling = the teacher's strength,
// further capped by the model's capacity, so use a strong teacher and a rich spec.
//
// Usage (NDJSON to stdout; progress to stderr):
//   node gen-distill.js [--agent ref-npat-softmax] [--teacher puct-distill] [--per-cand 32]
//                       [--size 13] [--samples N] [--epsilon 0.1] [--seed N]  > out/distill.ndjson

const { Game2, coordStr } = require('./game2.js');
const { makeRng } = require('./xorshift.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'],
  ['agent', 'teacher', 'per-cand', 'size', 'samples', 'epsilon', 'seed']);
if (opts.help) {
  console.log('Usage: node gen-distill.js [--agent ref-npat-softmax] [--teacher puct-distill] [--per-cand 32] [--size 13] [--samples N] [--epsilon 0.1] [--seed N] > FILE');
  process.exit(0);
}
const AGENT    = opts.agent || 'ref-npat-softmax';
const TEACHER  = opts.teacher || 'puct-distill';
const PER_CAND = parseInt(opts['per-cand'] || '32', 10);   // playouts per root candidate; total = PER_CAND × root width
const SIZE     = parseInt(opts.size || '13', 10);
const SAMPLES  = opts.samples !== undefined ? parseInt(opts.samples, 10) : Infinity;
const EPSILON  = opts.epsilon !== undefined ? parseFloat(opts.epsilon) : 0.1;
const SEED     = opts.seed !== undefined ? parseInt(opts.seed, 10)
                                         : ((Date.now() ^ (process.pid << 16) ^ (Math.random() * 0x7fffffff | 0)) >>> 0);

console.log = (...a) => process.stderr.write(a.join(' ') + '\n');   // some agents log load info via console.log; keep stdout pure NDJSON
const agent   = require('./ai/' + AGENT + '.js');
const teacher = require('./ai/' + TEACHER + '.js');
const rng     = makeRng(SEED);
const MAXMV   = SIZE * SIZE * 4;
const q3 = x => Math.round(x * 1000) / 1000;
const p2 = x => Math.round(x * 100) / 100;

function selfPlayGame() {
  const game = new Game2(SIZE), moves = [], options = { rng };
  while (!game.gameOver && moves.length < MAXMV) {
    const mv = rng.random() < EPSILON ? game.randomLegalMove(rng) : agent.getMove(game, 0, options).move;
    game.play(mv); moves.push(mv);
  }
  return moves;
}

// Run the teacher search at the replayed position; return { moves, q, n, qp, np } or null.
function evaluatePosition(prefix) {
  const base = new Game2(SIZE);
  for (const m of prefix) base.play(m);
  if (base.gameOver) return null;
  // Cheap candidate pre-check, so we never spend a search on a dead position.
  let any = false; const ec = base.emptyCount, eCells = base._emptyCells;
  for (let i = 0; i < ec; i++) { const idx = eCells[i]; if (base.isLegal(idx) && !base.isTrueEye(idx)) { any = true; break; } }
  if (!any) return null;

  const res = teacher.getMove(base, 0, { playoutsPerCand: PER_CAND, rng });
  if (!res || !res.children) return null;

  // The pass baseline comes straight from the main search: puct-distill always keeps PASS
  // as a root candidate, so its child's value (wins/visits) is the mover's win prob after
  // passing — already in the same perspective as the other candidates' q.  No second search.
  // Two channels per candidate, both in the mover's perspective: win ratio q (with qp the
  // pass baseline) and point margin p (with pp the pass baseline).  p comes from full/terminal
  // playouts only, so its sample count pn can be < n when truncated playouts are enabled.
  const moves = [], q = [], n = [], p = [], pn = [];
  let qp = null, np = 0, pp = 0, ppn = 0;
  for (const ch of res.children) {
    if (ch.move.type === 'pass') {
      qp = ch.wins / ch.visits; np = Math.round(ch.visits);
      pp = ch.pointN > 0 ? ch.pointSum / ch.pointN : 0; ppn = Math.round(ch.pointN);
      continue;
    }
    const idx = ch.move.y * SIZE + ch.move.x;
    if (base.isTrueEye(idx)) continue;            // match featurepol's legal-non-eye candidate set
    moves.push(coordStr(idx, SIZE));
    q.push(q3(ch.wins / ch.visits));
    n.push(Math.round(ch.visits));
    p.push(p2(ch.pointN > 0 ? ch.pointSum / ch.pointN : 0));
    pn.push(Math.round(ch.pointN));
  }
  if (!moves.length || qp === null) return null;  // qp===null means pass wasn't a root child (skip)
  return { moves, q, n, qp: q3(qp), np, p, pp: p2(pp), pn, ppn };
}

// ── Generate (NDJSON to stdout; progress/settings to stderr) ─────────────────────
process.stderr.write(`gen-distill: sample-agent: ${AGENT}  teacher: ${TEACHER}  per-cand: ${PER_CAND}  size: ${SIZE}  samples: ${SAMPLES === Infinity ? 'unlimited' : SAMPLES}  epsilon: ${EPSILON}  seed: ${SEED}\n`);

const COLS = ['elapsed', 'posns', 'cands', 'pos/s'];
const COLW = [7, 7, 8, 6];
const printRow = c => process.stderr.write(c.map((v, i) => String(v).padStart(COLW[i])).join('  ') + '\n');
printRow(COLS);

const t0 = Date.now();
let written = 0, candTotal = 0, nextReport = 1;
function report() {
  const secs = (Date.now() - t0) / 1000;
  printRow([Util.fmtMs(Date.now() - t0), Util.fmt4i(written), Util.fmt4i(candTotal), secs > 0 ? (written / secs).toFixed(1) : '0']);
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
    moves: res.moves.join(','),
    q: res.q, n: res.n, qp: res.qp, np: res.np,
    p: res.p, pp: res.pp, pn: res.pn, ppn: res.ppn,
  };
  process.stdout.write(JSON.stringify(rec) + '\n');
  written++; candTotal += res.moves.length;
  if (written >= nextReport) { report(); nextReport = Math.max(nextReport + 1, Math.ceil(nextReport * 1.5)); }
}
report();
process.stderr.write(`done: ${written} positions\n`);
