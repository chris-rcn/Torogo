#!/usr/bin/env node
'use strict';

// move-agreement.js — do two agent configurations actually make different
// moves?
//
// A head-to-head that finishes level admits two readings: the two sides are
// genuinely different and equally strong, or they are making the same moves and
// the match could never have separated them.  This tool tells them apart, and
// it needs a null, because these agents are stochastic — ppat playouts, root
// dither, softmax sampling — so even one agent against itself agrees well below
// 100%.  The agent's own run-to-run agreement IS that null.
//
// Per position, four decisions: two independent runs of each side.  That gives
// two SELF comparisons (P1 vs P1, P2 vs P2) and four CROSS comparisons, all on
// the same position, so the difference is paired and the SE is computed over
// positions rather than over comparisons (the four cross pairs of one position
// are not independent).
//
// Read the GAP, not the levels.  gap ~ 0 means the two configurations are
// interchangeable in play and a level match says nothing about the models.
// gap clearly positive means they diverge, and a level match is then a real
// statement that the divergence does not help.
//
// Agents are slot-aware exactly as in selfplay.js, so the P1_*/P2_* environment
// of a match script can be pasted in front of this command unchanged.
//
// Usage:
//   node move-agreement.js --p1 <agent> --p2 <agent> --corpus <games.txt> [options]
//
//   --p1 NAME        agent under ai/ for slot 1                     (required)
//   --p2 NAME        agent under ai/ for slot 2                     (required)
//   --corpus PATH    gen-games corpus to sample positions from      (required)
//   --positions N    positions to sample                            (default 200)
//   --budget MS      per-move time budget, both sides               (default 2000)
//   --min-phase F    sample positions with phase in [min, max] —    (default 0)
//   --max-phase F    use the match's phase window                   (default 0.4)
//   --size N         board size; corpus lines of other sizes skipped (default 13)
//   --seed N         rng seed                                       (default 1)

const fs   = require('fs');
const path = require('path');
const Util = require('./util.js');
const { Game2, parseMove } = require('./game2.js');
const { makeRng } = require('./xorshift.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'],
  ['p1', 'p2', 'corpus', 'positions', 'budget', 'min-phase', 'max-phase', 'size', 'seed']);
if (opts.help || !opts.p1 || !opts.p2 || !opts.corpus) {
  console.error(`Usage: node move-agreement.js --p1 <agent> --p2 <agent> --corpus <games.txt> [options]

How often do two agent configurations pick the same move, against the null of
one configuration against itself?  Agents are slot-aware (P1_*/P2_* env), so a
match script's environment can be pasted in front of this command.

  --p1 NAME       agent under ai/ for slot 1                      (required)
  --p2 NAME       agent under ai/ for slot 2                      (required)
  --corpus PATH   gen-games corpus to sample positions from       (required)
  --positions N   positions to sample                             (default 200)
  --budget MS     per-move time budget, both sides                (default 2000)
  --min-phase F   sample positions with phase in [min, max]       (default 0)
  --max-phase F                                                   (default 0.4)
  --size N        board size                                      (default 13)
  --seed N        rng seed                                        (default 1)
  --help          show this message`);
  process.exit(opts.help ? 0 : 1);
}

const SIZE      = parseInt(opts.size || '13', 10);
const POSITIONS = parseInt(opts.positions || '200', 10);
const BUDGET    = parseInt(opts.budget || '2000', 10);
const MIN_PH    = parseFloat(opts['min-phase'] !== undefined ? opts['min-phase'] : '0');
const MAX_PH    = parseFloat(opts['max-phase'] !== undefined ? opts['max-phase'] : '0.4');
const SEED      = parseInt(opts.seed || '1', 10);
const area      = SIZE * SIZE;

function loadAgent(name, slot) {
  const mod = require(path.join(__dirname, 'ai', name + '.js'));
  const inst = (typeof mod.create === 'function') ? mod.create(Util.makeCfg(slot)) : mod;
  return inst.getMove;
}
const p1 = loadAgent(opts.p1, 1);
const p2 = loadAgent(opts.p2, 2);

// Bounded prefix read: the game corpora run to ~100MB and readFileSync throws
// ERR_STRING_TOO_LONG well before that.  A prefix is all this needs — one
// position is sampled per line and the sample stops at --positions.
const CORPUS_BYTES = 64 << 20;
const corpus = (() => {
  const fd = fs.openSync(opts.corpus, 'r');
  const size = fs.fstatSync(fd).size;
  const n = Math.min(size, CORPUS_BYTES);
  const buf = Buffer.allocUnsafe(n);
  fs.readSync(fd, buf, 0, n, 0);
  fs.closeSync(fd);
  const lines = buf.toString('utf-8').split('\n');
  if (n < size) lines.pop();            // drop the truncated tail line
  return lines.filter(l => l && l[0] !== '#');
})();
const pick = makeRng(SEED);

// One position per corpus line, reservoir-sampled over the moves whose phase is
// in the window — the same band the match lets these agents decide in.
function samplePosition(line) {
  const sp = line.indexOf(' ');
  if (parseInt(line.slice(0, sp), 10) !== SIZE) return null;
  const toks = line.slice(sp + 1).split(',');
  const walk = new Game2(SIZE);
  let chosen = null, nEligible = 0;
  for (let i = 0; i < toks.length; i++) {
    const ph = 1 - walk.emptyCount / area;
    if (ph > MAX_PH) break;
    if (ph >= MIN_PH) { nEligible++; if (pick.random() * nEligible < 1) chosen = walk.clone(); }
    walk.play(parseMove(toks[i], SIZE));
  }
  return chosen;
}

console.log(`move-agreement: p1 ${opts.p1}  p2 ${opts.p2}  corpus ${opts.corpus}  ` +
            `positions ${POSITIONS}  budget ${BUDGET}ms  phase [${MIN_PH}, ${MAX_PH}]  size ${SIZE}  seed ${SEED}`);

const COLS  = ['pos', 'elapsed', 'selfP1', 'selfP2', 'self', 'cross', 'gap', '+-SE'];
const COLW  = [6, 9, 8, 8, 8, 8, 8, 8];
let headerShown = false;
function printRow(cells) {
  if (!headerShown) { console.log(COLS.map((c, i) => c.padStart(COLW[i])).join('')); headerShown = true; }
  console.log(cells.map((c, i) => String(c).padStart(COLW[i])).join(''));
}

// Per-position rates, so the SE is over positions: the four cross pairs of one
// position share its board and are not independent observations.
const selfP1 = [], selfP2 = [], crossR = [], gapR = [];
const t0 = Date.now();
let next = 1, done = 0;   // first row immediately, then geometric x1.4

for (let li = 0; li < corpus.length && done < POSITIONS; li++) {
  const pos = samplePosition(corpus[li]);
  if (!pos) continue;
  const s = (done + 1) * 977;
  const a1 = p1(pos.clone(), BUDGET, { rng: makeRng(SEED + s + 1) }).move;
  const a2 = p1(pos.clone(), BUDGET, { rng: makeRng(SEED + s + 2) }).move;
  const b1 = p2(pos.clone(), BUDGET, { rng: makeRng(SEED + s + 3) }).move;
  const b2 = p2(pos.clone(), BUDGET, { rng: makeRng(SEED + s + 4) }).move;
  done++;

  const sa = a1 === a2 ? 1 : 0;
  const sb = b1 === b2 ? 1 : 0;
  const cr = ((a1 === b1) + (a1 === b2) + (a2 === b1) + (a2 === b2)) / 4;
  selfP1.push(sa); selfP2.push(sb); crossR.push(cr);
  gapR.push((sa + sb) / 2 - cr);

  if (done >= next || done === POSITIONS) {
    next = Math.max(done + 1, Math.round(done * 1.4));
    const mean = v => v.reduce((x, y) => x + y, 0) / v.length;
    const g = mean(gapR);
    const sd = Math.sqrt(gapR.reduce((x, y) => x + (y - g) * (y - g), 0) / Math.max(1, gapR.length - 1));
    printRow([done, Util.fmtMs(Date.now() - t0),
              (100 * mean(selfP1)).toFixed(1), (100 * mean(selfP2)).toFixed(1),
              (100 * (mean(selfP1) + mean(selfP2)) / 2).toFixed(1),
              (100 * mean(crossR)).toFixed(1), (100 * g).toFixed(1),
              (100 * sd / Math.sqrt(gapR.length)).toFixed(1)]);
  }
}

const mean = v => v.reduce((x, y) => x + y, 0) / v.length;
const g = mean(gapR);
const sd = Math.sqrt(gapR.reduce((x, y) => x + (y - g) * (y - g), 0) / Math.max(1, gapR.length - 1));
const se = sd / Math.sqrt(gapR.length);
console.log(`\n${done} positions, ${((Date.now() - t0) / 1000).toFixed(1)}s`);
console.log(`  self-agreement  p1 ${(100 * mean(selfP1)).toFixed(1)}%   p2 ${(100 * mean(selfP2)).toFixed(1)}%   mean ${(100 * (mean(selfP1) + mean(selfP2)) / 2).toFixed(1)}%`);
console.log(`  cross-agreement ${(100 * mean(crossR)).toFixed(1)}%`);
console.log(`  gap (self - cross) ${(100 * g).toFixed(1)}% +- ${(100 * se).toFixed(1)} (paired, SE over positions)`);
