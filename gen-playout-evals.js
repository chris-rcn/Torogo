'use strict';

// gen-playout-evals.js — generate positions with ref-featurepol-softmax
// self-play and label each with the mean outcome of standard playouts,
// producing (position, value) training data for a static evaluator.
//
// The label is the expected outcome of the engine's standard playout — uniform
// moves below board fullness 0.6, ppat policy after (prod's playout: move limit
// 3*empty+20, estimateWinner) — which is exactly the quantity a truncated
// playout discards.  An evaluator regressed on these labels is unbiased by
// construction with respect to the playout signal it replaces.
//
// Positions: one per generator game, so positions are independent.  Each game
// is ref-featurepol-softmax self-play from the empty board (softmax sampling
// provides diversity; no random opening needed); one ply is reservoir-sampled
// inside the phase window (--min-phase <= board fullness <= --max-phase).
//
// Output format (NOT train_ppat-compatible — the phase column displaces the
// gen_evals move list, and the best-move field is dropped):
//
//   <bsize> <phase> <move1,move2,...> <winRatio>
//
// where phase is the position's board fullness and winRatio is P(side-to-move
// wins).  Data to stdout (redirect to a file); progress/config to stderr.
// Non-deterministic.

const path = require('path');
const { Game2, BLACK, coordStr } = require('./game2.js');
const { makeRng } = require('./xorshift.js');
const Util = require('./util.js');
const PP = require('./ppat-lib.js');

// Agent modules may print a load banner to stdout, which would corrupt the
// emitted data stream.  Reroute console.log to '#' comment lines; the data
// itself is written via process.stdout.write.
console.log = (...a) => process.stdout.write('# ' + a.join(' ') + '\n');

const opts = Util.parseArgs(process.argv.slice(2), ['help'],
  ['bsize', 'min-phase', 'max-phase', 'playouts', 'limit']);
if (opts.help) {
  console.error(
    'Usage: node gen-playout-evals.js [options]  > out.txt\n' +
    '\n' +
    'Generate positions with ref-featurepol-softmax self-play and label each\n' +
    'with the mean outcome of standard playouts (uniform below phase 0.6,\n' +
    'ppat after — prod\'s playout).  One position per game.  Output line:\n' +
    '  <bsize> <phase> <move1,move2,...> <winRatio>    (winRatio: P(side-to-move wins))\n' +
    '\n' +
    'Options:\n' +
    '  --bsize <n>       board size (default 13)\n' +
    '  --min-phase <f>   sample plies at board fullness >= f (default 0)\n' +
    '  --max-phase <f>   sample plies at board fullness <= f (default 1)\n' +
    '  --playouts <n>    playouts per position; the label is their mean (default 200)\n' +
    '  --limit <n>       stop after emitting n positions (default: run until killed)\n' +
    '  --help            show this help');
  process.exit(opts.help ? 0 : 1);
}

const BSIZE    = opts.bsize !== undefined ? parseInt(opts.bsize, 10) : 13;
const minPhase = parseFloat(opts['min-phase'] !== undefined ? opts['min-phase'] : '0');
const maxPhase = parseFloat(opts['max-phase'] !== undefined ? opts['max-phase'] : '1');
const playouts = opts.playouts !== undefined ? parseInt(opts.playouts, 10) : 200;
const limit    = opts.limit !== undefined ? parseInt(opts.limit, 10) : Infinity;

// Position generator: ref-featurepol-softmax (hardcoded; banner goes to stderr).
const agent = require(path.join(__dirname, 'ai', 'ref-featurepol-softmax.js'));

// Standard-playout policy: prod's ppat checkpoint with the same deployment
// knob.  Hard failure, not a fallback — a silently-uniform playout would label
// every position with the wrong oracle.
const ppatModel = PP.loadWeights(path.join(__dirname, 'ppat-data.js'));
if (!ppatModel) throw new Error('gen-playout-evals: cannot load ppat weights from ppat-data.js');
ppatModel.uniformBelowPhase = 0.6;
const ppatState = PP.createState(BSIZE);

const rng = makeRng(((Date.now() ^ (process.pid << 16)) >>> 0) || 1);   // non-deterministic

// One standard playout from `pos` (clones; pos is not mutated).  Mirrors
// prod's playout: move limit 3*empty+20, estimateWinner.  Returns 1 if BLACK
// wins, else 0.
function playoutB(pos) {
  const g = pos.clone();
  const moveLimit = 3 * g.emptyCount + 20;
  let moves = 0;
  while (!g.gameOver && moves < moveLimit) {
    g.play(PP.ppatMove(g, ppatState, ppatModel, rng));
    moves++;
  }
  return g.estimateWinner() === BLACK ? 1 : 0;
}

process.stdout.write(`# gen-playout-evals: generator: ref-featurepol-softmax bsize: ${BSIZE} ` +
  `min-phase: ${minPhase} max-phase: ${maxPhase} playouts: ${playouts} ` +
  `date: ${new Date().toISOString().slice(0, 10)}\n`);
process.stderr.write(`gen-playout-evals: bsize: ${BSIZE} min-phase: ${minPhase} ` +
  `max-phase: ${maxPhase} playouts: ${playouts} limit: ${limit}\n`);

let emitted = 0, misses = 0;
const MAX_MISSES = 10000;            // consecutive games with no eligible position
const guard = BSIZE * BSIZE * 4;

// Progress table (stderr): geometric print schedule, capped at 4 h between
// rows (the JS-trainer convention).  tPosition is the interval mean.
const COLS = ['tElapsed', 'positions', 'tPosition'];
const COLW = [8, 9, 9];
const printRow = cells => process.stderr.write(
  cells.map((c, i) => String(c).padStart(COLW[i])).join('  ') + '\n');
printRow(COLS);
const MAX_PRINT_GAP_MS = 4 * 3600 * 1000;   // 4 h
const t0 = Date.now();
let nextPrintAt = t0 + 1000, lastPrintAt = t0, lastEmitted = 0;
function progressRow() {
  const now = Date.now();
  const n = emitted - lastEmitted;
  printRow([Util.fmtMs(now - t0), Util.fmt4i(emitted),
            Util.fmtMs(n > 0 ? (now - lastPrintAt) / n : 0)]);
  lastPrintAt = now; lastEmitted = emitted;
  nextPrintAt = Math.min(t0 + Math.round((now - t0) * 1.3), now + MAX_PRINT_GAP_MS);
}

while (emitted < limit) {
  // Self-play one game, reservoir-sampling a ply inside the phase window.
  // Stop the game once the window is behind us — the rest cannot qualify.
  const game = new Game2(BSIZE);     // free initial centre stone; replay with Game2(bsize)
  const moves = [];
  let chosenPos = -1, seen = 0;
  while (!game.gameOver && moves.length < guard) {
    const phase = game.phase();
    if (phase > maxPhase) break;
    if (phase >= minPhase) {
      seen++;
      if (rng.random() < 1 / seen) chosenPos = moves.length;
    }
    const m = agent.getMove(game, 0, { rng }).move;
    if (!game.play(m)) break;
    moves.push(m);
  }

  if (chosenPos < 0) {               // no eligible position this game
    if (++misses >= MAX_MISSES) {
      console.error(`no eligible position in ${MAX_MISSES} consecutive games ` +
                    `(phase window [${minPhase}, ${maxPhase}] likely never reached)`);
      process.exit(1);
    }
    continue;
  }
  misses = 0;

  // Replay to the chosen position and label it with the playout mean.
  const pos = new Game2(BSIZE);
  for (let i = 0; i < chosenPos; i++) pos.play(moves[i]);
  let wins = 0;
  for (let p = 0; p < playouts; p++) wins += playoutB(pos);
  // P(side-to-move wins), from the integer counts (1 - wins/playouts leaves
  // float dust in the data file).
  const winRatio = (pos.current === BLACK ? wins : playouts - wins) / playouts;

  const seq = Array.from(moves.slice(0, chosenPos), m => coordStr(m, BSIZE)).join(',');
  process.stdout.write(`${BSIZE} ${pos.phase().toFixed(3)} ${seq} ${winRatio}\n`);
  emitted++;

  if (Date.now() >= nextPrintAt) progressRow();
}
if (emitted > lastEmitted) progressRow();   // final partial interval
