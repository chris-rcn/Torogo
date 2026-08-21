'use strict';

// gen-agent-evals.js — generate (position, value) training data for Simulation
// Balancing (train_ppat), using any agent's valueB() method as the value oracle.
//
// Positions come from ref-npat-softmax self-play (fast, stochastic), with 4
// random opening moves per game for diversity.  Each game is sampled once, and
// only positions with --min-phase <= board phase <= --max-phase are eligible
// (board fullness 1-empty/area, e.g. to target the range where the oracle is
// trustworthy).  The label is the agent's
// value() (P(BLACK wins)) mapped to P(side-to-move wins), to match the gen_evals
// format consumed by train_ppat:
//
//   <size> <move1,move2,...> <winRatio> pass
//
// The agent module (ai/<name>.js) must export a valueB(game, options) -> P(BLACK wins).
// e.g. ref-vlibpat, mc-vlib.
//
// Output goes to stdout (redirect to a file); progress/config to stderr.
// Non-deterministic.  Usage:
//   node gen-agent-evals.js --agent <name> [--size 9] [--min-phase 0] [--max-phase 1] [--limit N]  > out.txt

const path = require('path');
const { Game2, BLACK, PASS, coordStr } = require('./game2.js');
const { makeRng } = require('./xorshift.js');
const Util = require('./util.js');

// Agent modules may print a load banner to stdout (e.g. puct-hybrid's "loaded N
// weights"), which would corrupt the emitted data stream.  Reroute console.log to
// '#' comment lines; the data itself is written via process.stdout.write.
console.log = (...a) => process.stdout.write('# ' + a.join(' ') + '\n');

const RefNpat = require('./ai/ref-npat-softmax.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['agent', 'size', 'min-phase', 'max-phase', 'limit']);
if (opts.help || !opts.agent) {
  console.error('Usage: node gen-agent-evals.js --agent <name> [--size 9] [--min-phase 0] [--max-phase 1] [--limit N]  > out.txt');
  process.exit(opts.help ? 0 : 1);
}
const agentName = opts.agent;
const size      = parseInt(opts.size || '9', 10);
const minPhase  = parseFloat(opts['min-phase'] !== undefined ? opts['min-phase'] : '0');
const maxPhase  = parseFloat(opts['max-phase'] !== undefined ? opts['max-phase'] : '1');
const limit     = opts.limit !== undefined ? parseInt(opts.limit, 10) : Infinity;

const agent = require(path.join(__dirname, 'ai', agentName + '.js'));
if (typeof agent.valueB !== 'function') {
  console.error(`Agent '${agentName}' does not export a valueB() method`);
  process.exit(1);
}

const area     = size * size;
const RAND_OPEN = 4;                 // random moves sprinkled at the start of each game
const MIN_POS   = RAND_OPEN + 1;     // sample at least one policy move past the random opening
const rng = makeRng(((Date.now() ^ (process.pid << 16)) >>> 0) || 1);   // non-deterministic

process.stderr.write(`gen-agent-evals: agent=${agentName} size=${size} min-phase=${minPhase} max-phase=${maxPhase} limit=${limit}\n`);

let emitted = 0;
const t0 = Date.now();

while (emitted < limit) {
  const game  = new Game2(size);     // free initial centre stone; replay with Game2(size)
  const moves = [];

  // 4 random opening moves.
  for (let i = 0; i < RAND_OPEN && !game.gameOver; i++) {
    const m = game.randomLegalMove(rng);
    if (!game.play(m)) break;
    moves.push(m);
  }

  // ref-npat-softmax self-play, reservoir-sampling one eligible position
  // (minPhase <= phase <= maxPhase); stop once the board passes the max cap
  // (board only fills, so no eligible positions remain past it).
  let chosenPos = -1, seen = 0;
  const guard = area * 4;
  while (!game.gameOver && moves.length < guard) {
    const phase = game.phase();
    const pos   = moves.length;
    if (phase > maxPhase && pos > RAND_OPEN) break;
    if (pos >= MIN_POS && phase >= minPhase && phase <= maxPhase) {
      seen++;
      if (rng.random() < 1 / seen) chosenPos = pos;
    }
    const m = RefNpat.getMove(game).move;
    if (!game.play(m)) break;
    moves.push(m);
  }
  if (chosenPos < 0) continue;       // no eligible position this game

  // Replay to the chosen position and label it with the agent's value().
  const pos = new Game2(size);
  for (let i = 0; i < chosenPos; i++) pos.play(moves[i]);
  const val = agent.valueB(pos, { rng });                     // P(BLACK wins)
  const winRatio = pos.current === BLACK ? val : 1 - val;     // P(side-to-move wins)

  const seq = moves.slice(0, chosenPos).map(m => coordStr(m, size)).join(',');
  process.stdout.write(`${size} ${seq} ${winRatio} pass\n`);
  emitted++;

  if (emitted % 200 === 0) {
    const s = (Date.now() - t0) / 1000;
    process.stderr.write(`  ${emitted} positions  ${(emitted / s).toFixed(1)}/s\n`);
  }
}
