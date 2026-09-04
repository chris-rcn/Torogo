'use strict';

// gen-games.js — generate a corpus of full game trajectories by agent self-play.
//
// Decouples position generation from evaluation: this tool plays whole games
// and stores only the trajectories; a labeler (gen-agent-evals.js with a
// corpus source) later evaluates positions drawn from them.  Since everything
// about a position is a pure function of the trajectory (result, phase, board,
// side to move — all recoverable by replay), each record is just:
//
//   <size> <move1,move2,...>
//
// Moves are the complete trajectory from the empty board (random opening
// included), comma-separated coordStr tokens, terminal passes included — a
// trajectory ending in pass,pass is a completed game; one that doesn't was
// cut by the move guard.  Two fields, deliberately incompatible with the
// eval-file readers (which require >= 3), so a corpus fed to a trainer by
// mistake fails loudly.  Line number is the game id.
//
// Output goes to stdout (redirect to a file); progress/config to stderr.
// Naming convention: out/games-s<size>-<generator-tag>.txt.  Usage:
//   node gen-games.js --agent <name> [--size 9] [--limit N] [--seed n]
//        [--rand-open 4]  > out.txt

const path = require('path');
const { Game2, coordStr } = require('./game2.js');
const { makeRng } = require('./xorshift.js');
const Util = require('./util.js');

// Agent modules may print a load banner to stdout, which would corrupt the
// emitted data stream.  Reroute console.log to '#' comment lines; the data
// itself is written via process.stdout.write.
console.log = (...a) => process.stdout.write('# ' + a.join(' ') + '\n');

const opts = Util.parseArgs(process.argv.slice(2), ['help'],
  ['agent', 'size', 'limit', 'seed', 'rand-open']);
if (opts.help || !opts.agent) {
  console.error(`Usage: node gen-games.js --agent <name> [options]  > out.txt

Generate a corpus of full game trajectories by agent self-play.  Positions
are labeled later (gen-agent-evals.js draws and labels plies from the
corpus), so each record is just the trajectory:

  <size> <move1,move2,...>

Two fields, deliberately unreadable by the eval-file consumers, so a corpus
fed to a trainer by mistake fails loudly.  A trajectory ending in pass,pass
is a completed game; one that does not was cut by the move guard.  Data to
stdout (redirect to a file; convention out/games-s<size>-<tag>.txt);
config/progress to stderr.

  --agent NAME      ai/<name>.js self-play policy — needs getMove(); a
                    stochastic agent (e.g. ref-featurepol-softmax) gives
                    game diversity (required)
  --size N          board size (default 9)
  --limit N         stop after N games (default: run until killed)
  --rand-open N     random opening moves per game for extra diversity
                    (default 4)
  --seed N          rng seed for the opening moves (default: time/pid).
                    Full games are still not seed-reproducible: softmax
                    agents sample via Math.random
  --help            show this message`);
  process.exit(opts.help ? 0 : 1);
}

const agentName = opts.agent;
const size      = parseInt(opts.size || '9', 10);
const limit     = opts.limit !== undefined ? parseInt(opts.limit, 10) : Infinity;
const randOpen  = parseInt(opts['rand-open'] !== undefined ? opts['rand-open'] : '4', 10);
const seed      = opts.seed !== undefined ? (parseInt(opts.seed, 10) >>> 0)
                                          : (((Date.now() ^ (process.pid << 16)) >>> 0) || 1);

// Prefer the create(cfg) factory so a slot-aware agent picks up its env
// config; fall back to a module-level getMove for the older agents.
const agentModule = require(path.join(__dirname, 'ai', agentName + '.js'));
const agent = typeof agentModule.create === 'function' ? agentModule.create(Util.makeCfg()) : agentModule;
if (typeof agent.getMove !== 'function') {
  console.error(`Agent '${agentName}' does not export getMove()`);
  process.exit(1);
}

const area = size * size;
// Drives the random opening and is passed to getMove (search agents honour
// options.rng; the softmax refs take only (game) and ignore it).  Full games
// are still not seed-reproducible: ppat-lib's ppatMove samples via
// Math.random regardless, and the softmax refs sample via Math.random too.
const rng  = makeRng(seed);

// Provenance header: the generator identity is the one thing replay cannot
// recover, and corpora outlive engine generations.
process.stdout.write(`# gen-games: agent: ${agentName} size: ${size} komi: 3.5 rand-open: ${randOpen} seed: ${seed} limit: ${limit} date: ${new Date().toISOString().slice(0, 10)}\n`);
process.stderr.write(`gen-games: agent: ${agentName} size: ${size} limit: ${limit} seed: ${seed} rand-open: ${randOpen}\n`);

let emitted = 0;

// Progress table (stderr): geometric print schedule, capped at 4 h between
// rows (the JS-trainer convention).  tGame is the interval mean.
const COLS = ['tElapsed', 'games', 'tGame'];
const COLW = [8, 8, 8];
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
  nextPrintAt = Math.min(t0 + Math.round((now - t0) * 1.4), now + MAX_PRINT_GAP_MS);
}

while (emitted < limit) {
  const game  = new Game2(size);     // free initial centre stone; replay with Game2(size)
  const moves = [];

  // Random opening moves for diversity.
  for (let i = 0; i < randOpen && !game.gameOver; i++) {
    const m = game.randomLegalMove(rng);
    if (!game.play(m)) break;
    moves.push(m);
  }

  // Agent self-play to the end of the game.
  const guard = area * 4;
  while (!game.gameOver && moves.length < guard) {
    const m = agent.getMove(game, 0, { rng }).move;
    if (!game.play(m)) break;
    moves.push(m);
  }

  process.stdout.write(`${size} ${moves.map(m => coordStr(m, size)).join(',')}\n`);
  emitted++;

  if (Date.now() >= nextPrintAt) progressRow();
}
if (emitted > lastEmitted) progressRow();   // final partial interval
