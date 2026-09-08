'use strict';

// measure-game-diversity.js — corpus diversity for gen-games files, two
// lenses matched to how corpora are consumed:
//
//   1. Duplicate-position rate under consumer-style sampling: one reservoir-
//      sampled ply per game inside the phase window (exactly gen-agent-evals'
//      selection), positions keyed by cells + side to move.  This is the
//      ground truth for "is the corpus effectively smaller than its record
//      count?"
//   2. Prefix-divergence curve: the fraction of games whose first-k moves are
//      unique.  The early-warning lens — a too-deterministic generator shows
//      a shared trunk before its stochastic moves branch, and the curve
//      localises WHERE diversity is missing.
//
// Board size comes from each record; mixed-size corpora work (positions can
// only collide within a size).  Reference points (500 games, 13x13):
// ref-featurepol-softmax with --rand-open 4: 0.00% duplicates, 98% unique
// prefixes at k=2; bare (rand-open 0): 0.00%, 100% at k=4.
//
// Usage: node measure-game-diversity.js (--file <games.txt> | --agent <name>) [options]

const fs = require('fs');
const path = require('path');
const { Game2, PASS, parseMove, coordStr } = require('./game2.js');
const { makeRng } = require('./xorshift.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'],
  ['file', 'agent', 'games', 'size', 'rand-open', 'min-phase', 'max-phase', 'seed']);
if (opts.help || (!opts.file && !opts.agent) || (opts.file && opts.agent)) {
  console.error(`Usage: node measure-game-diversity.js --file <games.txt> [options]

Duplicate-position rate under consumer-style sampling (one reservoir ply
per game in the phase window) and the unique-prefix curve.

  --file PATH       gen-games corpus ("<size> <move1,...>" lines)
  --agent NAME      ai/<name>.js — self-play the games in-process instead of
                    reading a file (exactly one of --file/--agent; env config
                    reaches the agent as usual)
  --games N         games: self-play count with --agent (default 500), or
                    a cap on games read with --file (default: all)
  --size N          board size for --agent games (default 13)
  --rand-open N     random opening moves per --agent game (default 0 — the
                    bare-agent diversity measurement; gen-games production
                    corpora usually use 4)
  --min-phase F     sampling window lower bound (default 0 — narrow to a
                    training band to mirror a specific consumer)
  --max-phase F     sampling window upper bound (default 1)
  --seed N          reservoir rng seed (default 23)
  --help            show this message`);
  process.exit(opts.help ? 0 : 1);
}
const FILE_CAP = opts.games !== undefined ? parseInt(opts.games, 10) : Infinity;
const MIN_PH = parseFloat(opts['min-phase'] !== undefined ? opts['min-phase'] : '0');
const MAX_PH = parseFloat(opts['max-phase'] !== undefined ? opts['max-phase'] : '1');
const rng = makeRng(parseInt(opts.seed || '23', 10) || 1);

let lines;
if (opts.file) {
  lines = fs.readFileSync(opts.file, 'utf8').split('\n')
    .filter(l => l && l[0] !== '#').slice(0, FILE_CAP);
  if (lines.length === 0) { console.error('no games in ' + opts.file); process.exit(1); }
} else {
  const GAMES = parseInt(opts.games || '500', 10);
  const SIZE = parseInt(opts.size || '13', 10);
  const RAND_OPEN = parseInt(opts['rand-open'] || '0', 10);
  const mod = require(path.join(__dirname, 'ai', opts.agent + '.js'));
  const inst = typeof mod.create === 'function' ? mod.create(Util.makeCfg()) : mod;
  console.error(`self-playing ${GAMES} games: agent ${opts.agent}  size ${SIZE}  rand-open ${RAND_OPEN}`);
  const tGen0 = Date.now();
  lines = [];
  for (let g = 0; g < GAMES; g++) {
    const game = new Game2(SIZE);
    const toks = [];
    for (let r = 0; r < RAND_OPEN && !game.gameOver; r++) {
      const m = game.randomLegalMove();
      game.play(m);
      toks.push(coordStr(m, SIZE));
    }
    let mv = 0;
    const lim = SIZE * SIZE * 3 + 20;
    while (!game.gameOver && mv < lim) {
      const m = inst.getMove(game, 0, {}).move;
      game.play(m);
      toks.push(coordStr(m, SIZE));
      mv++;
    }
    lines.push(SIZE + ' ' + toks.join(','));
    if ((g + 1) % 100 === 0) console.error((g + 1) + ' games');
  }
  const tGen = (Date.now() - tGen0) / 1000;
  console.error(`self-play done: ${GAMES} games in ${tGen.toFixed(1)}s (${(tGen / GAMES * 1000).toFixed(0)}ms/game)`);
}

const posSeen = new Set();
let dups = 0, sampled = 0, failed = 0;
const prefixes = [];
for (const line of lines) {
  const sp = line.indexOf(' ');
  const size = parseInt(line.slice(0, sp), 10);
  const toks = line.slice(sp + 1).split(',');
  prefixes.push(toks);
  const g = new Game2(size);
  let chosen = -1, seen = 0, ok = true;
  for (let i = 0; i < toks.length; i++) {
    const ph = g.phase();
    if (ph > MAX_PH) break;
    if (ph >= MIN_PH && i > 0) { seen++; if (rng.random() < 1 / seen) chosen = i; }
    if (!g.play(parseMove(toks[i], size))) { ok = false; break; }
  }
  if (!ok) { failed++; continue; }
  if (chosen < 0) continue;
  const g2 = new Game2(size);
  for (let i = 0; i < chosen; i++) g2.play(parseMove(toks[i], size));
  const key = size + '|' + g2.current + '|' + g2.cells.join('');
  sampled++;
  if (posSeen.has(key)) dups++; else posSeen.add(key);
}
if (failed) console.error(`replay-failed games skipped: ${failed}`);
console.log(`games: ${lines.length}  sampled positions (phase [${MIN_PH}, ${MAX_PH}]): ${sampled}  ` +
            `duplicates: ${dups} (${(100 * dups / Math.max(1, sampled)).toFixed(2)}%)`);

const out = [];
for (const k of [2, 4, 6, 8, 10, 14, 20, 30]) {
  const cnt = new Map();
  for (const p of prefixes) { const key = p.slice(0, k).join(','); cnt.set(key, (cnt.get(key) || 0) + 1); }
  let uniq = 0;
  for (const p of prefixes) if (cnt.get(p.slice(0, k).join(',')) === 1) uniq++;
  out.push(`k=${k}: ${(100 * uniq / prefixes.length).toFixed(0)}%`);
}
console.log('unique-prefix: ' + out.join('  '));
