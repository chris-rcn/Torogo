'use strict';

// measure-game-diversity.js — corpus diversity for gen-games files.
//
// Duplicate-position rate by ply: replay every game and, at each of a few ply
// points from 1..--max-ply, key the position by cells + side to move and count
// how many games share a board with an earlier game AT THAT SAME PLY.  The
// ground truth for "is the corpus effectively smaller than its record count?"
// Comparisons are always within one ply (same depth, so same stone count), so a
// collision is a genuine repeat — positions at different plies are never
// compared.  A game shorter than a ply point simply doesn't count toward it.
//
// Board size comes from each record; mixed-size corpora work (positions can
// only collide within a size).
//
// Usage: node measure-game-diversity.js (--file <games.txt> | --agent <name>) [options]

const fs = require('fs');
const path = require('path');
const { Game2, parseMove, coordStr } = require('./game2.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'],
  ['file', 'agent', 'games', 'size', 'rand-open', 'max-ply']);
if (opts.help || (!opts.file && !opts.agent) || (opts.file && opts.agent)) {
  console.error(`Usage: node measure-game-diversity.js --file <games.txt> [options]

Duplicate-position rate by ply: for a few ply points up to --max-ply, the
fraction of games whose board (at that ply) repeats an earlier game's.

  --file PATH       gen-games corpus ("<size> <move1,...>" lines)
  --agent NAME      ai/<name>.js — self-play the games in-process instead of
                    reading a file (exactly one of --file/--agent; env config
                    reaches the agent as usual)
  --games N         games: self-play count with --agent (default 200), or
                    a cap on games read with --file (default: all)
  --size N          board size for --agent games (default 13)
  --rand-open N     random opening moves per --agent game (default 0 — the
                    bare-agent diversity measurement; gen-games production
                    corpora usually use 4)
  --max-ply N       deepest ply to measure; stats are reported at a few points
                    from 1 to N (default 30)
  --help            show this message`);
  process.exit(opts.help ? 0 : 1);
}
const FILE_CAP = opts.games !== undefined ? parseInt(opts.games, 10) : Infinity;
const MAX_PLY = parseInt(opts['max-ply'] || '30', 10);

// The ply points reported: a fixed geometric-ish set, capped at MAX_PLY and
// always including MAX_PLY itself.
const PLY_POINTS = (() => {
  const pts = [];
  for (const p of [1, 2, 4, 6, 8, 10, 14, 20, 30, 40, 60, 80, 100]) if (p <= MAX_PLY) pts.push(p);
  if (pts.length === 0 || pts[pts.length - 1] !== MAX_PLY) pts.push(MAX_PLY);
  return pts;
})();

let lines;
if (opts.file) {
  lines = fs.readFileSync(opts.file, 'utf8').split('\n')
    .filter(l => l && l[0] !== '#').slice(0, FILE_CAP);
  if (lines.length === 0) { console.error('no games in ' + opts.file); process.exit(1); }
} else {
  const GAMES = parseInt(opts.games || '200', 10);
  const SIZE = parseInt(opts.size || '13', 10);
  const RAND_OPEN = parseInt(opts['rand-open'] || '0', 10);
  const mod = require(path.join(__dirname, 'ai', opts.agent + '.js'));
  const inst = typeof mod.create === 'function' ? mod.create(Util.makeCfg()) : mod;
  console.error(`self-playing ${GAMES} games: agent ${opts.agent}  size ${SIZE}  rand-open ${RAND_OPEN}  max-ply ${MAX_PLY}`);
  const tGen0 = Date.now();
  lines = [];
  for (let g = 0; g < GAMES; g++) {
    const game = new Game2(SIZE, true);
    const toks = [];
    for (let r = 0; r < RAND_OPEN && !game.gameOver; r++) {
      const m = game.randomLegalMove();
      game.play(m);
      toks.push(coordStr(m, SIZE));
    }
    // Only MAX_PLY moves are needed for the deepest measured point, so stop
    // there — no point generating the rest of the game.  (RAND_OPEN moves,
    // already in toks, count toward the ply.)
    while (!game.gameOver && toks.length < MAX_PLY) {
      const m = inst.getMove(game, 0, {}).move;
      game.play(m);
      toks.push(coordStr(m, SIZE));
    }
    lines.push(SIZE + ' ' + toks.join(','));
    if ((g + 1) % 100 === 0) console.error((g + 1) + ' games');
  }
  const tGen = (Date.now() - tGen0) / 1000;
  console.error(`self-play done: ${GAMES} games in ${tGen.toFixed(1)}s (${(tGen / GAMES * 1000).toFixed(0)}ms/game)`);
}

// One Set + duplicate/sample counter per ply point.  Each game is replayed once
// to MAX_PLY, recording its board key as it passes each point; keys are compared
// only within a point (same ply, same board size).
const seen = PLY_POINTS.map(() => new Set());
const dups = new Array(PLY_POINTS.length).fill(0);
const sampledAt = new Array(PLY_POINTS.length).fill(0);
let failed = 0;
for (const line of lines) {
  const sp = line.indexOf(' ');
  const size = parseInt(line.slice(0, sp), 10);
  const toks = line.slice(sp + 1).split(',');
  const g = new Game2(size, true);
  let pi = 0, ok = true;
  const last = Math.min(toks.length, MAX_PLY);
  for (let i = 0; i < last && pi < PLY_POINTS.length; i++) {
    if (!g.play(parseMove(toks[i], size))) { ok = false; break; }
    if (PLY_POINTS[pi] === i + 1) {         // i+1 moves now played
      const key = size + '|' + g.current + '|' + g.cells.join('');
      sampledAt[pi]++;
      if (seen[pi].has(key)) dups[pi]++; else seen[pi].add(key);
      pi++;
    }
  }
  if (!ok) failed++;
}
if (failed) console.error(`replay-failed games (partial): ${failed}`);
console.log(`duplicate-position rate by ply (${lines.length} games):`);
console.log('  ' + PLY_POINTS.map((P, i) =>
  `ply${P}: ${(100 * dups[i] / Math.max(1, sampledAt[i])).toFixed(1)}%`).join('   '));
const deepest = sampledAt[PLY_POINTS.length - 1];
if (deepest < lines.length)
  console.error(`note: ${deepest}/${lines.length} games reached ply ${MAX_PLY}`);
