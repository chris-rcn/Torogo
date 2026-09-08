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
// Usage: node measure-game-diversity.js --file <games.txt> [options]

const fs = require('fs');
const { Game2, parseMove } = require('./game2.js');
const { makeRng } = require('./xorshift.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'],
  ['file', 'limit', 'min-phase', 'max-phase', 'seed']);
if (opts.help || !opts.file) {
  console.error(`Usage: node measure-game-diversity.js --file <games.txt> [options]

Duplicate-position rate under consumer-style sampling (one reservoir ply
per game in the phase window) and the unique-prefix curve.

  --file PATH       gen-games corpus ("<size> <move1,...>" lines) (required)
  --limit N         games to read (default: all)
  --min-phase F     sampling window lower bound (default 0.2)
  --max-phase F     sampling window upper bound (default 0.6)
  --seed N          reservoir rng seed (default 23)
  --help            show this message`);
  process.exit(opts.help ? 0 : 1);
}
const LIMIT = opts.limit !== undefined ? parseInt(opts.limit, 10) : Infinity;
const MIN_PH = parseFloat(opts['min-phase'] !== undefined ? opts['min-phase'] : '0.2');
const MAX_PH = parseFloat(opts['max-phase'] !== undefined ? opts['max-phase'] : '0.6');
const rng = makeRng(parseInt(opts.seed || '23', 10) || 1);

const lines = fs.readFileSync(opts.file, 'utf8').split('\n')
  .filter(l => l && l[0] !== '#').slice(0, LIMIT);
if (lines.length === 0) { console.error('no games in ' + opts.file); process.exit(1); }

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
