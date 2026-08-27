'use strict';

// gen-agent-evals.js — label positions from a game corpus with an agent's
// valueB() oracle, producing (position, value) training data for Simulation
// Balancing (train_ppat).
//
// Positions come from a gen-games.js corpus (lines: "<size> <move1,move2,...>"):
// pick a uniform random game, replay it once recording phase per ply, reservoir-
// sample one ply inside the phase window (--min-phase <= board fullness <=
// --max-phase), label it.  One position per selection; games are drawn with
// replacement, so independent labelers on the same corpus need no coordination
// beyond distinct processes.  Size comes from each record, so mixed-size corpora
// work.  Games are validated on selection (the phase scan is the integrity
// check); a game that fails replay is counted, dropped from the pool, and
// reported on stderr.
//
// The position source matters for a reason worth stating: SB's objective is
// defined at the playout ROOT — E[outcome of a playout from s] should equal
// v*(s) — and at deployment those roots are search-tree nodes, i.e. positions
// reached by reasonably strong play.  The corpus's generating agent controls
// that realism; this tool just samples and labels.  The label is the agent's
// valueB() (P(BLACK wins)) mapped to P(side-to-move wins), to match the
// gen_evals format consumed by train_ppat:
//
//   <size> <move1,move2,...> <winRatio> pass
//
// The agent module (ai/<name>.js) must export a valueB(game, options) -> P(BLACK wins).
// e.g. ref-vlibpat, mc-vlib.
//
// Output goes to stdout (redirect to a file); progress/config to stderr.
// Non-deterministic.  Usage:
//   node gen-agent-evals.js --agent <name> --corpus <file>
//        [--min-phase 0] [--max-phase 1] [--limit N]  > out.txt

const fs = require('fs');
const path = require('path');
const { Game2, BLACK, PASS, coordStr, parseMove } = require('./game2.js');
const { makeRng } = require('./xorshift.js');
const Util = require('./util.js');

// Agent modules may print a load banner to stdout (e.g. puct-hybrid's "loaded N
// weights"), which would corrupt the emitted data stream.  Reroute console.log to
// '#' comment lines; the data itself is written via process.stdout.write.
console.log = (...a) => process.stdout.write('# ' + a.join(' ') + '\n');

const opts = Util.parseArgs(process.argv.slice(2), ['help'],
  ['agent', 'corpus', 'min-phase', 'max-phase', 'limit']);
if (opts.help || !opts.agent || !opts.corpus) {
  console.error('Usage: node gen-agent-evals.js --agent <name> --corpus <file>\n' +
                '                               [--min-phase 0] [--max-phase 1] [--limit N]  > out.txt');
  process.exit(opts.help ? 0 : 1);
}

const agentName  = opts.agent;
const corpusPath = opts.corpus;
const minPhase   = parseFloat(opts['min-phase'] !== undefined ? opts['min-phase'] : '0');
const maxPhase   = parseFloat(opts['max-phase'] !== undefined ? opts['max-phase'] : '1');
const limit      = opts.limit !== undefined ? parseInt(opts.limit, 10) : Infinity;

const agent = require(path.join(__dirname, 'ai', agentName + '.js'));
if (typeof agent.valueB !== 'function') {
  console.error(`Agent '${agentName}' does not export a valueB() method`);
  process.exit(1);
}

const rng = makeRng(((Date.now() ^ (process.pid << 16)) >>> 0) || 1);   // non-deterministic

// Load the corpus up front (token->index only; each game is replay-validated
// when it is first selected, not eagerly).
const corpus = [];                   // [{ size, moves: Int16Array }]
{
  let malformed = 0;
  for (const line of fs.readFileSync(corpusPath, 'utf8').split('\n')) {
    if (!line) continue;
    if (line[0] === '#') {
      // Chain provenance: the corpus's own generation header goes into ours.
      if (line.startsWith('# gen-games:')) process.stdout.write(line + '\n');
      continue;
    }
    const p = line.split(/\s+/);
    const gsize = p.length === 2 ? parseInt(p[0], 10) : NaN;
    if (!Number.isFinite(gsize)) { malformed++; continue; }
    const toks = p[1].split(',');
    const moves = new Int16Array(toks.length);
    let ok = true;
    for (let i = 0; i < toks.length; i++) {
      const m = parseMove(toks[i], gsize);
      // Guard the Int16Array store: NaN (torn token) would coerce to 0 = a1,
      // silently turning a corrupt record into a playable game.
      if (!Number.isInteger(m) || m < PASS || m >= gsize * gsize) { ok = false; break; }
      moves[i] = m;
    }
    if (!ok) { malformed++; continue; }
    corpus.push({ size: gsize, moves });
  }
  if (malformed) process.stderr.write(`corpus: ${malformed} malformed line(s) skipped\n`);
  if (corpus.length === 0) {
    console.error(`corpus '${corpusPath}' contains no games`);
    process.exit(1);
  }
  process.stdout.write(`# corpus: ${corpusPath} (${corpus.length} games)\n`);
}

process.stderr.write(`gen-agent-evals: agent=${agentName} corpus=${corpusPath} (${corpus.length} games) min-phase=${minPhase} max-phase=${maxPhase} limit=${limit}\n`);

let emitted = 0, misses = 0;
const MAX_MISSES = 10000;            // consecutive games with no eligible position
const t0 = Date.now();

while (emitted < limit) {
  // Pick a uniform random game; replay it once, reservoir-sampling one ply
  // inside the phase window (the same replay validates the record).
  const gi = (rng.random() * corpus.length) | 0;
  const g  = corpus[gi];
  const { size, moves } = g;
  const game = new Game2(size);
  let chosenPos = -1, seen = 0, ok = true;
  for (let i = 0; i < moves.length; i++) {
    const phase = game.phase();
    if (phase > maxPhase) break;     // board only fills; nothing eligible past the cap
    if (phase >= minPhase) {
      seen++;
      if (rng.random() < 1 / seen) chosenPos = i;
    }
    if (!game.play(moves[i])) { ok = false; break; }
  }
  if (!ok) {
    // Bad record (e.g. torn tail): drop it from the pool, loudly.
    process.stderr.write(`corpus: game ${gi} failed replay, dropped (${corpus.length - 1} left)\n`);
    corpus[gi] = corpus[corpus.length - 1];
    corpus.pop();
    if (corpus.length === 0) { console.error('corpus: no valid games left'); process.exit(1); }
    continue;
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

  // Replay to the chosen position and label it with the agent's value().
  const pos = new Game2(size);
  for (let i = 0; i < chosenPos; i++) pos.play(moves[i]);
  const val = agent.valueB(pos, { rng });                     // P(BLACK wins)
  const winRatio = pos.current === BLACK ? val : 1 - val;     // P(side-to-move wins)

  const seq = Array.from(moves.slice(0, chosenPos), m => coordStr(m, size)).join(',');
  process.stdout.write(`${size} ${seq} ${winRatio} pass\n`);
  emitted++;

  if (emitted % 200 === 0) {
    const s = (Date.now() - t0) / 1000;
    process.stderr.write(`  ${emitted} positions  ${(emitted / s).toFixed(1)}/s\n`);
  }
}
