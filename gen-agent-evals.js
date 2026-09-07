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
// reported on stderr.  Ply 0 (the bare free initial stone) is never eligible:
// its record would have an empty move list.
//
// The position source matters for a reason worth stating: SB's objective is
// defined at the playout ROOT — E[outcome of a playout from s] should equal
// v*(s) — and at deployment those roots are search-tree nodes, i.e. positions
// reached by reasonably strong play.  The corpus's generating agent controls
// that realism; this tool just samples and labels.  The label is the agent's
// valueB() (P(BLACK wins)) mapped to P(side-to-move wins).  Output format
// (shared with the *-playout-eval trainers; announced by a '# format:' header
// line so legacy consumers can detect it — NOT the old train_ppat gen_evals
// format, which had no phase column and a trailing best-move token):
//
//   <size> <phase> <move1,move2,...> <winRatio>
//
// With --prefix-delta D each sampled ply is instead treated as a playout-
// START: a standard-playout prefix (ppat policy, uniform below fullness 0.6
// — the deployed playout) advances the position until board fullness has
// gained D, and the ENDPOINT is what gets labeled and emitted (its move
// list includes the prefix).  This produces the truncation deployment
// distribution — coherent substrate under a playout crust — with the phase
// window selecting playout-starts.
//
// The agent module (ai/<name>.js) must export a valueB(game, options) -> P(BLACK wins).
// e.g. mc-ppat (mean of PLAYOUTS standard playouts), ref-vlibpat, mc-vlib.
//
// Output goes to stdout (redirect to a file); progress/config to stderr.
// Non-deterministic.  Usage:
//   node gen-agent-evals.js --agent <name> --corpus <file>
//        [--min-phase 0] [--max-phase 1] [--prefix-delta D] [--limit N]  > out.txt

const fs = require('fs');
const path = require('path');
const { Game2, BLACK, PASS, coordStr, parseMove, setKomi } = require('./game2.js');
const { makeRng } = require('./xorshift.js');
const Util = require('./util.js');

// Agent modules may print a load banner to stdout (e.g. puct-hybrid's "loaded N
// weights"), which would corrupt the emitted data stream.  Reroute console.log to
// '#' comment lines; the data itself is written via process.stdout.write.
console.log = (...a) => process.stdout.write('# ' + a.join(' ') + '\n');

const opts = Util.parseArgs(process.argv.slice(2), ['help'],
  ['agent', 'corpus', 'min-phase', 'max-phase', 'prefix-delta', 'limit', 'komi']);
if (opts.help || !opts.agent || !opts.corpus) {
  console.error(`Usage: node gen-agent-evals.js --agent <name> --corpus <file> [options]  > out.txt

Label positions from a game corpus with an agent's valueB() oracle,
producing (position, value) training data.  Each emitted position comes
from a fresh corpus game (drawn with replacement, one reservoir-sampled
ply per game inside the phase window), so positions are independent and
parallel labelers on the same corpus need no coordination.  Output line
(announced by a '# format:' header; readable by the *-playout-eval
trainers and train_ppat):

  <bsize> <phase> <move1,move2,...> <winRatio>    (winRatio: P(side-to-move wins))

Data goes to stdout (redirect to a file); config and a progress table go
to stderr.  Non-deterministic; runs until --limit or killed.

  --agent NAME      ai/<name>.js — must export valueB(game, opts) ->
                    P(BLACK wins), e.g. mc-ppat (mean of PLAYOUTS standard
                    playouts; set PLAYOUTS in the env), vpatsearch,
                    ref-vlibpat (required)
  --corpus FILE     gen-games.js corpus ("<size> <move1,move2,...>" lines;
                    mixed sizes fine — size comes from each record) (required)
  --min-phase F     sample plies at board fullness >= F (default 0)
  --max-phase F     sample plies at board fullness <= F (default 1)
  --komi K          scoring komi for the labeling playouts (default 3.5;
                    applied to every board size via setKomi before the
                    agent loads).  Integer komi allows tied scores on this
                    area scoring — prefer half-integer values
  --prefix-delta D  treat each sampled ply as a playout-START: advance a
                    standard-playout prefix (ppat policy, uniform below
                    fullness 0.6 — the deployed playout) until board
                    fullness has gained D, then label the ENDPOINT; the
                    emitted move list includes the prefix.  The phase
                    window still selects the start, so endpoints land
                    near [min+D, max+D].  Samples whose game ends inside
                    the prefix are skipped
  --limit N         stop after emitting N positions (default: run until
                    killed)
  --help            show this message`);
  process.exit(opts.help ? 0 : 1);
}

const agentName  = opts.agent;
const corpusPath = opts.corpus;
const minPhase   = parseFloat(opts['min-phase'] !== undefined ? opts['min-phase'] : '0');
const maxPhase   = parseFloat(opts['max-phase'] !== undefined ? opts['max-phase'] : '1');
const limit      = opts.limit !== undefined ? parseInt(opts.limit, 10) : Infinity;
const PREFIX_DELTA = opts['prefix-delta'] !== undefined ? parseFloat(opts['prefix-delta']) : null;
if (PREFIX_DELTA !== null && !(PREFIX_DELTA > 0 && PREFIX_DELTA < 1)) {
  console.error('--prefix-delta: must be in (0, 1)'); process.exit(1);
}
// Labeling komi: applied before the agent loads so its playouts score with
// it.  Set for every plausible board size (setKomi is per-size).
const KOMI_ARG = opts.komi !== undefined ? parseFloat(opts.komi) : null;
if (KOMI_ARG !== null) {
  if (!(KOMI_ARG > -100 && KOMI_ARG < 100)) { console.error('--komi: bad value'); process.exit(1); }
  if (Number.isInteger(KOMI_ARG)) { console.error('--komi: komi must be half-integer'); process.exit(1); }
  for (let n = 5; n <= 19; n++) setKomi(n, KOMI_ARG);
}

const agent = require(path.join(__dirname, 'ai', agentName + '.js'));
if (typeof agent.valueB !== 'function') {
  console.error(`Agent '${agentName}' does not export a valueB() method`);
  process.exit(1);
}

const rng = makeRng(((Date.now() ^ (process.pid << 16)) >>> 0) || 1);   // non-deterministic

// Standard-playout prefix machinery (--prefix-delta): the same policy the
// deployed playout uses, matching measure-trunc-bias's defaults.
let PPat = null, ppatModel = null;
const ppatStates = new Map();                 // size -> ppat scratch state
if (PREFIX_DELTA !== null) {
  PPat = require('./ppat-lib.js');
  ppatModel = PPat.loadWeights(path.join(__dirname, 'out', 'ppat-data-233162-best-ref-candidate.js'));
  ppatModel.uniformBelowPhase = 0.6;
}

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
  process.stdout.write(`# format: bsize phase moves winRatio\n`);
  process.stdout.write(`# corpus: ${corpusPath} (${corpus.length} games)\n`);
  process.stdout.write(`# komi: ${KOMI_ARG !== null ? KOMI_ARG : '3.5 (default)'}\n`);
  if (PREFIX_DELTA !== null) process.stdout.write(
    `# prefix-delta: ${PREFIX_DELTA} (standard-playout prefix from each sampled ply; endpoint labeled)\n`);
}

const corpusSizes = [...new Set(corpus.map(g => g.size))].sort((a, b) => a - b).join(',');
process.stderr.write(`gen-agent-evals: agent: ${agentName}  corpus: ${corpusPath} (${corpus.length} games)  size: ${corpusSizes}  min-phase: ${minPhase}  max-phase: ${maxPhase}${PREFIX_DELTA !== null ? `  prefix-delta: ${PREFIX_DELTA}` : ''}  komi: ${KOMI_ARG !== null ? KOMI_ARG : '3.5 (default)'}  limit: ${limit === Infinity ? 'none' : limit}\n`);

let emitted = 0, misses = 0, prefixSkips = 0, prefixStreak = 0;

// Progress table (stderr): geometric print schedule, capped at 4 h between
// rows (the JS-trainer convention).  tPosition is the interval mean.
const COLS = ['tElapsed', 'positions', 'tPosition', 'blkWR'];
const COLW = [8, 9, 9, 6];
const printRow = cells => process.stderr.write(
  cells.map((c, i) => String(c).padStart(COLW[i])).join('  ') + '\n');
printRow(COLS);
const MAX_PRINT_GAP_MS = 4 * 3600 * 1000;   // 4 h
const t0 = Date.now();
let nextPrintAt = t0 + 1000, lastPrintAt = t0, lastEmitted = 0;
let pBlackSum = 0;   // running P(BLACK) over every emitted label — the
                     // symmetry gauge the labeling komi is tuned against
function progressRow() {
  const now = Date.now();
  const n = emitted - lastEmitted;
  printRow([Util.fmtMs(now - t0), Util.fmt4i(emitted),
            Util.fmtMs(n > 0 ? (now - lastPrintAt) / n : 0),
            emitted > 0 ? Util.fmtRatio4(pBlackSum / emitted) : '-']);
  lastPrintAt = now; lastEmitted = emitted;
  nextPrintAt = Math.min(t0 + Math.round((now - t0) * 1.3), now + MAX_PRINT_GAP_MS);
}
const MAX_MISSES = 10000;            // consecutive games with no eligible position

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
    // Ply 0 is ineligible: an empty move list is unparseable, and the
    // position is the same degenerate one every time.
    if (phase >= minPhase && i > 0) {
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
  let seq = Array.from(moves.slice(0, chosenPos), m => coordStr(m, size)).join(',');

  if (PREFIX_DELTA !== null) {
    // Advance the standard-playout prefix; the endpoint is the labeled
    // position.  Captures delay the fullness gain, as deployed; a game
    // that ends before the gain is reached has no endpoint — skip it.
    let state = ppatStates.get(size);
    if (!state) { state = PPat.createState(size); ppatStates.set(size, state); }
    const area = size * size;
    const stopEmpty = pos.emptyCount - Math.ceil(PREFIX_DELTA * area);
    const moveLimit = 3 * pos.emptyCount + 20;
    let n = 0;
    while (!pos.gameOver && pos.emptyCount > stopEmpty && n < moveLimit) {
      const m = PPat.ppatMove(pos, state, ppatModel, rng);
      pos.play(m);
      seq += ',' + coordStr(m, size);
      n++;
    }
    if (pos.emptyCount > stopEmpty) {
      prefixSkips++;
      if (++prefixStreak >= MAX_MISSES) {
        console.error(`game ended inside the prefix in ${MAX_MISSES} consecutive samples ` +
                      `(--max-phase + --prefix-delta likely past reachable fullness)`);
        process.exit(1);
      }
      continue;
    }
    prefixStreak = 0;
  }

  const val = agent.valueB(pos, { rng });                     // P(BLACK wins)
  pBlackSum += val;
  const winRatio = pos.current === BLACK ? val : 1 - val;     // P(side-to-move wins)

  process.stdout.write(`${size} ${pos.phase().toFixed(3)} ${seq} ${winRatio}\n`);
  emitted++;

  if (Date.now() >= nextPrintAt) progressRow();
}
if (emitted > lastEmitted) progressRow();   // final partial interval
if (prefixSkips) process.stderr.write(
  `prefix-delta: ${prefixSkips} sample(s) skipped (game ended inside the prefix)\n`);
