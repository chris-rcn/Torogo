'use strict';

// gen-agent-evals.js — label positions with an agent's valueB() oracle,
// producing (position, value) training data for Simulation Balancing
// (train_ppat).  Positions come from a game corpus (--corpus) or from novel
// self-play games generated on the fly by an agent (--position-agent) — the latter
// skips the separate gen-games.js corpus step.
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
//   node gen-agent-evals.js --value-agent <name> (--corpus <file> | --position-agent <name>)
//        [--min-phase 0] [--max-phase 1] [--prefix-delta D] [--limit N]  > out.txt

const path = require('path');
const { Game2, BLACK, coordStr, setKomi } = require('./game2.js');
const { makeRng } = require('./xorshift.js');
const Util = require('./util.js');

// Agent modules may print a load banner to stdout (e.g. puct-hybrid's "loaded N
// weights"), which would corrupt the emitted data stream.  Reroute console.log to
// '#' comment lines; the data itself is written via process.stdout.write.
console.log = (...a) => process.stdout.write('# ' + a.join(' ') + '\n');

const opts = Util.parseArgs(process.argv.slice(2), ['help', 'corpus-ordered'],
  ['value-agent', 'corpus', 'position-agent', 'size', 'rand-open', 'min-phase', 'max-phase', 'prefix-delta', 'limit', 'komi']);
if (opts.help || !opts['value-agent'] || (!opts.corpus && !opts['position-agent'])) {
  console.error(`Usage: node gen-agent-evals.js --value-agent <name> (--corpus <file> | --position-agent <name>) [options]  > out.txt

Label positions with an agent's valueB() oracle, producing (position, value)
training data.  Positions come from either a game corpus (--corpus) or novel
self-play games generated on the fly (--position-agent) — one reservoir-sampled ply
per game inside the phase window.  Positions are independent, so parallel
labelers need no coordination.  Output line (announced by a '# format:' header;
readable by the *-playout-eval trainers and train_ppat):

  <bsize> <phase> <move1,move2,...> <winRatio>    (winRatio: P(side-to-move wins))

Data goes to stdout (redirect to a file); config and a progress table go
to stderr.  Non-deterministic; runs until --limit or killed.

  --value-agent NAME
                    ai/<name>.js labeling oracle — must export valueB(game,
                    opts) -> P(BLACK wins), e.g. mc-ppat (mean of PLAYOUTS
                    standard playouts; set PLAYOUTS in the env), vpatsearch,
                    ref-vlibpat (required)
  --corpus FILE     gen-games.js corpus ("<size> <move1,move2,...>" lines;
                    mixed sizes fine — size comes from each record).  One of
                    --corpus / --position-agent is required (mutually exclusive)
  --corpus-ordered  take the corpus games in file order, looping, instead of
                    at random; eval k's ply (and --prefix-delta moves) come
                    from an rng seeded by k.  Runs with the same corpus, phase
                    window and prefix then emit the same positions line for
                    line, whatever the value agent and its effort (labels use
                    their own rng)
  --position-agent NAME
                    ai/<name>.js self-play policy (needs getMove()) that
                    generates a novel game per position — no corpus file
                    needed.  Uses --size and --rand-open
  --size N          board size for --position-agent games (default 13)
  --rand-open N     random opening moves per generated game, for diversity
                    (default 4; --position-agent only)
  --min-phase F     sample plies at board fullness >= F (default 0)
  --max-phase F     sample plies at board fullness <= F (default 1)
  --komi K          scoring komi for the labeling playouts (default 3.5;
                    applied to every board size via setKomi before the
                    agent loads).  Must be half-integer.
  --prefix-delta D  treat each sampled ply as a playout-START: advance a
                    standard-playout prefix until board fullness has gained
                    D, then label the ENDPOINT; the
                    emitted move list includes the prefix.  The phase
                    window still selects the start, so endpoints land
                    near [min+D, max+D].  Samples whose game ends inside
                    the prefix are skipped
  --limit N         stop after emitting N positions (default: run until
                    killed)
  --help            show this message`);
  process.exit(opts.help ? 0 : 1);
}

const agentName    = opts['value-agent'];
const corpusPath   = opts.corpus || null;
const gameAgentName = opts['position-agent'] || null;
const GAME_MODE     = !!gameAgentName;
if (corpusPath && gameAgentName) {
  console.error('--corpus and --position-agent are mutually exclusive'); process.exit(1);
}
const ORDERED = opts['corpus-ordered'] === true;
if (ORDERED && !corpusPath) { console.error('--corpus-ordered needs --corpus'); process.exit(1); }
const GAME_SIZE = opts.size !== undefined ? parseInt(opts.size, 10) : 13;
const randOpen = opts['rand-open'] !== undefined ? parseInt(opts['rand-open'], 10) : 4;
if (!GAME_MODE && (opts.size !== undefined || opts['rand-open'] !== undefined)) {
  console.error('--size/--rand-open apply only with --position-agent'); process.exit(1);
}
if (GAME_MODE) {
  if (!Number.isInteger(GAME_SIZE) || GAME_SIZE < 3) { console.error('--size must be an integer >= 3'); process.exit(1); }
  if (!Number.isInteger(randOpen) || randOpen < 0) { console.error('--rand-open must be a non-negative integer'); process.exit(1); }
}
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

// --position-agent: the self-play policy that generates novel games (getMove).
// Prefer its create(cfg) factory so a slot-aware agent reads its env config.
let gameAgent = null;
if (GAME_MODE) {
  const gm = require(path.join(__dirname, 'ai', gameAgentName + '.js'));
  gameAgent = typeof gm.create === 'function' ? gm.create(Util.makeCfg()) : gm;
  if (typeof gameAgent.getMove !== 'function') {
    console.error(`Position agent '${gameAgentName}' does not export getMove()`);
    process.exit(1);
  }
}

const rng = makeRng(Util.randomSeed());   // non-deterministic

// Standard-playout prefix machinery (--prefix-delta): the same policy the
// deployed playout uses, matching measure-trunc-bias's defaults.
let PPat = null, ppatModel = null;
const ppatStates = new Map();                 // size -> ppat scratch state
if (PREFIX_DELTA !== null) {
  PPat = require('./ppat-lib.js');
  ppatModel = PPat.loadWeights(path.join(__dirname, 'out', 'ppat-data-233162-best-ref-candidate.js'));
  ppatModel.ppatMinPhase = 0.6;
}

// Corpus mode: load games up front (token->index only; each game is replay-
// validated when first selected).  Gen-agent mode uses no corpus.
const corpus = [];                   // corpus mode: [{ size, moves: Int16Array }]
if (!GAME_MODE) {
  const { games, malformed, provenance } = require('./games-corpus.js').loadGamesCorpus(corpusPath);
  // Chain provenance: the corpus's own generation header goes into ours.
  for (const line of provenance) process.stdout.write(line + '\n');
  for (const gm of games) corpus.push(gm);
  if (malformed) process.stderr.write(`corpus: ${malformed} malformed line(s) skipped\n`);
  if (corpus.length === 0) {
    console.error(`corpus '${corpusPath}' contains no games`);
    process.exit(1);
  }
}

// Provenance / format headers (stdout), both modes.
process.stdout.write(`# format: bsize phase moves winRatio\n`);
process.stdout.write(GAME_MODE
  ? `# position-agent: ${gameAgentName} size: ${GAME_SIZE} rand-open: ${randOpen}\n`
  : `# corpus: ${corpusPath} (${corpus.length} games)${ORDERED ? ' ordered (eval k seeded by k)' : ''}\n`);
process.stdout.write(`# komi: ${KOMI_ARG !== null ? KOMI_ARG : '3.5 (default)'}\n`);
if (PREFIX_DELTA !== null) process.stdout.write(
  `# prefix-delta: ${PREFIX_DELTA} (standard-playout prefix from each sampled ply; endpoint labeled)\n`);

const sizeStr   = GAME_MODE ? String(GAME_SIZE)
                           : [...new Set(corpus.map(g => g.size))].sort((a, b) => a - b).join(',');
const sourceStr = GAME_MODE ? `position-agent: ${gameAgentName} (rand-open ${randOpen})`
                           : `corpus: ${corpusPath} (${corpus.length} games)`;
process.stderr.write(`gen-agent-evals: agent: ${agentName}  ${sourceStr}  size: ${sizeStr}  min-phase: ${minPhase}  max-phase: ${maxPhase}${PREFIX_DELTA !== null ? `  prefix-delta: ${PREFIX_DELTA}` : ''}  komi: ${KOMI_ARG !== null ? KOMI_ARG : '3.5 (default)'}  limit: ${limit === Infinity ? 'none' : limit}\n`);

let emitted = 0, misses = 0, prefixSkips = 0, prefixStreak = 0;
// --corpus-ordered: the next game (looping), and eval k's position rng, seeded
// by k so every run makes the same choices; the labels keep using `rng`.
let nextGame = 0;
const evalRng = k => makeRng((Math.imul(k + 1, 0x9E3779B1) >>> 0) || 1);

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

// --position-agent: play a novel game (random opening + self-play) only as far as the
// phase cap needs — the reservoir sampler below still sees every eligible ply.
// Returns { size, moves } shaped like a corpus record.
function generateTrajectory() {
  const game = new Game2(GAME_SIZE, true);
  const moves = [];
  for (let i = 0; i < randOpen && !game.gameOver && game.phase() <= maxPhase; i++) {
    const m = game.randomLegalMove(rng);
    if (!game.play(m)) break;
    moves.push(m);
  }
  const guard = GAME_SIZE * GAME_SIZE * 4;
  while (!game.gameOver && moves.length < guard && game.phase() <= maxPhase) {
    const m = gameAgent.getMove(game, 0, { rng }).move;
    if (!game.play(m)) break;
    moves.push(m);
  }
  return { size: GAME_SIZE, moves };
}

while (emitted < limit) {
  // A trajectory: a corpus game (drawn with replacement) or a freshly generated
  // one.  Replay it once, reservoir-sampling one ply inside the phase window
  // (the same replay validates a corpus record).
  let gi = -1, size, moves;
  const prng = ORDERED ? evalRng(emitted) : rng;   // position choices (ply, prefix)
  if (GAME_MODE) {
    ({ size, moves } = generateTrajectory());
  } else if (ORDERED) {
    gi = nextGame;
    nextGame = (nextGame + 1) % corpus.length;
    ({ size, moves } = corpus[gi]);
  } else {
    gi = (rng.random() * corpus.length) | 0;
    ({ size, moves } = corpus[gi]);
  }
  const game = new Game2(size, true);
  let chosenPos = -1, seen = 0, ok = true;
  for (let i = 0; i < moves.length; i++) {
    const phase = game.phase();
    if (phase > maxPhase) break;     // board only fills; nothing eligible past the cap
    // Ply 0 is ineligible: an empty move list is unparseable, and the
    // position is the same degenerate one every time.
    if (phase >= minPhase && i > 0) {
      seen++;
      if (prng.random() < 1 / seen) chosenPos = i;
    }
    if (!game.play(moves[i])) { ok = false; break; }
  }
  if (!ok) {
    if (GAME_MODE) { console.error('position-agent: generated game failed replay (bug)'); process.exit(1); }
    if (ORDERED) {   // keep the order: skip the record (every pass), loudly
      process.stderr.write(`corpus: game ${gi} failed replay, skipped\n`);
      if (++misses >= MAX_MISSES) { console.error(`no eligible position in ${MAX_MISSES} consecutive games`); process.exit(1); }
      continue;
    }
    // Bad corpus record (e.g. torn tail): drop it from the pool, loudly.
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
  const pos = new Game2(size, true);
  for (let i = 0; i < chosenPos; i++) pos.play(moves[i]);
  let seq = Array.from(moves.slice(0, chosenPos), m => coordStr(m, size)).join(',');

  if (PREFIX_DELTA !== null) {
    // Advance the standard-playout prefix; the endpoint is the labeled
    // position.  A FIXED number of moves, as deployed — delta is a fullness
    // fraction only so one number carries across board sizes, and a fullness
    // check repeated per move descends further whenever the prefix captures.
    // A game that ends before the prefix completes has no endpoint — skip it.
    let state = ppatStates.get(size);
    if (!state) { state = PPat.createState(size); ppatStates.set(size, state); }
    const prefixLen = Math.ceil(PREFIX_DELTA * size * size);
    let n = 0;
    while (!pos.gameOver && n < prefixLen) {
      const m = PPat.ppatMove(pos, state, ppatModel, prng);
      pos.play(m);
      seq += ',' + coordStr(m, size);
      n++;
    }
    if (n < prefixLen) {
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
