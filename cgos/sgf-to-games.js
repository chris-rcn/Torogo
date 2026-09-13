'use strict';

// sgf-to-games.js — convert the CGOS server's SGF archive into a gen-games
// corpus ("<size> <move1,move2,...>" lines) for use as position source by
// gen-agent-evals / measure-trunc-bias.
//
// Coordinate mapping inverts the server's SGF writer (gogame/game.py sgf()):
// col = c1 - 'a' (the GTP 'i'-skip is already undone by the writer), row =
// N - (c2 - 'a'), Game2 idx = (row - 1) * N + col.  The first SGF move is
// Game2's auto-played black center stone and is dropped (gen-games records
// begin at the second stone); a game whose first move is NOT the center
// cannot map onto Game2 and is dropped, loudly.  Pass and resign both
// encode as "[]" in this archive, so conversion truncates each game at the
// first "[]" — the stone-placement history is complete by then and trailing
// passes add nothing for phase sampling.  Every converted game is replayed
// through Game2 before it is emitted; replay failures are dropped and
// reported.
//
// Output to stdout (redirect to a file); stats to stderr.  Usage:
//   node cgos/sgf-to-games.js [--dir cgos/data] [--min-elo E]  > games.txt
//
//   --min-elo E   keep only games where BOTH players' at-game-time ratings
//                 (the SGF WR/BR tags) are >= E; games with a missing or
//                 unparseable rating tag are excluded when this is set

const fs = require('fs');
const path = require('path');
const { Game2, coordStr } = require('../game2.js');
const Util = require('../util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['dir', 'min-elo']);
if (opts.help) {
  console.error('Usage: node cgos/sgf-to-games.js [--dir cgos/data] [--min-elo E]  > games.txt');
  process.exit(0);
}
const DIR = opts.dir || path.join(__dirname, 'data');
const MIN_ELO = opts['min-elo'] !== undefined ? parseFloat(opts['min-elo']) : null;

const files = [];
(function walk(d) {
  for (const e of fs.readdirSync(d, { withFileTypes: true })) {
    const p = path.join(d, e.name);
    if (e.isDirectory()) walk(p);
    else if (e.name.endsWith('.sgf')) files.push(p);
  }
})(DIR);
// Deterministic output order: by gid (the numeric SGF basename).
files.sort((a, b) => parseInt(path.basename(a), 10) - parseInt(path.basename(b), 10));

let kept = 0, belowElo = 0, badFirst = 0, replayFail = 0, malformed = 0;
process.stdout.write(`# sgf-to-games: dir: ${DIR} min-elo: ${MIN_ELO !== null ? MIN_ELO : 'none'} scanned: ${files.length} date: ${new Date().toISOString().slice(0, 10)}\n`);

for (const f of files) {
  const sgf = fs.readFileSync(f, 'utf8');
  const sz = /SZ\[(\d+)\]/.exec(sgf);
  if (!sz) { malformed++; process.stderr.write(`${f}: no SZ tag, dropped\n`); continue; }
  const N = parseInt(sz[1], 10);

  if (MIN_ELO !== null) {
    const wr = /WR\[([\d.]+)\]/.exec(sgf), br = /BR\[([\d.]+)\]/.exec(sgf);
    if (!wr || !br || +wr[1] < MIN_ELO || +br[1] < MIN_ELO) { belowElo++; continue; }
  }

  const toks = [];
  const re = /;[BW]\[([a-z]{2})?\]/g;
  let m, first = true, ok = true;
  while ((m = re.exec(sgf)) !== null) {
    if (m[1] === undefined) break;               // pass/resign: history complete
    const col = m[1].charCodeAt(0) - 97;
    const row = N - (m[1].charCodeAt(1) - 97);   // 1-based
    if (col < 0 || col >= N || row < 1 || row > N) { ok = false; break; }
    const idx = (row - 1) * N + col;
    if (first) {
      first = false;
      if (idx !== (N >> 1) * N + (N >> 1)) { ok = false; break; }  // not the center stone
      continue;                                  // auto-played by the Game2 constructor
    }
    toks.push(idx);
  }
  if (!ok) { badFirst++; process.stderr.write(`${f}: unmappable (bad coord or non-center first move), dropped\n`); continue; }
  if (toks.length === 0) { malformed++; continue; }

  const g = new Game2(N, true);
  let legal = true;
  for (const idx of toks) if (!g.play(idx)) { legal = false; break; }
  if (!legal) { replayFail++; process.stderr.write(`${f}: replay failed, dropped\n`); continue; }

  process.stdout.write(`${N} ${toks.map(i => coordStr(i, N)).join(',')}\n`);
  kept++;
}

process.stderr.write(`sgf-to-games: kept ${kept} of ${files.length}` +
  `  (below min-elo: ${belowElo}, unmappable: ${badFirst}, replay-failed: ${replayFail}, malformed: ${malformed})\n`);
