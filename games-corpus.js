'use strict';

// games-corpus.js — reader for gen-games.js corpora: one game per line,
//   <size> <move1,move2,...>
// the complete trajectory after gen-games' start position, the free black
// centre stone (terminal passes included; a line ending in pass,pass is a
// completed game, any other was cut by the move guard).  '#' lines are comments; gen-games' own '# gen-games:'
// provenance lines are returned so a consumer can chain them.
//
//   loadGamesCorpus(path) -> { games: [{ size, moves: Int16Array, line }], malformed, provenance: [line] }
//                            (line: the record's 1-based line number in the file)
//   isCompleted(moves)    -> the trajectory ends in pass,pass
//   startGame(size)       -> a Game2 at the start position the moves replay from

const fs = require('fs');
const { Game2, PASS, parseMove } = require('./game2.js');

// Calls fn(line, lineNo) for each line, reading the file in chunks: a corpus
// can exceed V8's ~512M-character string limit, so it is never read whole.
// Lines split at byte 0x0A, so a multi-byte character never straddles a chunk.
function _forEachLine(filePath, fn) {
  const fd = fs.openSync(filePath, 'r');
  const chunk = Buffer.allocUnsafe(1 << 24);
  let carry = null, lineNo = 0, n;
  try {
    while ((n = fs.readSync(fd, chunk, 0, chunk.length, null)) > 0) {
      const data = carry ? Buffer.concat([carry, chunk.subarray(0, n)]) : chunk.subarray(0, n);
      const cut = data.lastIndexOf(10);
      if (cut < 0) { carry = Buffer.from(data); continue; }
      for (const line of data.toString('utf8', 0, cut).split('\n')) fn(line, ++lineNo);
      carry = cut + 1 < data.length ? Buffer.from(data.subarray(cut + 1)) : null;
    }
  } finally {
    fs.closeSync(fd);
  }
  if (carry) fn(carry.toString('utf8'), ++lineNo);
}

function loadGamesCorpus(filePath) {
  const games = [], provenance = [];
  let malformed = 0;
  _forEachLine(filePath, (line, lineNo) => {
    if (!line) return;
    if (line[0] === '#') {
      if (line.startsWith('# gen-games:')) provenance.push(line);
      return;
    }
    const p = line.split(/\s+/);
    const size = p.length === 2 ? parseInt(p[0], 10) : NaN;
    if (!Number.isFinite(size)) { malformed++; return; }
    const toks = p[1].split(',');
    const moves = new Int16Array(toks.length);
    let ok = true;
    for (let i = 0; i < toks.length; i++) {
      const m = parseMove(toks[i], size);
      // Guard the Int16Array store: NaN (torn token) would coerce to 0 = a1,
      // silently turning a corrupt record into a playable game.
      if (!Number.isInteger(m) || m < PASS || m >= size * size) { ok = false; break; }
      moves[i] = m;
    }
    if (!ok) { malformed++; return; }
    games.push({ size, moves, line: lineNo });
  });
  return { games, malformed, provenance };
}

function isCompleted(moves) {
  const n = moves.length;
  return n >= 2 && moves[n - 1] === PASS && moves[n - 2] === PASS;
}

// gen-games' start: the free black centre stone, white to move.
function startGame(size) {
  const g = new Game2(size, false);
  g.play((size >> 1) * size + (size >> 1));
  return g;
}

module.exports = { loadGamesCorpus, isCompleted, startGame };
