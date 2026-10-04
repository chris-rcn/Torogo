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

function loadGamesCorpus(filePath) {
  const games = [], provenance = [];
  let malformed = 0;
  const lines = fs.readFileSync(filePath, 'utf8').split('\n');
  for (let li = 0; li < lines.length; li++) {
    const line = lines[li];
    if (!line) continue;
    if (line[0] === '#') {
      if (line.startsWith('# gen-games:')) provenance.push(line);
      continue;
    }
    const p = line.split(/\s+/);
    const size = p.length === 2 ? parseInt(p[0], 10) : NaN;
    if (!Number.isFinite(size)) { malformed++; continue; }
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
    if (!ok) { malformed++; continue; }
    games.push({ size, moves, line: li + 1 });
  }
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
