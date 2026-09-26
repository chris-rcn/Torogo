'use strict';
// Shared parser/serializer for the movedetails data format (*.md).
//
// One position per line, five space-separated fields:
//   <rowFormatId> <boardSize> <positionPhase> <moveHistoryCsv> <candidatesCsv>
//   e.g.  f1 13 0.4571 g7,a1,c9,l1 d4:483,i12:483,k7:481
//
// - rowFormatId: schema tag; only 'f1' is defined.  Gating parse on it lets a
//   future schema (f2, …) share tooling and even coexist in a file.
// - positionPhase: board fullness 1 - empty/area, STORED (not re-derived).
// - moveHistoryCsv: EVERY stone in play order, comma-separated coords, STARTING
//   with the centre stone — so a position replays from an EMPTY board
//   (Game2(N, false)), with no implicit free stone to remember.  '-' if empty.
// - candidatesCsv: comma-separated move:kwr, best-first (kwr = win ratio x1000,
//   integer; a terminal move carrying no rating is 'move:' with an empty value).
//   '-' if none.
// Lines starting with '#' are comments; blank lines are ignored.

const { Game2, parseMove, coordStr } = require('./game2.js');

const ROW_ID = 'f1';

// The free centre stone, as an explicit first move (so it lives in the history).
function centerMove(N) { return coordStr((N >> 1) * N + (N >> 1), N); }

function formatRow(pos) {
  const hist = pos.history.length ? pos.history.join(',') : '-';
  const cands = pos.candidates.length
    ? pos.candidates.map(c => `${c.m}:${c.kwr == null ? '' : c.kwr}`).join(',')
    : '-';
  const phase = typeof pos.phase === 'number' ? pos.phase.toFixed(4) : pos.phase;
  return `${ROW_ID} ${pos.boardSize} ${phase} ${hist} ${cands}`;
}

// Parse one line -> { boardSize, phase, history, candidates:[{m,kwr}] }, or null
// for a comment/blank.  Throws on a malformed line (incl. the old JSON format).
function parseRow(line) {
  const l = line.trim();
  if (!l || l[0] === '#') return null;
  if (l[0] === '{') throw new Error('movedetails: old JSON format — run convert-movedetails.js to migrate to *.md');
  const f = l.split(' ');
  if (f.length !== 5) throw new Error(`movedetails: expected 5 fields, got ${f.length}: "${l.slice(0, 64)}"`);
  const [id, sizeS, phaseS, histS, candsS] = f;
  if (id !== ROW_ID) throw new Error(`movedetails: unknown row format '${id}' (expected '${ROW_ID}')`);
  const history = histS === '-' ? [] : histS.split(',');
  const candidates = candsS === '-' ? [] : candsS.split(',').map(t => {
    const j = t.indexOf(':');
    const k = t.slice(j + 1);
    return { m: t.slice(0, j), kwr: k === '' ? null : parseInt(k, 10) };
  });
  return { boardSize: parseInt(sizeS, 10), phase: parseFloat(phaseS), history, candidates };
}

// Reconstruct the position's board from an EMPTY board plus the full history.
function buildGame(pos) {
  const g = new Game2(pos.boardSize, false);
  for (const m of pos.history) g.play(parseMove(m, pos.boardSize));
  return g;
}

module.exports = { ROW_ID, centerMove, formatRow, parseRow, buildGame };
