'use strict';

// gen-ladder-grid.js — sweep the ladder-test-case grid.
//   for example 1..N:
//     for chain size 1..MAX_STONES:
//       for each of the 4 types (kill, escape, futile-attack, futile-extend):
//         find a matching position (random legal play + ladder2 + agent confirm)
//         and display it, centered on and marking the critical move(s).
//
// Usage:  node gen-ladder-grid.js [boardSize=13] [N=1] [MAX_STONES=10]

process.env.DITHER = '0';   // deterministic confirmation-agent moves

const { Game2, BLACK, PASS, coordStr } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');
const { getLadderStatus } = require('./ladder2.js');
const Util = require('./util.js');

// Usage: node gen-ladders.js [--size 13] [--examples 1] [--max-stones 10] [--playouts 4000] [--min-depth 10] [--min-nodes 0] [--agent prod]
// Writes text-block cases (consumed by evalladders2.js) to stdout; redirect as needed.
const opts       = Util.parseArgs(process.argv.slice(2), ['help'], ['agent', 'examples', 'max-stones', 'min-depth', 'min-nodes', 'playouts', 'size']);
const SIZE       = parseInt(opts.size       || '13',   10);
const N          = parseInt(opts.examples   || '1',    10);   // examples per (chain, type) cell
const MAX_STONES = parseInt(opts['max-stones'] || '10', 10);
const PLAYOUTS   = parseInt(opts.playouts   || '10000', 10);
const MIN_DEPTH  = parseInt(opts['min-depth'] || '10', 10);   // reject ladders read shallower than this
const MIN_NODES  = parseInt(opts['min-nodes'] || '50',  10);   // reject ladders read in fewer nodes than this
const AGENT      = opts.agent || 'prod';                       // confirmation agent in ai/ that must pick the ladder move
const confirmAgent = require(`./ai/${AGENT}.js`);

const TYPES = [
  { name: 'kill',          wantDef: false, fail: false },
  { name: 'escape',        wantDef: true,  fail: false },
  { name: 'futile-attack', wantDef: false, fail: true  },
  { name: 'futile-extend', wantDef: true,  fail: true  },
];

// representative stones of every size-`chain`, 2-liberty group
function chainGroups(game, chain) {
  const cap = SIZE * SIZE, seen = new Set(), out = [];
  for (let i = 0; i < cap; i++) {
    if (game.cells[i] === 0) continue;
    const gid = game._gid[i];
    if (seen.has(gid)) continue;
    seen.add(gid);
    if (game.groupSize(gid) !== chain) continue;
    if (game.groupLibertyCount(gid) !== 2) continue;
    out.push(i);
  }
  return out;
}

// Would the defender extending the chain (playing one of `libs`) leave the
// group in atari (self-atari)?  Such futile-extend cases are trivial blunders,
// not real ladder tests, so we reject them.
function extendSelfAtari(game, stoneIdx, libs) {
  const g3 = game3FromGame2(game);
  for (const lib of libs) {
    if (!g3.play(lib)) continue;
    const atari = g3.groupLibs2(stoneIdx).count === 1;
    g3.undo();
    if (atari) return true;
  }
  return false;
}

// Does playing `move` capture any opponent stones?  (A plain fill lowers
// emptyCount by exactly 1; a capture frees the taken cells, so it drops less.)
function moveCaptures(game, move) {
  const g3 = game3FromGame2(game);
  const before = g3.emptyCount;
  if (!g3.play(move)) return false;
  const captured = g3.emptyCount >= before;
  g3.undo();
  return captured;
}

// A confirmed position is "decided" — and thus rejected — when the agent reports
// a root win ratio outside the contested band, i.e. abs(v - 0.5) >= 0.2.  Agents
// that don't return rootWinRatio impose no such condition (the check is skipped).
function decided(r) {
  return r.rootWinRatio !== undefined && !(Math.abs(r.rootWinRatio - 0.5) < 0.2);
}

// scan ONE position for an agent-confirmed case of (chain, type); return hit or null
function scanPos(game, chain, type) {
  for (const stoneIdx of chainGroups(game, chain)) {
    const isDef = game.cells[stoneIdx] === game.current;
    if (isDef !== type.wantDef) continue;
    const st = getLadderStatus(game3FromGame2(game), stoneIdx);
    if (!st) continue;
    // Require a non-trivial ladder read: shallow/small reads are easy cases that
    // don't discriminate.  readDepth = deepest recursion, readNodes = total
    // positions read.
    if (st.readDepth < MIN_DEPTH) continue;
    if (st.readNodes < MIN_NODES) continue;
    if (!type.fail) {
      if (!st.moverSucceeds || st.urgentLibs.length !== 1) continue;
      // escape: reject if the saving move is a capture (escapes via capture, not a real ladder) — before the agent
      if (type.wantDef && moveCaptures(game, st.urgentLibs[0])) continue;
      // reject if a random legal non-eye move hits the answer (too easy to guess) — before the agent
      if (game.randomLegalMove() === st.urgentLibs[0]) continue;
      const r = confirmAgent.getMove(game, 0, { playoutLimit: PLAYOUTS });
      if (decided(r)) return null;   // skip won/lost positions; keep only contested ones
      if (r.move === st.urgentLibs[0]) return { stoneIdx, color: game.cells[stoneIdx], require: st.urgentLibs[0] };
    } else {
      if (st.moverSucceeds) continue;   // mover can't succeed → futile
      // moves-to-avoid: keep only legal non-eye liberties; the set must be non-empty
      const prohibit = st.libs.filter(m => game.isLegal(m) && !game.isTrueEye(m));
      if (prohibit.length === 0) continue;
      // futile-extend: reject self-atari extends (trivial, not a ladder) — before the agent
      if (type.wantDef && extendSelfAtari(game, stoneIdx, prohibit)) continue;
      const r = confirmAgent.getMove(game, 0, { playoutLimit: PLAYOUTS });
      if (decided(r)) return null;   // skip won/lost positions; keep only contested ones
      if (!prohibit.includes(r.move)) return { stoneIdx, color: game.cells[stoneIdx], prohibit };
    }
  }
  return null;
}

// search random legal play until a case is found (or budget exhausted)
function findCase(chain, type) {
  const t0 = Date.now();
  let scanned = 0;
  while (true) {   // search until found (no time limit)
    const game = new Game2(SIZE);
    const maxMoves = SIZE * SIZE * 2;
    let moves = 0;
    while (!game.gameOver && moves < maxMoves) {
      scanned++;
      const hit = scanPos(game, chain, type);
      if (hit) return { game, hit, scanned, ms: Date.now() - t0 };
      const mv = game.randomLegalMove();
      if (mv === PASS) game.play(PASS); else if (!game.play(mv)) break;
      moves++;
    }
  }
}

// Quantized toroidal center of gravity of the chain containing stoneIdx.
// Each axis wraps, so use the circular mean (average unit vectors, not raw coords).
function chainCentroid(game, stoneIdx) {
  const N = game.N, gid = game._gid[stoneIdx];
  let sx = 0, cx = 0, sy = 0, cy = 0;
  for (let i = 0; i < N * N; i++) {
    if (game._gid[i] !== gid) continue;
    const ax = 2 * Math.PI * (i % N) / N, ay = 2 * Math.PI * ((i / N | 0)) / N;
    cx += Math.cos(ax); sx += Math.sin(ax);
    cy += Math.cos(ay); sy += Math.sin(ay);
  }
  const ang = (s, c) => { let a = Math.atan2(s, c); if (a < 0) a += 2 * Math.PI; return Math.round(a / (2 * Math.PI) * N) % N; };
  return ang(sy, cy) * N + ang(sx, cx);
}

// FNV-1a 32-bit string hash → 8-char hex.  Used to give each case a short,
// stable id derived from its board position.
function hashStr(s) {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}

function emit(res, type, chain) {
  const { game, hit } = res;
  // Physically recenter the board on the chain's centroid, then translate the
  // cited coordinates by the same shift so they match the shifted board.
  const { dx, dy } = game.recenter(chainCentroid(game, hit.stoneIdx));
  const tr = idx => (idx % SIZE + dx) % SIZE + (((idx / SIZE | 0) + dy) % SIZE) * SIZE;
  const toPlay = game.current === BLACK ? 'B' : 'W';
  let answer, marks;
  if (!type.fail) {
    marks = [tr(hit.require)];
    answer = `require=${coordStr(marks[0], SIZE)}`;
  } else {
    marks = hit.prohibit.map(tr);
    answer = `prohibit=${marks.map(m => coordStr(m, SIZE)).join(',')}`;
  }
  // Block: metadata header line, then the labeled board with the cited
  // required/avoid coordinates marked, blank-line separated.  id = hash of the
  // (recentered) position, giving each case a unique, stable handle.
  const id = hashStr(game.toString(PASS));
  const header = `id=${id} type=${type.name} chainSize=${chain} toPlay=${toPlay} ${answer} by=${AGENT}`;
  process.stdout.write(`${header}\n${game.toString(marks, { labels: true })}\n\n`);
}

// Per-type tallies for the end-of-run summary (cases, positions scanned, time).
const stats = new Map(TYPES.map(t => [t.name, { cases: 0, scanned: 0, ms: 0 }]));
const tStart = Date.now();

for (let i = 1; i <= N; i++) {
  for (let chain = 1; chain <= MAX_STONES; chain++) {
    for (const type of TYPES) {
      const res = findCase(chain, type);
      const s = stats.get(type.name);
      s.cases++; s.scanned += res.scanned; s.ms += res.ms;
      emit(res, type, chain);
    }
  }
}

// Summary (stderr, so it doesn't pollute the case data on stdout).
const wall = (Date.now() - tStart) / 1000;
let totCases = 0, totScanned = 0;
process.stderr.write('\n=== gen-ladders summary ===\n');
process.stderr.write(`board=${SIZE} examples=${N} maxStones=${MAX_STONES} playouts=${PLAYOUTS} minDepth=${MIN_DEPTH} minNodes=${MIN_NODES}\n`);
for (const t of TYPES) {
  const s = stats.get(t.name);
  totCases += s.cases; totScanned += s.scanned;
  const perCase = s.cases ? (s.scanned / s.cases).toFixed(0) : '-';
  const secs = (s.ms / 1000).toFixed(1);
  process.stderr.write(`  ${t.name.padEnd(14)} cases=${String(s.cases).padStart(4)}  scanned=${String(s.scanned).padStart(9)}  (${perCase}/case)  ${secs}s\n`);
}
process.stderr.write(`  ${'TOTAL'.padEnd(14)} cases=${String(totCases).padStart(4)}  scanned=${String(totScanned).padStart(9)}  wall=${wall.toFixed(1)}s  (${(wall / Math.max(1, totCases)).toFixed(2)}s/case)\n`);
