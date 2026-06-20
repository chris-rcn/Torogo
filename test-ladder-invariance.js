#!/usr/bin/env node
'use strict';

// test-ladder-invariance.js
//
// Invariant-based audit of ladder2.js (JS only).  ladder2's verdict for a
// 1–2-liberty group is a function of the board graph, the stone colours, and
// the side to move.  Three transformations leave the underlying tactical truth
// unchanged, so the verdicts must transform covariantly — any deviation is a
// topology / parity bug in ladder2, found with no oracle:
//
//   • D4 symmetry      — the 8 rotations/reflections of the square torus.
//   • Toroidal shift   — translate every stone by (dx,dy) (mod N).
//   • Colour/turn flip — swap every stone's colour AND the side to move.
//
// Under all three, a group's (moverSucceeds, urgent-move set) is invariant;
// only the stone indices move (via the transform) and, for the colour flip,
// the colour label flips.  We build each transformed position by replaying the
// transformed stones in the BASE position's known-good index order (a symmetry
// preserves legality), then compare per-group verdicts.
//
// Usage:
//   node test-ladder-invariance.js [--games N] [--size N] [--seed N]
//        [--shifts K] [--max-report N]

const { Game2, coordStr } = require('./game2.js');
const { Game3 } = require('./game3.js');
const { getLadderStatus } = require('./ladder2.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help']);
if (opts.help) {
  console.log('Usage: node test-ladder-invariance.js [--games N] [--size N] [--seed N] [--shifts K] [--max-report N]');
  process.exit(0);
}
const GAMES      = parseInt(opts.games || '300', 10);
const N          = parseInt(opts.size  || '9',   10);
const SEED       = parseInt(opts.seed  || '1',   10);
const SHIFTS     = parseInt(opts.shifts || '2',  10);   // random toroidal shifts per position
const MAX_REPORT = parseInt(opts['max-report'] || '20', 10);

const cap = N * N;
let _s = (SEED >>> 0) || 1;
const rng = { random() { _s = (_s * 1664525 + 1013904223) >>> 0; return _s / 4294967296; } };

// ── Transforms: each is { name, idx(i)->i, color(c)->c, current(c)->c } ────────
const X = i => i % N, Y = i => (i / N) | 0;
const at = (x, y) => ((y % N + N) % N) * N + (x % N + N) % N;
const D4 = [
  ['identity',  i => i],
  ['rot90',     i => at(N - 1 - Y(i), X(i))],
  ['rot180',    i => at(N - 1 - X(i), N - 1 - Y(i))],
  ['rot270',    i => at(Y(i), N - 1 - X(i))],
  ['flipX',     i => at(N - 1 - X(i), Y(i))],
  ['flipY',     i => at(X(i), N - 1 - Y(i))],
  ['transpose', i => at(Y(i), X(i))],
  ['antidiag',  i => at(N - 1 - Y(i), N - 1 - X(i))],
];
const idC = c => c, negC = c => -c;
const IDENTITY = { name: 'base', idx: i => i, color: idC, current: idC };

function transformsFor() {
  const ts = [];
  for (const [name, fn] of D4) ts.push({ name, idx: fn, color: idC, current: idC });
  for (let k = 0; k < SHIFTS; k++) {
    const dx = 1 + ((rng.random() * (N - 1)) | 0);
    const dy = 1 + ((rng.random() * (N - 1)) | 0);
    ts.push({ name: `shift(${dx},${dy})`, idx: i => at(X(i) + dx, Y(i) + dy), color: idC, current: idC });
  }
  ts.push({ name: 'colorTurn', idx: i => i, color: negC, current: negC });
  return ts;
}

// Replay the transformed stones in the base's index order (known legal for the
// base; a symmetry preserves legality and the no-capture property).
function buildG3(srcCells, srcCurrent, t) {
  const g = new Game3(N);
  let placed = 0;
  for (let i = 0; i < cap; i++) {
    if (srcCells[i] === 0) continue;
    const ti = t.idx(i);
    g.current = t.color(srcCells[i]);
    if (!g.isLegal(ti)) throw new Error(`buildG3: illegal replay of ${coordStr(ti, N)} under ${t.name}`);
    g.play(ti);
    placed++;
    if (g.emptyCount !== cap - placed) throw new Error(`buildG3: capture during replay under ${t.name}`);
  }
  g.current = t.current(srcCurrent);
  return g;
}

// Per-group verdict map, keyed by the group's minimum stone index.
function verdictMap(g) {
  const seen = new Set(), m = new Map();
  for (let i = 0; i < cap; i++) {
    if (g.cells[i] === 0) continue;
    const gid = g._gid[i];
    if (seen.has(gid)) continue;
    seen.add(gid);
    const lc = g.groupLibs2(i).count;
    if (lc === 0 || lc > 2) continue;
    const stones = [];
    let mn = Infinity;
    for (let j = 0; j < cap; j++) if (g.cells[j] !== 0 && g._gid[j] === gid) { stones.push(j); if (j < mn) mn = j; }
    const st = getLadderStatus(g, i);
    m.set(mn, { stones: new Set(stones), color: g.cells[i], ms: st.moverSucceeds, urgent: new Set(st.urgentLibs) });
  }
  return m;
}

const setEq = (a, b) => a.size === b.size && [...a].every(x => b.has(x));
const mapSet = (s, fn) => new Set([...s].map(fn));

// ── urgentLib consistency ─────────────────────────────────────────────────────
// An urgentLib secures the mover's outcome (after it, the group's fate is
// locked).  So after the mover plays one, the opponent is on the losing side of
// a decided position and MUST read as a non-urgent loser: urgentLibs = [].  We
// play a RANDOM urgentLib and re-read; a non-empty opponent urgentLib set means
// it can reverse the just-secured outcome — a self-inconsistency.  (A single
// move suffices: the chain either escapes, dies, or settles with the opponent
// locked out — there is no multi-move urgentLib chain to follow.)
const RUN_PV = !opts['no-pv'];

// Play one random urgentLib (play/undo, restored on return).
// Returns { moved, result: 'escape'|'die'|'settled', failure|null }.
function pvWalk(g, idx) {
  const st = getLadderStatus(g, idx);
  if (!st || st.urgentLibs.length === 0) return { moved: 0, result: 'settled', failure: null };
  const u = st.urgentLibs;
  g.play(u[(rng.random() * u.length) | 0]);
  let result = 'settled', failure = null;
  if (g.cells[idx] === 0) result = 'die';
  else if (g.groupLibs2(idx).count >= 3) result = 'escape';
  else {
    const st2 = getLadderStatus(g, idx);
    if (st2 && st2.urgentLibs.length > 0) {
      failure = { cells: Int8Array.from(g.cells), idx, urgent: st2.urgentLibs.slice() };
    }
  }
  g.undo();
  return { moved: 1, result, failure };
}

function renderCells(cells, marks) {
  const mk = new Set(marks || []);
  const rows = [];
  for (let y = N - 1; y >= 0; y--) {
    let row = '';
    for (let x = 0; x < N; x++) {
      const i = y * N + x;
      const ch = cells[i] > 0 ? '●' : cells[i] < 0 ? '○' : '·';
      row += (mk.has(i) ? `(${ch})` : ` ${ch} `);
    }
    rows.push(row);
  }
  return rows.join('\n');
}

// ── Run ───────────────────────────────────────────────────────────────────────
console.log(`ladder2 invariance audit (JS)`);
console.log(`games=${GAMES}  size=${N}  seed=${SEED}  shifts/pos=${SHIFTS}`);
console.log();

let positions = 0, groupChecks = 0, mismatches = 0, reported = 0;
let pvWalks = 0, pvEscape = 0, pvDie = 0, pvSettled = 0, pvFailures = 0;

outer:
for (let g = 0; g < GAMES; g++) {
  const g2 = new Game2(N);
  let mv = 0;
  while (!g2.gameOver && mv < cap * 4) {
    const srcCells = Int8Array.from(g2.cells);
    const srcCurrent = g2.current;
    const base = buildG3(srcCells, srcCurrent, IDENTITY);
    const baseMap = verdictMap(base);
    if (baseMap.size > 0) {
      positions++;
      for (const t of transformsFor()) {
        const tg = buildG3(srcCells, srcCurrent, t);
        const tMap = verdictMap(tg);
        if (tMap.size !== baseMap.size) {
          report(t, 'group count changed', srcCells, null, baseMap.size, tMap.size);
          continue;
        }
        for (const [, v] of baseMap) {
          groupChecks++;
          const expStones = mapSet(v.stones, t.idx);
          const expMin = Math.min(...expStones);
          const expColor = t.color(v.color);
          const expUrgent = mapSet(v.urgent, t.idx);
          const tv = tMap.get(expMin);
          let bad = null;
          if (!tv)                                bad = 'no corresponding group';
          else if (!setEq(tv.stones, expStones))  bad = 'stone set differs';
          else if (tv.color !== expColor)         bad = `color ${tv.color} != ${expColor}`;
          else if (tv.ms !== v.ms)                bad = `moverSucceeds ${tv.ms} != ${v.ms}`;
          else if (!setEq(tv.urgent, expUrgent))  bad = 'urgent-move set differs';
          if (bad) {
            report(t, bad, srcCells, [...v.stones], v, tv);
            if (reported >= MAX_REPORT) break outer;
          }
        }
      }
      // Forward urgent-move consistency for each group (base restored after each).
      if (RUN_PV) {
        for (const [mn] of baseMap) {
          const { moved, result, failure } = pvWalk(base, mn);
          if (moved === 0) continue;              // chain had no urgentLibs
          pvWalks++;
          if (result === 'escape') pvEscape++;
          else if (result === 'die') pvDie++;
          else pvSettled++;
          if (failure) { pvReport(failure); if (reported >= MAX_REPORT) break outer; }
        }
      }
    }
    g2.play(g2.randomLegalMove(rng));
    mv++;
  }
}

function pvReport(f) {
  pvFailures++;
  if (reported >= MAX_REPORT) return;
  reported++;
  let stones = 0;
  for (let i = 0; i < cap; i++) if (f.cells[i] !== 0) stones++;
  const fullness = (stones / cap).toFixed(2);
  console.log(`── PV-INCONSISTENCY: after one urgentLib the opponent still has urgentLibs {${f.urgent.map(m => coordStr(m, N)).join(',')}} — it can reverse the secured outcome  (group@${coordStr(f.idx, N)}, fullness=${fullness})`);
  console.log(renderCells(f.cells, [f.idx]));
  console.log();
}

function report(t, reason, srcCells, baseStones, a, b) {
  mismatches++;
  if (reported >= MAX_REPORT) return;
  reported++;
  console.log(`── MISMATCH under ${t.name}: ${reason}`);
  if (baseStones) {
    console.log(`   base group: {${baseStones.map(s => coordStr(s, N)).join(' ')}}  ` +
      `ms=${a.ms} urgent={${[...a.urgent].map(s => coordStr(s, N)).join(' ')}}`);
    if (b) console.log(`   transformed: ms=${b.ms} urgent={${[...b.urgent].map(s => coordStr(s, N)).join(' ')}}`);
    console.log(renderCells(srcCells, baseStones));
  } else {
    console.log(`   base groups=${a}  transformed groups=${b}`);
    console.log(renderCells(srcCells));
  }
  console.log();
}

console.log();
console.log(`transform invariance : positions=${positions}  group-checks=${groupChecks}  mismatches=${mismatches}`);
if (RUN_PV) {
  console.log(`urgentLib consistency: walks=${pvWalks}  (escape=${pvEscape} die=${pvDie} settled=${pvSettled})  failures=${pvFailures}`);
}
const total = mismatches + pvFailures;
if (total > MAX_REPORT) console.log(`(showed first ${MAX_REPORT} of ${total})`);
process.exit(total > 0 ? 1 : 0);
