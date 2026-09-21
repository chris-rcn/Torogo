'use strict';

// Tests for symmetry.js.  The oracle is an exhaustive brute force over the whole
// toroidal group (8 point ops x N^2 translations), checking EVERY cell (not just
// stones) — so it independently validates both the library's stone-matching
// shortcut and its orbit dedup.

const { parseBoard, PASS, EMPTY } = require('./game2.js');
const Symmetry = require('./symmetry.js');

let passed = 0, failed = 0;
function section(name) { console.log(`\n── ${name} ──`); }
function check(cond, msg) { if (cond) { passed++; } else { failed++; console.log(`  FAIL: ${msg}`); } }

// ── Brute-force oracle ───────────────────────────────────────────────────────
const NEG = (x, N) => (N - x) % N;
const T = [
  (r, c, N) => [r, c],
  (r, c, N) => [r, NEG(c, N)],
  (r, c, N) => [NEG(r, N), c],
  (r, c, N) => [NEG(r, N), NEG(c, N)],
  (r, c, N) => [c, r],
  (r, c, N) => [c, NEG(r, N)],
  (r, c, N) => [NEG(c, N), r],
  (r, c, N) => [NEG(c, N), NEG(r, N)],
];
// Return the stabiliser as an array of permutations (Int32Array cell -> image).
function bruteStab(game) {
  const N = game.N, area = N * N, cells = game.cells;
  const ko = game.ko, hasKo = ko !== PASS && ko >= 0 && ko < area;
  const perms = [];
  for (let op = 0; op < 8; op++) {
    for (let tr = 0; tr < N; tr++) for (let tc = 0; tc < N; tc++) {
      const perm = new Int32Array(area);
      let ok = true;
      for (let i = 0; i < area && ok; i++) {
        const rc = T[op]((i / N) | 0, i % N, N);
        const img = ((rc[0] + tr) % N) * N + (rc[1] + tc) % N;
        perm[i] = img;
        if (cells[img] !== cells[i]) ok = false;   // checks empties too
      }
      if (ok && hasKo) {
        const rc = T[op]((ko / N) | 0, ko % N, N);
        if (((rc[0] + tr) % N) * N + (rc[1] + tc) % N !== ko) ok = false;
      }
      if (ok) perms.push(perm);
    }
  }
  return perms;
}
function bruteCanonical(perms, cell) {
  let best = cell;
  for (const p of perms) if (p[cell] < best) best = p[cell];
  return best;
}
function emptyCells(game) {
  const out = []; for (let i = 0; i < game.N * game.N; i++) if (game.cells[i] === EMPTY) out.push(i);
  return out;
}

// Assert the library agrees with the oracle on a board, and optionally that the
// oracle's stabiliser has an expected size (documents the board).
function agrees(label, game, expectedSize) {
  const perms = bruteStab(game);
  const sym = Symmetry.of(game);
  const area = game.N * game.N;
  // size
  check(sym.ops().length === perms.length,
    `${label}: stabiliser size ${sym.ops().length} vs brute ${perms.length}`);
  if (expectedSize !== undefined)
    check(perms.length === expectedSize, `${label}: brute size ${perms.length}, expected ${expectedSize}`);
  // hasSymmetry
  check(sym.hasSymmetry() === (perms.length > 1), `${label}: hasSymmetry`);
  // canonical over every cell (pins the orbit structure)
  let canonOk = true;
  for (let i = 0; i < area; i++) if (sym.canonical(i) !== bruteCanonical(perms, i)) canonOk = false;
  check(canonOk, `${label}: canonical matches oracle for all cells`);
  // distinctMoves over the empty cells == number of distinct orbits
  const empties = emptyCells(game);
  const dm = sym.distinctMoves(empties);
  const orbits = new Set(empties.map(c => bruteCanonical(perms, c)));
  check(dm.length === orbits.size, `${label}: distinctMoves ${dm.length} vs orbit count ${orbits.size}`);
  // every kept move is a legal candidate and one per orbit
  const keptOrbits = new Set(dm.map(c => bruteCanonical(perms, c)));
  check(keptOrbits.size === dm.length && keptOrbits.size === orbits.size,
    `${label}: distinctMoves picks exactly one per orbit`);
  return { perms, sym, dm, empties };
}

// ── Single centre stone: 8-fold, and countB != countW (the opening) ──────────
section('single centre stone (opening; countB != countW)');
{
  const g = parseBoard(`
    . . . . .
    . . . . .
    . . X . .
    . . . . .
    . . . . .`);
  const r = agrees('centre stone', g, 8);
  // this is exactly the count-mismatch case: 1 black, 0 white, yet symmetric
  check(r.sym.hasSymmetry(), 'centre stone: symmetric despite countB(1) != countW(0)');
}

// ── Off-centre mirror (the user's example): translated half-integer axis ─────
section('off-centre mirror (translated axis)');
{
  const g = parseBoard(`
    . . . . .
    . O X . .
    . O X . .
    . . . . .
    . . . . .`);
  const r = agrees('off-centre mirror', g, 2);
  check(r.dm.length < r.empties.length, 'off-centre mirror: some moves pruned');
}

// ── Two equal stones: 180 about their midpoint + the two diagonal mirrors ─────
// (a translated Klein-four stabiliser, size 4 — includes a 180 about a
// translated centre, the case grid-centre-only D4 would miss).
section('two equal stones (translated Klein four)');
{
  const g = parseBoard(`
    X . . . .
    . . . . .
    . . X . .
    . . . . .
    . . . . .`);
  agrees('translated 180', g, 4);
}

// ── Asymmetric control ───────────────────────────────────────────────────────
section('asymmetric board');
{
  const g = parseBoard(`
    . X . . .
    X . . . .
    . . . O .
    . . . . .
    . . . . .`);
  const r = agrees('asymmetric', g, 1);
  check(!r.sym.hasSymmetry(), 'asymmetric: hasSymmetry is false');
  check(r.dm.length === r.empties.length, 'asymmetric: distinctMoves returns all moves');
}

// ── ko point must be fixed ───────────────────────────────────────────────────
section('ko point fixing');
{
  // A mirror-symmetric board (order-2 stabiliser).  Derive a fixed cell and a
  // moved cell of that mirror straight from the oracle, so we don't have to
  // reason about parseBoard's row flip.
  const base = `
    . . . . .
    . O X . .
    . O X . .
    . . . . .
    . . . . .`;
  const gBase = parseBoard(base);
  const mirror = bruteStab(gBase).find(p => p.some((img, i) => img !== i));
  const empt = emptyCells(gBase);
  const fixedCell = empt.find(c => mirror[c] === c);   // the mirror fixes it
  const movedCell = empt.find(c => mirror[c] !== c);   // the mirror moves it
  {
    const g = parseBoard(base); g.ko = fixedCell;
    const r = agrees('ko on axis', g);
    check(r.perms.length === 2, 'ko on a fixed cell keeps the mirror (size 2)');
  }
  {
    const g = parseBoard(base); g.ko = movedCell;
    const r = agrees('ko off axis', g);
    check(r.perms.length === 1, 'ko on a moved cell breaks the mirror (size 1)');
  }
}

// ── PASS passes through distinctMoves ────────────────────────────────────────
section('PASS handling');
{
  const g = parseBoard(`
    . . . . .
    . . . . .
    . . X . .
    . . . . .
    . . . . .`);
  const sym = Symmetry.of(g);
  const withPass = sym.distinctMoves([...emptyCells(g), PASS]);
  check(withPass.includes(PASS), 'PASS is retained');
  check(withPass.filter(m => m === PASS).length === 1, 'PASS retained exactly once');
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
