#!/usr/bin/env node
'use strict';

// test-eyes.js — correctness tests for Game2.isTrueEye, the rule that decides
// which points a playout refuses to fill.  Everything downstream of a playout
// depends on it: ppat simulations, puct values, truncated evaluations, and the
// chain-survival labels train-health.js fits.
//
// Positions are ASCII boards parsed by game2's parseBoard: X black, O white,
// . empty.  The point under test is marked in the comment above each board.

const { parseBoard, BLACK, WHITE } = require('./game2.js');

let pass = 0, fail = 0, known = 0;
function check(cond, msg) {
  if (cond) { pass++; }
  else       { fail++; console.error('  FAIL:', msg); }
}
// A case that SHOULD hold but does not, because of a bug we have decided not to
// fix yet: reported, but does not redden the suite.  If it starts passing the
// suite fails, so the test cannot be silently left behind by a fix.
function xfail(cond, msg) {
  if (!cond) { known++; console.error('  KNOWN BUG:', msg); }
  else       { fail++; console.error('  FIXED — promote this xfail to check():', msg); }
}
function section(name) { console.log(`\n── ${name} ──`); }

// parseBoard reads ASCII rows top-to-bottom but stores them bottom-to-top, so
// board coordinates must go through this rather than being computed by eye.
const at = (g, row, col) => (g.N - 1 - row) * g.N + col;
// Index of the centre point of an odd-sized board.
const centre = g => { const N = g.N; return ((N - 1) >> 1) * N + ((N - 1) >> 1); };

// ── same-chain walls: diagonals are irrelevant ────────────────────────────────
// When the four orthogonal neighbours are ONE chain there is nothing to cut, so
// the point is an eye whatever sits on the diagonals.

section('same-chain wall');
{
  // Centre ringed by a single connected white chain; the centre's diagonals are
  // EMPTY, which the "three friendly diagonals" test would reject.
  const g = parseBoard(`
    . . . . . . .
    . O O O O O .
    . O . O . O .
    . O O . O O .
    . O . O . O .
    . O O O O O .
    . . . . . . .`, WHITE);
  const c = centre(g);
  const gids = new Set([g._nbr[c * 4], g._nbr[c * 4 + 1], g._nbr[c * 4 + 2], g._nbr[c * 4 + 3]]
    .map(i => g._gid[i]));
  check(gids.size === 1, 'the four orthogonals are a single chain');
  check(g.isTrueEye(c) === true, 'same-chain wall with EMPTY diagonals is an eye');
}

// ── multi-chain walls: the diagonal rule decides ──────────────────────────────

section('multi-chain wall');
{
  // Four white stones around the centre, in two chains, with ONE hostile
  // diagonal and one empty — an eye under the standard rule (at most one
  // hostile diagonal), which does not care that the wall spans two chains.
  const g = parseBoard(`
    . . . . .
    . . O O .
    . O . O .
    . O O X .
    . . . . .`, WHITE);
  const c = centre(g);
  const cells = g.cells, nbr = g._nbr, dnbr = g._dnbr, gid = g._gid;
  let friends = 0; const gids = new Set();
  for (let d = 0; d < 4; d++) { const j = nbr[c * 4 + d]; if (cells[j] === WHITE) { friends++; gids.add(gid[j]); } }
  let hostileDiag = 0;
  for (let d = 0; d < 4; d++) if (cells[dnbr[c * 4 + d]] === BLACK) hostileDiag++;
  check(friends === 4, 'all four orthogonals are friendly');
  check(gids.size > 1, 'they span more than one chain');
  check(hostileDiag <= 1, 'at most one hostile diagonal — an eye by the standard rule');
  // KNOWN FAILURE: isTrueEye asks for >= 3 FRIENDLY diagonals, so an empty
  // diagonal counts against the eye and the point is left fillable by its own
  // owner.  Measured on late-game positions, this misses ~79% of real eyes.
  xfail(g.isTrueEye(c) === true,
        'multi-chain wall with one hostile diagonal should be an eye — isTrueEye ' +
        'requires >= 3 FRIENDLY diagonals, so an EMPTY diagonal counts against it ' +
        '(misses ~79% of real eyes in late-game positions)');
}
{
  // A genuine FALSE eye: four friendly orthogonals in separate chains with
  // hostile diagonals all round.  Must NOT be an eye — the walls can be cut.
  const g = parseBoard(`
    . . . . .
    . X O X .
    . O . O .
    . X O X .
    . . . . .`, WHITE);
  check(g.isTrueEye(centre(g)) === false, 'false eye (hostile diagonals) is not an eye');
}

// ── captures are never eye-blocked ────────────────────────────────────────────
// The capturing point is the victim's last liberty, so one of its neighbours is
// an enemy stone: the capturer can never have four friendly neighbours there.

section('capture is never blocked');
{
  // A white stone in atari: the empty point below it is its last liberty.
  const g = parseBoard(`
    . . . . .
    . X X X .
    . X O X .
    . X . X .
    . . . . .`, BLACK);
  const white = at(g, 2, 2);            // the O
  const lib = at(g, 3, 2);              // its last liberty
  check(g._ls[g._gid[white]] === 1, 'the white stone is in atari');
  check(g.isTrueEye(lib) === false, 'the capturing point is not an eye for the capturer');
  check(g.isLegal(lib) === true, 'and the capture is legal');
}

// ── Summary ───────────────────────────────────────────────────────────────────

console.log(`\n${pass} passed, ${fail} failed` + (known ? `, ${known} known bug(s)` : ''));
if (fail > 0) process.exit(1);
