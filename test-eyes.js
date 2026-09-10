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
const { game3FromGame2 } = require('./game3.js');

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
  // The four orthogonals of the centre belong to ONE white chain (connected the
  // long way round), and two of the diagonals are hostile.  That fails the
  // current test (fewer than three friendly diagonals) AND the standard rule
  // (more than one hostile diagonal), so the point can only be recognised via
  // the same-chain shortcut — which is right, because a wall that is already
  // one chain cannot be cut, whatever sits on the diagonals.
  const g = parseBoard(`
    . . . . . . . . .
    . O O O O . . . .
    . O . . O . . . .
    . O . X O O . . .
    . O O O . O . . .
    . . . O O X . . .
    . . . . . . . . .
    . . . . . . . . .
    . . . . . . . . .`, WHITE);
  const c = centre(g);
  const cells = g.cells, nbr = g._nbr, dnbr = g._dnbr, gid = g._gid;
  const orth = [0, 1, 2, 3].map(d => nbr[c * 4 + d]);
  let hostileDiag = 0, friendlyDiag = 0;
  for (let d = 0; d < 4; d++) {
    const v = cells[dnbr[c * 4 + d]];
    if (v === BLACK) hostileDiag++; else if (v === WHITE) friendlyDiag++;
  }
  check(orth.every(i => cells[i] === WHITE), 'all four orthogonals are friendly');
  check(new Set(orth.map(i => gid[i])).size === 1, 'and they are a single chain');
  check(hostileDiag >= 2, 'two hostile diagonals — fails the standard rule too');
  check(friendlyDiag < 3, 'and fewer than three friendly diagonals');
  check([at(g, 3, 3), at(g, 5, 5)].every(i => g._ls[gid[i]] > 0),
        'the hostile diagonal stones are alive (not a captured-stone artefact)');
  check(g.isTrueEye(c) === true,
        'same-chain wall is an eye regardless of the diagonals');
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
  check(g.isTrueEye(c) === true,
        'multi-chain wall with one hostile diagonal is an eye');
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

// ── game3 agrees with game2 ───────────────────────────────────────────────────
// game3 carries its own copy of the rule for the tactical passes, so the two
// must decide every point identically or a ladder read and a playout disagree
// about what is fillable.

section('game3 parity');
{
  // Every board used above, every empty point, both colours to move.
  const boards = [
    `. . . . . . . . .
     . O O O O . . . .
     . O . . O . . . .
     . O . X O O . . .
     . O O O . O . . .
     . . . O O X . . .
     . . . . . . . . .
     . . . . . . . . .
     . . . . . . . . .`,
    `. . . . .
     . . O O .
     . O . O .
     . O O X .
     . . . . .`,
    `. . . . .
     . X O X .
     . O . O .
     . X O X .
     . . . . .`,
    `. . . . .
     . X X X .
     . X O X .
     . X . X .
     . . . . .`,
  ];
  let compared = 0, disagreed = 0;
  for (const b of boards) {
    const g2 = parseBoard(b, BLACK);
    const g3 = game3FromGame2(g2);
    for (let i = 0; i < g2.N * g2.N; i++) {
      if (g2.cells[i] !== 0) continue;
      for (const col of [BLACK, WHITE]) {
        g2.current = col; g3.current = col;
        compared++;
        if (g2.isTrueEye(i) !== g3.isTrueEye(i)) disagreed++;
      }
    }
  }
  check(compared > 0, `compared ${compared} (point, colour) pairs`);
  check(disagreed === 0, `game3 matches game2 everywhere (${disagreed} disagreements)`);
}

// ── Summary ───────────────────────────────────────────────────────────────────

console.log(`\n${pass} passed, ${fail} failed` + (known ? `, ${known} known bug(s)` : ''));
if (fail > 0) process.exit(1);
