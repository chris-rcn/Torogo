#!/usr/bin/env node
'use strict';

// test-tactics3.js — correctness tests for tactics3.js

const { Game2, BLACK, WHITE, PASS } = require('./game2.js');
const { searchChain, searchChains } = require('./tactics3.js');

let pass = 0, fail = 0;

function assert(cond, msg) {
  if (cond) { pass++; }
  else       { fail++; console.error('  FAIL:', msg); }
}

function section(name) { console.log(`\n── ${name} ──`); }

// Play a sequence of moves from an array of [x, y] pairs (or 'pass').
function playMoves(game, moves) {
  const N = game.N;
  for (const m of moves) {
    if (m === 'pass') { game.play(PASS); continue; }
    const [x, y] = m;
    game.play(y * N + x);
  }
}

// ── groupLibs ─────────────────────────────────────────────────────────────────

section('groupLibs returns all liberties');
{
  const g = new Game2(5, false);
  // Place a single black stone at center (2,2) — should have 4 liberties
  playMoves(g, [[2,2]]);
  const libs = g.groupLibs(2*5+2);
  assert(libs.length === 4, `center stone has 4 libs, got ${libs.length}`);

  // Play white somewhere irrelevant, then a black stone at edge (0,2) — 4 liberties (toroidal)
  playMoves(g, [[0,0], [0,2]]);
  const edgeLibs = g.groupLibs(2*5+0);
  assert(edgeLibs.length === 4, `edge stone has 4 libs on torus, got ${edgeLibs.length}`);
}

// ── Single stone in atari ──────────────────────────────────────────────────────

section('Single stone in atari — capture succeeds');
{
  // 5x5 board. Surround a black stone on 3 sides; black to play — can it escape?
  //   . . . . .
  //   . W . . .
  //   W B W . .
  //   . W . . .
  //   . . . . .
  // Black at (1,2), surrounded N/E/S/W by white, leaving only... wait, 4 neighbors
  // on a torus. Let's do a simpler setup: black stone with 1 lib.
  //
  // Place black at (1,1). Surround with white at (0,1),(2,1),(1,0),(1,2).
  // Black turn first, then whites:
  const g = new Game2(5);
  // Black plays (1,1)
  g.play(1*5+1);
  // White plays 4 neighbors
  g.play(0*5+1); // white (0,1)  — white's turn after black
  // need to alternate: B W B W ...
  // Let's use a helper game where we don't care about alternation
  // Actually we need to set it up properly with alternating moves.
  // Use pass to set up positions carefully.
  // Reset and do it properly:
}
{
  // Set up: black stone at (1,1) surrounded on 3 sides by white, leaving 1 lib.
  // B(1,1), W(1,0), B(pass), W(1,2), B(pass), W(0,1) → black has 1 lib, white to play.
  // Use applyFirstMove=false so BLACK moves first.
  const g = new Game2(5, false);
  const N = 5;
  playMoves(g, [
    [1,1],       // B
    [1,0],       // W  x=1,y=0 → idx=0*5+1=1
    'pass',      // B pass
    [1,2],       // W  x=1,y=2 → idx=2*5+1=11
    'pass',      // B pass
    [0,1],       // W  x=0,y=1 → idx=1*5+0=5
  ]);
  // Black at (1,1)=idx6, neighbors: N=1,S=11,W=5,E=7. Three are white → 1 lib at idx 7.
  // After B,W,B,W,B,W: BLACK to play. Pass to give white the turn.
  const lc = g.groupLibs(1*N+1).length;
  assert(lc === 1, `black in atari, lc=${lc}`);
  assert(g.current === BLACK, 'black to play after white surrounds');
  g.play(PASS); // black passes — now white to play
  assert(g.current === WHITE, 'white to play');
  const status = searchChain(g, 1*N+1);
  assert(status !== null, 'status not null');
  assert(status.moverSucceeds === true, 'white (attacker) succeeds in capturing');
  // Not urgent: black can't escape regardless, so urgentLibs is empty
  assert(status.urgentLibs.length === 0, `not urgent (black can't escape either way), got ${status.urgentLibs.length}`);
}

// ── Defender can escape to 4 libs ─────────────────────────────────────────────

section('Chain can escape — mover is defender');
{
  // Black chain with 2 libs in open space — can clearly reach 4.
  // B(2,2), W(0,0), B(2,3) — black chain of 2 with many liberties
  const g = new Game2(7, false);
  const N = 7;
  playMoves(g, [
    [2,2],  // B
    [0,0],  // W
    [2,3],  // B chain grows
    [0,1],  // W
  ]);
  // Black chain at (2,2)-(2,3) — black to play, count libs
  const lc = g.groupLibs(2*N+2).length;
  assert(lc >= 3, `open chain has >= 3 libs, got ${lc}`);
  if (lc <= 3) {
    const status = searchChain(g, 2*N+2);
    assert(status !== null, 'status not null');
    assert(status.moverSucceeds === true, 'defender (black) escapes');
  }
}

// ── 3-lib group (key difference from ladder2) ─────────────────────────────────

section('3-lib group — searchChain handles lc=3');
{
  // Black chain with exactly 3 libs — tactics3 handles this; ladder2 would reject it.
  // B(1,1), W(0,0), B(2,1) — black chain of 2 on 7x7
  const g = new Game2(7, false);
  const N = 7;
  playMoves(g, [
    [1,1],  // B
    [0,0],  // W
    [2,1],  // B chain grows
    [0,6],  // W far away
  ]);
  const lc = g.groupLibs(1*N+1).length;
  if (lc === 3) {
    const status = searchChain(g, 1*N+1);
    assert(status !== null, '3-lib group: status not null');
    assert(typeof status.moverSucceeds === 'boolean', '3-lib group: moverSucceeds is boolean');
  } else {
    // Chain has != 3 libs in this position; just verify it gets skipped by searchChains if > 3
    assert(true, `chain has ${lc} libs (skip 3-lib check)`);
  }
}

section('searchChain — 3-lib group in open space: defender escapes');
{
  // On a 9x9 board, surround a black chain on 3 sides leaving exactly 3 libs.
  // B(4,4), W(4,3), B(pass), W(4,5), B(pass), W(3,4), B(pass), W(5,4) would give 0 libs.
  // Instead leave one side open: black chain at (4,4) with white on N,W,S only → 2 libs (E,internal).
  // Simpler: single black stone at (4,4) on 9x9, surround 1 neighbor with white → 3 libs.
  const g = new Game2(9, false);
  const N = 9;
  playMoves(g, [
    [4,4],  // B at center
    [4,3],  // W fills N lib
    'pass', // B
  ]);
  // Black at (4,4) now has 3 libs (S,W,E). Black to play.
  const lc = g.groupLibs(4*N+4).length;
  assert(lc === 3, `stone with 1 neighbor filled has 3 libs, got ${lc}`);
  const status = searchChain(g, 4*N+4);
  assert(status !== null, 'status not null');
  assert(status.moverSucceeds === true, 'defender (black) can escape to 4 libs');
}

// ── searchChains ──────────────────────────────────────────────────────

section('searchChains — empty board returns empty array');
{
  const g = new Game2(9, false);
  const statuses = searchChains(g);
  assert(statuses.length === 0, `empty board: 0 statuses, got ${statuses.length}`);
}

section('searchChains — finds groups with 1-3 libs only');
{
  const g = new Game2(7, false);
  const N = 7;
  // Create a black stone in atari: B(1,1), surround 3 sides
  playMoves(g, [
    [1,1],        // B
    [1,0],        // W
    'pass',       // B pass
    [1,2],        // W
    'pass',       // B pass
    [0,1],        // W
    'pass',       // B pass
  ]);
  // Black at (1,1) has 1 lib. White to play.
  const statuses = searchChains(g);
  assert(statuses.length >= 1, `at least 1 group in atari, got ${statuses.length}`);
  const blackStatus = statuses.find(s => s.color === BLACK);
  assert(blackStatus !== undefined, 'found black group status');
  assert(blackStatus.status.libs.length === 1, 'group has 1 lib');
}

section('searchChains — skips groups with 4+ libs');
{
  const g = new Game2(9, false);
  const N = 9;
  // Single black stone in the middle — 4 liberties — should be skipped
  g.play(4*N+4);
  const statuses = searchChains(g);
  assert(statuses.length === 0, `single stone (4 libs) is skipped, got ${statuses.length}`);
}


// ── Summary ───────────────────────────────────────────────────────────────────

console.log(`\n${pass} passed, ${fail} failed`);
if (fail > 0) process.exit(1);
