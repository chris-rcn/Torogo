#!/usr/bin/env node
'use strict';

// test-ladder2.js — correctness tests for ladder2.js (getLadderStatus),
// including the defender's capture of an adjacent atari'd enemy chain.

const { Game2, BLACK, WHITE, PASS, parseBoard, parseMove, coordStr } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');
const { getLadderStatus } = require('./ladder2.js');
const { makeRng } = require('./xorshift.js');

let pass = 0, fail = 0;
const check = (cond, msg) => { if (cond) pass++; else { fail++; console.error('  FAIL:', msg); } };
const eqSet = (a, b) => a.length === b.length && a.every(x => b.includes(x));

// ── 1. Atari group: extend / capture by the last liberty ──────────────────────
// Black d4 in atari (only liberty d3).
{
  const board = `
. . . . . . .
. . . . . . .
. . . O . . .
. . O X O . .
. . . . . . .
. . . . . . .
. . . . . . .`;
  const d4 = parseMove('d4', 7), d3 = parseMove('d3', 7);
  const atk = getLadderStatus(game3FromGame2(parseBoard(board, WHITE)), d4);   // white attacker to move
  check(atk.moverSucceeds && eqSet(atk.urgentLibs, [d3]), 'atari + attacker to move: captures at last liberty d3');
  const def = getLadderStatus(game3FromGame2(parseBoard(board, BLACK)), d4);   // black defender to move
  check(def.moverSucceeds && eqSet(def.urgentLibs, [d3]), 'atari + defender to move: escapes by extending d3');
}

// ── 2. NEW: defender saves the group by CAPTURING an adjacent atari'd enemy ────
// A saving move (urgentLib) that is NOT one of the group's own liberties must be
// a capture: it removes >=1 enemy stone and the chased group survives.
{
  const N = 13, rng = makeRng(123);
  let found = null, scanned = 0;
  search:
  while (scanned < 2000000) {
    const game = new Game2(N); let moves = 0;
    while (!game.gameOver && moves < N * N * 2) {
      scanned++;
      const seen = new Set();
      for (let i = 0; i < N * N; i++) {
        if (game.cells[i] === 0 || game.cells[i] !== game.current) continue;   // defender to move
        const gid = game._gid[i]; if (seen.has(gid)) continue; seen.add(gid);
        const lc = game.groupLibertyCount(gid); if (lc < 1 || lc > 2) continue;
        const st = getLadderStatus(game3FromGame2(game), i);
        if (!st || !st.moverSucceeds) continue;
        const caps = st.urgentLibs.filter(m => !st.libs.includes(m));   // saving moves that aren't liberties
        if (caps.length) { found = { game, idx: i, st, cap: caps[0] }; break search; }
      }
      const mv = game.randomLegalMove(rng);
      if (mv === PASS) game.play(PASS); else if (!game.play(mv)) break;
      moves++;
    }
  }
  check(found !== null, `defender capture-escape is generated (searched ${scanned} positions)`);
  if (found) {
    const { game, idx, st, cap } = found;
    check(!st.libs.includes(cap), 'the capture saving move is not one of the group liberties');
    const g3 = game3FromGame2(game), before = g3.emptyCount;
    check(g3.play(cap), 'the capture saving move is legal');
    check(g3.emptyCount >= before, 'the capture saving move removes >=1 enemy stone');   // a plain fill would drop emptyCount by 1
    check(g3.cells[idx] !== 0, 'the chased group survives the capture');
    console.log(`  capture case: group ${coordStr(idx, N)} (${game.cells[idx] === BLACK ? 'B' : 'W'}), libs=${st.libs.map(m => coordStr(m, N)).join(',')}, capture move=${coordStr(cap, N)}`);
  }
}

// ── 3. Invariants over random positions ───────────────────────────────────────
{
  const N = 13, rng = makeRng(99);
  let nChecked = 0, bad = 0;
  for (let g = 0; g < 30 && nChecked < 30000; g++) {
    const game = new Game2(N); let moves = 0;
    while (!game.gameOver && moves < N * N * 2) {
      const seen = new Set();
      for (let i = 0; i < N * N; i++) {
        if (game.cells[i] === 0) continue;
        const gid = game._gid[i]; if (seen.has(gid)) continue; seen.add(gid);
        const lc = game.groupLibertyCount(gid); if (lc < 1 || lc > 2) continue;
        const st = getLadderStatus(game3FromGame2(game), i);
        if (!st) { bad++; continue; }
        nChecked++;
        if (typeof st.moverSucceeds !== 'boolean') bad++;
        if (st.libs.some(m => game.cells[m] !== 0)) bad++;          // liberties must be empty
        if (st.urgentLibs.some(m => game.cells[m] !== 0)) bad++;    // saving/killing moves must be empty points
      }
      const mv = game.randomLegalMove(rng);
      if (mv === PASS) game.play(PASS); else if (!game.play(mv)) break;
      moves++;
    }
  }
  check(bad === 0, `invariants hold over ${nChecked} group reads (${bad} violations)`);
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
