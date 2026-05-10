'use strict';

// ladder2.js — Ladder detection using Game2.

(function() {

const { PASS } = typeof require === 'function' ? require('./game2.js') : window.Game2;

// Returns true when the group at stoneIdx can reach 3+ liberties despite best
// attacker play.
function _canReach3Libs(game2, idx) {
  const { count: lc, lib0, lib1 } = game2.groupLibs2(idx);
  if (lc >= 3) return true;
  if (lc === 0) return false;

  const defColor = game2.cells[idx];

  if (game2.current === defColor) {
    // Adjacent attacker in atari with liberty outside ours?
    const capMoves = _captureEscapes(game2, idx, defColor, lib0, lib1);
    // 2 libs: capturing doesn't consume any of our libs → 2 + freed ≥ 3.
    if (lc === 2 && capMoves.length > 0) return true;

    // Defender's turn: play one of the liberties; succeed if any leads to safety.
    if (lc === 1 && capMoves.length === 0) {
      // Only one move — play directly, no clone needed.
      if (!game2.play(lib0)) return false;
      if (game2.cells[idx] === 0) return false;
      return _canReach3Libs(game2, idx);
    }
    const libs = lc === 1 ? [lib0] : [lib0, lib1];
    for (const libIdx of libs) {
      const g = game2.clone();
      if (!g.play(libIdx)) continue;      // suicide — skip
      if (g.cells[idx] === 0) continue;   // captured — skip
      if (_canReach3Libs(g, idx)) return true;
    }
    // Also try capturing adjacent attacker groups.
    for (const cm of capMoves) {
      const g = game2.clone();
      if (!g.play(cm)) continue;
      if (g.cells[idx] === 0) continue;
      if (_canReach3Libs(g, idx)) return true;
    }
    return false;
  }

  // Attacker's turn: 1 lib = fill last liberty = capture (if legal).
  if (lc === 1) return !game2.isLegal(lib0);
  for (const libIdx of [lib0, lib1]) {
    const g = game2.clone();
    if (!g.play(libIdx)) continue;        // illegal for attacker — skip
    if (g.cells[idx] === 0) return false; // captured immediately
    const afterLc = g.groupLibs2(idx).count;
    if (afterLc === 0) return false;
    if (afterLc <= 2 && !_canReach3Libs(g, idx)) return false;
  }

  return true;
}

// Find liberties of adjacent attacker groups in atari, excluding the
// defender's own liberties.  These are moves the defender can play to
// capture an attacker group and potentially escape.
function _captureEscapes(game2, idx, defColor, dlib0, dlib1) {
  const atkColor = -defColor;
  const gid = game2._gid[idx];
  const W = game2._W;
  const sw = game2._sw;
  const nbr = game2._nbr;
  const cells = game2.cells;
  const gidArr = game2._gid;
  const ls = game2._ls;
  const cap = game2.N * game2.N;
  const gb = gid * W;
  const result = [];
  let seen0 = -1, seen1 = -1, seen2 = -1, seen3 = -1;
  for (let wi = 0; wi < W; wi++) {
    let w = sw[gb + wi];
    while (w) {
      const lsb = w & -w;
      const si = wi * 32 + (31 - Math.clz32(lsb));
      w ^= lsb;
      if (si >= cap) continue;
      for (let d = 0; d < 4; d++) {
        const ni = nbr[si * 4 + d];
        if (cells[ni] !== atkColor) continue;
        const agid = gidArr[ni];
        if (agid === seen0 || agid === seen1 || agid === seen2 || agid === seen3) continue;
        if      (seen0 === -1) seen0 = agid;
        else if (seen1 === -1) seen1 = agid;
        else if (seen2 === -1) seen2 = agid;
        else                   seen3 = agid;
        if (ls[agid] !== 1) continue;
        const { lib0: alib } = game2.groupLibs2(ni);
        if (alib !== dlib0 && alib !== dlib1) result.push(alib);
      }
    }
  }
  return result;
}

// getAllLadderStatuses(game2, minChainSize) — run getLadderStatus2 on every
// group with 1 or 2 liberties and return an array of { gid, color, status }
// objects, one per group (groups with 0 or 3+ liberties are skipped).
// minChainSize: skip groups smaller than this (default 1).
function getAllLadderStatuses(game2, minChainSize = 1) {
  const cap  = game2.N * game2.N;
  const results = [];
  const visited = new Set();
  for (let i = 0; i < cap; i++) {
    if (game2.cells[i] === 0) continue;
    const gid = game2._gid[i];
    if (visited.has(gid)) continue;
    visited.add(gid);
    if (game2.groupSize(gid) < minChainSize) continue;
    const { count: lc } = game2.groupLibs2(i);
    if (lc === 0 || lc > 2) continue;
    const status = getLadderStatus2(game2, i);
    results.push({ gid, color: game2.cells[i], status });
  }
  return results;
}

// Examines the group containing the stone at stoneIdx (must have 1 or 2
// liberties).  For each liberty, simulates both colours playing it first.
//
// Returns { libs: [], moverSucceeds: boolean, urgentLibs: [] }
//
// Logs a warning and returns null when the group has more than 2 liberties.
function getLadderStatus2(game2, stoneIdx) {
  const { count: lc, lib0, lib1 } = game2.groupLibs2(stoneIdx);
  if (lc < 1 || lc > 2) {
    const N = game2.N;
    console.warn(`getLadderStatus2: group at ${stoneIdx % N},${(stoneIdx / N) | 0} has ${lc} liberties (expected ≤ 2)`);
    return null;
  }
  const atari = lc === 1;
  const libs = atari ? [lib0] : [lib0, lib1];
  const gColor = game2.cells[stoneIdx];
  const mover = game2.current;   // BLACK or WHITE
  const defending = gColor === mover;

  let escape;
  // Try opponent playing first.
  if (defending && atari) {
    escape = false;
  } else {
    const g = game2.clone();
    g.play(PASS);
    escape = _canReach3Libs(g, stoneIdx);
  }
  if (defending === escape) {
    // group is not urgent
    return { libs, moverSucceeds: true, urgentLibs: [] };
  }

  // Try mover playing first.
  let moverSucceeds = false;
  let urgentLibs = [];
  for (const libIdx of libs) {
    if (!defending && atari) {
      escape = false;
    } else {
      const g = game2.clone();
      if (!g.play(libIdx)) continue;  // illegal move — skip
      escape = _canReach3Libs(g, stoneIdx);
    }
    if (defending === escape) {
      moverSucceeds = true;
      urgentLibs.push(libIdx);
    }
  }
  return { libs, moverSucceeds, urgentLibs };
}

const _exports = { getLadderStatus2, getAllLadderStatuses, _canReach3Libs, _captureEscapes };
if (typeof module !== 'undefined') module.exports = _exports;
else window.Ladder2 = _exports;

})();
