'use strict';

// ladder2.js — Ladder detection using Game2.

const { PASS } = typeof require === 'function' ? require('./game2.js') : window.Game2;

// Module-level output slots for _readLibs.  Written once, then immediately
// copied to locals by the caller before any recursive call can overwrite them.
let _lib0 = -1, _lib1 = -1;

// Scan the liberty bitset for the group at idx.  Writes up to two liberty
// indices into _lib0 / _lib1 (-1 when absent) and returns the group's full
// liberty count (may exceed 2).  Returns 0 when idx has no group.
function _readLibs(game2, idx) {
  _lib0 = _lib1 = -1;
  const gid = game2._gid[idx];
  if (gid === -1) return 0;
  const lc = game2._ls[gid];
  if (lc === 0) return 0;
  const W   = game2._W;
  const lw  = game2._lw;
  const lb  = gid * W;
  const cap = game2.N * game2.N;
  let found = 0;
  for (let wi = 0; wi < W; wi++) {
    let w = lw[lb + wi];
    while (w) {
      const i = wi * 32 + (31 - Math.clz32(w & -w));
      if (i < cap) {
        if (found === 0) _lib0 = i;
        else             _lib1 = i;
        if (++found === 2) return lc;
      }
      w &= w - 1;
    }
  }
  return lc;
}

// Returns true when the group at stoneIdx can reach 3+ liberties despite best
// attacker play.  Same logic as _canReach3Libs in ladder.js.
function _canReach3Libs(game2, idx) {
  const lc = _readLibs(game2, idx);
  if (lc >= 3) return true;
  if (lc === 0) return false;

  // Copy liberty indices before any recursive call overwrites _lib0/_lib1.
  const lib0 = _lib0, lib1 = _lib1;
  const defColor = game2.cells[idx];

  if (game2.current === defColor) {
    // Defender's turn: play one of the liberties; succeed if any leads to safety.
    const libs = lc === 1 ? [lib0] : [lib0, lib1];
    for (const libIdx of libs) {
      const g = game2.clone();
      if (!g.play(libIdx)) continue;      // suicide — skip
      if (g.cells[idx] === 0) continue;   // captured — skip
      if (_canReach3Libs(g, idx)) return true;
    }
    return false;
  }

  // Attacker's turn (1 or 2 libs): tries each liberty; succeeds if any leads to capture.
  const libs = lc === 1 ? [lib0] : [lib0, lib1];
  for (const libIdx of libs) {
    const g = game2.clone();
    if (!g.play(libIdx)) continue;        // illegal for attacker — skip
    if (g.cells[idx] === 0) return false; // captured immediately
    const agid = g._gid[idx];
    const afterLc = agid === -1 ? 0 : g._ls[agid];
    if (afterLc === 0) return false;
    if (afterLc === 1 && !_canReach3Libs(g, idx)) return false;
  }

  return true;
}

// getAllLadderStatuses(game2) — run getLadderStatus2 on every group with 1 or 2
// liberties and return an array of { gid, color, status } objects,
// one per group (groups with 0 or 3+ liberties are skipped).
function getAllLadderStatuses(game2) {
  const cap  = game2.N * game2.N;
  const results = [];
  const visited = new Set();
  for (let i = 0; i < cap; i++) {
    if (game2.cells[i] === 0) continue;
    const gid = game2._gid[i];
    if (visited.has(gid)) continue;
    visited.add(gid);
    const lc = game2._ls[gid];
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
  const lc = _readLibs(game2, stoneIdx);
  if (lc < 1 || lc > 2) {
    const N = game2.N;
    console.warn(`getLadderStatus2: group at ${stoneIdx % N},${(stoneIdx / N) | 0} has ${lc} liberties (expected ≤ 2)`);
    return null;
  }
  const atari = lc === 1;
  const libs = atari ? [_lib0] : [_lib0, _lib1];  // Set by _readLibs.  Nasty stuff.
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
      escape = g.play(libIdx) && _canReach3Libs(g, stoneIdx);
    }
    if (defending === escape) {
      moverSucceeds = true;
      urgentLibs.push(libIdx);
    }
  }
  return { libs, moverSucceeds, urgentLibs };
}

if (typeof module !== 'undefined') module.exports = { getLadderStatus2, getAllLadderStatuses };
