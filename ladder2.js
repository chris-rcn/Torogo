'use strict';

// ladder2.js — Ladder detection using Game2.

(function() {

const { PASS } = typeof require === 'function' ? require('./game2.js') : window.Game2;

// Returns true when the group at stoneIdx can reach 3+ liberties despite best
// attacker play.
function _canReach3Libs(game2, idx) {
  const { count: lc, lib0, lib1 } = game2.groupLibs(idx);
  if (lc >= 3) return true;
  if (lc === 0) return false;

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
    const afterLc = g.groupLibs(idx).count;
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
    const { count: lc } = game2.groupLibs(i);
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
  const { count: lc, lib0, lib1 } = game2.groupLibs(stoneIdx);
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
      escape = g.play(libIdx) && _canReach3Libs(g, stoneIdx);
    }
    if (defending === escape) {
      moverSucceeds = true;
      urgentLibs.push(libIdx);
    }
  }
  return { libs, moverSucceeds, urgentLibs };
}

const _exports = { getLadderStatus2, getAllLadderStatuses };
if (typeof module !== 'undefined') module.exports = _exports;
else window.Ladder2 = _exports;

})();
