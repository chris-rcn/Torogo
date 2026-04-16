'use strict';

// ladder2.js — Ladder detection using Game3-Precise.

(function() {

const { PASS } = typeof require === 'function' ? require('./game3-precise.js') : window.Game3Precise;

// Returns true when the group at stoneIdx can reach 3+ liberties despite best
// attacker play.
function _canReach3Libs(game, idx) {
  const { count: lc, lib0, lib1 } = game.groupLibs2(idx);
  if (lc >= 3) return true;
  if (lc === 0) return false;

  const defColor = game.cells[idx];

  if (game.current === defColor) {
    // Defender's turn: play one of the liberties; succeed if any leads to safety.
    const libs = lc === 1 ? [lib0] : [lib0, lib1];
    for (const libIdx of libs) {
      const g = game.clone();
      if (!g.play(libIdx)) continue;      // suicide — skip
      if (g.cells[idx] === 0) continue;   // captured — skip
      if (_canReach3Libs(g, idx)) return true;
    }
    return false;
  }

  // Attacker's turn (1 or 2 libs): tries each liberty; succeeds if any leads to capture.
  const libs = lc === 1 ? [lib0] : [lib0, lib1];
  for (const libIdx of libs) {
    const g = game.clone();
    if (!g.play(libIdx)) continue;        // illegal for attacker — skip
    if (g.cells[idx] === 0) return false; // captured immediately
    const afterLc = g.groupLibs2(idx).count;
    if (afterLc === 0) return false;
    if (afterLc === 1 && !_canReach3Libs(g, idx)) return false;
  }

  return true;
}

// getAllLadderStatuses(game, minChainSize) — run getLadderStatus on every
// group with 1 or 2 liberties and return an array of { gid, color, status }
// objects, one per group (groups with 0 or 3+ liberties are skipped).
// minChainSize: skip groups smaller than this (default 1).
function getAllLadderStatuses(game, minChainSize = 1) {
  const cap  = game.N * game.N;
  const results = [];
  const visited = new Set();
  for (let i = 0; i < cap; i++) {
    if (game.cells[i] === 0) continue;
    const gid = game._gid[i];
    if (visited.has(gid)) continue;
    visited.add(gid);
    if (game.groupSize(gid) < minChainSize) continue;
    const { count: lc } = game.groupLibs2(i);
    if (lc === 0 || lc > 2) continue;
    const status = getLadderStatus(game, i);
    results.push({ gid, color: game.cells[i], status });
  }
  return results;
}

// Examines the group containing the stone at stoneIdx (must have 1 or 2
// liberties).  For each liberty, simulates both colours playing it first.
//
// Returns { libs: [], moverSucceeds: boolean, urgentLibs: [] }
//
// Logs a warning and returns null when the group has more than 2 liberties.
function getLadderStatus(game, stoneIdx) {
  const { count: lc, lib0, lib1 } = game.groupLibs2(stoneIdx);
  if (lc < 1 || lc > 2) {
    const N = game.N;
    console.warn(`getLadderStatus: group at ${stoneIdx % N},${(stoneIdx / N) | 0} has ${lc} liberties (expected ≤ 2)`);
    return null;
  }
  const atari = lc === 1;
  const libs = atari ? [lib0] : [lib0, lib1];
  const gColor = game.cells[stoneIdx];
  const mover = game.current;   // BLACK or WHITE
  const defending = gColor === mover;

  let escape;
  // Try opponent playing first.
  if (defending && atari) {
    escape = false;
  } else {
    const g = game.clone();
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
      const g = game.clone();
      escape = g.play(libIdx) && _canReach3Libs(g, stoneIdx);
    }
    if (defending === escape) {
      moverSucceeds = true;
      urgentLibs.push(libIdx);
    }
  }
  return { libs, moverSucceeds, urgentLibs };
}

const _exports = { getLadderStatus, getAllLadderStatuses };
if (typeof module !== 'undefined') module.exports = _exports;
else window.Ladder2 = _exports;

})();
