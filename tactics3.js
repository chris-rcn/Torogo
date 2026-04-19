'use strict';

// tactics3.js — Tactical search using Game3.
// Like ladder2.js but terminates when the chain reaches 4+ liberties (or is captured).

(function() {

const { PASS } = typeof require === 'function' ? require('./game3.js') : window.Game3;

// Returns [result, remainingCredits] where result is:
//   true  — defender can reach 4+ liberties despite best attacker play
//   false — attacker can capture the chain
//   null  — budget exhausted before a conclusion, or depth limit reached
//
// Enforces both nodeLimit (via credits) and depth limit to prevent
// excessive recursion.
function canReach4Libs(game, idx, credits, depth = 0, depthLimit = 20) {
  if (depth > depthLimit) return [null, credits];  // Depth limit
  if (credits <= 0) return [null, 0];
  credits--;

  const libs = game.groupLibs(idx);
  const lc = libs.length;
  if (lc >= 4) return [true,  credits];
  if (lc === 0) return [false, credits];

  const defColor = game.cells[idx];

  if (game.current === defColor) {
    // Defender's turn: succeed if any branch is definitely true; unknown if
    // any branch is null and none is true; false only if all are false.
    let hasUnknown = false;
    for (let k = 0; k < lc; k++) {
      const libIdx = libs[k];
      if (!game.play(libIdx)) {
        game.undo();
        continue;      // suicide — skip
      }
      if (game.cells[idx] === 0) {
        game.undo();
        continue;    // captured — skip
      }
      const budget = Math.floor(credits / (lc - k));
      credits -= budget;
      let result, unused;
      [result, unused] = canReach4Libs(game, idx, budget, depth + 1, depthLimit);
      credits += unused;
      game.undo();
      if (result === true)  return [true,    credits];
      if (result === null)  hasUnknown = true;
    }
    return [hasUnknown ? null : false, credits];
  }

  // Attacker's turn: succeed (return false) if any branch definitely captures;
  // unknown if any branch is null and none captures; true only if all survive.
  let hasUnknown = false;
  for (let k = 0; k < lc; k++) {
    const libIdx = libs[k];
    if (!game.play(libIdx)) {
      game.undo();
      continue;        // illegal for attacker — skip
    }
    if (game.cells[idx] === 0) {
      game.undo();
      return [false, credits]; // captured immediately
    }
    const afterLc = game.groupLibs(idx).length;
    if (afterLc === 0) {
      game.undo();
      return [false, credits];
    }
    if (afterLc < 4) {
      const budget = Math.floor(credits / (lc - k));
      credits -= budget;
      let result, unused;
      [result, unused] = canReach4Libs(game, idx, budget, depth + 1, depthLimit);
      credits += unused;
      game.undo();
      if (result === false) return [false, credits];
      if (result === null)  hasUnknown = true;
    } else {
      game.undo();
    }
  }

  return [hasUnknown ? null : true, credits];
}

// searchChains(game, nodeLimit, depthLimit) — run searchChain on every group with 1–3
// liberties and return an array of { gid, color, status } objects, one per
// group (groups with 0 or 4+ liberties are skipped).
// nodeLimit: max nodes per sub-search per liberty in searchChain (default Infinity).
// depthLimit: maximum recursion depth for canReach4Libs (default 20).
function searchChains(game, nodeLimit = Infinity, depthLimit = 20) {
  const cap  = game.N * game.N;
  const results = [];
  const visited = new Set();
  for (let i = 0; i < cap; i++) {
    if (game.cells[i] === 0) continue;
    const gid = game._gid[i];
    if (visited.has(gid)) continue;
    visited.add(gid);
    const lc = game.groupLibs(i).length;
    if (lc === 0 || lc > 3) continue;
    const status = searchChain(game, i, nodeLimit, depthLimit);
    results.push({ gid, color: game.cells[i], status });
  }
  return results;
}

// Examines the group containing the stone at stoneIdx (must have 1–3 liberties).
//
// Returns { libs, moverSucceeds, urgentLibs } where moverSucceeds is
//   true  — mover achieves their goal (capture or escape)
//   false — mover fails
//   null  — inconclusive (node budget exhausted before a definitive result)
//
// nodeLimit: fresh credit budget given to each canReach4Libs sub-search
// (one per liberty). Default Infinity (unbounded).
// depthLimit: maximum recursion depth for canReach4Libs. Default 20.
// Logs a warning and returns null when the group has more than 3 liberties.
function searchChain(game, stoneIdx, nodeLimit = Infinity, depthLimit = 20) {
  const libs = game.groupLibs(stoneIdx);
  const lc = libs.length;
  if (lc < 1 || lc > 3) {
    const N = game.N;
    console.warn(`searchChain: group at ${stoneIdx % N},${(stoneIdx / N) | 0} has ${lc} liberties (expected 1–3)`);
    return null;
  }
  const gColor = game.cells[stoneIdx];
  const mover = game.current;
  const defending = gColor === mover;
  const atari = lc === 1;

  // Count canReach4Libs calls we will make, for credit division.
  const opponentCalls = (defending && atari) ? 0 : 1;
  const moverCalls    = (!defending && atari) ? 0 : lc;
  let callsLeft = opponentCalls + moverCalls;
  let credits   = nodeLimit;

  let escape;
  // Try opponent playing first.
  if (defending && atari) {
    escape = false;
  } else {
    const budget = Math.floor(credits / callsLeft);
    credits -= budget;
    callsLeft--;
    game.play(PASS);
    let unused;
    [escape, unused] = canReach4Libs(game, stoneIdx, budget, 0, depthLimit);
    credits += unused;
    game.undo();
  }
  // escape===null means inconclusive; skip the early-return optimisation.
  if (escape !== null && defending === escape) {
    // group is not urgent — outcome same regardless of who plays first
    return { libs, moverSucceeds: true, urgentLibs: [] };
  }

  // Try mover playing first.
  let moverSucceeds = false;
  let hasUnknown = escape === null;  // propagate unknown from opponent-first
  let urgentLibs = [];
  for (let k = 0; k < lc; k++) {
    const libIdx = libs[k];
    if (!defending && atari) {
      escape = false;
    } else {
      const budget = Math.floor(credits / callsLeft);
      credits -= budget;
      callsLeft--;
      const played = game.play(libIdx);
      if (played) {
        let unused;
        [escape, unused] = canReach4Libs(game, stoneIdx, budget, 0, depthLimit);
        credits += unused;
      } else {
        escape = false;
        credits += budget;  // return unspent budget to pool
      }
      game.undo();
    }
    if (escape !== null && defending === escape) {
      moverSucceeds = true;
      urgentLibs.push(libIdx);
    } else if (escape === null) {
      hasUnknown = true;
    }
  }
  return {
    libs,
    moverSucceeds: moverSucceeds ? true : (hasUnknown ? null : false),
    urgentLibs,
  };
}

const _exports = { searchChain, searchChains };
if (typeof module !== 'undefined') module.exports = _exports;
else window.Tactics3 = _exports;

})();
