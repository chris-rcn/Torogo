'use strict';

// tactics3.js — Tactical search using Game2.
// Like ladder2.js but terminates when the chain reaches 4+ liberties (or is captured).

(function() {

const { PASS } = typeof require === 'function' ? require('./game2.js') : window.Game2;

// Returns [result, remainingCredits] where result is:
//   true  — defender can reach 4+ liberties despite best attacker play
//   false — attacker can capture the chain
//   null  — budget exhausted before a conclusion
//
// Sibling branches share credits: each call returns what's left so the next
// sibling can use it.
function canReach4Libs(game2, idx, credits) {
  if (credits <= 0) return [null, 0];
  credits--;

  const libs = game2.groupLibs(idx);
  const lc = libs.length;
  if (lc >= 4) return [true,  credits];
  if (lc === 0) return [false, credits];

  const defColor = game2.cells[idx];

  if (game2.current === defColor) {
    // Defender's turn: succeed if any branch is definitely true; unknown if
    // any branch is null and none is true; false only if all are false.
    let hasUnknown = false;
    for (let k = 0; k < lc; k++) {
      const g = game2.clone();
      if (!g.play(libs[k])) continue;      // suicide — skip
      if (g.cells[idx] === 0) continue;    // captured — skip
      const budget = Math.floor(credits / (lc - k));
      credits -= budget;
      let result, unused;
      [result, unused] = canReach4Libs(g, idx, budget);
      credits += unused;
      if (result === true)  return [true,    credits];
      if (result === null)  hasUnknown = true;
    }
    return [hasUnknown ? null : false, credits];
  }

  // Attacker's turn: succeed (return false) if any branch definitely captures;
  // unknown if any branch is null and none captures; true only if all survive.
  let hasUnknown = false;
  for (let k = 0; k < lc; k++) {
    const g = game2.clone();
    if (!g.play(libs[k])) continue;        // illegal for attacker — skip
    if (g.cells[idx] === 0) return [false, credits]; // captured immediately
    const afterLc = g.groupLibs(idx).length;
    if (afterLc === 0) return [false, credits];
    if (afterLc < 4) {
      const budget = Math.floor(credits / (lc - k));
      credits -= budget;
      let result, unused;
      [result, unused] = canReach4Libs(g, idx, budget);
      credits += unused;
      if (result === false) return [false, credits];
      if (result === null)  hasUnknown = true;
    }
  }

  return [hasUnknown ? null : true, credits];
}

// searchChains(game2, nodeLimit) — run searchChain on every group with 1–3
// liberties and return an array of { gid, color, status } objects, one per
// group (groups with 0 or 4+ liberties are skipped).
// nodeLimit: max nodes per sub-search per liberty in searchChain (default Infinity).
function searchChains(game2, nodeLimit = Infinity) {
  const cap  = game2.N * game2.N;
  const results = [];
  const visited = new Set();
  for (let i = 0; i < cap; i++) {
    if (game2.cells[i] === 0) continue;
    const gid = game2._gid[i];
    if (visited.has(gid)) continue;
    visited.add(gid);
    const lc = game2.groupLibs(i).length;
    if (lc === 0 || lc > 3) continue;
    const status = searchChain(game2, i, nodeLimit);
    results.push({ gid, color: game2.cells[i], status });
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
// Logs a warning and returns null when the group has more than 3 liberties.
function searchChain(game2, stoneIdx, nodeLimit = Infinity) {
  const libs = game2.groupLibs(stoneIdx);
  const lc = libs.length;
  if (lc < 1 || lc > 3) {
    const N = game2.N;
    console.warn(`searchChain: group at ${stoneIdx % N},${(stoneIdx / N) | 0} has ${lc} liberties (expected 1–3)`);
    return null;
  }
  const gColor = game2.cells[stoneIdx];
  const mover = game2.current;
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
    const g = game2.clone();
    g.play(PASS);
    let unused;
    [escape, unused] = canReach4Libs(g, stoneIdx, budget);
    credits += unused;
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
      const g = game2.clone();
      const played = g.play(libIdx);
      if (played) {
        let unused;
        [escape, unused] = canReach4Libs(g, stoneIdx, budget);
        credits += unused;
      } else {
        escape = false;
        credits += budget;  // return unspent budget to pool
      }
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
