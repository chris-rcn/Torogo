'use strict';

// ladder2.js — Ladder detection using Game2 (fast typed-array engine).
//
// Mirrors the API of ladder.js but operates on Game2 instances.
// getLadderStatus2(game2, stoneIdx) → same shape as getLadderStatus.

const { PASS, BLACK, WHITE } = require('../game2.js');

// Return liberty count for the group containing stoneIdx.
function _libCount(game2, idx) {
  const gid = game2._gid[idx];
  return gid === -1 ? 0 : game2._ls[gid];
}

// Return an array of liberty indices for the group containing stoneIdx.
function _libIndices(game2, idx) {
  const gid = game2._gid[idx];
  if (gid === -1) return [];
  const W   = game2._W;
  const lw  = game2._lw;
  const lb  = gid * W;
  const cap = game2.N * game2.N;
  const out = [];
  for (let wi = 0; wi < W; wi++) {
    let w = lw[lb + wi];
    while (w) {
      const bit = 31 - Math.clz32(w & -w);
      const i   = wi * 32 + bit;
      if (i < cap) out.push(i);
      w &= w - 1;
    }
  }
  return out;
}

// Returns true when the group at stoneIdx can reach 3+ liberties despite best
// attacker play.  Same logic as _canReach3Libs in ladder.js.
function _canReach3Libs(game2, idx) {
  const lc = _libCount(game2, idx);
  if (lc >= 3) return true;
  if (lc === 0) return false;

  const defColor = game2.cells[idx];
  const libs     = _libIndices(game2, idx);

  if (lc === 1 && game2.current === defColor) {
    // Defender's turn in atari: play the only liberty.
    const g = game2.clone();
    if (!g.play(libs[0])) return false;     // suicide
    if (g.cells[idx] === 0) return false;   // captured
    return _canReach3Libs(g, idx);
  }

  // 1 lib (attacker's turn) or 2 libs: attacker tries each liberty.
  for (const libIdx of libs) {
    const g = game2.clone();
    g.current = 3 - defColor;              // attacker's turn
    if (!g.play(libIdx)) continue;          // illegal for attacker — skip
    if (g.cells[idx] === 0) return false;  // captured immediately
    const afterLc = _libCount(g, idx);
    if (afterLc === 0) return false;
    if (afterLc === 1 && !_canReach3Libs(g, idx)) return false;
  }

  return true;
}

// Examines the group containing the stone at stoneIdx (must have 1 or 2
// liberties).  For each liberty, simulates both colours playing it first.
//
// Returns an array — one entry per liberty — of:
//   { liberty: {x, y}, canEscape: boolean, canEscapeAfterPass: boolean }
//
// Logs a warning and returns null when the group has more than 2 liberties.
function getLadderStatus2(game2, stoneIdx) {
  if (game2.cells[stoneIdx] === 0) return [];

  const lc = _libCount(game2, stoneIdx);
  if (lc > 2) {
    const N = game2.N;
    console.warn(`getLadderStatus2: group at ${stoneIdx % N},${(stoneIdx / N) | 0} has ${lc} liberties (expected ≤ 2)`);
    return null;
  }

  const libs  = _libIndices(game2, stoneIdx);
  const mover = game2.current;   // BLACK or WHITE
  const opp   = 3 - mover;
  const N     = game2.N;
  const results = [];

  for (const libIdx of libs) {
    const entry = { liberty: { x: libIdx % N, y: (libIdx / N) | 0 } };
    for (const color of [mover, opp]) {
      const g = game2.clone();
      g.current = color;
      let escaped;
      if (!g.play(libIdx)) {
        escaped = false;
      } else {
        escaped = g.cells[stoneIdx] !== 0 && _canReach3Libs(g, stoneIdx);
      }
      entry[color === mover ? 'canEscape' : 'canEscapeAfterPass'] = escaped;
    }
    results.push(entry);
  }
  return results;
}

module.exports = { getLadderStatus2 };
