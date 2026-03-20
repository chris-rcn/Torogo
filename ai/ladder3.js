'use strict';

// ladder3.js — Ladder detection using Game3 (play + undo instead of clone).
//
// Same API as ladder2.js but avoids allocation in the recursive search.
// getLadderStatus3(game3, stoneIdx) → same shape as getLadderStatus2.

// Module-level output slots for _readLibs.  Written once, then immediately
// copied to locals by the caller before any recursive call can overwrite them.
let _lib0 = -1, _lib1 = -1;

// Same as ladder2: scan liberty bitset, write up to two indices into _lib0/_lib1.
function _readLibs(game3, idx) {
  _lib0 = _lib1 = -1;
  const gid = game3._gid[idx];
  if (gid === -1) return 0;
  const lc = game3._ls[gid];
  if (lc === 0) return 0;
  const W   = game3._W;
  const lw  = game3._lw;
  const lb  = gid * W;
  const cap = game3.N * game3.N;
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

// Returns true when the group at idx can reach 3+ liberties despite best
// attacker play.  Uses game3.play()/undo() instead of clone — leaves game3
// in exactly the state it was called with.
function _canReach3Libs(game3, idx) {
  const lc = _readLibs(game3, idx);
  if (lc >= 3) return true;
  if (lc === 0) return false;

  // Copy before any recursive call overwrites _lib0/_lib1.
  const lib0 = _lib0, lib1 = _lib1;
  const defColor = game3.cells[idx];

  if (lc === 1 && game3.current === defColor) {
    // Defender's turn in atari: play the only liberty.
    if (!game3.play(lib0)) return false;       // suicide — play rejected, no undo needed
    if (game3.cells[idx] === 0) { game3.undo(); return false; }  // captured
    const result = _canReach3Libs(game3, idx);
    game3.undo();
    return result;
  }

  // 1 lib (attacker's turn) or 2 libs: attacker tries each liberty.
  const libs = lc === 1 ? [lib0] : [lib0, lib1];
  const atkColor = 3 - defColor;

  for (const libIdx of libs) {
    const savedCurrent = game3.current;
    game3.current = atkColor;

    if (!game3.play(libIdx)) {
      // Illegal for attacker — restore current (play didn't push a snap).
      game3.current = savedCurrent;
      continue;
    }

    // play() succeeded: undo() will restore current to atkColor, so we
    // must fix it back to savedCurrent ourselves after undo.
    if (game3.cells[idx] === 0) {
      game3.undo(); game3.current = savedCurrent;
      return false;  // captured immediately
    }

    const agid   = game3._gid[idx];
    const afterLc = agid === -1 ? 0 : game3._ls[agid];

    if (afterLc === 0) {
      game3.undo(); game3.current = savedCurrent;
      return false;
    }

    if (afterLc === 1) {
      const result = _canReach3Libs(game3, idx);
      game3.undo(); game3.current = savedCurrent;
      if (!result) return false;
    } else {
      // afterLc >= 2: attacker failed to re-atari — try next liberty.
      game3.undo(); game3.current = savedCurrent;
    }
  }

  return true;
}

// Examines the group containing the stone at stoneIdx (1–2 liberties).
// Mutates game3 transiently (play/undo) but leaves it unchanged on return.
//
// Returns an array — one entry per liberty — of:
//   { liberty: {x, y}, canEscape: boolean, canEscapeAfterPass: boolean }
//
// Returns null (with a warning) when the group has more than 2 liberties.
function getLadderStatus3(game3, stoneIdx) {
  if (game3.cells[stoneIdx] === 0) return [];

  const lc = _readLibs(game3, stoneIdx);
  if (lc > 2) {
    const N = game3.N;
    console.warn(`getLadderStatus3: group at ${stoneIdx % N},${(stoneIdx / N) | 0} has ${lc} liberties (expected ≤ 2)`);
    return null;
  }

  const lib0 = _lib0, lib1 = _lib1;
  const mover = game3.current;
  const opp   = 3 - mover;
  const N     = game3.N;
  const libs  = lc === 1 ? [lib0] : [lib0, lib1];
  const results = [];

  for (const libIdx of libs) {
    const entry = { liberty: { x: libIdx % N, y: (libIdx / N) | 0 } };

    for (const color of [mover, opp]) {
      const savedCurrent = game3.current;
      game3.current = color;

      let escaped;
      if (!game3.play(libIdx)) {
        game3.current = savedCurrent;
        escaped = false;
      } else {
        escaped = game3.cells[stoneIdx] !== 0 && _canReach3Libs(game3, stoneIdx);
        game3.undo();
        game3.current = savedCurrent;
      }

      entry[color === mover ? 'canEscape' : 'canEscapeAfterPass'] = escaped;
    }

    results.push(entry);
  }
  return results;
}

module.exports = { getLadderStatus3 };
