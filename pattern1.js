'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs (require, process, etc.).
// Loaded as a plain <script> tag; ladder2.js must be loaded before this file.

(function() {

//const Util = require('./util.js');
//const PAT1_MAX_STONES   = Util.envInt('PAT1_MAX_STONES', 1);

const minChainSize  = 1;
const maxStoneCount = 8;

const { getAllLadderStatuses } = typeof require === 'function' ? require('./ladder2.js') : window.Ladder2;

// Random value table: flat Int32Array. Generated with xorshift32.
// Values are accumulated with += (not ^=) to avoid cancellation when multiple
// chains share a liberty.
function makeZobrist(seed, size) {
  const t = new Int32Array(size);
  let s = seed;
  for (let p = 0; p < size; p++) {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    t[p] = s;
  }
  return t;
};

const killUrgentZ = makeZobrist(936265824, maxStoneCount+1);
const saveUrgentZ = makeZobrist(785247856, maxStoneCount+1);
const killWastedZ = makeZobrist(614738291, maxStoneCount+1);
const saveWastedZ = makeZobrist(492837465, maxStoneCount+1);
const openZ       = makeZobrist(371829463, 5);
const friendZ     = makeZobrist(258463719, 5);
const enemyZ      = makeZobrist(193847261, 5);

function getPatternHashes(game2, indices, ladderStatuses) {
  const N    = game2.N;
  const cap  = N * N;
  const mover = game2.current;

  const killUrgent = new Int32Array(cap);
  const saveUrgent = new Int32Array(cap);
  const killWasted = new Int32Array(cap);
  const saveWasted = new Int32Array(cap);

  const statuses = ladderStatuses ?? getAllLadderStatuses(game2, minChainSize);
  for (const { gid, color, status } of statuses) {
    const chainSize = game2.groupSize(gid);
    const isOpponent = color !== mover;
    if (status.moverSucceeds) {
      const urgent = isOpponent ? killUrgent : saveUrgent;
      for (const lib of status.urgentLibs) {
        urgent[lib] += chainSize;
      }
    } else {
      const wasted = isOpponent ? killWasted : saveWasted;
      for (const lib of status.libs) {
        wasted[lib] += chainSize;
      }
    }
  }

  const cells = game2.cells;
  const result = [];
  for (let idx of indices) {
//    const base = idx * 4;
//    let open = 0;
//    const friendGids = new Set(), enemyGids = new Set();
//    for (let i = 0; i < 4; i++) {
//      const n = game2._nbr[base + i];
//      const c = cells[n];
//      if (c === 0) open++; else if (c === mover) friendGids.add(game2._gid[n]); else enemyGids.add(game2._gid[n]);
//    }
//    const friend = friendGids.size, enemy = enemyGids.size;
//
    const hash = 0
      ^ killUrgentZ[Math.min(killUrgent[idx], maxStoneCount)]
      ^ saveUrgentZ[Math.min(saveUrgent[idx], maxStoneCount)]
      ^ killWastedZ[Math.min(killWasted[idx], maxStoneCount)]
      ^ saveWastedZ[Math.min(saveWasted[idx], maxStoneCount)]
//      ^ friendZ[friend]
//      ^ enemyZ[enemy]
//      ^ openZ[open]
    ;
    result.push({ idx, hash });
  }
  return result;
}

const _exports = { getPatternHashes };
if (typeof module !== 'undefined') module.exports = _exports;
else window.Pattern1 = _exports;

})();

