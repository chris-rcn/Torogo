'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs (require, process, etc.).
// Loaded as a plain <script> tag; ladder2.js must be loaded before this file.

(function() {

const { getAllLadderStatuses } = typeof require === 'function' ? require('./ladder2.js') : window.Ladder2;

// Zobrist random table: flat Int32Array. Generated with xorshift32.
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

// Size 9*3, indexed [pos*3 + cellCode].
// cellCode 0 = empty, cellCode 1 = mover stone, 2 = opponent stone.
const POINTS = 13;
const ZOBRIST = makeZobrist(0x12345678, POINTS * 3);
const colorZ = makeZobrist(675425835, 3);
const moverSucceedsZ = makeZobrist(763158854, 3);
const urgentZ = makeZobrist(936265824, 5);
const wastedZ = makeZobrist(785247856, 5);

// The 8 symmetries of the square (dihedral group D4) as index permutations.
// Indices 0-8 are the 3×3 core; indices 9-12 are the cardinal arms:
//   9=north(-2,0), 10=west(0,-2), 11=east(0,+2), 12=south(+2,0).
const SYMMETRY_PERMS = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8,  9, 10, 11, 12], // identity
  [6, 3, 0, 7, 4, 1, 8, 5, 2, 10, 12,  9, 11], // rotate 90° CW
  [8, 7, 6, 5, 4, 3, 2, 1, 0, 12, 11, 10,  9], // rotate 180°
  [2, 5, 8, 1, 4, 7, 0, 3, 6, 11,  9, 12, 10], // rotate 270° CW
  [2, 1, 0, 5, 4, 3, 8, 7, 6,  9, 11, 10, 12], // reflect horizontal
  [6, 7, 8, 3, 4, 5, 0, 1, 2, 12, 10, 11,  9], // reflect vertical
  [0, 3, 6, 1, 4, 7, 2, 5, 8, 10,  9, 12, 11], // reflect main diagonal
  [8, 5, 2, 7, 4, 1, 6, 3, 0, 11, 12,  9, 10], // reflect anti-diagonal
];

// patternHashes2(game2, indices) — batch hash with ladder urgency flags.
// Returns an array of { idx, pHash } in the same order as `indices`.
function patternHashes2(game2, indices, ladderStatuses) {
  const N     = game2.N;
  const cap   = N * N;
  const cells = game2.cells;
  const mover = game2.current;
  const statuses   = ladderStatuses ?? getAllLadderStatuses(game2);

  const moverSucceeds = new Int8Array(cap);
  const urgent = new Int8Array(cap);
  const wasted = new Int8Array(cap);
  for (const { gid, color, status } of statuses) {
    if (status.moverSucceeds) {
      const urgencyFlag = status.urgentLibs.length > 0 ? 1 : 2;
      for (const stone of game2.groupStones(gid)) {
        moverSucceeds[stone] = urgencyFlag;
      }
      for (const lib of status.urgentLibs) {
        urgent[lib]++;
      }
    } else {
      for (const lib of status.libs) {
        wasted[lib]++;
      }
    }
  }
  const relColor = c => c === 0 ? 0 : (c === game2.current ? 1 : 2);
  const cellHash = new Int32Array(cap);
  for (let i = 0; i < cap; i++) {
//    cellHash[i] = colorZ[relColor(cells[i])];
    cellHash[i] = colorZ[relColor(cells[i])] ^ moverSucceedsZ[moverSucceeds[i]];
//    cellHash[i] = colorZ[relColor(cells[i])] ^ urgentZ[urgent[i]];
//    cellHash[i] = colorZ[relColor(cells[i])] ^ wastedZ[wasted[i]];
//    cellHash[i] = colorZ[relColor(cells[i])] ^ moverSucceedsZ[moverSucceeds[i]] ^ urgentZ[urgent[i]];
//    cellHash[i] = colorZ[relColor(cells[i])] ^ moverSucceedsZ[moverSucceeds[i]] ^ wastedZ[wasted[i]];
//    cellHash[i] = colorZ[relColor(cells[i])] ^ moverSucceedsZ[moverSucceeds[i]] ^ urgentZ[urgent[i]] ^ wastedZ[wasted[i]];
  }
  const result = [];
  for (let idx of indices) {
    const x = idx % N;
    const y = (idx / N) | 0;
    const rNN = ((y - 2 + N) % N) * N,  rN = ((y - 1 + N) % N) * N,  r0 = y * N,  rS = ((y + 1) % N) * N,  rSS = ((y + 2) % N) * N;
    const cWW =  (x - 2 + N) % N,       cW =  (x - 1 + N) % N,       c0 = x,      cE =  (x + 1) % N,       cEE =  (x + 2) % N
    const region = new Int32Array(POINTS);
                                                                 region[ 9] = cellHash[rNN+c0];
                                   region[ 0] = cellHash[rN+cW]; region[ 1] = cellHash[rN +c0]; region[ 2] = cellHash[rN+cE];
    region[10] = cellHash[r0+cWW]; region[ 3] = cellHash[r0+cW]; region[ 4] = cellHash[r0 +c0]; region[ 5] = cellHash[r0+cE]; region[11] = cellHash[r0+cEE];
                                   region[ 6] = cellHash[rS+cW]; region[ 7] = cellHash[rS +c0]; region[ 8] = cellHash[rS+cE];
                                                                 region[12] = cellHash[rSS+c0];
    // Try all 8 symmetry transforms; keep the minimum hash.
    let pHash = 0xFFFFFFFF;
    for (const perm of SYMMETRY_PERMS) {
      let h = 0;
      for (let k = 0; k < POINTS; k++) {
        h = Math.imul(h, 0x9e3779b9) ^ region[perm[k]];
      }
      h = h >>> 0;  // treat as unsigned uint32
      if (h < pHash) pHash = h;
    }
    result.push({ idx, pHash });
  }
  return result;
}

const _exports = { patternHashes2 };
if (typeof module !== 'undefined') module.exports = _exports;
else window.Patterns12 = _exports;

})();
