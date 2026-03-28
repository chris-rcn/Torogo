'use strict';

// pattern9.js — pattern recognition helpers for Game2 (integer-move engine).
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
const ZOBRIST = makeZobrist(259752658, 9 * 3);
const colorZ = makeZobrist(675425835, 3);
const moverSucceedsZ = makeZobrist(763158854, 3);
const urgentZ = makeZobrist(936265824, 5);
const wastedZ = makeZobrist(785247856, 5);
// Center-cell atari flags: an orthogonally adjacent friendly/enemy group has exactly 1 liberty.
const friendAtariZ = makeZobrist(966278351, 2);
const enemyAtariZ  = makeZobrist(179872158, 2);

// The 8 symmetries of the square (dihedral group D4) as index permutations.
const SYMMETRY_PERMS = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8], // identity
  [6, 3, 0, 7, 4, 1, 8, 5, 2], // rotate 90° CW
  [8, 7, 6, 5, 4, 3, 2, 1, 0], // rotate 180°
  [2, 5, 8, 1, 4, 7, 0, 3, 6], // rotate 270° CW
  [2, 1, 0, 5, 4, 3, 8, 7, 6], // reflect horizontal
  [6, 7, 8, 3, 4, 5, 0, 1, 2], // reflect vertical
  [0, 3, 6, 1, 4, 7, 2, 5, 8], // reflect main diagonal
  [8, 5, 2, 7, 4, 1, 6, 3, 0], // reflect anti-diagonal
];

// patternHash2(game2, idx, mover) → rotation- and reflection-invariant Zobrist hash
// of the 3×3 neighbourhood centred on the cell at flat index idx = y*N+x.
//
// mover — game2.current (BLACK=1 or WHITE=2)
function patternHash2(game2, idx, mover) {
  const N     = game2.N;
  const cells = game2.cells;
  const x = idx % N;
  const y = (idx / N) | 0;

  const rN = ((y - 1 + N) % N) * N,  r0 = y * N,  rS = ((y + 1) % N) * N;
  const cW =  (x - 1 + N) % N,       c0 = x,      cE =  (x + 1) % N;

  const cc = new Uint8Array(9);
  const relativeColor = c => c === 0 ? 0 : c === mover ? 1 : 2;
  cc[0] = relativeColor(cells[rN + cW]); cc[1] = relativeColor(cells[rN + c0]); cc[2] = relativeColor(cells[rN + cE]);
  cc[3] = relativeColor(cells[r0 + cW]); cc[4] = relativeColor(cells[r0 + c0]); cc[5] = relativeColor(cells[r0 + cE]);
  cc[6] = relativeColor(cells[rS + cW]); cc[7] = relativeColor(cells[rS + c0]); cc[8] = relativeColor(cells[rS + cE]);

  // Adjacent-to-center atari flags (D4-invariant: center and its 4 neighbors are preserved).
  let friendAtari = 0, enemyAtari = 0;
  const nbr4 = game2._nbr;
  const gids = game2._gid;
  const ls   = game2._ls;
  const base = idx * 4;
  for (let d = 0; d < 4; d++) {
    const nb = nbr4[base + d];
    const nc = cells[nb];
    if (nc === 0) continue;
    if (ls[gids[nb]] === 1) {
      if (nc === mover) friendAtari = 1;
      else              enemyAtari  = 1;
    }
  }
  const atariZ = friendAtariZ[friendAtari] ^ enemyAtariZ[enemyAtari];

  // Try all 8 symmetry transforms; keep the minimum Zobrist hash.
  let minHash = 0xFFFFFFFF;
  for (const perm of SYMMETRY_PERMS) {
    let h = 0;
    for (let k = 0; k < 9; k++) {
      h ^= ZOBRIST[k * 3 + cc[perm[k]]];
    }
    h = (h ^ atariZ) >>> 0;  // treat as unsigned uint32
    if (h < minHash) minHash = h;
  }
  return minHash;
}

// patternHashes2(game2, indices) — batch hash with ladder urgency flags.
// Returns an array of { idx, pHash } in the same order as `indices`.
function patternHashes2(game2, indices, ladderStatuses) {
  const N     = game2.N;
  const cap   = N * N;
  const cells = game2.cells;
  const mover = game2.current;

//  const statuses   = ladderStatuses ?? getAllLadderStatuses(game2);
//  const moverSucceeds = new Int8Array(cap);
//  const urgent = new Int8Array(cap);
//  const wasted = new Int8Array(cap);
//  for (const { gid, color, status } of statuses) {
//    if (status.moverSucceeds) {
//      const urgencyFlag = status.urgentLibs.length > 0 ? 1 : 2;
//      for (const stone of game2.groupStones(gid)) {
//        moverSucceeds[stone] = urgencyFlag;
//      }
//      for (const lib of status.urgentLibs) {
//        urgent[lib]++;
//      }
//    } else {
//      for (const lib of status.libs) {
//        wasted[lib]++;
//      }
//    }
//  }

  const relativeColor = c => c === 0 ? 0 : (c === game2.current ? 1 : 2);
  const cellHash = new Int32Array(cap);
  for (let i = 0; i < cap; i++) {
    cellHash[i] = colorZ[relativeColor(cells[i])];
//    cellHash[i] = colorZ[relativeColor(cells[i])] ^ moverSucceedsZ[moverSucceeds[i]];
//    cellHash[i] = colorZ[relativeColor(cells[i])] ^ urgentZ[urgent[i]];
//    cellHash[i] = colorZ[relativeColor(cells[i])] ^ wastedZ[wasted[i]];
//    cellHash[i] = colorZ[relativeColor(cells[i])] ^ moverSucceedsZ[moverSucceeds[i]] ^ urgentZ[urgent[i]];
//    cellHash[i] = colorZ[relativeColor(cells[i])] ^ moverSucceedsZ[moverSucceeds[i]] ^ wastedZ[wasted[i]];
//    cellHash[i] = colorZ[relativeColor(cells[i])] ^ moverSucceedsZ[moverSucceeds[i]] ^ urgentZ[urgent[i]] ^ wastedZ[wasted[i]];
  }
  const result = [];
  for (let idx of indices) {
    const x = idx % N;
    const y = (idx / N) | 0;
    const rN = ((y - 1 + N) % N) * N,  r0 = y * N,  rS = ((y + 1) % N) * N;
    const cW =  (x - 1 + N) % N,       c0 = x,      cE =  (x + 1) % N;
    const region = new Int32Array(9);
    region[0] = cellHash[rN + cW]; region[1] = cellHash[rN + c0]; region[2] = cellHash[rN + cE];
    region[3] = cellHash[r0 + cW]; region[4] = cellHash[r0 + c0]; region[5] = cellHash[r0 + cE];
    region[6] = cellHash[rS + cW]; region[7] = cellHash[rS + c0]; region[8] = cellHash[rS + cE];
    // Try all 8 symmetry transforms; keep the minimum hash.
    let pHash = 0xFFFFFFFF;
    for (const perm of SYMMETRY_PERMS) {
      let h = 0;
      for (let k = 0; k < 9; k++) {
        h = (h<<3) ^ region[perm[k]];
      }
      h = h >>> 0;  // treat as unsigned uint32
      if (h < pHash) pHash = h;
    }
    result.push({ idx, pHash });
  }
  return result;
}

const _exports = { patternHash2, patternHashes2 };
if (typeof module !== 'undefined') module.exports = _exports;
else window.Pattern9 = _exports;

})();
