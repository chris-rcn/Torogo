'use strict';

// patterns2.js — pattern recognition helpers for Game2 (integer-move engine).
// BROWSER-COMPATIBLE: no Node.js-only APIs (require, process, etc.).
// Loaded as a plain <script> tag; ladder2.js must be loaded before this file.
if (typeof require === 'function') { var { getLadderStatus2 } = require('./ai/ladder2.js'); }

// Liberty counts above this threshold are treated as equivalent.
const MAX_LIBS = 1;

// Powers of 3 for weighting the 9 cells (base-3 hash).
const POW3 = [1, 3, 9, 27, 81, 243, 729, 2187, 6561]; // 3^0 … 3^8

// Powers of (MAX_LIBS+1) for weighting the 4 orthogonal liberty counts.
// Orthogonal neighbours occupy flat positions 1, 3, 5, 7 in row-major order:
//
//   0 1 2
//   3 4 5   ← 1,3,5,7 are the 4 orthogonal neighbours of centre (4)
//   6 7 8
//
const _M1 = MAX_LIBS + 1;
const LIB_WEIGHT = [0, 1, 0, _M1, 0, _M1 * _M1, 0, _M1 * _M1 * _M1, 0];

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

// Clamped liberty count for the group at idx, or 0 if idx is empty.
function cellLiberties2(game2, idx) {
  const gid = game2._gid[idx];
  if (gid < 0) return 0;
  return Math.min(game2._ls[gid], MAX_LIBS);
}

// patternHash2(game2, idx, mover) → canonical integer in [0, 3^9 · (MAX_LIBS+1)^4).
//
// Returns a rotation- and reflection-invariant hash of the 3×3 neighbourhood
// centred on the cell at flat index idx = y*N+x.  Produces identical values
// to patternHash(game, x, y, mover) in patterns.js for the same board state.
//
// mover — game2.current (BLACK=1 or WHITE=2)
function patternHash2(game2, idx, mover) {
  const N     = game2.N;
  const cells = game2.cells;
  const x = idx % N;
  const y = (idx / N) | 0;

  // Collect cell codes and liberty counts for all 9 positions in row-major
  // order: (dy,dx) = (-1,-1),(−1,0),(−1,+1),(0,−1),(0,0),(0,+1),(+1,−1),(+1,0),(+1,+1).
  const cellCodes = new Uint8Array(9);
  const libs      = new Uint8Array(9);
  let i = 0;
  for (let dy = -1; dy <= 1; dy++) {
    const ry = (y + dy + N) % N;
    for (let dx = -1; dx <= 1; dx++) {
      const rx   = (x + dx + N) % N;
      const ridx = ry * N + rx;
      const c    = cells[ridx];
      cellCodes[i] = c === 0 ? 0 : c === mover ? 1 : 2;
      if (cellCodes[i] !== 0) libs[i] = cellLiberties2(game2, ridx);
      i++;
    }
  }

  // Try all 8 symmetry transforms; keep the minimum combined hash.
  let minHash = Infinity;
  for (const perm of SYMMETRY_PERMS) {
    let cellHash = 0, libHash = 0;
    for (let k = 0; k < 9; k++) {
      const src = perm[k];
      cellHash += cellCodes[src] * POW3[k];
      libHash  += libs[src]      * LIB_WEIGHT[k];
    }
    const combined = cellHash + 19683 * libHash;
    if (combined < minHash) minHash = combined;
  }
  return minHash;
}

// patternHashes2(game2, indices) — batch hash with ladder urgency flags.
// Returns an array of { idx, pHash } in the same order as `indices`.
// Produces identical pHash values to patternHashes(game, coords) in patterns.js
// for corresponding positions.
function patternHashes2(game2, indices) {
  const N     = game2.N;
  const cap   = N * N;
  const cells = game2.cells;
  const mover = game2.current;

  // Start every cell's ladder-flag at 1 (neutral; above-1 means urgent).
  const ladderFlag  = new Int32Array(cap).fill(1);
  const visitedGids = new Set();

  for (let i = 0; i < cap; i++) {
    const color = cells[i];
    if (color === 0) continue;
    const gid = game2._gid[i];
    if (visitedGids.has(gid)) continue;
    if (game2._ss[gid] < 2) continue;
    visitedGids.add(gid);
    if (game2._ls[gid] > 2) continue;

    const statusEntries = getLadderStatus2(game2, i);
    if (!statusEntries) continue;

    for (const entry of statusEntries) {
      const { x: lx, y: ly } = entry.liberty;
      const li = ly * N + lx;
      if (color === mover && !entry.canEscape) {
        ladderFlag[li]++;  // mover's group is doomed: avoid extending it
      } else if (color !== mover && entry.canEscape) {
        ladderFlag[li]++;  // opponent's group will escape: avoid chasing it
      }
    }
  }

  return indices.map(idx => {
    const stoneLibHash = patternHash2(game2, idx, mover);
    return { idx, pHash: stoneLibHash * 14836251 + ladderFlag[idx] * 49576351 };
  });
}

if (typeof module !== 'undefined') module.exports = { patternHash2, patternHashes2, MAX_LIBS };
