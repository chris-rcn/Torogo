'use strict';

// patterns2.js — pattern recognition helpers for Game2 (integer-move engine).
// BROWSER-COMPATIBLE: no Node.js-only APIs (require, process, etc.).
// Loaded as a plain <script> tag; ladder2.js must be loaded before this file.
if (typeof require === 'function') { var { getAllLadderStatuses } = require('./ladder2.js'); }

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
const ZOBRIST = makeZobrist(0x12345678, 9 * 3);

// XOR'd into the final hash when the ladder flag is urgent (ladderFlag > 1).
const LADDER_ZOBRIST = makeZobrist(874245861, 5);

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
  const encode = c => c === 0 ? 0 : c === mover ? 1 : 2;
  cc[0] = encode(cells[rN + cW]); cc[1] = encode(cells[rN + c0]); cc[2] = encode(cells[rN + cE]);
  cc[3] = encode(cells[r0 + cW]); cc[4] = encode(cells[r0 + c0]); cc[5] = encode(cells[r0 + cE]);
  cc[6] = encode(cells[rS + cW]); cc[7] = encode(cells[rS + c0]); cc[8] = encode(cells[rS + cE]);

  // Try all 8 symmetry transforms; keep the minimum Zobrist hash.
  let minHash = 0xFFFFFFFF;
  for (const perm of SYMMETRY_PERMS) {
    let h = 0;
    for (let k = 0; k < 9; k++) {
      h ^= ZOBRIST[k * 3 + cc[perm[k]]];
    }
    h = h >>> 0;  // treat as unsigned uint32
    if (h < minHash) minHash = h;
  }
  return minHash;
}

// patternHashes2(game2, indices) — batch hash with ladder urgency flags.
// Returns an array of { idx, pHash } in the same order as `indices`.
function patternHashes2(game2, indices, ladderStatuses) {
  const N     = game2.N;
  const cap   = N * N;
  const mover = game2.current;

  const ladderFlag = new Int32Array(cap);
  const statuses   = ladderStatuses ?? getAllLadderStatuses(game2);

  for (const { color, entries } of statuses) {
    for (const entry of entries) {
      const { x: lx, y: ly } = entry.liberty;
      const li = ly * N + lx;
      if (color === mover && !entry.canEscape) {        // mover's group is doomed
        ladderFlag[li]++;
      } else if (color !== mover && entry.canEscape) {  // opponent's group will escape
        ladderFlag[li]++;
      }
    }
  }

  return indices.map(idx => {
    const hash = patternHash2(game2, idx, mover);
    const center = LADDER_ZOBRIST[ladderFlag[idx]];
    return { idx, pHash: hash ^ center };
  });
}

if (typeof module !== 'undefined') module.exports = { patternHash2, patternHashes2 };
