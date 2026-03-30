'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.
// Wrapped in an IIFE to avoid polluting the global namespace.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

const { BLACK } = _isNode ? require('./game2.js') : window.Game2;

// ── Constants ─────────────────────────────────────────────────────────────────

// Cell state encoding (color-canonicalized):
//   0               = empty
//   1 .. maxLibs   = own stone with that many liberties (capped)
//   maxLibs+1 .. 2*maxLibs = opponent stone with that many liberties (capped)

// ── D4 symmetry permutations ─────────────────────────────────────────────────

// 2×2 grid, row-major: TL=0, TR=1, BL=2, BR=3
const PERMS_2x2 = [
  [0, 1, 2, 3],   // Identity
  [2, 0, 3, 1],   // Rot90CW:   new(r,c) ← old(N-1-c, r), N=2
  [3, 2, 1, 0],   // Rot180
  [1, 3, 0, 2],   // Rot270CW
  [1, 0, 3, 2],   // FlipH:     new(r,c) ← old(r, N-1-c)
  [2, 3, 0, 1],   // FlipV
  [0, 2, 1, 3],   // TransposeMD
  [3, 1, 2, 0],   // TransposeAD
];

// 3×3 grid, row-major: 0=TL, 4=center, 8=BR
const PERMS_3x3 = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8],   // Identity
  [6, 3, 0, 7, 4, 1, 8, 5, 2],   // Rot90CW
  [8, 7, 6, 5, 4, 3, 2, 1, 0],   // Rot180
  [2, 5, 8, 1, 4, 7, 0, 3, 6],   // Rot270CW
  [2, 1, 0, 5, 4, 3, 8, 7, 6],   // FlipH
  [6, 7, 8, 3, 4, 5, 0, 1, 2],   // FlipV
  [0, 3, 6, 1, 4, 7, 2, 5, 8],   // TransposeMD
  [8, 5, 2, 7, 4, 1, 6, 3, 0],   // TransposeAD
];

// ── Core encoding ─────────────────────────────────────────────────────────────

// Returns the raw state of the cell at idx: 0 for empty, 1-maxLibs for BLACK
// with that many liberties (capped), maxLibs+1 .. 2*maxLibs for WHITE.
function rawState(game2, maxLibs, idx) {
  const color = game2.cells[idx];
  if (color === 0) return 0;
  const libs = Math.min(game2.groupLibs(idx).length, maxLibs);
  if (color === BLACK) return libs;
  return -libs;
}

function flipState(s) {
  return -s;
}

// Given an array of raw cell states and a set of D4 permutation arrays,
// returns { key, polarity } where key is the minimum 
// over all 16 transforms (8 D4 × 2 color-flips), and polarity is +1 or -1
// (whether the minimum-achieving transform used the original or flipped cells).
// Returns null if all cells are empty.
function canonicalize(cells, perms, mixer) {
  const n = cells.length;

  let hasStone = false;
  for (let i = 0; i < n; i++) if (cells[i] !== 0) { hasStone = true; break; }
  if (!hasStone) return null;

  // Compute the minimum encoding over all 16 transforms.
  // c=0: original cells (own=1); c=1: color-flipped cells (own↔opp).
  let key = Infinity, polarity = 1;
  for (let c = 0; c < 2; c++) {
    const flipper = (c === 0) ? 1 : -1;
    for (let p = 0; p < perms.length; p++) {
      const perm = perms[p];
      let val = mixer + 17;
      for (let i = 0; i < n; i++) val = (val * mixer + flipper * cells[perm[i]] + 13) | 0;
      if (val < key) { key = val; polarity = c === 0 ? 1 : -1; }
    }
  }

  return { key: key, polarity };
}

// ── Pattern extractors ────────────────────────────────────────────────────────

// 1×1 pattern: single cell.
// Returns { key, polarity } or null if empty.
// key = liberty count (1-maxLibs), polarity = +1 for BLACK, -1 for WHITE.
function pattern1(game2, maxLibs, idx) {
  const color = game2.cells[idx];
  if (color === 0) return null;
  const libs = Math.min(game2.groupLibs(idx).length, maxLibs);
  const polarity = color === BLACK ? 1 : -1;
  return { key: libs, polarity };
}

// 2×2 pattern: anchor is top-left cell.
// Cells in row-major order: [TL, TR, BL, BR].
// Returns { key, polarity } or null if all empty.
function pattern2(game2, maxLibs, idx) {
  const nbr  = game2._nbr;
  const dnbr = game2._dnbr;

  // TL=idx, TR=right-of-idx, BL=bottom-of-idx, BR=bottom-right-of-idx
  const tr = nbr[idx * 4 + 3];  // right
  const bl = nbr[idx * 4 + 1];  // bottom
  const br = dnbr[idx * 4 + 3]; // bottom-right

  const raw = [
    rawState(game2, maxLibs, idx), rawState(game2, maxLibs, tr),
    rawState(game2, maxLibs,  bl), rawState(game2, maxLibs, br),
  ];

  return canonicalize(raw, PERMS_2x2, 131);
}

// 3×3 pattern: idx is the top-left (anchor) cell.
// Cells in row-major order: row 0 left→right, row 1 left→right, row 2 left→right.
// Returns { key, polarity } or null if all empty.
function pattern3(game2, maxLibs, idx) {
  const nbr = game2._nbr;

  // Navigate by chaining right (nbr[*4+3]) and bottom (nbr[*4+1]) steps.
  const c01 = nbr[idx * 4 + 3];    // right of anchor
  const c02 = nbr[c01 * 4 + 3];    // right of c01
  const c10 = nbr[idx * 4 + 1];    // below anchor
  const c11 = nbr[c10 * 4 + 3];    // right of c10
  const c12 = nbr[c11 * 4 + 3];    // right of c11
  const c20 = nbr[c10 * 4 + 1];    // below c10
  const c21 = nbr[c20 * 4 + 3];    // right of c20
  const c22 = nbr[c21 * 4 + 3];    // right of c21

  const raw = [
    rawState(game2, maxLibs, idx), rawState(game2, maxLibs, c01), rawState(game2, maxLibs, c02),
    rawState(game2, maxLibs, c10), rawState(game2, maxLibs, c11), rawState(game2, maxLibs, c12),
    rawState(game2, maxLibs, c20), rawState(game2, maxLibs, c21), rawState(game2, maxLibs, c22),
  ];

  return canonicalize(raw, PERMS_3x3, 537);
}

// ── Exports ───────────────────────────────────────────────────────────────────

const Patterns = {
  rawState,
  flipState,
  canonicalize,
  pattern1,
  pattern2,
  pattern3,
  PERMS_2x2,
  PERMS_3x3,
};

if (typeof module !== 'undefined') module.exports = Patterns;
else window.Patterns = Patterns;

})();
