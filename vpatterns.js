'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.
// Wrapped in an IIFE to avoid polluting the global namespace.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

const { BLACK } = _isNode ? require('./game2.js') : window.Game2;

// ── Constants ─────────────────────────────────────────────────────────────────

// Cell state encoding (signed, color-canonicalized):
//   0            = empty
//  +1 .. +maxLibs = BLACK stone with that many liberties (capped)
//  -1 .. -maxLibs = WHITE stone with that many liberties (capped)

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
  const libs = maxLibs === 1 ? 1 : Math.min(game2.groupLibertyCount(game2.groupIdAt(idx)), maxLibs);
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

  // A pattern is color-symmetric if some D4 transform maps each cell to its
  // negation: cells[perm[i]] === -cells[i] for all i.  Such patterns are
  // zero-value by symmetry (neither colour has an advantage), so return null.
  // Also handles all-empty patterns (0 === -0).
  for (let p = 0; p < perms.length; p++) {
    const perm = perms[p];
    let sym = true;
    for (let i = 0; i < n; i++) { if (cells[perm[i]] !== -cells[i]) { sym = false; break; } }
    if (sym) return null;
  }

  // Compute the key as the minimum hash over all 16 transforms (8 D4 × 2 color-flips).
  let minPos = Infinity, minNeg = Infinity;
  for (let p = 0; p < perms.length; p++) {
    const perm = perms[p];
    let valPos = mixer + 17, valNeg = mixer + 17;
    for (let i = 0; i < n; i++) {
      valPos = (valPos * mixer + cells[perm[i]] + 13) | 0;
      valNeg = (valNeg * mixer - cells[perm[i]] + 13) | 0;
    }
    if (valPos < minPos) minPos = valPos;
    if (valNeg < minNeg) minNeg = valNeg;
  }

  return minPos < minNeg ? { key: minPos, polarity:  1 }
                         : { key: minNeg, polarity: -1 };
}

// ── Multi-spec extraction ─────────────────────────────────────────────────────

// Given an array of specs [{ size: 1|2|3, maxLibs: N }, ...], scans every cell
// and returns a flat array of { key, polarity } for all matching patterns.
//
// Optimisations vs calling pattern1/2/3 individually:
//   - Raw cell states are precomputed once per unique maxLibs value.
//   - Scratch arrays for the cell windows are reused across calls.
//   - pattern1 is inlined (raw[idx] already holds the capped liberty count).
function extractFeatures(game2, specs) {
  const cap  = game2.N * game2.N;
  const nbr  = game2._nbr;
  const dnbr = game2._dnbr;
  const out  = [];

  // Group specs by maxLibs; sort descending so we compute raw once at the
  // highest maxLibs and clamp in-place for each lower value.
  const byMaxLibs = new Map();
  for (const spec of specs) {
    if (!byMaxLibs.has(spec.maxLibs)) byMaxLibs.set(spec.maxLibs, []);
    byMaxLibs.get(spec.maxLibs).push(spec.size);
  }
  const sortedMaxLibs = [...byMaxLibs.keys()].sort((a, b) => b - a);

  const buf2 = [0, 0, 0, 0];              // reusable scratch for 2×2 window
  const buf3 = [0, 0, 0, 0, 0, 0, 0, 0, 0];  // reusable scratch for 3×3 window

  let raw = null;
  for (const maxLibs of sortedMaxLibs) {
    if (raw === null) {
      raw = new Int8Array(cap);
      for (let i = 0; i < cap; i++) raw[i] = rawState(game2, maxLibs, i);
    } else {
      // Clamp in-place: rawState(maxLibs) = sign * min(|rawState(prevMaxLibs)|, maxLibs).
      for (let i = 0; i < cap; i++) {
        if      (raw[i] >  maxLibs) raw[i] =  maxLibs;
        else if (raw[i] < -maxLibs) raw[i] = -maxLibs;
      }
    }
    const sizes = byMaxLibs.get(maxLibs);

    const do1 = sizes.includes(1);
    const do2 = sizes.includes(2);
    const do3 = sizes.includes(3);
    const mix2 = 131 * maxLibs;
    const mix3 = 537 * maxLibs;

    for (let idx = 0; idx < cap; idx++) {
      if (do1) {
        const s = raw[idx];
        if (s !== 0) {
          const libs = s > 0 ? s : -s;
          out.push({ key: libs + 131 * maxLibs, polarity: s > 0 ? 1 : -1 });
        }
      }

      if (do2) {
        const tr = nbr[idx * 4 + 3];
        const bl = nbr[idx * 4 + 1];
        const br = dnbr[idx * 4 + 3];
        buf2[0] = raw[idx]; buf2[1] = raw[tr];
        buf2[2] = raw[bl];  buf2[3] = raw[br];
        const r2 = canonicalize(buf2, PERMS_2x2, mix2);
        if (r2 !== null) out.push(r2);
      }

      if (do3) {
        const c01 = nbr[idx * 4 + 3];
        const c02 = nbr[c01 * 4 + 3];
        const c10 = nbr[idx * 4 + 1];
        const c11 = nbr[c10 * 4 + 3];
        const c12 = nbr[c11 * 4 + 3];
        const c20 = nbr[c10 * 4 + 1];
        const c21 = nbr[c20 * 4 + 3];
        const c22 = nbr[c21 * 4 + 3];
        buf3[0] = raw[idx]; buf3[1] = raw[c01]; buf3[2] = raw[c02];
        buf3[3] = raw[c10]; buf3[4] = raw[c11]; buf3[5] = raw[c12];
        buf3[6] = raw[c20]; buf3[7] = raw[c21]; buf3[8] = raw[c22];
        const r3 = canonicalize(buf3, PERMS_3x3, mix3);
        if (r3 !== null) out.push(r3);
      }
    }
  }
  return out;
}

// ── Value function (pure) ─────────────────────────────────────────────────────

// V(s) = σ(Σ polarity_i · w[key_i]) = P(BLACK wins)
// weights: Map<key, float>  (missing keys treated as 0)
function evaluateFeatures(features, weights) {
  let z = 0;
  for (const f of features) {
    const w = weights.get(f.key);
    if (w !== undefined) z += f.polarity * w;
  }
  return 1 / (1 + Math.exp(-z));
}

// Convenience: extract features and evaluate in one call.
function evaluate(game2, model) {
  return evaluateFeatures(extractFeatures(game2, model.specs), model.weights);
}

// ── Persistence ───────────────────────────────────────────────────────────────

// Loads a model JS file and returns { weights: Map<number,float>, specs: [...] }.
// Always returns a fresh copy so multiple callers don't share the same Map.
function loadWeights(filePath) {
  const raw = require(require('path').resolve(filePath));
  return { specs: raw.specs, weights: new Map(raw.weights) };
}

// Writes a model { weights, specs } to a JS file (browser-includable).
function saveWeights(filePath, model) {
  const fs         = require('fs');
  const specStr    = JSON.stringify(model.specs);
  const weightsStr = '[' + [...model.weights].map(([k, v]) => `[${k},${+v.toFixed(6)}]`).join(',') + ']';
  const src = [
    "'use strict';",
    '// Auto-generated by train-vpatterns.js — do not edit by hand.',
    `const vpatternsModel = { specs: ${specStr}, weights: new Map(${weightsStr}) };`,
    "if (typeof module !== 'undefined') module.exports = vpatternsModel;",
    "else window.vpatternsModel = vpatternsModel;",
  ].join('\n') + '\n';
  fs.writeFileSync(filePath, src);
}

// ── Exports ───────────────────────────────────────────────────────────────────

const Patterns = {
  rawState,
  flipState,
  canonicalize,
  extractFeatures,
  evaluateFeatures,
  evaluate,
  loadWeights,
  saveWeights,
  PERMS_2x2,
  PERMS_3x3,
};

if (typeof module !== 'undefined') module.exports = Patterns;
else window.VPatterns = Patterns;

})();
