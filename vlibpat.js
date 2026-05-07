'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.
// Wrapped in an IIFE to avoid polluting the global namespace.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

const { BLACK, EMPTY, PASS } = _isNode ? require('./game2.js') : window.game;
const { game3FromGame2 } = _isNode ? require('./game3.js') : window.Game3;
const { getAllLadderStatuses } = _isNode ? require('./ladder2.js') : window.Ladder2;

// ── Constants ─────────────────────────────────────────────────────────────────

// Cell state encoding (signed, color-canonicalized) — ladder-aware,
// turn-independent.  5 states (open / dead / other × {empty, B, W}):
//   0   = empty (open)
//  ±1   = dead  (1-2 lib group captured regardless of turn)
//  ±2   = other (alive: libs > 2 or 1-2 lib survives; unsettled: tempo-dependent)
// Sign encodes color (+ BLACK, − WHITE).  Spec is { size: 1|2|3, ladder?: bool }.

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

// 4×4 grid removed; only sizes 1, 2, 3 are supported.

// ── Core encoding ─────────────────────────────────────────────────────────────

// Compute per-cell ladder-aware codes for `game3` (Int8Array(cap)) — turn-
// independent (the encoding does NOT depend on whose move it is).
//   0          empty (open)
//  ±1          dead:  1-2 lib group captured regardless of who moves
//  ±2          other: alive (libs > 2 or 1-2 lib survives) or unsettled
// Sign encodes color (+ BLACK, − WHITE).  |code| ∈ {0, 1, 2}.
function computeLadderCodes(game3) {
  const N = game3.N;
  const cap = N * N;
  const codes = new Int8Array(cap);
  // Default: stones get ±2 (other = "alive or unsettled"), empties stay 0.
  for (let i = 0; i < cap; i++) {
    const c = game3.cells[i];
    if (c !== 0) codes[i] = 2 * c;
  }
  if (game3.emptyCount === cap) return codes;

  // Single ladder pass.  For each 1-2 lib group, classify dead vs other from
  // the {moverSucceeds, urgentLibs} pair (turn-independent because:
  //   urgentLibs.length === 0 → mover doesn't need to act, so the same outcome
  //   holds even if the other side moves first).
  const infos = getAllLadderStatuses(game3);
  if (infos.length === 0) return codes;
  const cur = game3.current;
  const codeByGid = new Map();
  for (const info of infos) {
    if (!info.status) continue;
    const { moverSucceeds, urgentLibs } = info.status;
    const groupColor = info.color;
    const defending  = groupColor === cur;
    const sign       = groupColor > 0 ? 1 : -1;
    // Reduce to dead vs other:
    //   urgentLibs > 0     → unsettled → other (mag=2)
    //   moverSucceeds=true → defender wins (alive=other) / attacker captures (dead)
    //   moverSucceeds=false→ defender can't save (dead) / attacker can't kill (other)
    let mag;
    if (urgentLibs.length > 0)  mag = 2;                 // unsettled → other
    else if (moverSucceeds)     mag = defending ? 2 : 1; // alive vs dead
    else                        mag = defending ? 1 : 2; // dead vs alive
    codeByGid.set(info.gid, mag * sign);
  }
  if (codeByGid.size === 0) return codes;
  for (let i = 0; i < cap; i++) {
    if (game3.cells[i] === 0) continue;
    const code = codeByGid.get(game3.groupIdAt(i));
    if (code !== undefined) codes[i] = code;
  }
  return codes;
}

// Single-cell convenience wrapper (slow: rebuilds game3 + ladder analysis each
// call).  Kept for tests / external callers.  Internally extractFeatures uses
// computeLadderCodes once per position.
function rawState(game, idx) {
  if (game.cells[idx] === 0) return 0;
  const game3 = game3FromGame2(game);
  const codes = computeLadderCodes(game3);
  return codes[idx];
}

// Given an array of raw cell states and a set of D4 permutation arrays,
// returns { key, polarity } where key is the minimum 
// over all 16 transforms (8 D4 × 2 color-flips), and polarity is +1 or -1
// (whether the minimum-achieving transform used the original or flipped cells).
// Returns null if all cells are empty.
function canonicalize(cells, perms, mixer) {
  const n = perms[0].length;

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

// Ladder-aware code range: |c| ≤ 2 (5 states: empty / dead-{B,W} / other-{B,W}).
// No-ladder (presence-only): |c| ≤ 1 (3 states, just sign of game.cells[]).
const ML       = 2, BASE    = 2 * ML + 1;        // ladder:    5
const ML_NL    = 1, BASE_NL = 2 * ML_NL + 1;     // no-ladder: 3

// All LUTs are sparse Map-based caches, populated on first lookup of each
// distinct raw pattern.  Storage scales with patterns actually encountered,
// not the worst-case raw configuration count.
function _makeSparseLUT(cellCount, perms, ml, mix) {
  const base = 2 * ml + 1;
  const lut = {
    cacheKey: new Map(),
    cachePol: new Map(),
    perms, mix, base, ml,
  };
  // Precompute base powers (b1..b{cellCount-1}) for index computation.
  let p = 1;
  for (let j = 1; j < cellCount; j++) { p *= base; lut['b' + j] = p; }
  return lut;
}

// Spec = { size: 1|2|3, ladder?: boolean }.  ladder defaults to true.
//   ladder:true  → 7-state ladder-aware encoding (alive/dead/unsettled/+sign)
//   ladder:false → 3-state presence-only encoding (empty/black/white)
// Different mixer constants ensure no key collisions across variants.
function prepareSpecs(specs) {
  const need = { s1L:false, s2L:false, s3L:false,
                 s1NL:false, s2NL:false, s3NL:false };
  for (const spec of specs) {
    if (spec.size === 4) throw new Error('vlibpat: size 4 is not supported');
    const ladder = spec.ladder !== false;  // default true
    if      (spec.size === 1 && ladder)  need.s1L  = true;
    else if (spec.size === 1 && !ladder) need.s1NL = true;
    else if (spec.size === 2 && ladder)  need.s2L  = true;
    else if (spec.size === 2 && !ladder) need.s2NL = true;
    else if (spec.size === 3 && ladder)  need.s3L  = true;
    else if (spec.size === 3 && !ladder) need.s3NL = true;
  }

  // Mixer constants (chosen distinct to avoid cross-variant key collisions).
  const mixL2  = 131 * ML;     // size 2 ladder
  const mixL3  = 537 * ML;     // size 3 ladder
  const mixNL2 = 1093;         // size 2 no-ladder
  const mixNL3 = 2741;         // size 3 no-ladder

  const lut2L  = need.s2L  ? _makeSparseLUT(4, PERMS_2x2, ML,    mixL2)  : null;
  const lut3L  = need.s3L  ? _makeSparseLUT(9, PERMS_3x3, ML,    mixL3)  : null;
  const lut2NL = need.s2NL ? _makeSparseLUT(4, PERMS_2x2, ML_NL, mixNL2) : null;
  const lut3NL = need.s3NL ? _makeSparseLUT(9, PERMS_3x3, ML_NL, mixNL3) : null;

  let totalSizes = 0;
  for (const k of Object.keys(need)) if (need[k]) totalSizes++;

  return { need, lut2L, lut3L, lut2NL, lut3NL, totalSizes };
}

// Sparse-LUT-driven size-2 extraction.  raw is a per-cell code array (Int8/Uint8);
// lut is from _makeSparseLUT (with cacheKey/cachePol Maps + base/ml/perms/mix +
// b1=base, b2=base^2, b3=base^3).
function _extractSize2(raw, cap, N, lut, outKeys, outPols, count, buf) {
  const { cacheKey, cachePol, perms, mix, base, b2, b3, ml } = lut;
  for (let idx = 0; idx < cap; idx++) {
    buf[0] = raw[idx];
    buf[1] = raw[(idx+1)  %cap];
    buf[2] = raw[(idx+N)  %cap];
    buf[3] = raw[(idx+N+1)%cap];
    const li = (buf[0]+ml) + base*(buf[1]+ml) + b2*(buf[2]+ml) + b3*(buf[3]+ml);
    let pol = cachePol.get(li);
    if (pol === undefined) {
      const r = canonicalize(buf, perms, mix);
      if (r === null) { cachePol.set(li, 0); continue; }
      cacheKey.set(li, r.key);
      cachePol.set(li, r.polarity);
      outKeys[count] = r.key; outPols[count] = r.polarity; count++;
    } else if (pol !== 0) {
      outKeys[count] = cacheKey.get(li); outPols[count] = pol; count++;
    }
  }
  return count;
}

// Sparse-LUT-driven size-3 extraction.  Mirrors _extractSize2 with a 3×3 window.
function _extractSize3(raw, cap, N, lut, outKeys, outPols, count, buf) {
  const { cacheKey, cachePol, perms, mix, base, b2, b3, b4, b5, b6, b7, b8, ml } = lut;
  for (let idx = 0; idx < cap; idx++) {
    buf[0] = raw[idx];
    buf[1] = raw[(idx+1)    %cap]; buf[2] = raw[(idx+2)    %cap];
    buf[3] = raw[(idx+N)    %cap]; buf[4] = raw[(idx+N+1)  %cap]; buf[5] = raw[(idx+N+2)  %cap];
    buf[6] = raw[(idx+2*N)  %cap]; buf[7] = raw[(idx+2*N+1)%cap]; buf[8] = raw[(idx+2*N+2)%cap];
    const li =
      (buf[0]+ml)    + base*(buf[1]+ml) + b2*(buf[2]+ml) +
      b3*(buf[3]+ml) + b4  *(buf[4]+ml) + b5*(buf[5]+ml) +
      b6*(buf[6]+ml) + b7  *(buf[7]+ml) + b8*(buf[8]+ml);
    let pol = cachePol.get(li);
    if (pol === undefined) {
      const r = canonicalize(buf, perms, mix);
      if (r === null) { cachePol.set(li, 0); continue; }
      cacheKey.set(li, r.key);
      cachePol.set(li, r.polarity);
      outKeys[count] = r.key; outPols[count] = r.polarity; count++;
    } else if (pol !== 0) {
      outKeys[count] = cacheKey.get(li); outPols[count] = pol; count++;
    }
  }
  return count;
}

// and returns a flat array of { key, polarity } for all matching patterns.
//
// Optimisations vs calling pattern1/2/3 individually:
//   - Raw cell states are precomputed once per position via computeLadderCodes.
//   - size:2 and size:3 use precomputed lookup tables (built by prepareSpecs) to
//     replace the 8-permutation canonicalize loop with a single array index.
//   - pattern1 is inlined (raw[idx] already holds the capped liberty count).
function extractFeatures(game, prepSpecs, doSetNext, nextMove) {
  const cap   = game.N * game.N;
  const N     = game.N;

  // Pre-allocate flat typed output arrays; max one feature per cell per size entry.
  const maxF   = cap * prepSpecs.totalSizes;
  const outKeys = new Int32Array(maxF);
  const outPols = new Int8Array(maxF);
  let   count   = 0;

  if (nextMove === PASS) doSetNext = false;

  // Build a Game3 mirror so we can apply candidate moves with full play/undo
  // and run ladder analysis on the resulting position.
  const game3 = game3FromGame2(game);
  let movePlayed = false;
  if (doSetNext) {
    movePlayed = game3.play(nextMove);
  }

  const { need, lut2L, lut3L, lut2NL, lut3NL, mixL2, mixL3, mixNL2, mixNL3 } = prepSpecs;

  // Compute ladder codes only if any ladder spec is requested.
  const rawL  = (need.s1L || need.s2L || need.s3L)    ? computeLadderCodes(game3) : null;
  // No-ladder codes = sign-only cell values (-1, 0, +1) — i.e. game3.cells.
  const rawNL = (need.s1NL || need.s2NL || need.s3NL) ? game3.cells : null;

  const buf = [0, 0, 0, 0, 0, 0, 0, 0, 0];  // scratch for size-3 canonicalize fallback

  // ── size 1 ─────────────────────────────────────────────────────────────────
  if (need.s1L) {
    const k1base = 131 * ML;  // ladder magnitude ∈ {1, 2, 3} → keys k1base+1..3
    for (let idx = 0; idx < cap; idx++) {
      const s = rawL[idx];
      if (s !== 0) {
        outKeys[count] = (s > 0 ? s : -s) + k1base;
        outPols[count] = s > 0 ? 1 : -1;
        count++;
      }
    }
  }
  if (need.s1NL) {
    const k1base = 131 * ML_NL;  // distinct from ladder size-1 base
    for (let idx = 0; idx < cap; idx++) {
      const s = rawNL[idx];
      if (s !== 0) {
        outKeys[count] = 1 + k1base;  // magnitude is always 1 for presence-only
        outPols[count] = s > 0 ? 1 : -1;
        count++;
      }
    }
  }

  // ── size 2 ─────────────────────────────────────────────────────────────────
  if (need.s2L)  count = _extractSize2(rawL,  cap, N, lut2L,  outKeys, outPols, count, buf);
  if (need.s2NL) count = _extractSize2(rawNL, cap, N, lut2NL, outKeys, outPols, count, buf);

  // ── size 3 ─────────────────────────────────────────────────────────────────
  if (need.s3L)  count = _extractSize3(rawL,  cap, N, lut3L,  outKeys, outPols, count, buf);
  if (need.s3NL) count = _extractSize3(rawNL, cap, N, lut3NL, outKeys, outPols, count, buf);

  if (movePlayed) game3.undo();
  return { keys: outKeys, pols: outPols, count, val: 0.5 };
}

// ── Value function (pure) ─────────────────────────────────────────────────────

// V(s) = σ(Σ polarity_i · w[key_i]) = P(BLACK wins)
// features: { keys: Int32Array, pols: Int8Array, count, val }  (from extractFeatures)
// weights: Map<key, float>  (missing keys treated as 0)
function evaluateFeatures(features, weights) {
  let z = 0;
  const { keys, pols, count } = features;
  for (let i = 0; i < count; i++) {
    const w = weights.get(keys[i]) ?? 0;
    z += pols[i] * w;
  }
  features.val = 1 / (1 + Math.exp(-z));
  return features.val;
}

// Convenience: extract features and evaluate in one call.
// model must have a preparedSpecs property (see prepareSpecs).
function evaluate(game, model) {
  return evaluateFeatures(extractFeatures(game, model.preparedSpecs), model.weights);
}

// ── Persistence ───────────────────────────────────────────────────────────────

// Loads a model JS file and returns { weights: Map<number,float>, specs: [...] }.
// Always returns a fresh copy so multiple callers don't share the same Map.
function loadWeights(filePath) {
  const raw = require(require('path').resolve(filePath));
  const specs = raw.specs;
  return { specs, preparedSpecs: prepareSpecs(specs), weights: new Map(raw.weights) };
}

// Writes a model { weights, specs } to a JS file (browser-includable).
function saveWeights(filePath, model) {
  const fs         = require('fs');
  const specStr    = JSON.stringify(model.specs);
  const weightsStr = '[' + [...model.weights].map(([k, v]) => `[${k},${+v.toFixed(6)}]`).join(',') + ']';
  const src = [
    "'use strict';",
    '// Auto-generated by train-vlibpat.js — do not edit by hand.',
    `const vlibpatModel = { specs: ${specStr}, weights: new Map(${weightsStr}) };`,
    "if (typeof module !== 'undefined') module.exports = vlibpatModel;",
    "else window.vlibpatModel = vlibpatModel;",
  ].join('\n') + '\n';
  fs.writeFileSync(filePath, src);
}

// ── Exports ───────────────────────────────────────────────────────────────────

const Patterns = {
  rawState,
  canonicalize,
  prepareSpecs,
  extractFeatures,
  evaluateFeatures,
  evaluate,
  loadWeights,
  saveWeights,
  PERMS_2x2,
  PERMS_3x3,
};

if (typeof module !== 'undefined') module.exports = Patterns;
else window.VlibPat = Patterns;

})();
