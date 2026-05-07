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
// turn-independent:
//   0   = empty
//  ±1   = alive  (libs > 2, or 1-2 lib group surviving regardless of turn)
//  ±2   = dead   (1-2 lib group captured regardless of turn)
//  ±3   = unsettled (1-2 lib group whose fate depends on whose turn it is)
// Sign encodes color (+ BLACK, − WHITE).  Spec is just { size: 1|2|3 }.

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
//   0          empty
//  ±1          alive: group with > 2 liberties, OR 1-2 lib group that survives
//              optimal play regardless of who moves next
//  ±2          dead: 1-2 lib group that gets captured regardless of who moves
//  ±3          unsettled: 1-2 lib group whose fate depends on whose turn it is
// Sign encodes color (+ BLACK, − WHITE).  |code| ∈ {0, 1, 2, 3}.
function computeLadderCodes(game3) {
  const N = game3.N;
  const cap = N * N;
  const codes = new Int8Array(cap);
  // Default: stones get ±1 (alive), empties stay 0.
  for (let i = 0; i < cap; i++) codes[i] = game3.cells[i];
  if (game3.emptyCount === cap) return codes;

  // Single ladder pass.  For each 1-2 lib group, classify alive/dead/unsettled
  // from the {moverSucceeds, urgentLibs} pair (turn-independent because:
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
    let mag;
    if (urgentLibs.length > 0) {
      // Either side can flip the outcome by playing an urgent lib first.
      mag = 3;  // unsettled
    } else if (moverSucceeds) {
      // Mover's preferred outcome is already locked in (no action needed).
      // Defender's wish is "live", attacker's wish is "die".
      mag = defending ? 1 : 2;
    } else {
      // Mover can't change the outcome; result is opposite of mover's wish.
      mag = defending ? 2 : 1;
    }
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

// Ladder-aware code range: |c| ≤ 3 (7 states).  No-ladder (presence-only):
// |c| ≤ 1 (3 states, just sign of game.cells[]).
const ML       = 3, BASE    = 2 * ML + 1;        // ladder:    7
const ML_NL    = 1, BASE_NL = 2 * ML_NL + 1;     // no-ladder: 3

// Build a LUT: enumerate all base^cellCount raw configs, canonicalize each.
function _buildLUT(cellCount, perms, ml, mix) {
  const base = 2 * ml + 1;
  const n    = base ** cellCount;
  const keys = new Int32Array(n);
  const pols = new Int8Array(n);
  const c    = new Array(cellCount).fill(0);
  for (let i = 0; i < n; i++) {
    let tmp = i;
    for (let j = 0; j < cellCount; j++) { c[j] = (tmp % base) - ml; tmp = (tmp / base) | 0; }
    const r = canonicalize(c, perms, mix);
    if (r !== null) { keys[i] = r.key; pols[i] = r.polarity; }
  }
  return { keys, pols, base, ml };
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

  let lut2L  = null, lut3L  = null;
  let lut2NL = null, lut3NL = null;

  if (need.s2L) {
    const lut = _buildLUT(4, PERMS_2x2, ML, mixL2);
    const b = BASE;
    lut2L = { ...lut, b2: b*b, b3: b*b*b };
  }
  if (need.s3L && BASE ** 9 <= 4000000) {
    // BASE=7, 7^9=40M — currently disabled; LUT path stays on the fallback
    // canonicalize (kept for parity with prior versions).
    const lut = _buildLUT(9, PERMS_3x3, ML, mixL3);
    const b = BASE;
    lut3L = { ...lut, b2: b*b, b3: b**3, b4: b**4, b5: b**5, b6: b**6, b7: b**7, b8: b**8 };
  }
  if (need.s2NL) {
    const lut = _buildLUT(4, PERMS_2x2, ML_NL, mixNL2);
    const b = BASE_NL;
    lut2NL = { ...lut, b2: b*b, b3: b*b*b };
  }
  if (need.s3NL) {  // 3^9 = 19683, easily enumerable
    const lut = _buildLUT(9, PERMS_3x3, ML_NL, mixNL3);
    const b = BASE_NL;
    lut3NL = { ...lut, b2: b*b, b3: b**3, b4: b**4, b5: b**5, b6: b**6, b7: b**7, b8: b**8 };
  }

  let totalSizes = 0;
  for (const k of Object.keys(need)) if (need[k]) totalSizes++;

  return { need, lut2L, lut3L, lut2NL, lut3NL, mixL2, mixL3, mixNL2, mixNL3, totalSizes };
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

  const { need, lut2L, lut2NL, lut3NL, mixL2, mixL3, mixNL2, mixNL3 } = prepSpecs;

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
  if (need.s2L && lut2L) {
    const { keys, pols, b2, b3, ml } = lut2L;
    for (let idx = 0; idx < cap; idx++) {
      const li = (rawL[idx]+ml) + BASE*(rawL[(idx+1)%cap]+ml) + b2*(rawL[(idx+N)%cap]+ml) + b3*(rawL[(idx+N+1)%cap]+ml);
      const pol = pols[li];
      if (pol !== 0) { outKeys[count] = keys[li]; outPols[count] = pol; count++; }
    }
  }
  if (need.s2NL && lut2NL) {
    const { keys, pols, b2, b3, ml } = lut2NL;
    for (let idx = 0; idx < cap; idx++) {
      const li = (rawNL[idx]+ml) + BASE_NL*(rawNL[(idx+1)%cap]+ml) + b2*(rawNL[(idx+N)%cap]+ml) + b3*(rawNL[(idx+N+1)%cap]+ml);
      const pol = pols[li];
      if (pol !== 0) { outKeys[count] = keys[li]; outPols[count] = pol; count++; }
    }
  }

  // ── size 3 ─────────────────────────────────────────────────────────────────
  if (need.s3L) {  // ladder: no LUT (BASE^9 = 40M too big to enumerate); fallback canonicalize
    for (let idx = 0; idx < cap; idx++) {
      buf[0] = rawL[idx];
      buf[1] = rawL[(idx+1)    %cap]; buf[2] = rawL[(idx+2)    %cap];
      buf[3] = rawL[(idx+N)    %cap]; buf[4] = rawL[(idx+N+1)  %cap]; buf[5] = rawL[(idx+N+2)  %cap];
      buf[6] = rawL[(idx+2*N)  %cap]; buf[7] = rawL[(idx+2*N+1)%cap]; buf[8] = rawL[(idx+2*N+2)%cap];
      const r = canonicalize(buf, PERMS_3x3, mixL3);
      if (r !== null) { outKeys[count] = r.key; outPols[count] = r.polarity; count++; }
    }
  }
  if (need.s3NL && lut3NL) {  // no-ladder: 3^9 = 19683, full LUT
    const { keys, pols, b2, b3, b4, b5, b6, b7, b8, ml } = lut3NL;
    for (let idx = 0; idx < cap; idx++) {
      const li =
        (rawNL[idx]            +ml)       +
        BASE_NL*(rawNL[(idx+1)    %cap]+ml) +
        b2     *(rawNL[(idx+2)    %cap]+ml) +
        b3     *(rawNL[(idx+N)    %cap]+ml) +
        b4     *(rawNL[(idx+N+1)  %cap]+ml) +
        b5     *(rawNL[(idx+N+2)  %cap]+ml) +
        b6     *(rawNL[(idx+2*N)  %cap]+ml) +
        b7     *(rawNL[(idx+2*N+1)%cap]+ml) +
        b8     *(rawNL[(idx+2*N+2)%cap]+ml);
      const pol = pols[li];
      if (pol !== 0) { outKeys[count] = keys[li]; outPols[count] = pol; count++; }
    }
  }

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
