'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.
// Wrapped in an IIFE to avoid polluting the global namespace.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

const { BLACK, WHITE, EMPTY, PASS } = _isNode ? require('./game2.js') : window.game;
const { game3FromGame2 } = _isNode ? require('./game3.js') : window.Game3;
const { getAllLadderStatuses } = _isNode ? require('./ladder2.js') : window.Ladder2;

// ── Constants ─────────────────────────────────────────────────────────────────

// Cell state encoding (signed, color-canonicalized) — ladder-aware,
// turn-independent:
//   0   = empty
//  ±1   = alive  (libs > 2, or 1-2 lib group surviving regardless of turn)
//  ±2   = dead   (1-2 lib group captured regardless of turn)
//  ±3   = unsettled (1-2 lib group whose fate depends on whose turn it is)
// Sign encodes color (+ BLACK, − WHITE).  maxLibs in spec caps |code|.

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
// Sign encodes color (+ BLACK, − WHITE).  Caller may clamp via |code| ≤ maxLibs.
function computeLadderCodes(game3) {
  const N = game3.N;
  const cap = N * N;
  const codes = new Int8Array(cap);
  // Default: stones get ±1 (alive), empties stay 0.
  for (let i = 0; i < cap; i++) codes[i] = game3.cells[i];
  if (game3.emptyCount === cap) return codes;

  // Run ladder analysis from BOTH colours' perspectives so the encoding is
  // independent of whose turn it is.  getLadderStatus uses game.current to
  // decide "mover", and play/undo restore game.current — so manually toggling
  // game3.current between the two passes is safe.
  const origCurrent = game3.current;
  game3.current = BLACK;
  const infosB = getAllLadderStatuses(game3);
  game3.current = WHITE;
  const infosW = getAllLadderStatuses(game3);
  game3.current = origCurrent;

  if (infosB.length === 0) return codes;

  // Both passes visit the same gids (group structure & liberties don't depend
  // on whose turn it is).  Pair them by gid.
  const msW = new Map();
  for (const info of infosW) {
    if (info.status) msW.set(info.gid, info.status.moverSucceeds);
  }

  const codeByGid = new Map();
  for (const info of infosB) {
    if (!info.status) continue;
    const gid = info.gid;
    if (!msW.has(gid)) continue;
    const ms_b = info.status.moverSucceeds;
    const ms_w = msW.get(gid);
    const groupColor = info.color;
    // "lives" if the group avoids capture from that side's tempo:
    //   BLACK to move → BLACK group: defender succeeds = ms_b
    //                   WHITE group: attacker (BLACK) fails = !ms_b
    //   WHITE to move → BLACK group: attacker (WHITE) fails = !ms_w
    //                   WHITE group: defender succeeds = ms_w
    const livesIfB = (groupColor === BLACK) ?  ms_b : !ms_b;
    const livesIfW = (groupColor === BLACK) ? !ms_w :  ms_w;
    const sign = groupColor > 0 ? 1 : -1;
    let mag;
    if      ( livesIfB &&  livesIfW) mag = 1;  // alive
    else if (!livesIfB && !livesIfW) mag = 2;  // dead
    else                              mag = 3;  // unsettled
    codeByGid.set(gid, mag * sign);
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
function rawState(game, maxLibs, idx) {
  if (game.cells[idx] === 0) return 0;
  if (maxLibs === 1) return game.cells[idx];
  const game3 = game3FromGame2(game);
  const codes = computeLadderCodes(game3);
  let v = codes[idx];
  if (v >  maxLibs) v =  maxLibs;
  if (v < -maxLibs) v = -maxLibs;
  return v;
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

// Given an array of specs [{ size: 1|2|3, maxLibs: N }, ...], scans every cell
// prepareSpecs: convert a specs array into the internal structure used by
// extractFeatures.  Call once per unique specs array and reuse the result.
// Also precomputes lookup tables for size:2 and size:3:
//   lut2/lut3: Map<maxLibs, { keys: Int32Array, pols: Int8Array, base, b2, b3[, ...], ml }>
//   Index = Σ (cell[i]+maxLibs) * base^i.  pols[i]===0 → skip (symmetric/empty).
function prepareSpecs(specs) {
  const byMaxLibs = new Map();
  for (const spec of specs) {
    if (!byMaxLibs.has(spec.maxLibs)) byMaxLibs.set(spec.maxLibs, []);
    byMaxLibs.get(spec.maxLibs).push(spec.size);
  }
  const sortedMaxLibs = [...byMaxLibs.keys()].sort((a, b) => b - a);

  const lut2 = new Map();
  const lut3 = new Map();
  for (const maxLibs of sortedMaxLibs) {
    const sizes = byMaxLibs.get(maxLibs);
    const base  = 2 * maxLibs + 1;

    if (sizes.includes(2)) {
      const mix  = 131 * maxLibs;
      const n    = base ** 4;
      const keys = new Int32Array(n);
      const pols = new Int8Array(n);
      const c    = [0, 0, 0, 0];
      for (let i = 0; i < n; i++) {
        let tmp = i;
        for (let j = 0; j < 4; j++) { c[j] = (tmp % base) - maxLibs; tmp = (tmp / base) | 0; }
        const r = canonicalize(c, PERMS_2x2, mix);
        if (r !== null) { keys[i] = r.key; pols[i] = r.polarity; }
      }
      const b2 = base * base, b3 = b2 * base;
      lut2.set(maxLibs, { keys, pols, base, b2, b3, ml: maxLibs });
    }

    if (sizes.includes(3) && base ** 9 <= 4000000) {
      const mix  = 537 * maxLibs;
      const n    = base ** 9;
      const keys = new Int32Array(n);
      const pols = new Int8Array(n);
      const c    = [0, 0, 0, 0, 0, 0, 0, 0, 0];
      for (let i = 0; i < n; i++) {
        let tmp = i;
        for (let j = 0; j < 9; j++) { c[j] = (tmp % base) - maxLibs; tmp = (tmp / base) | 0; }
        const r = canonicalize(c, PERMS_3x3, mix);
        if (r !== null) { keys[i] = r.key; pols[i] = r.polarity; }
      }
      const b2 = base*base, b3 = b2*base, b4 = b3*base, b5 = b4*base,
            b6 = b5*base,   b7 = b6*base, b8 = b7*base;
      lut3.set(maxLibs, { keys, pols, base, b2, b3, b4, b5, b6, b7, b8, ml: maxLibs });
    }

    if (sizes.includes(4)) {
      throw new Error('vlibpat: size 4 is no longer supported');
    }
  }

  let totalSizes = 0;
  for (const sizes of byMaxLibs.values()) totalSizes += sizes.length;

  return { byMaxLibs, sortedMaxLibs, lut2, lut3, totalSizes };
}

// and returns a flat array of { key, polarity } for all matching patterns.
//
// Optimisations vs calling pattern1/2/3 individually:
//   - Raw cell states are precomputed once per unique maxLibs value.
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

  // Compute ladder codes once for this (possibly post-move) position.
  const codes = computeLadderCodes(game3);

  const { byMaxLibs, sortedMaxLibs, lut2, lut3 } = prepSpecs;

  const buf = [0, 0, 0, 0, 0, 0, 0, 0, 0];  // scratch for fallback canonicalize (up to 3×3)

  let raw = null;
  for (const maxLibs of sortedMaxLibs) {
    if (raw === null) {
      raw = new Int8Array(cap);
      for (let i = 0; i < cap; i++) {
        let v = codes[i];
        if      (v >  maxLibs) v =  maxLibs;
        else if (v < -maxLibs) v = -maxLibs;
        raw[i] = v;
      }
    } else {
      // Clamp in-place: code(maxLibs) = sign * min(|code(prevMaxLibs)|, maxLibs).
      for (let i = 0; i < cap; i++) {
        if      (raw[i] >  maxLibs) raw[i] =  maxLibs;
        else if (raw[i] < -maxLibs) raw[i] = -maxLibs;
      }
    }
    const sizes = byMaxLibs.get(maxLibs);
    const do1   = sizes.includes(1);
    const do2   = sizes.includes(2);
    const do3   = sizes.includes(3);

    if (do1) {
      const k1base = 131 * maxLibs;
      for (let idx = 0; idx < cap; idx++) {
        const s = raw[idx];
        if (s !== 0) {
          const libs = s > 0 ? s : -s;
          outKeys[count] = libs + k1base;
          outPols[count] = s > 0 ? 1 : -1;
          count++;
        }
      }
    }

    if (do2) {
      const lut = lut2.get(maxLibs);
      if (lut) {
        const { keys, pols, base, b2, b3, ml } = lut;
        for (let idx = 0; idx < cap; idx++) {
          const li = (raw[idx]+ml) + base*(raw[(idx+1)%cap]+ml) + b2*(raw[(idx+N)%cap]+ml) + b3*(raw[(idx+N+1)%cap]+ml);
          const pol = pols[li];
          if (pol !== 0) { outKeys[count] = keys[li]; outPols[count] = pol; count++; }
        }
      } else {
        const mix2 = 131 * maxLibs;
        for (let idx = 0; idx < cap; idx++) {
          buf[0] = raw[idx]; buf[1] = raw[(idx+1)%cap];
          buf[2] = raw[(idx+N)%cap]; buf[3] = raw[(idx+N+1)%cap];
          const r = canonicalize(buf, PERMS_2x2, mix2);
          if (r !== null) { outKeys[count] = r.key; outPols[count] = r.polarity; count++; }
        }
      }
    }

    if (do3) {
      const lut = lut3.get(maxLibs);
      if (false && lut) {
        const { keys, pols, base, b2, b3, b4, b5, b6, b7, b8, ml } = lut;
        for (let idx = 0; idx < cap; idx++) {
          const li =
            (raw[idx]           +ml)       +
            base*(raw[(idx+1)  %cap]+ml)   +
            b2  *(raw[(idx+2)  %cap]+ml)   +
            b3  *(raw[(idx+N)  %cap]+ml)   +
            b4  *(raw[(idx+N+1)%cap]+ml)   +
            b5  *(raw[(idx+N+2)%cap]+ml)   +
            b6  *(raw[(idx+2*N)  %cap]+ml) +
            b7  *(raw[(idx+2*N+1)%cap]+ml) +
            b8  *(raw[(idx+2*N+2)%cap]+ml);
          const pol = pols[li];
          if (pol !== 0) { outKeys[count] = keys[li]; outPols[count] = pol; count++; }
        }
      } else {
        const mix3 = 537 * maxLibs;
        for (let idx = 0; idx < cap; idx++) {
          buf[0] = raw[idx];
          buf[1] = raw[(idx+1)    %cap]; buf[2] = raw[(idx+2)    %cap];
          buf[3] = raw[(idx+N)    %cap]; buf[4] = raw[(idx+N+1)  %cap]; buf[5] = raw[(idx+N+2)  %cap];
          buf[6] = raw[(idx+2*N)  %cap]; buf[7] = raw[(idx+2*N+1)%cap]; buf[8] = raw[(idx+2*N+2)%cap];

          const r = canonicalize(buf, PERMS_3x3, mix3);
          if (r !== null) { outKeys[count] = r.key; outPols[count] = r.polarity; count++; }
        }
      }
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
