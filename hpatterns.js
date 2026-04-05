#!/usr/bin/env node
'use strict';

// hpatterns.js — Hierarchical pattern feature extraction
//
// Extracts 2×2 … maxSize×maxSize pattern features from a Go board position.
// Unlike vpatterns.js: raw cell values only (no liberty counting), all sizes
// included automatically, D4 canonicalisation via algorithmic rotations,
// hash collisions accepted (rare and harmless in a linear learning system).
//
// API:
//   const m = createModel(maxStones, maxSize);
//   const f = extractFeatures(game, m [, maxSearch [, nextMove]]);
//   const v = evaluateFeatures(f, m.weights);
//   const f = evaluate(game, m [, maxSearch [, nextMove]]);
//
// maxStones: plain object { [size]: number } — per-size stone limit; absent sizes are inactive.
// maxSearch: optional cap on the largest pattern size extracted this call.
//            Used by the training loop to skip levels known to have no eligible patterns.
//            f.topLevel (returned) = highest size that had eligible patterns this call.
// nextMove:  if >= 0, speculatively place game.current at nextMove before extracting
//            (captures handled), then restore.

(function () {
  const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
  const { EMPTY, PASS } = _isNode ? require('./game2.js') : window.game;

  // ── D4 permutations ────────────────────────────────────────────────────────
  // perm[dest] = src: applying perm gives rotated[dest] = cells[perm[dest]].

  const d4PermsCache = new Map();

  function makeD4Perms(M) {
    const transforms = [
      (r, c) => [r,       c      ],  // Identity
      (r, c) => [c,       M-1-r  ],  // Rot90CW
      (r, c) => [M-1-r,   M-1-c  ],  // Rot180
      (r, c) => [M-1-c,   r      ],  // Rot270CW
      (r, c) => [r,       M-1-c  ],  // FlipH
      (r, c) => [M-1-r,   c      ],  // FlipV
      (r, c) => [c,       r      ],  // TransposeMD
      (r, c) => [M-1-c,   M-1-r  ],  // TransposeAD
    ];
    return transforms.map(t => {
      const perm = new Int32Array(M * M);
      for (let r = 0; r < M; r++)
        for (let c = 0; c < M; c++) {
          const [nr, nc] = t(r, c);
          perm[nr * M + nc] = r * M + c;
        }
      return perm;
    });
  }

  function getD4Perms(M) {
    if (!d4PermsCache.has(M)) d4PermsCache.set(M, makeD4Perms(M));
    return d4PermsCache.get(M);
  }

  // ── Hash ───────────────────────────────────────────────────────────────────
  // FNV-1a style, 32-bit via Math.imul. +2 shifts {-1,0,1} → {1,2,3}.

  function hash4(a, b, c, d) {
    let h = 2166136261;
    h = Math.imul(h ^ (a + 2), 16777619);
    h = Math.imul(h ^ (b + 2), 16777619);
    h = Math.imul(h ^ (c + 2), 16777619);
    h = Math.imul(h ^ (d + 2), 16777619);
    return h;
  }

  // Iterative hierarchical hash of a flat M×M cell array (stride = M).
  // Builds level by level: each pass reduces the side length by 1, combining
  // four adjacent values into one hash4. O(M³) total — vs O(4^(M−2)) recursive.
  // Used only during canon computation (amortised — once per unique pattern).

  function hierHash(cells, M) {
    let buf = cells;
    let sz  = M;
    while (sz > 2) {
      sz--;
      const stride = sz + 1;  // stride of current buf
      const next = new Array(sz * sz);
      for (let r = 0; r < sz; r++)
        for (let c = 0; c < sz; c++)
          next[r * sz + c] = hash4(
            buf[ r      * stride + c],
            buf[ r      * stride + c + 1],
            buf[(r + 1) * stride + c],
            buf[(r + 1) * stride + c + 1]
          );
      buf = next;
    }
    return hash4(buf[0], buf[1], buf[2], buf[3]);
  }

  // Cross-shaped hash for M×M windows (M≥4): covers M×M minus the 4 corner cells.
  // Uses four standard (M-2)×(M-2) sub-window hashes at the cross positions
  // (0,1), (1,0), (1,2), (2,1) — matching the board-level rawKey formula.
  // Called only from computeAndStoreCanon (amortised).

  function crossHierHash(winCells, M) {
    const K = M - 2;
    function subHash(rOff, cOff) {
      const sub = new Array(K * K);
      for (let r = 0; r < K; r++)
        for (let c = 0; c < K; c++)
          sub[r * K + c] = winCells[(rOff + r) * M + (cOff + c)];
      return hierHash(sub, K);
    }
    return hash4(subHash(0, 1), subHash(1, 0), subHash(1, 2), subHash(2, 1));
  }

  // ── Canonicalisation ───────────────────────────────────────────────────────
  // Generates 16 variants (8 D4 rotations × 2 color flips), finds the minimum
  // hash as the canonical key, detects color-twin patterns, and stores all 16
  // rawKey → encoded mappings in canonMap for future O(1) lookups.
  //
  // Encoding: a single integer per rawKey, always a V8 Smi (fast path, no heap boxing).
  //   twin              → 0
  //   polarity +1       → abs(canonKey) + 1   (positive)
  //   polarity -1       → -(abs(canonKey) + 1) (negative)
  // Decoding (hot path — comparisons only, no division or modulo):
  //   isTwin   = enc === 0
  //   polarity = enc > 0 ? 1 : -1
  //   outKey   = (enc > 0 ? enc : -enc) - 1   i.e. abs(enc) - 1

  function computeAndStoreCanon(winCells, M, canonMap) {
    const perms = getD4Perms(M);
    const n2 = M * M;
    const rotated   = new Array(n2);
    const vHashes   = new Array(16);
    const vParities = new Array(16);  // +1 = non-inverted, -1 = color-inverted
    const hashFn    = M >= 4 ? crossHierHash : hierHash;

    for (let pi = 0; pi < 8; pi++) {
      const perm = perms[pi];

      // Non-inverted rotation
      for (let i = 0; i < n2; i++) rotated[i] =  winCells[perm[i]];
      vHashes[pi * 2]     = hashFn(rotated, M);
      vParities[pi * 2]   = 1;

      // Color-inverted rotation
      for (let i = 0; i < n2; i++) rotated[i] = -winCells[perm[i]];
      vHashes[pi * 2 + 1]   = hashFn(rotated, M);
      vParities[pi * 2 + 1] = -1;
    }

    // Canonical key = minimum signed int32 across all 16 variants.
    let canonKey    = vHashes[0];
    let canonParity = vParities[0];
    for (let i = 1; i < 16; i++) {
      if (vHashes[i] < canonKey) {
        canonKey    = vHashes[i];
        canonParity = vParities[i];
      }
    }

    // Color-twin: any non-inverted hash equals any inverted hash →
    // pattern is symmetric under color-inversion + some D4 rotation,
    // so it contributes zero to the value function.
    let isTwin = false;
    outer: for (let i = 0; i < 8; i++) {
      for (let j = 0; j < 8; j++) {
        if (vHashes[i * 2] === vHashes[j * 2 + 1]) { isTwin = true; break outer; }
      }
    }

    // Polarity formula: polarity[i] = colorParity[i] * canonParity
    // Guarantees polarity[canonVariant] === +1.
    for (let i = 0; i < 16; i++) {
      if (canonMap.has(vHashes[i])) continue;  // earlier entry (smaller M) wins
      const pol    = vParities[i] * canonParity;
      const absKey = canonKey < 0 ? -canonKey : canonKey;
      canonMap.set(vHashes[i], isTwin ? 0 : (pol === 1 ? absKey + 1 : -(absKey + 1)));
    }

    return canonMap.get(vHashes[0]);
  }

  // ── Feature extraction ─────────────────────────────────────────────────────

  // maxSearch: optional upper bound on pattern size to extract (caller-managed optimisation).
  // nextMove:  if >= 0, speculatively place game.current at nextMove before extracting
  //            (captures handled), then restore.
  function extractFeatures(game, model, maxSearch, nextMove) {
    const { canonMap, maxStones } = model;
    const N   = game.N;
    const cap = N * N;
    const cells = game.cells;  // Int8Array: +1=BLACK, -1=WHITE, 0=EMPTY
    const effMaxSize = Math.min(
      model.maxSize === Infinity ? N : model.maxSize,
      N
    );
    const searchMaxSize = maxSearch !== undefined ? Math.min(maxSearch, effMaxSize) : effMaxSize;

    // Speculative stone placement: temporarily modify cells, then restore.
    if (nextMove >= 0) {
      const captures = game.captureList(nextMove);
      for (let i = 0; i < captures.length; i++) cells[captures[i]] = EMPTY;
      cells[nextMove] = game.current;

      const result = _extractCore(game, model, effMaxSize, N, cap, cells, canonMap, maxStones, searchMaxSize);

      cells[nextMove] = EMPTY;
      for (let i = 0; i < captures.length; i++) cells[captures[i]] = -game.current;
      return result;
    }

    return _extractCore(game, model, effMaxSize, N, cap, cells, canonMap, maxStones, searchMaxSize);
  }

  function _extractCore(game, model, effMaxSize, N, cap, cells, canonMap, maxStones, searchMaxSize) {
    if (searchMaxSize === undefined) searchMaxSize = effMaxSize;
    // Per-level hash arrays (reused across calls for the same board/size config).
    if (!model._hBufs || model._hCap !== cap || model._hMaxSize !== effMaxSize) {
      model._hBufs = new Array(effMaxSize - 1);
      for (let m = 0; m < effMaxSize - 1; m++) model._hBufs[m] = new Int32Array(cap);
      model._hCap     = cap;
      model._hMaxSize = effMaxSize;
    }
    const hBufs = model._hBufs;

    const maxFeatures = cap * (effMaxSize - 1);
    if (!model._outKeys || model._outKeys.length < maxFeatures) {
      model._outKeys = new Int32Array(maxFeatures);
      model._outPols = new Int8Array(maxFeatures);
    }
    const outKeys = model._outKeys;
    const outPols = model._outPols;
    let count = 0;

    // 2D prefix sum of stone presence over the N×N board (non-toroidal part).
    // P[(r+1)*(N+1)+(c+1)] = number of non-empty cells in rows [0..r], cols [0..c].
    // Used for O(1) stone counting in non-wrapping windows.
    const Np1 = N + 1;
    if (!model._prefixBuf || model._prefixBuf.length < Np1 * Np1) {
      model._prefixBuf = new Int32Array(Np1 * Np1);
    }
    const P = model._prefixBuf;
    for (let r = 0; r < N; r++) {
      let rowSum = 0;
      for (let c = 0; c < N; c++) {
        rowSum += cells[r * N + c] !== 0 ? 1 : 0;
        P[(r + 1) * Np1 + (c + 1)] = rowSum + P[r * Np1 + (c + 1)];
      }
    }

    const winCells = new Array(effMaxSize * effMaxSize);  // scratch for canon
    let topActive = 1;  // highest M where anyEligible was true
    for (let M = 2; M <= searchMaxSize; M++) {
      const hM    = hBufs[M - 2];
      const hPrev = M > 2 ? hBufs[M - 3] : null;
      const hPrev2 = M > 3 ? hBufs[M - 4] : null;
      const limit = maxStones[M] ?? 0;
      let anyEligible = false;

      for (let idx = 0; idx < cap; idx++) {
        // Standard hash: always stored in hBufs so higher levels can use it.
        const stdKey = M === 2
          ? hash4(cells[idx], cells[(idx + 1) % cap], cells[(idx + N) % cap], cells[(idx + N + 1) % cap])
          : hash4(hPrev[idx], hPrev[(idx + 1) % cap], hPrev[(idx + N) % cap], hPrev[(idx + N + 1) % cap]);
        hM[idx] = stdKey;
        if (limit === 0) continue;

        // For M≥4 use cross-shaped rawKey (M×M minus corners); else rawKey = stdKey.
        const rawKey = M < 4 ? stdKey
          : hash4(hPrev2[(idx + 1) % cap], hPrev2[(idx + N) % cap], hPrev2[(idx + N + 2) % cap], hPrev2[(idx + 2*N + 1) % cap]);

        // Stone count over the active cells (M≥4: exclude 4 corners; else all M×M).
        // O(1) via prefix sum for non-wrapping windows, O(M²) fallback for toroidal wrap.
        const row = (idx / N) | 0;
        const col = idx % N;
        let stones;
        if (col + M <= N && row + M <= N) {
          stones = P[(row + M) * Np1 + (col + M)] - P[row * Np1 + (col + M)]
                 - P[(row + M) * Np1 + col]       + P[row * Np1 + col];
          if (M >= 4) {
            if (cells[ row          * N + col        ] !== 0) stones--;
            if (cells[ row          * N + col + M - 1] !== 0) stones--;
            if (cells[(row + M - 1) * N + col        ] !== 0) stones--;
            if (cells[(row + M - 1) * N + col + M - 1] !== 0) stones--;
          }
        } else {
          stones = 0;
          for (let dr = 0; dr < M; dr++)
            for (let dc = 0; dc < M; dc++) {
              if (M >= 4 && (dr === 0 || dr === M - 1) && (dc === 0 || dc === M - 1)) continue;
              if (cells[(idx + dr * N + dc) % cap]) stones++;
            }
        }
        if (stones === 0 || stones > limit) continue;
        anyEligible = true;

        // Canon lookup — decode: 0=twin, positive=pol+1, negative=pol-1; outKey=abs(enc)-1.
        let enc = canonMap.get(rawKey);
        if (enc !== undefined) {
          if (enc !== 0) { outKeys[count] = (enc > 0 ? enc : -enc) - 1; outPols[count] = enc > 0 ? 1 : -1; count++; }
          continue;
        }

        // First encounter: compute and store canonical form for all 16 variants.
        for (let dr = 0; dr < M; dr++)
          for (let dc = 0; dc < M; dc++)
            winCells[dr * M + dc] = cells[(idx + dr * N + dc) % cap];
        enc = computeAndStoreCanon(winCells, M, canonMap);
        if (enc === 0) continue;

        outKeys[count] = (enc > 0 ? enc : -enc) - 1;
        outPols[count] = enc > 0 ? 1 : -1;
        count++;
      }

      if (anyEligible) topActive = M;
      if (!anyEligible && limit > 0) break;
    }

    return { keys: outKeys, pols: outPols, count, topLevel: topActive, val: 0.5 };
  }

  // ── Evaluation ─────────────────────────────────────────────────────────────

  function evaluateFeatures(features, weights) {
    let z = 0;
    const { keys, pols, count } = features;
    for (let i = 0; i < count; i++) {
      const w = weights.get(keys[i]);
      if (w !== undefined) z += pols[i] * w;
    }
    return 1 / (1 + Math.exp(-z));
  }

  function evaluate(game, model, maxSearch, nextMove) {
    const f = extractFeatures(game, model, maxSearch, nextMove);
    f.val = evaluateFeatures(f, model.weights);
    return f;
  }

  // ── Model ──────────────────────────────────────────────────────────────────

  // maxStones: plain object {2: n, 3: m, …} — per-size stone limit.
  // maxSize:   largest window size to extract (defaults to board size).
  function createModel(maxStones, maxSize) {
    return {
      weights:   new Map(),
      canonMap:  new Map(),
      maxStones: maxStones !== undefined ? maxStones : {},
      maxSize:   maxSize   !== undefined ? maxSize   : Infinity,
    };
  }

  // ── Exports ────────────────────────────────────────────────────────────────

  const HPatterns = { createModel, extractFeatures, evaluateFeatures, evaluate };
  if (typeof module !== 'undefined') module.exports = HPatterns;
  else window.HPatterns = HPatterns;
})();
