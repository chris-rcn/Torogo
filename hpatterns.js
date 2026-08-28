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

  // ── Hash ───────────────────────────────────────────────────────────────────
  // Symmetry-invariant hierarchical "X" hash.  No D4 enumeration, no canonical-
  // form memo: the hash is invariant by construction, so the value it returns
  // IS the canonical key.
  //
  // uh(a,b) is an unordered (commutative) combiner.  A window's hash pairs its
  // two diagonals and combines them unordered:
  //
  //     p0 p1        xh4 = uh( uh(p0,p3), uh(p1,p2) )
  //     p2 p3
  //
  // That is invariant exactly to the group preserving the partition
  // {{p0,p3},{p1,p2}} — order 8, which on a square is precisely D4.  So at 2×2
  // it collapses the D4 orbits and nothing else: 21 distinct values for the 21
  // orbits of a 3-state 2×2, lossless.
  //
  // Larger sizes recurse on the four (M−1)×(M−1) sub-windows at the corners of
  // the M×M window, in the same X arrangement.  Each sub-hash is itself
  // D4-invariant, so the four permute exactly as the cells of a 2×2 do and the
  // combiner's invariance carries up — every level is D4-invariant.
  //
  // This is deliberately lossy above 2×2: a sub-hash has already discarded its
  // own orientation, so a parent cannot tell a sub-window's stone at top-left
  // from top-right.  Measured fidelity at 3×3 is 2379 distinct values against
  // 2862 true D4 orbits (83%).  Distinct shapes therefore share weights — the
  // trade bought for O(1) memory and no per-window canonicalisation.

  function mix32(x) {
    x |= 0;
    x ^= x >>> 16; x = Math.imul(x, 0x7feb352d);
    x ^= x >>> 15; x = Math.imul(x, 0x846ca68b);
    x ^= x >>> 16;
    return x | 0;
  }

  // Unordered: uh(a,b) === uh(b,a).  Operands are mixed before the commutative
  // combine so that structurally different pairs with equal sums don't alias.
  function uh(a, b) {
    return mix32((mix32(a) + mix32(b)) | 0);
  }

  // X-hash of a 2×2 arrangement: diagonals {tl,br} and {tr,bl}, both unordered,
  // then combined unordered.
  function xh4(tl, tr, bl, br) {
    return uh(uh(tl, br), uh(tr, bl));
  }

  // Hash of the all-empty M×M window, per level: every level of an empty
  // window's tree is uniform, so the value is a per-size constant.  Used to
  // drop empty windows without counting stones.
  const emptyHash = [0, 0];            // [0],[1] unused
  for (let m = 2, h = 0; m <= 32; m++) {
    h = m === 2 ? xh4(0, 0, 0, 0) : xh4(h, h, h, h);
    emptyHash.push(h);
  }

  // ── Feature extraction ─────────────────────────────────────────────────────

  // maxSearch: optional upper bound on pattern size to extract (caller-managed optimisation).
  // nextMove:  if >= 0, speculatively place game.current at nextMove before extracting
  //            (captures handled), then restore.
  function extractFeatures(game, model, maxSearch, nextMove) {
    const { maxStones } = model;
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

      const result = _extractCore(game, model, effMaxSize, N, cap, cells, maxStones, searchMaxSize);

      cells[nextMove] = EMPTY;
      for (let i = 0; i < captures.length; i++) cells[captures[i]] = -game.current;
      return result;
    }

    return _extractCore(game, model, effMaxSize, N, cap, cells, maxStones, searchMaxSize);
  }

  function _extractCore(game, model, effMaxSize, N, cap, cells, maxStones, searchMaxSize) {
    if (searchMaxSize === undefined) searchMaxSize = effMaxSize;
    // Per-level hash arrays (reused across calls for the same board/size config).
    // Two parallel stacks: the board as-is, and colour-inverted.  Comparing the
    // two gives the colour canonicalisation the D4 enumeration used to provide,
    // for one extra tree build instead of 16 variant hashes.
    if (!model._hBufs || model._hCap !== cap || model._hMaxSize !== effMaxSize) {
      model._hBufs    = new Array(effMaxSize - 1);
      model._hBufsInv = new Array(effMaxSize - 1);
      for (let m = 0; m < effMaxSize - 1; m++) {
        model._hBufs[m]    = new Int32Array(cap);
        model._hBufsInv[m] = new Int32Array(cap);
      }
      model._hCap     = cap;
      model._hMaxSize = effMaxSize;
    }
    const hBufs    = model._hBufs;
    const hBufsInv = model._hBufsInv;

    const maxFeatures = cap * (effMaxSize - 1);
    if (!model._outKeys || model._outKeys.length < maxFeatures) {
      model._outKeys = new Int32Array(maxFeatures);
      model._outPols = new Int8Array(maxFeatures);
      model._outSizes = new Int8Array(maxFeatures);
    }
    const outKeys = model._outKeys;
    const outPols = model._outPols;
    const outSizes = model._outSizes;
    let count = 0;

    // 2D prefix sum of stone presence over the N×N board (non-toroidal part).
    // P[(r+1)*(N+1)+(c+1)] = number of non-empty cells in rows [0..r], cols [0..c].
    // Used for O(1) stone counting in non-wrapping windows.  The fill below only
    // writes interior cells and relies on the top-row / left-column borders
    // being zero, so the buffer must be reallocated (fresh zeros) whenever the
    // board size changes — a grow-only reuse would leave a smaller board reading
    // stale borders from a larger previous board, corrupting stone counts.
    const Np1 = N + 1;
    // Only needed by sizes with a binding (0 < limit < M²) stone cap.
    let needCounts = false;
    for (let M = 2; M <= searchMaxSize; M++) {
      const lim = maxStones[M] ?? 0;
      if (lim > 0 && lim < M * M) { needCounts = true; break; }
    }
    if (!model._prefixBuf || model._prefixBuf.length !== Np1 * Np1) {
      model._prefixBuf = new Int32Array(Np1 * Np1);
    }
    const P = model._prefixBuf;
    if (needCounts) {
      for (let r = 0; r < N; r++) {
        let rowSum = 0;
        for (let c = 0; c < N; c++) {
          rowSum += cells[r * N + c] !== 0 ? 1 : 0;
          P[(r + 1) * Np1 + (c + 1)] = rowSum + P[r * Np1 + (c + 1)];
        }
      }
    }

    let topActive = 1;  // highest M where anyEligible was true
    for (let M = 2; M <= searchMaxSize; M++) {
      const hM     = hBufs[M - 2];
      const hPrev  = M > 2 ? hBufs[M - 3] : null;
      const hMI    = hBufsInv[M - 2];
      const hPrevI = M > 2 ? hBufsInv[M - 3] : null;
      const limit = maxStones[M] ?? 0;
      const uncapped = limit >= M * M;   // limit can't exclude any non-empty window
      let anyEligible = false;

      // Toroidal windows: wrap row and column independently.  Flat (idx+1)%cap
      // arithmetic would spill the right column into the next row.
      for (let row = 0; row < N; row++) {
        const rowStart  = row * N;
        const downStart = (row + 1 < N ? row + 1 : 0) * N;   // row below, wrapped
        for (let col = 0; col < N; col++) {
          const idx = rowStart + col;
          const col1 = col + 1 < N ? col + 1 : 0;            // column right, wrapped
          const tr = rowStart + col1;                         // right neighbor (same row)
          const bl = downStart + col;                         // down neighbor
          const br = downStart + col1;                         // down-right neighbor
          // X-hash for both colourings; always stored so higher levels can use them.
          const kN = M === 2
            ? xh4(cells[idx], cells[tr], cells[bl], cells[br])
            : xh4(hPrev[idx], hPrev[tr], hPrev[bl], hPrev[br]);
          const kI = M === 2
            ? xh4(-cells[idx], -cells[tr], -cells[bl], -cells[br])
            : xh4(hPrevI[idx], hPrevI[tr], hPrevI[bl], hPrevI[br]);
          hM[idx]  = kN;
          hMI[idx] = kI;
          if (limit === 0) continue;

          if (uncapped) {
            // A saturated limit (limit ≥ M²) can only exclude the empty
            // window, and empty hashes to a per-size constant — no stone
            // count needed.  (A non-empty window colliding with emptyHash[M]
            // is dropped; same accepted risk as any other hash collision.)
            if (kN === emptyHash[M]) continue;
          } else {
            // Stone count over the full M×M window.  O(1) via prefix sum for
            // non-wrapping windows, O(M²) toroidal fallback (per-axis wrap).
            let stones;
            if (col + M <= N && row + M <= N) {
              stones = P[(row + M) * Np1 + (col + M)] - P[row * Np1 + (col + M)]
                     - P[(row + M) * Np1 + col]       + P[row * Np1 + col];
            } else {
              stones = 0;
              for (let dr = 0; dr < M; dr++) {
                const rr = (row + dr) % N;
                for (let dc = 0; dc < M; dc++) {
                  if (cells[rr * N + (col + dc) % N]) stones++;
                }
              }
            }
            if (stones === 0 || stones > limit) continue;
          }
          anyEligible = true;

          // Colour-twin: the pattern equals its own inverse, so it contributes
          // nothing to an antisymmetric value function.  Eligible, but no feature.
          if (kN === kI) continue;

          // Canonical key = the smaller of the two colourings; polarity says which.
          outKeys[count]  = kN < kI ? kN : kI;
          outPols[count]  = kN < kI ? 1 : -1;
          outSizes[count] = M;
          count++;
        }
      }

      if (anyEligible) topActive = M;
      if (!anyEligible && limit > 0) break;
    }

    return { keys: outKeys, pols: outPols, sizes: outSizes, count, topLevel: topActive, val: 0.5 };
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
      weights:    new Map(),                  // live TD-updated weights
      weightsEMA: new Map(),                  // Polyak-averaged shadow (eval-quality)
      weightsEMAInit: false,                  // first applyEMA seeds EMA = weights
      maxStones:  maxStones !== undefined ? maxStones : {},
      maxSize:    maxSize   !== undefined ? maxSize   : Infinity,
    };
  }

  // Polyak / SWA averaging.  Updates m.weightsEMA in-place to track a
  // smoothed copy of m.weights.  First call seeds EMA = weights so the
  // average doesn't include the zero initialisation.  Subsequent calls do
  //   weightsEMA[k] = alpha * weightsEMA[k] + (1 - alpha) * weights[k]
  // for every interned weight.  Caller picks alpha by desired window — at
  // one snapshot per game, alpha=0.999 averages over ~1000 recent games.
  function applyEMA(m, alpha) {
    const w = m.weights, e = m.weightsEMA;
    if (!m.weightsEMAInit) {
      for (const [k, v] of w) e.set(k, v);
      m.weightsEMAInit = true;
      return;
    }
    const beta = 1 - alpha;
    // Track keys that exist only in weights but not yet in EMA (newly interned
    // since the last applyEMA — they get seeded from current weights value).
    for (const [k, v] of w) {
      const eOld = e.get(k);
      e.set(k, eOld === undefined ? v : alpha * eOld + beta * v);
    }
  }

  // ── Persistence helpers ──────────────────────────────────────────────────────

  // Normalise a raw hpatterns data file's weight table to { count, forEach(cb) },
  // where cb(key, val) is called once per entry.  Supports three forms:
  //   - int16-quantised  ({ keys: Int32Array, qvals: Int16Array, scale, count })
  //     — produced by train-hpatterns.js saveModel; weight = qvals[i] / scale.
  //   - float32 typed arrays ({ keys: Int32Array, vals: Float32Array, count })
  //   - legacy literal Map   ({ weights: Map })
  function modelWeights(raw) {
    if (raw.keys && raw.qvals) {
      const keys = raw.keys, qvals = raw.qvals;
      const count = raw.count != null ? raw.count : keys.length;
      const inv = 1 / raw.scale;
      return { count, forEach(cb) { for (let i = 0; i < count; i++) cb(keys[i], qvals[i] * inv); } };
    }
    if (raw.keys && raw.vals) {
      const keys = raw.keys, vals = raw.vals;
      const count = raw.count != null ? raw.count : keys.length;
      return { count, forEach(cb) { for (let i = 0; i < count; i++) cb(keys[i], vals[i]); } };
    }
    const m = raw.weights;
    return { count: m.size, forEach(cb) { for (const [k, v] of m) cb(k, v); } };
  }

  // Build a Map<key, float> from a raw model in any supported form.
  function weightsMap(raw) {
    const w = new Map();
    modelWeights(raw).forEach((k, v) => w.set(k, v));
    return w;
  }

  // ── Exports ────────────────────────────────────────────────────────────────

  const HPatterns = { createModel, extractFeatures, evaluateFeatures, evaluate, applyEMA, modelWeights, weightsMap };
  if (typeof module !== 'undefined') module.exports = HPatterns;
  else window.HPatterns = HPatterns;
})();
