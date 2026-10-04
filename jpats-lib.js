#!/usr/bin/env node
'use strict';

// jpats-lib.js — gated hierarchical pattern features.
//
// A sibling of hpatterns.js, sharing its X-hash but replacing the per-size
// stone ceiling (maxStones) with two data-driven mechanisms:
//
//   1. GATED PROMOTION.  A window is "present" when the model already knows
//      it.  The (k+1)x(k+1) window at anchor a is only hashed and looked up
//      when all four of its kxk children — at a, a+right, a+down, a+right+down
//      — are present.  So the receptive field is adaptive per anchor: deep
//      where the local shape is familiar, shallow where it is not.  Depth
//      stops where the model's knowledge stops, instead of at a hand-set
//      stone count.
//
//   2. STOCHASTIC ADMISSION.  A window whose hash is absent is admitted with
//      probability pAdd, so a pattern seen n times enters with probability
//      1-(1-pAdd)^n — a frequency filter costing no counters (median entry at
//      ~7 sightings for pAdd 0.1).  Singleton patterns, which dominate memory
//      and teach nothing, are mostly rejected.  Admission is DEFERRED: the
//      extraction pass only records candidates, and admitPending() creates
//      them.  Two consequences, both wanted:
//        - extraction never mutates the model, so evaluating 169 candidate
//          moves in a 1-ply search cannot admit patterns from moves that were
//          never played (which would inflate the admission rate ~169x and
//          defeat the filter entirely);
//        - a pattern admitted this pass was recorded absent, so it cannot
//          gate its own parent until a LATER pass.  The pyramid grows one
//          level per encounter-generation rather than cascading on a single
//          sighting.
//
// PRESENCE, precisely: a window is present when kN === kI (its two colourings
// hash alike — the all-empty window and colour-twins, both value-neutral: they
// contribute nothing to an antisymmetric value function and so carry no
// weight) OR when its canonical key is in the weight table.  The first case is
// what lets the climb run for free through empty space: an empty board
// promotes to maxPSize on constant-time comparisons, emitting no features.
//
// DEPTH, not a boolean, is the per-anchor state.  Presence is downward-closed
// (a window is only present if its own children were), so depth[a] = the
// largest size present at a captures the whole history, and the gate tests
// depth >= k.  Every write sets a neighbour to k or k+1, both of which satisfy
// the test, so updates are safe in place and the result does not depend on
// scan order.  That matters twice: rotations of a position must extract the
// same features (the X-hash is D4-invariant, and an order-dependent gate would
// throw that away), and an incremental evaluation over a subset of anchors
// must agree with a full sweep.
//
// The hash, emptyHash table and the int16 persistence format are hpatterns'.
// See hpatterns.js for the X-hash derivation and its fidelity trade.
//
// API:
//   const m = createModel({ minPSize, maxPSize, pAdd, maxWeights });
//   const f = extractFeatures(game, m [, opts]);   opts: { nextMove, collectPending }
//   const v = evaluateFeatures(f, m.weights);
//   const f = evaluate(game, m [, opts]);
//   admitPending(m, f);                            // update-time only

(function () {
  const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
  const { EMPTY, PASS } = _isNode ? require('./game2.js') : window.game;
  const { makeIntFloat64Map } = _isNode ? require('./int-map.js') : window.IntMap;

  function makeWeights(minCap) {
    const m = makeIntFloat64Map(minCap);
    m.suppressZeroWarning();
    return m;
  }

  // ── Hash (shared with hpatterns.js) ────────────────────────────────────────
  function uh(a, b) { return (1234567 + a + b + Math.imul(a, b)) | 0; }
  function xh4(tl, tr, bl, br) { return uh(uh(tl, br), uh(tr, bl)); }

  const emptyHash = [0, 0];
  for (let m = 2, h = 0; m <= 32; m++) {
    h = m === 2 ? xh4(2, 2, 2, 2) : xh4(h, h, h, h);
    emptyHash.push(h);
  }

  // ── Model ──────────────────────────────────────────────────────────────────
  // minPSize:    smallest size that emits features and anchors the gate.  Sizes
  //             below it are pure hash scaffolding (the level-M hash is built
  //             from level M-1), computed but never emitted or looked up.
  // maxPSize:    largest size to climb to; defaults to boardSize-1.  Not N: on a
  //             torus the NxN window at every anchor is the whole board, merely
  //             translated, so all N^2 anchors would describe one position.
  // pAdd:       admission probability per sighting of an absent pattern.
  // maxWeights: table ceiling.  Reaching it triggers a prune; the table never
  //             exceeds it — see pruneWeights().
  // pruneRatio: fraction of the weights admitted since the last prune that the
  //             next prune drops.
  function createModel(opts = {}) {
    return {
      weights:    makeWeights(1024),
      weightsEMA: makeWeights(1024),
      weightsEMAInit: false,
      minPSize:    opts.minPSize    !== undefined ? opts.minPSize    : 3,
      maxPSize:    opts.maxPSize    !== undefined ? opts.maxPSize    : Infinity,
      pAdd:       opts.pAdd       !== undefined ? opts.pAdd       : 0.1,
      maxWeights: opts.maxWeights !== undefined ? opts.maxWeights : 4000000,
      pruneRatio: opts.pruneRatio !== undefined ? opts.pruneRatio : 0.5,
      addedSincePrune: 0,               // admissions since the last prune
      pruned:     0,                    // weights dropped over the run
      prunes:     0,                    // prune passes run
      admitted:   new Int32Array(33),   // admitted patterns per size, for diagnostics
      seen:       new Int32Array(33),   // windows examined per size (last extraction)
      fired:      new Int32Array(33),   // features emitted per size (last extraction)
    };
  }

  // Flat: every sighting of an absent pattern is admitted with probability
  // pAdd, whatever the table size.  The ceiling is held by pruning, not by
  // throttling admission — so a shape that only becomes common once the policy
  // is strong can still enter and displace one that has gone dead.
  function admissionP(model) { return model.pAdd; }

  // Triggered on reaching maxWeights: drop pruneRatio of the weights admitted
  // since the last prune, taking those nearest zero.  Nearest-zero is the right
  // criterion for VALUE — those weights contribute nothing to z — but note it
  // is not free structurally: presence gates promotion, so dropping a small
  // pattern also silently disables every larger pattern that had it as a child.
  // Those parents stop firing and become dead entries, reclaimed by a later
  // pass only if their own weight is small.  int-map has no delete, so
  // survivors are rebuilt into a fresh table (also compacts the probe
  // sequence).
  function pruneWeights(model, count) {
    const w = model.weights, n = w.size;
    // Nothing to drop: skip the whole O(n) collect/sort/rebuild and leave
    // addedSincePrune standing, so it accumulates until a pass is worth making.
    if (count < 1) return 0;
    const drop = Math.min(Math.floor(count), n - 1);
    const mags = new Float64Array(n);
    let i = 0;
    w.forEach((k, v) => { mags[i++] = v < 0 ? -v : v; });
    mags.sort();                                   // ascending |w|
    const cut = mags[drop - 1];                    // drop |w| <= cut
    const kept = makeWeights(Math.max(1024, (n - drop) * 2));
    w.forEach((k, v) => { if ((v < 0 ? -v : v) > cut) kept.set(k, v); });
    const dropped = n - kept.size;
    model.weights = kept;
    // The EMA shadow indexes the same keys; drop the entries that no longer
    // exist so it cannot resurrect a pruned weight or grow without bound.
    if (model.weightsEMAInit) {
      const keptE = makeWeights(Math.max(1024, kept.size * 2));
      model.weightsEMA.forEach((k, v) => { if (kept.get(k) !== undefined) keptE.set(k, v); });
      model.weightsEMA = keptE;
    }
    model.pruned += dropped; model.prunes++; model.addedSincePrune = 0;
    return dropped;
  }

  // ── Extraction ─────────────────────────────────────────────────────────────
  function extractFeatures(game, model, opts = {}) {
    const nextMove = opts.nextMove !== undefined ? opts.nextMove : -1;
    const cells = game.cells;
    if (nextMove >= 0) {
      const captures = game.captureList(nextMove);
      for (let i = 0; i < captures.length; i++) cells[captures[i]] = EMPTY;
      cells[nextMove] = game.current;
      const result = _extractCore(game, model, opts);
      cells[nextMove] = EMPTY;
      for (let i = 0; i < captures.length; i++) cells[captures[i]] = -game.current;
      return result;
    }
    return _extractCore(game, model, opts);
  }

  function _extractCore(game, model, opts) {
    const N = game.N, cap = N * N, cells = game.cells;
    const weights = model.weights;
    const minPSize = Math.max(2, model.minPSize);
    const top  = Math.min(model.maxPSize === Infinity ? N - 1 : model.maxPSize, N - 1);
    // collectPending gates the admission rolls.  Off by default so search /
    // candidate evaluation cannot grow the model; the trainer turns it on for
    // the positions it actually updates on.
    const collect = opts.collectPending === true;
    const pAdd = collect ? admissionP(model) : 0;

    if (!model._hBufs || model._hCap !== cap || model._hTop !== top) {
      model._hBufs = new Array(Math.max(top - 1, 1));
      model._hBufsInv = new Array(Math.max(top - 1, 1));
      for (let m = 0; m < Math.max(top - 1, 1); m++) {
        model._hBufs[m] = new Int32Array(cap);
        model._hBufsInv[m] = new Int32Array(cap);
      }
      model._depth = new Int8Array(cap);
      model._hCap = cap; model._hTop = top;
    }
    const hBufs = model._hBufs, hBufsInv = model._hBufsInv, depth = model._depth;

    const maxFeatures = cap * Math.max(top - minPSize + 1, 1);
    if (!model._outKeys || model._outKeys.length < maxFeatures) {
      model._outKeys  = new Int32Array(maxFeatures);
      model._outPols  = new Int8Array(maxFeatures);
      model._outSizes = new Int8Array(maxFeatures);
      model._pending  = new Int32Array(maxFeatures);
      model._pendSize = new Int8Array(maxFeatures);
    }
    const outKeys = model._outKeys, outPols = model._outPols, outSizes = model._outSizes;
    const pending = model._pending, pendSize = model._pendSize;
    let count = 0, pendCount = 0;
    model.seen.fill(0); model.fired.fill(0);

    // ── Scaffolding + base level: every anchor ────────────────────────────────
    for (let M = 2; M <= minPSize && M <= top; M++) {
      const hM = hBufs[M - 2], hMI = hBufsInv[M - 2];
      const hPrev = M > 2 ? hBufs[M - 3] : null, hPrevI = M > 2 ? hBufsInv[M - 3] : null;
      for (let row = 0; row < N; row++) {
        const rowStart = row * N, downStart = (row + 1 < N ? row + 1 : 0) * N;
        for (let col = 0; col < N; col++) {
          const idx = rowStart + col, col1 = col + 1 < N ? col + 1 : 0;
          const tr = rowStart + col1, bl = downStart + col, br = downStart + col1;
          const kN = M === 2 ? xh4(cells[idx] + 2, cells[tr] + 2, cells[bl] + 2, cells[br] + 2)
                             : xh4(hPrev[idx], hPrev[tr], hPrev[bl], hPrev[br]);
          const kI = M === 2 ? xh4(2 - cells[idx], 2 - cells[tr], 2 - cells[bl], 2 - cells[br])
                             : xh4(hPrevI[idx], hPrevI[tr], hPrevI[bl], hPrevI[br]);
          hM[idx] = kN; hMI[idx] = kI;
          if (M !== minPSize) continue;                   // below minPSize: scaffolding only

          model.seen[M]++;
          if (kN === kI) { depth[idx] = M; continue; } // empty or colour-twin: known, weightless
          const key = kN < kI ? kN : kI;
          if (weights.get(key) !== undefined) {
            depth[idx] = M;
            outKeys[count] = key; outPols[count] = kN < kI ? 1 : -1; outSizes[count] = M; count++;
            model.fired[M]++;
          } else {
            depth[idx] = 0;                            // absent: blocks its parent this pass
            if (pAdd > 0 && Math.random() < pAdd) {
              pending[pendCount] = key; pendSize[pendCount] = M; pendCount++;
            }
          }
        }
      }
    }

    // ── Climb: only anchors still at the current depth are examined ───────────
    let reached = minPSize;
    for (let k = minPSize; k < top; k++) {
      const hK = hBufs[k - 2], hKI = hBufsInv[k - 2];
      const hN = hBufs[k - 1], hNI = hBufsInv[k - 1];
      let advanced = 0;
      for (let row = 0; row < N; row++) {
        const rowStart = row * N, downStart = (row + 1 < N ? row + 1 : 0) * N;
        for (let col = 0; col < N; col++) {
          const idx = rowStart + col;
          if (depth[idx] !== k) continue;
          const col1 = col + 1 < N ? col + 1 : 0;
          const tr = rowStart + col1, bl = downStart + col, br = downStart + col1;
          // Gate on depth >= k: a neighbour already advanced this level reads
          // k+1, one not yet visited reads k, and both pass — so the outcome
          // is independent of scan order.
          if (depth[tr] < k || depth[bl] < k || depth[br] < k) continue;
          const kN = xh4(hK[idx], hK[tr], hK[bl], hK[br]);
          const kI = xh4(hKI[idx], hKI[tr], hKI[bl], hKI[br]);
          hN[idx] = kN; hNI[idx] = kI;
          model.seen[k + 1]++;
          if (kN === kI) { depth[idx] = k + 1; advanced++; continue; }
          const key = kN < kI ? kN : kI;
          if (weights.get(key) !== undefined) {
            depth[idx] = k + 1; advanced++;
            outKeys[count] = key; outPols[count] = kN < kI ? 1 : -1; outSizes[count] = k + 1; count++;
            model.fired[k + 1]++;
          } else if (pAdd > 0 && Math.random() < pAdd) {
            pending[pendCount] = key; pendSize[pendCount] = k + 1; pendCount++;
          }
        }
      }
      if (advanced === 0) break;
      reached = k + 1;
    }

    // meanPSize: the mean largest-present pattern SIZE over the anchors that
    // are present at all.  Anchors the model does not know even at minPSize are
    // excluded rather than counted as zero — zero is not a size, and folding
    // "ignorant here" into a mean of sizes makes the number mean neither thing.
    // presentFrac reports how much of the board that mean is drawn from.
    // topPSize (the deepest single anchor) is kept but is a poor progress
    // signal: it tracks the largest coherent region, so it falls as the board
    // fills regardless of what the model has learned.
    let pSum = 0, pN = 0;
    for (let i = 0; i < cap; i++) if (depth[i] >= minPSize) { pSum += depth[i]; pN++; }
    return { keys: outKeys, pols: outPols, sizes: outSizes, count,
             pending, pendSize, pendCount, depth,
             topPSize: reached, meanPSize: pN > 0 ? pSum / pN : 0,
             presentFrac: pN / cap, val: 0.5 };
  }

  // ── Incremental 1-ply evaluation ───────────────────────────────────────────
  // Placing a stone at p changes only the windows CONTAINING p — M^2 anchors per
  // level instead of N^2 — and that set is closed under the gate: a window's
  // children are its own sub-windows, so a parent can only be disturbed if it
  // contains p too.  Both facts together let a candidate move be scored as
  // z_base + delta over the affected cone.
  //
  // jpats needs more than hpatterns' hash patching, because presence is
  // hierarchical: a changed hash can move an anchor's DEPTH, which changes
  // whether its parent may promote.  So the walk carries a patched copy of the
  // depth array and, at each level, gates on a mix of freshly recomputed depths
  // (children inside the cone) and untouched ones (children outside it).
  //
  // Requires the caller to have run extractFeatures on the CURRENT position
  // immediately before, and to not let anything overwrite the buffers in
  // between.  Capture moves disturb windows around every captured stone as
  // well, so those return NaN and the caller falls back to full extraction.

  // Raw z (pre-sigmoid) of an extraction, the base for delta scoring.
  function zOf(features, weights) {
    let z = 0;
    const { keys, pols, count } = features;
    for (let i = 0; i < count; i++) {
      const w = weights.get(keys[i]);
      if (w !== undefined) z += pols[i] * w;
    }
    return z;
  }

  function contribution(kN, kI, weights) {
    if (kN === kI) return 0;                       // empty or colour-twin: weightless
    const w = weights.get(kN < kI ? kN : kI);
    return w === undefined ? 0 : (kN < kI ? w : -w);
  }

  function deltaZ(game, model, weights, move) {
    if (move < 0) return 0;                                    // PASS
    if (game.captureList(move).length > 0) return NaN;         // caller falls back
    const N = game.N, cap = N * N, cells = game.cells, cur = game.current;
    const minPSize = Math.max(2, model.minPSize);
    const top  = Math.min(model.maxPSize === Infinity ? N - 1 : model.maxPSize, N - 1);
    if (top < minPSize || !model._hBufs) return NaN;

    const hB = model._hBufs, hBI = model._hBufsInv, depth = model._depth;
    const pr = (move / N) | 0, pc = move % N;

    // Scratch: per level, the M^2 affected anchors indexed by the offset
    // (dr, dc) of p inside that window — anchor = p - dr*N - dc (toroidal).
    if (!model._dKN || model._dTop !== top) {
      model._dKN = new Array(top + 1); model._dKI = new Array(top + 1);
      for (let M = 2; M <= top; M++) { model._dKN[M] = new Int32Array(M * M); model._dKI[M] = new Int32Array(M * M); }
      model._dDepth = new Int8Array(cap);
      model._dTop = top;
    }
    const dKN = model._dKN, dKI = model._dKI, nd = model._dDepth;
    nd.set(depth);                                 // patched copy; cheap (N^2 bytes)

    let delta = 0;
    for (let M = 2; M <= top; M++) {
      const kNs = dKN[M], kIs = dKI[M];
      const prevKN = M > 2 ? dKN[M - 1] : null, prevKI = M > 2 ? dKI[M - 1] : null;
      const bufN = M > 2 ? hB[M - 3] : null, bufI = M > 2 ? hBI[M - 3] : null;
      let anyLive = false;
      for (let dr = 0; dr < M; dr++) {
        // Row arithmetic depends only on dr, so it is hoisted out of the column
        // loop; the +1 wraps are a compare instead of a modulo (a division).
        const ar  = pr - dr >= 0 ? pr - dr : pr - dr + N;
        const ar1 = ar + 1 < N ? ar + 1 : 0;
        const rowBase = ar * N, downBase = ar1 * N;
        for (let dc = 0; dc < M; dc++) {
          const ac  = pc - dc >= 0 ? pc - dc : pc - dc + N;
          const ac1 = ac + 1 < N ? ac + 1 : 0;
          const a   = rowBase + ac;
          // The three neighbour anchors the gate needs, hoisted: they are plain
          // anchor arithmetic and are read below whether or not M > 2.
          const b1 = rowBase + ac1, b2 = downBase + ac, b3 = downBase + ac1;
          // Gate first.  A window whose gate fails is not present, so no parent
          // can need its hash (a parent needs ALL four children present) — the
          // xh4 pair is pure waste there.  Below minPSize every hash is scaffolding
          // the minPSize level needs, so it is always computed.
          const gated = M < minPSize ? true
                      : M === minPSize ? true
                      : (nd[a] >= M - 1 && nd[b1] >= M - 1 && nd[b2] >= M - 1 && nd[b3] >= M - 1);
          if (!gated) {
            if (depth[a] >= M) { delta -= contribution(hB[M - 2][a], hBI[M - 2][a], weights); anyLive = true; }
            if (nd[a] >= M) nd[a] = M - 1;
            continue;
          }
          let kN, kI;
          if (M === 2) {
            const i1 = a, i2 = b1, i3 = b2, i4 = b3;
            const v1 = i1 === move ? cur : cells[i1], v2 = i2 === move ? cur : cells[i2];
            const v3 = i3 === move ? cur : cells[i3], v4 = i4 === move ? cur : cells[i4];
            kN = xh4(v1 + 2, v2 + 2, v3 + 2, v4 + 2);
            kI = xh4(2 - v1, 2 - v2, 2 - v3, 2 - v4);
          } else {
            // Four (M-1) children; each is inside the cone (use the fresh value)
            // or outside it (unchanged, read the extraction's buffer).
            const P = M - 1;
            let cn0, ci0, cn1, ci1, cn2, ci2, cn3, ci3;
            // child at a: p sits at (dr, dc) if that is inside an (M-1) window
            if (dr < P && dc < P) { const j = dr * P + dc; cn0 = prevKN[j]; ci0 = prevKI[j]; }
            else                  { cn0 = bufN[a];        ci0 = bufI[a]; }
            // child at a+right
            if (dr < P && dc - 1 >= 0 && dc - 1 < P) { const j = dr * P + (dc - 1); cn1 = prevKN[j]; ci1 = prevKI[j]; }
            else                                    { cn1 = bufN[b1];       ci1 = bufI[b1]; }
            // child at a+down
            if (dr - 1 >= 0 && dr - 1 < P && dc < P) { const j = (dr - 1) * P + dc; cn2 = prevKN[j]; ci2 = prevKI[j]; }
            else                                     { cn2 = bufN[b2];      ci2 = bufI[b2]; }
            // child at a+right+down
            if (dr - 1 >= 0 && dr - 1 < P && dc - 1 >= 0 && dc - 1 < P) { const j = (dr - 1) * P + (dc - 1); cn3 = prevKN[j]; ci3 = prevKI[j]; }
            else                                                        { cn3 = bufN[b3]; ci3 = bufI[b3]; }
            kN = xh4(cn0, cn1, cn2, cn3);
            kI = xh4(ci0, ci1, ci2, ci3);
          }
          kNs[dr * M + dc] = kN; kIs[dr * M + dc] = kI;
          if (M < minPSize) continue;                  // scaffolding: no presence, no feature

          // OLD contribution at this level: the extraction emitted one exactly
          // when the anchor was present here and the window was not weightless.
          if (depth[a] >= M) { delta -= contribution(hB[M - 2][a], hBI[M - 2][a], weights); anyLive = true; }

          // One lookup serves both the presence test and the contribution:
          // presence for a non-twin means "in the table", which is the same
          // probe the weight value comes from.
          let present = gated;
          if (present && kN !== kI) {
            const w = weights.get(kN < kI ? kN : kI);
            if (w === undefined) present = false;
            else { nd[a] = M; delta += (kN < kI ? w : -w); anyLive = true; continue; }
          }
          if (present) { nd[a] = M; anyLive = true; }   // twin: present, weightless
          else if (nd[a] >= M) nd[a] = M - 1;
        }
      }
      // Nothing in the cone is present at this level and nothing was present
      // here before, so no higher level can gain or lose a feature: every
      // cone anchor above has at least one child in this cone (the four
      // children tile the parent window, which contains p), and that child
      // now fails the gate.
      if (M >= minPSize && !anyLive) break;
    }
    return delta;
  }

  // ── Admission (the only writer of new keys) ────────────────────────────────
  // Called from the TD/update path, never from search.  New weights start at 0
  // so the value function is continuous at the moment a pattern enters.
  function admitPending(model, f) {
    const w = model.weights;
    let added = 0;
    for (let i = 0; i < f.pendCount; i++) {
      // At the ceiling there is no point admitting: the weight would only be
      // prune fodder, and admission is the sole growth path, so refusing here
      // makes maxWeights an absolute bound rather than a trigger.  Room is made
      // by the next maybePrune().
      if (w.size >= model.maxWeights) break;
      const key = f.pending[i];
      if (w.get(key) !== undefined) continue;
      w.set(key, 0);
      model.admitted[f.pendSize[i]]++;
      added++;
    }
    model.addedSincePrune += added;
    return added;
  }

  // Prune check.  Call ONCE PER TRAINING GAME, not per admission: a prune is an
  // O(n) collect/sort/rebuild of the whole table, so testing after every
  // admitted pattern turns the converged regime into thousands of full rebuilds
  // per hundred games for a couple of weights each.
  function maybePrune(model) {
    if (model.weights.size < model.maxWeights) return 0;
    return pruneWeights(model, model.addedSincePrune * model.pruneRatio);
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

  function evaluate(game, model, opts) {
    const f = extractFeatures(game, model, opts);
    f.val = evaluateFeatures(f, model.weights);
    return f;
  }

  // ── Polyak averaging / persistence (shared with hpatterns.js) ──────────────
  function applyEMA(m, alpha) {
    const w = m.weights, e = m.weightsEMA;
    if (!m.weightsEMAInit) { w.forEach((k, v) => e.set(k, v)); m.weightsEMAInit = true; return; }
    const beta = 1 - alpha;
    w.forEach((k, v) => { const eOld = e.get(k); e.set(k, eOld === undefined ? v : alpha * eOld + beta * v); });
  }

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

  function weightsMap(raw) {
    const mw = modelWeights(raw);
    const w = makeWeights(mw.count * 2);
    mw.forEach((k, v) => w.set(k, v));
    return w;
  }

  const JPats = { createModel, extractFeatures, evaluateFeatures, evaluate,
                  admitPending, maybePrune, admissionP, pruneWeights, applyEMA,
                  modelWeights, weightsMap, emptyHash, xh4, zOf, deltaZ };
  if (typeof module !== 'undefined') module.exports = JPats;
  else window.JPats = JPats;
})();
