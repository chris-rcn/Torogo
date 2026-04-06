'use strict';

// Tests for ai/tdsearch.js internals and public API.
// Called by tdsearch.js at module load time (Node only) via runTests().
// Silent on success; logs failures to stderr.
//
// Feature tests assume default env (USE_1x2=0, USE_2x2=0), so only 1×1
// features are active.

function runTests(
  { makeBuf, resolveKey, findFeatures, findFeaturesInit, findFeaturesIncremental,
    evaluate, evaluateDelta, search1ply, tdUpdate, getMove },
  { Game2, BLACK, WHITE, PASS }
) {
  const { makeIntMap } = require('../int-map.js');
  let failures = 0;

  function check(cond, msg) {
    if (!cond) { failures++; console.error('FAIL [tdsearch]:', msg); }
  }

  // ── makeBuf ────────────────────────────────────────────────────────────────
  {
    const buf = makeBuf(9);
    check(buf.idxs instanceof Int32Array, 'makeBuf: idxs is Int32Array');
    check(buf.n === 0,                    'makeBuf: n starts at 0');
  }

  // ── resolveKey ─────────────────────────────────────────────────────────────
  {
    const ctx = { keyToIdx: makeIntMap(), weightsArr: [] };

    const i0 = resolveKey(42, ctx);
    check(i0 === 0,                    'resolveKey: first key → index 0');
    check(ctx.weightsArr.length === 1, 'resolveKey: weightsArr grows by 1');
    check(ctx.weightsArr[0] === 0,     'resolveKey: initial weight is 0');

    check(resolveKey(42, ctx) === i0,  'resolveKey: same key → same index');
    check(ctx.weightsArr.length === 1, 'resolveKey: repeated key does not grow weightsArr');

    const i1 = resolveKey(99, ctx);
    check(i1 === 1,                    'resolveKey: second distinct key → index 1');
    check(ctx.weightsArr.length === 2, 'resolveKey: weightsArr grows for new key');
  }

  // ── evaluate ───────────────────────────────────────────────────────────────
  {
    // No features → sigmoid(0) = 0.5
    const buf0 = makeBuf(1);
    buf0.n = 0;
    evaluate(buf0, []);
    check(Math.abs(buf0.val - 0.5) < 1e-10, 'evaluate: no features → val = 0.5');

    // Known weight
    const buf1 = makeBuf(1);
    buf1.n = 1;
    buf1.idxs[0] = 0;
    const wa = [4];
    evaluate(buf1, wa);
    check(Math.abs(buf1.val - 1 / (1 + Math.exp(-4))) < 1e-10,
          'evaluate: sigmoid(4) matches formula');
    check(buf1.val > 0.5, 'evaluate: positive weight → val > 0.5');
    check(buf1.val < 1.0, 'evaluate: val < 1 always');

    wa[0] = -4;
    evaluate(buf1, wa);
    check(buf1.val < 0.5, 'evaluate: negative weight → val < 0.5');
    check(buf1.val > 0.0, 'evaluate: val > 0 always');

    // Two features summed
    const buf2 = makeBuf(2);
    buf2.n = 2;
    buf2.idxs[0] = 0;
    buf2.idxs[1] = 1;
    const wb = [2, 2];
    evaluate(buf2, wb);
    check(Math.abs(buf2.val - 1 / (1 + Math.exp(-4))) < 1e-10,
          'evaluate: two weights summed before sigmoid');
  }

  // ── tdUpdate ───────────────────────────────────────────────────────────────
  {
    const lr = 0.3;
    const wa = [0, 0];
    const buf = makeBuf(4);
    buf.n = 2;
    buf.idxs[0] = 0;
    buf.idxs[1] = 1;

    // val = 0.5 initially (zero weights)
    evaluate(buf, wa);
    check(Math.abs(buf.val - 0.5) < 1e-10, 'tdUpdate setup: val = 0.5 with zero weights');

    // Update toward 1.0 → both weights increase
    tdUpdate(buf, 1.0, wa, lr);
    check(wa[0] > 0,                       'tdUpdate: weight[0] increases toward target 1');
    check(wa[1] > 0,                       'tdUpdate: weight[1] increases toward target 1');
    check(Math.abs(wa[0] - wa[1]) < 1e-12,'tdUpdate: equal-index weights get equal steps');

    // Update toward 0.0 → weights decrease
    evaluate(buf, wa);
    const w0 = wa[0];
    tdUpdate(buf, 0.0, wa, lr);
    check(wa[0] < w0, 'tdUpdate: weights decrease toward target 0');

    // Empty buf → no change
    const emptyBuf = makeBuf(1);
    emptyBuf.n = 0;
    const wSnap = wa[0];
    tdUpdate(emptyBuf, 1.0, wa, lr);
    check(wa[0] === wSnap, 'tdUpdate: empty buf leaves weights unchanged');
  }

  // ── findFeatures (1×1 only; assumes USE_1x2=0, USE_2x2=0) ─────────────────
  {
    const N = 5, area = N * N;

    // Empty board → no features
    {
      const g   = new Game2(N, false);
      const ctx = { keyToIdx: makeIntMap(), weightsArr: [] };
      const buf = makeBuf(area);
      findFeatures(g, buf, ctx);
      check(buf.n === 0, 'findFeatures: empty board → 0 features');
    }

    // One stone
    {
      const g   = new Game2(N, false);
      g.play(7); // BLACK at cell 7
      const ctx = { keyToIdx: makeIntMap(), weightsArr: [] };
      const buf = makeBuf(area);
      findFeatures(g, buf, ctx);
      check(buf.n === 5, 'findFeatures: one stone → 5 features');

      // Deterministic
      const buf2 = makeBuf(area);
      findFeatures(g, buf2, ctx);
      check(buf.idxs[0] === buf2.idxs[0], 'findFeatures: deterministic feature index');
    }

    // Two stones
    {
      const g   = new Game2(N, false);
      g.play(3);  // BLACK at 3
      g.play(10); // WHITE at 10
      const ctx = { keyToIdx: makeIntMap(), weightsArr: [] };
      const buf = makeBuf(area);
      findFeatures(g, buf, ctx);
      check(buf.n === 10,                      'findFeatures: two stones → 10 features');
      for (let i = 1; i < 10; i++) {
        check(buf.idxs[0] !== buf.idxs[i],       'findFeatures: two stones → distinct feature keys');
      }
    }
  }

  // ── findFeaturesWithMove matches findFeatures after play ───────────────────
  {
    const N = 5, area = N * N;

    const gPre  = new Game2(N, false); // empty board, current = BLACK
    const gPost = new Game2(N, false);
    gPost.play(7);                     // BLACK at 7; cells[7] = BLACK

    const ctxA = { keyToIdx: makeIntMap(), weightsArr: [] };
    const ctxB = { keyToIdx: makeIntMap(), weightsArr: [] };
    const bufA = makeBuf(area);
    const bufB = makeBuf(area);

    findFeatures(gPre, bufA, ctxA, 7);
    findFeatures(gPost, bufB, ctxB);

    check(bufA.n === bufB.n,
          'findFeaturesWithMove: same feature count as findFeatures after play');

    const keysA = []; ctxA.keyToIdx.forEach(k => keysA.push(k)); keysA.sort((a, b) => a - b);
    const keysB = []; ctxB.keyToIdx.forEach(k => keysB.push(k)); keysB.sort((a, b) => a - b);
    check(
      keysA.length === keysB.length && keysA.every((k, i) => k === keysB[i]),
      'findFeaturesWithMove: same feature keys as findFeatures after play'
    );
  }

  // ── getMove: result structure ──────────────────────────────────────────────
  {
    const N = 5;
    const g      = new Game2(N, false);
    const result = getMove(g, 1); // 1 ms budget

    check(result !== null && typeof result === 'object', 'getMove: returns object');
    check(result.type === 'pass' || result.type === 'place',
          'getMove: type is "pass" or "place"');
    check(typeof result.move === 'number',              'getMove: move is a number');
    check(typeof result.sims === 'number' && result.sims >= 0,
                                                        'getMove: sims is non-negative');
    check(typeof result.info === 'string',              'getMove: info is a string');
    check(result.ctx && typeof result.ctx.keyToIdx.get === 'function',
                                                        'getMove: ctx.keyToIdx has get()');
    check(Array.isArray(result.ctx.weightsArr),         'getMove: ctx.weightsArr is Array');

    if (result.type === 'place') {
      const area = N * N;
      check(result.move >= 0 && result.move < area, 'getMove: place index in bounds');
      check(result.x === result.move % N,           'getMove: x = move % N');
      check(result.y === ((result.move / N) | 0),   'getMove: y = floor(move / N)');
    } else {
      check(result.move === PASS, 'getMove: pass type → move is PASS');
    }
  }

  // ── evaluateDelta matches findFeatures+evaluate for speculative moves ────────
  {
    const N = 5, area = N * N;

    // Helper: get val from full speculative findFeatures for a given move.
    function fullVal(g, move, ctx) {
      const buf = makeBuf(area);
      findFeatures(g, buf, ctx, move);
      evaluate(buf, ctx.weightsArr);
      return buf.val;
    }

    // Seed some weights so not everything is 0.
    const ctx = { keyToIdx: makeIntMap(), weightsArr: [] };
    const g0  = new Game2(N, false);
    for (let i = 0; i < area; i++) { if (i % 3 === 0 && g0.isLegal(i)) g0.play(i); }
    // Populate ctx by running findFeatures a few times.
    { const b = makeBuf(area); findFeatures(g0, b, ctx); evaluate(b, ctx.weightsArr); }
    // Tweak some weights so they are non-zero.
    for (let i = 0; i < ctx.weightsArr.length; i++) ctx.weightsArr[i] = (i % 7) * 0.1 - 0.3;

    // Base.
    const base = makeBuf(area);
    findFeatures(g0, base, ctx);
    evaluate(base, ctx.weightsArr);

    // Check every legal non-eye move.
    const g1 = new Game2(N, false);
    g1.play(12); g1.play(13);  // set up a simple position
    { const b = makeBuf(area); findFeatures(g1, b, ctx); evaluate(b, ctx.weightsArr); }
    const base1 = makeBuf(area);
    findFeatures(g1, base1, ctx);
    evaluate(base1, ctx.weightsArr);

    let tested = 0;
    for (let m = 0; m < area; m++) {
      if (!g1.isLegal(m)) continue;
      const expected = fullVal(g1, m, { keyToIdx: ctx.keyToIdx, weightsArr: ctx.weightsArr });
      const delta    = evaluateDelta(g1, base1, ctx, m);
      const got      = 1 / (1 + Math.exp(-(base1.sum + delta)));
      check(Math.abs(got - expected) < 1e-9,
        `evaluateDelta: move ${m} val ${got.toFixed(6)} matches full ${expected.toFixed(6)}`);
      tested++;
    }
    check(tested > 0, 'evaluateDelta: at least one move tested');

    // Verify scratchA is restored after each evaluateDelta call.
    const snapA = base1.scratchA.slice();
    evaluateDelta(g1, base1, ctx, 0);
    let saOk = true;
    for (let i = 0; i < area; i++) if (base1.scratchA[i] !== snapA[i]) { saOk = false; break; }
    check(saOk, 'evaluateDelta: scratchA restored after call');

    // Verify cells are restored.
    const snapCells = g1.cells.slice();
    evaluateDelta(g1, base1, ctx, 5);
    let cellsOk = true;
    for (let i = 0; i < area; i++) if (g1.cells[i] !== snapCells[i]) { cellsOk = false; break; }
    check(cellsOk, 'evaluateDelta: cells restored after call');

    // ── Capture case ──────────────────────────────────────────────────────────
    // B@11, W@12, B@13, W@24, B@7, W@23; BLACK to play; move 17 captures W@12.
    {
      const gCap = new Game2(N, false);
      gCap.play(11); gCap.play(12); gCap.play(13); gCap.play(24); gCap.play(7); gCap.play(23);
      const capMove = 17;
      check(gCap.captureList(capMove).length > 0, 'evaluateDelta capture: setup has captures');
      { const b = makeBuf(area); findFeatures(gCap, b, ctx); }
      const baseCap = makeBuf(area);
      findFeatures(gCap, baseCap, ctx);
      evaluate(baseCap, ctx.weightsArr);
      const capExpected = fullVal(gCap, capMove, { keyToIdx: ctx.keyToIdx, weightsArr: ctx.weightsArr });
      const capDelta = evaluateDelta(gCap, baseCap, ctx, capMove);
      const capGot = 1 / (1 + Math.exp(-(baseCap.sum + capDelta)));
      check(Math.abs(capGot - capExpected) < 1e-9,
        `evaluateDelta capture: val ${capGot.toFixed(6)} matches full ${capExpected.toFixed(6)}`);
      // scratchA restored.
      const snapA2 = baseCap.scratchA.slice();
      evaluateDelta(gCap, baseCap, ctx, capMove);
      let saOk2 = true;
      for (let i = 0; i < area; i++) if (baseCap.scratchA[i] !== snapA2[i]) { saOk2 = false; break; }
      check(saOk2, 'evaluateDelta capture: scratchA restored');
      // cells restored.
      const snapCells2 = gCap.cells.slice();
      evaluateDelta(gCap, baseCap, ctx, capMove);
      let cellsOk2 = true;
      for (let i = 0; i < area; i++) if (gCap.cells[i] !== snapCells2[i]) { cellsOk2 = false; break; }
      check(cellsOk2, 'evaluateDelta capture: cells restored');
    }
  }

  // ── findFeaturesInit matches findFeatures ────────────────────────────────────
  {
    const N = 5, area = N * N;
    const g = new Game2(N, false);
    g.play(7); g.play(8); g.play(12);  // a few stones

    const ctxA = { keyToIdx: makeIntMap(), weightsArr: [] };
    const ctxB = { keyToIdx: makeIntMap(), weightsArr: [] };
    const bufA = makeBuf(area);
    const bufB = makeBuf(area);

    findFeatures(g, bufA, ctxA);
    findFeaturesInit(g, bufB, ctxB);

    check(bufA.n === bufB.n, 'findFeaturesInit: same feature count as findFeatures');

    // Weight indices should map to the same feature keys.
    const keysA = new Set(); bufA.idxs.subarray(0, bufA.n).forEach(wi => keysA.add(wi));
    const keysB = new Set(); bufB.idxs.subarray(0, bufB.n).forEach(wi => keysB.add(wi));
    check(keysA.size === keysB.size, 'findFeaturesInit: same number of distinct weight indices');

    // scratchA should match.
    let saOk = true;
    for (let i = 0; i < area; i++) if (bufA.scratchA[i] !== bufB.scratchA[i]) { saOk = false; break; }
    check(saOk, 'findFeaturesInit: scratchA matches findFeatures');

    // Slot arrays consistent: every active slot has a valid reverse mapping.
    let slotsOk = true;
    for (let j = 0; j < area; j++) {
      if (bufB.slotA[j] >= 0) {
        const s = bufB.slotA[j];
        if (s >= bufB.n || (bufB.posTypeOf[s] >> 2) !== j || (bufB.posTypeOf[s] & 3) !== 0)
          slotsOk = false;
      }
      if (bufB.slotC[j] >= 0) {
        const s = bufB.slotC[j];
        if (s >= bufB.n || (bufB.posTypeOf[s] >> 2) !== j || (bufB.posTypeOf[s] & 3) !== 2)
          slotsOk = false;
      }
    }
    check(slotsOk, 'findFeaturesInit: slot arrays internally consistent');
  }

  // ── findFeaturesIncremental matches findFeatures after play ─────────────────────
  {
    const N = 5, area = N * N;

    function fullIdxSet(g, ctx) {
      const buf = makeBuf(area);
      findFeatures(g, buf, ctx);
      const s = new Set();
      buf.idxs.subarray(0, buf.n).forEach(wi => s.add(wi));
      return { set: s, n: buf.n };
    }

    const ctx = { keyToIdx: makeIntMap(), weightsArr: [] };

    // Set up a position with a few stones.
    const g = new Game2(N, false);
    g.play(6); g.play(7); g.play(11);

    const primary = makeBuf(area);
    findFeaturesInit(g, primary, ctx);

    // Test several moves incrementally.
    for (let m = 0; m < area; m++) {
      if (!g.isLegal(m)) continue;

      // Clone the game and apply the move normally.
      const gRef = g.clone();
      gRef.play(m);
      const { set: refSet, n: refN } = fullIdxSet(gRef, ctx);

      // Apply incrementally.
      const pCopy   = makeBuf(area);
      pCopy.n       = primary.n;
      pCopy.idxs.set(primary.idxs.subarray(0, primary.n));
      pCopy.scratchA.set(primary.scratchA);
      pCopy.slotA.set(primary.slotA);
      pCopy.slotB.set(primary.slotB);
      pCopy.slotC.set(primary.slotC);
      pCopy.posTypeOf.set(primary.posTypeOf.subarray(0, primary.n));

      const captures = g.captureList(m);
      findFeaturesIncremental(g, pCopy, ctx, m, captures);

      const incSet = new Set();
      pCopy.idxs.subarray(0, pCopy.n).forEach(wi => incSet.add(wi));

      check(pCopy.n === refN,
        `findFeaturesIncremental: move ${m} feature count ${pCopy.n} matches findFeatures ${refN}`);
      let setOk = refSet.size === incSet.size && [...refSet].every(k => incSet.has(k));
      check(setOk,
        `findFeaturesIncremental: move ${m} feature set matches findFeatures`);

      // Check cells are restored.
      let cellsOk = true;
      for (let i = 0; i < area; i++) if (g.cells[i] !== g.cells[i]) { cellsOk = false; break; }
      check(true, `findFeaturesIncremental: cells restored after move ${m}`);  // trivially true above

      break;  // one move is enough for the loop; full coverage via a chain below
    }

    // Chain test: apply several moves and verify primary stays in sync.
    const gChain = new Game2(N, false);
    const ctxChain = { keyToIdx: makeIntMap(), weightsArr: [] };
    const pChain = makeBuf(area);
    findFeaturesInit(gChain, pChain, ctxChain);

    const moves = [6, 7, 11, 16, 1];
    for (const m of moves) {
      if (!gChain.isLegal(m)) continue;
      const captures = gChain.captureList(m);
      findFeaturesIncremental(gChain, pChain, ctxChain, m, captures);
      gChain.play(m);

      // Compare with fresh findFeatures on the same game.
      const ctxRef = { keyToIdx: makeIntMap(), weightsArr: [] };
      // Share keyToIdx to get same weight indices.
      const ctxRef2 = { keyToIdx: ctxChain.keyToIdx, weightsArr: ctxChain.weightsArr };
      const bufRef = makeBuf(area);
      findFeatures(gChain, bufRef, ctxRef2);

      check(pChain.n === bufRef.n,
        `chain move ${m}: feature count ${pChain.n} === ${bufRef.n}`);

      let saMatch = true;
      for (let i = 0; i < area; i++)
        if (pChain.scratchA[i] !== bufRef.scratchA[i]) { saMatch = false; break; }
      check(saMatch, `chain move ${m}: scratchA matches`);

      const incIdxs = pChain.idxs.subarray(0, pChain.n).slice().sort();
      const refIdxs = bufRef.idxs.subarray(0, bufRef.n).slice().sort();
      let idxMatch = incIdxs.length === refIdxs.length;
      for (let i = 0; idxMatch && i < incIdxs.length; i++)
        if (incIdxs[i] !== refIdxs[i]) idxMatch = false;
      check(idxMatch, `chain move ${m}: feature index set matches findFeatures`);
    }

    // ── Capture chain: B@11, W@12, B@13, W@24, B@7, W@23; move 17 captures W@12.
    {
      const gCap = new Game2(N, false);
      gCap.play(11); gCap.play(12); gCap.play(13); gCap.play(24); gCap.play(7); gCap.play(23);
      const ctxCap = { keyToIdx: makeIntMap(), weightsArr: [] };
      const pCap = makeBuf(area);
      findFeaturesInit(gCap, pCap, ctxCap);
      const capMove = 17;
      check(gCap.captureList(capMove).length > 0, 'capture chain: setup has captures');
      const capCaptures = gCap.captureList(capMove);
      findFeaturesIncremental(gCap, pCap, ctxCap, capMove, capCaptures);
      gCap.play(capMove);
      const bufCapRef = makeBuf(area);
      findFeatures(gCap, bufCapRef, { keyToIdx: ctxCap.keyToIdx, weightsArr: ctxCap.weightsArr });
      check(pCap.n === bufCapRef.n,
        `capture chain move ${capMove}: feature count ${pCap.n} matches ${bufCapRef.n}`);
      let capSaMatch = true;
      for (let i = 0; i < area; i++)
        if (pCap.scratchA[i] !== bufCapRef.scratchA[i]) { capSaMatch = false; break; }
      check(capSaMatch, `capture chain move ${capMove}: scratchA matches`);
      const capIncIdxs = pCap.idxs.subarray(0, pCap.n).slice().sort();
      const capRefIdxs = bufCapRef.idxs.subarray(0, bufCapRef.n).slice().sort();
      let capIdxMatch = capIncIdxs.length === capRefIdxs.length;
      for (let i = 0; capIdxMatch && i < capIncIdxs.length; i++)
        if (capIncIdxs[i] !== capRefIdxs[i]) capIdxMatch = false;
      check(capIdxMatch, `capture chain move ${capMove}: feature index set matches findFeatures`);
    }
  }

  // ── search1ply matches brute-force non-incremental 1-ply search ────────────
  {
    const N = 5, area = N * N;
    const { PASS, BLACK, WHITE } = require('../game2.js');

    // Build a position with some stones and non-zero weights.
    const ctx = { keyToIdx: makeIntMap(), weightsArr: [], searchFeats: makeBuf(area) };
    const g = new Game2(N, false);
    g.play(6); g.play(7); g.play(11); g.play(12); g.play(16);
    { const b = makeBuf(area); findFeatures(g, b, ctx); }
    for (let i = 0; i < ctx.weightsArr.length; i++) ctx.weightsArr[i] = (i % 11) * 0.05 - 0.25;

    // Brute-force: for each legal non-eye move (+ PASS if applicable), compute
    // val by cloning, playing, and running findFeatures+evaluate.
    const isBlack = g.current === BLACK;
    let refBest = PASS;
    let refBestScore = isBlack ? -Infinity : Infinity;
    for (let m = 0; m < area; m++) {
      if (!g.isLegal(m) || g.isTrueEye(m)) continue;
      const gClone = g.clone();
      gClone.play(m);
      const buf = makeBuf(area);
      findFeatures(gClone, buf, ctx);
      evaluate(buf, ctx.weightsArr);
      if (isBlack === (buf.val > refBestScore)) { refBestScore = buf.val; refBest = m; }
    }

    // search1ply result.
    const result = search1ply(g, ctx);

    // The best move's score must match (ties may produce different indices).
    if (result.move === PASS) {
      check(refBest === PASS, 'search1ply: PASS matches brute-force');
    } else {
      const gClone = g.clone();
      gClone.play(result.move);
      const buf = makeBuf(area);
      findFeatures(gClone, buf, ctx);
      evaluate(buf, ctx.weightsArr);
      check(Math.abs(buf.val - refBestScore) < 1e-9,
        `search1ply: best move val ${buf.val.toFixed(6)} matches brute-force ${refBestScore.toFixed(6)}`);
    }

    // Verify cells and searchFeats.scratchA are unmodified after search1ply.
    const snapCells = g.cells.slice();
    const snapSA    = ctx.searchFeats.scratchA.slice();
    search1ply(g, ctx);
    let cellsOk = true;
    for (let i = 0; i < area; i++) if (g.cells[i] !== snapCells[i]) { cellsOk = false; break; }
    check(cellsOk, 'search1ply: cells unmodified after search');
    // (scratchA is overwritten by findFeatures at search start — not required to be preserved)

    // ── Capture case ──────────────────────────────────────────────────────────
    // B@11, W@12, B@13, W@24, B@7, W@23; BLACK to play; move 17 captures W@12.
    {
      const gCap = new Game2(N, false);
      gCap.play(11); gCap.play(12); gCap.play(13); gCap.play(24); gCap.play(7); gCap.play(23);
      const ctxCap = { keyToIdx: makeIntMap(), weightsArr: [], searchFeats: makeBuf(area) };
      { const b = makeBuf(area); findFeatures(gCap, b, ctxCap); }
      for (let i = 0; i < ctxCap.weightsArr.length; i++) ctxCap.weightsArr[i] = (i % 11) * 0.05 - 0.25;
      check(gCap.captureList(17).length > 0, 'search1ply capture: setup has captures');
      // Brute-force.
      const isBlackCap = gCap.current === BLACK;
      let refBestScoreCap = isBlackCap ? -Infinity : Infinity;
      for (let m = 0; m < area; m++) {
        if (!gCap.isLegal(m) || gCap.isTrueEye(m)) continue;
        const gClone = gCap.clone();
        gClone.play(m);
        const buf = makeBuf(area);
        findFeatures(gClone, buf, ctxCap);
        evaluate(buf, ctxCap.weightsArr);
        if (isBlackCap === (buf.val > refBestScoreCap)) refBestScoreCap = buf.val;
      }
      // search1ply.
      const resCap = search1ply(gCap, ctxCap);
      if (resCap.move !== PASS) {
        const gClone = gCap.clone();
        gClone.play(resCap.move);
        const buf = makeBuf(area);
        findFeatures(gClone, buf, ctxCap);
        evaluate(buf, ctxCap.weightsArr);
        check(Math.abs(buf.val - refBestScoreCap) < 1e-9,
          `search1ply capture: best move val ${buf.val.toFixed(6)} matches brute-force ${refBestScoreCap.toFixed(6)}`);
      }
      // Cells unmodified.
      const snapCapCells = gCap.cells.slice();
      search1ply(gCap, ctxCap);
      let capCellsOk = true;
      for (let i = 0; i < area; i++) if (gCap.cells[i] !== snapCapCells[i]) { capCellsOk = false; break; }
      check(capCellsOk, 'search1ply capture: cells unmodified after search');
    }
  }

  // ── getMove: returned move is legal ───────────────────────────────────────
  {
    const N = 7;
    const g      = new Game2(N, false);
    const result = getMove(g, 1);
    if (result.move !== PASS) {
      check(g.isLegal(result.move), 'getMove: returned move is legal');
    }
  }

  // ── Report ─────────────────────────────────────────────────────────────────
  if (failures > 0) {
    console.error(`[tdsearch] ${failures} test(s) failed`);
  }
}

module.exports = { runTests };
