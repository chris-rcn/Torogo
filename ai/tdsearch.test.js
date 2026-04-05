'use strict';

// Tests for ai/tdsearch.js internals and public API.
// Called by tdsearch.js at module load time (Node only) via runTests().
// Silent on success; logs failures to stderr.
//
// Feature tests assume default env (USE_1x2=0, USE_2x2=0), so only 1×1
// features are active.

function runTests(
  { makeBuf, resolveKey, findFeatures, evaluate, tdUpdate, getMove },
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
