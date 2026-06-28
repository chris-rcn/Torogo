'use strict';

// Tests for npats-lib.js.
// Called by npats-lib.js at module load time (Node only) via runTests().
// Silent on success; logs failures to stderr.

function runTests(NPats) {
  let failures = 0;

  function check(cond, msg) {
    if (!cond) { failures++; console.error('FAIL [npats-lib]:', msg); }
  }

  const NPat = require('./npat-lib.js');
  const { Game2 } = require('./game2.js');
  const { makeRng } = require('./xorshift.js');

  const CFG = { useTactical: true, use33c: true, useP12: true, tactStoneLimit: NPat.TACT_STONE_LIMIT };

  // ── makeWeights: raw-key store, including the real key 0 ───────────────────
  {
    const w = NPats.makeWeights();
    check(w.get(7) === undefined,        'weights: miss returns undefined');
    w.set(0, 0.25);                      // all-empty 3×3 window — a real key
    w.set(7, -0.5);
    check(w.get(0) === 0.25,             'weights: raw key 0 stores and reads');
    check(w.get(7) === -0.5,             'weights: ordinary key');
    check(w.size === 2,                  'weights: size');
    let seen = 0, sawZero = false;
    w.forEach((k, v) => { seen++; if (k === 0) sawZero = v === 0.25; });
    check(seen === 2 && sawZero,         'weights: forEach yields unbiased raw keys');
  }

  // ── pack → require → prepareWeights roundtrip (incl. key 0, f32 values) ────
  {
    const fs = require('fs');
    const path = require('path');
    const os = require('os');

    const w = NPats.makeWeights();
    const rng = makeRng(99);
    const entries = [[0, 0.125]];                      // exact in f32
    for (let i = 0; i < 200; i++) {
      entries.push([1 + ((rng.random() * 1e6) | 0), Math.fround(rng.random() * 2 - 1)]);
    }
    const byKey = new Map(entries);                    // dedupe random keys
    for (const [k, v] of byKey) w.set(k, v);

    const file = path.join(os.tmpdir(), `npats-lib-test-${process.pid}.js`);
    fs.writeFileSync(file, NPats.packModel(w, CFG, 'roundtrip test'));
    const raw = require(file);
    fs.unlinkSync(file);
    delete require.cache[require.resolve(file)];

    check(raw.version === NPats.VERSION,   'roundtrip: version');
    check(raw.count === byKey.size,         'roundtrip: count');
    check(raw.cfg.use33c === true,          'roundtrip: cfg');
    const { weights: w2 } = NPats.prepareWeights(raw, 'test');
    let ok = true;
    for (const [k, v] of byKey) {
      if (w2.get(k) !== Math.fround(v)) { ok = false; break; }
    }
    check(ok && w2.size === byKey.size,     'roundtrip: every entry identical after f32');
  }

  // ── empty model: valid, uniform softmax ────────────────────────────────────
  {
    const { weights } = NPats.prepareWeights(
      { version: NPats.VERSION, count: 0, cfg: CFG, keys: new Int32Array(0), vals: new Float32Array(0) },
      'test');
    const model = { weights, cfg: CFG };
    const g = new Game2(9);
    const rng = makeRng(5);
    for (let i = 0; i < 20; i++) g.play(g.randomLegalMove(rng));
    const state = NPat.createState(9);
    const n = NPats.computeProbs(g, state, undefined, model);
    check(n > 0, 'empty model: candidates extracted');
    let sum = 0, uniform = true;
    for (let i = 0; i < n; i++) {
      sum += state.probs[i];
      if (Math.abs(state.probs[i] - 1 / n) > 1e-12) uniform = false;
    }
    check(Math.abs(sum - 1) < 1e-9, 'empty model: probs sum to 1');
    check(uniform,                  'empty model: probs uniform');
  }

  // ── scoring equivalence with npat-lib ──────────────────────────────────────
  // Assign identical weights to an npat store and a policy store; the softmax
  // distributions must match on random positions.
  {
    const rng = makeRng(11);
    let positions = 0, agree = true;
    for (let trial = 0; trial < 3 && agree; trial++) {
      const g = new Game2(9);
      const len = 8 + ((rng.random() * 40) | 0);
      for (let i = 0; i < len && !g.gameOver; i++) g.play(g.randomLegalMove(rng));
      if (g.gameOver) continue;

      // Collect this position's raw keys via a raw-mode extraction.
      const probe = NPat.createState(9);
      NPat.extractFeatures(g, probe, undefined, undefined, null);
      const rawKeys = new Set();
      for (let i = 0; i < probe.count; i++) {
        rawKeys.add(probe.patIdsP8[i]);
        rawKeys.add(probe.patIdsP12[i]);
      }
      for (let k = 0; k < NPat.N_TACT_SLOTS; k++) rawKeys.add(NPat.TACT_RAW_BASE + k);

      // Same random weight per raw key in both stores.
      const npatW = NPat.createWeights({ useP8: true, useP12: true });
      const polW  = NPats.makeWeights();
      for (const k of rawKeys) {
        const v = rng.random() * 2 - 1;
        npatW.vals[NPat.internWeight(npatW, k)] = v;
        polW.set(k, v);
      }

      const s1 = NPat.createState(9);
      NPat.extractFeatures(g, s1, undefined, undefined, npatW);
      NPat.computeSoftmax(s1, npatW);

      const s2 = NPat.createState(9);
      NPats.computeProbs(g, s2, undefined, { weights: polW, cfg: CFG });

      if (s1.count !== s2.count) { agree = false; break; }
      // npat stores weights in a Float32Array; npats keeps them float64 in a Map.
      // The softmax math is identical, so the two agree only to f32 precision
      // (~1e-7) — a real feature/logic divergence would be O(1e-2+), far above this.
      for (let i = 0; i < s1.count; i++) {
        if (s1.moves[i] !== s2.moves[i] || Math.abs(s1.probs[i] - s2.probs[i]) > 1e-5) {
          agree = false;
          break;
        }
      }
      positions++;
    }
    check(agree && positions > 0, 'equivalence: npats softmax matches npat softmax (' + positions + ' positions)');
  }

  // ── crossEntropyUpdate: converges to a representable target (convexity) ────
  // An arbitrary distribution is NOT representable (candidates with identical
  // feature sets are forced to equal probabilities), so the target is the
  // softmax of a reference weight vector — train fresh weights to match it.
  {
    const g = new Game2(9);
    const rng = makeRng(31);
    for (let i = 0; i < 16; i++) g.play(g.randomLegalMove(rng));
    const state = NPat.createState(9);

    // Reference model with random weights on this position's raw keys.
    const refW = NPats.makeWeights();
    NPats.computeProbs(g, state, undefined, { weights: refW, cfg: CFG });
    for (let i = 0; i < state.count; i++) {
      refW.set(state.patIdsP8[i], rng.random() * 2 - 1);
      refW.set(state.patIdsP12[i], rng.random() * 2 - 1);
    }
    for (let k = 0; k < NPat.N_TACT_SLOTS; k++) {
      refW.set(NPat.TACT_RAW_BASE + k, rng.random() - 0.5);
    }
    const n0 = NPats.computeProbs(g, state, undefined, { weights: refW, cfg: CFG });
    const target = Float64Array.from(state.probs.subarray(0, n0));

    const model = { weights: NPats.makeWeights(), cfg: CFG };
    NPats.computeProbs(g, state, undefined, model);
    function ce() {
      let l = 0;
      for (let i = 0; i < n0; i++) l -= target[i] * Math.log(Math.max(state.probs[i], 1e-12));
      return l;
    }
    const ce0 = ce();
    for (let iter = 0; iter < 400; iter++) {
      NPats.computeProbs(g, state, undefined, model);
      NPats.crossEntropyUpdate(state, model, target, 0.5);
    }
    NPats.computeProbs(g, state, undefined, model);
    let maxDiff = 0;
    for (let i = 0; i < n0; i++) maxDiff = Math.max(maxDiff, Math.abs(state.probs[i] - target[i]));
    check(ce() < ce0,     'crossEntropyUpdate: loss decreased');
    check(maxDiff < 0.005, 'crossEntropyUpdate: probs converge to representable target (maxDiff ' + maxDiff.toFixed(5) + ')');
  }

  // ── Report ──────────────────────────────────────────────────────────────────
  if (failures > 0) {
    console.error(`[npats-lib] ${failures} test(s) failed`);
  }
}

module.exports = { runTests };
