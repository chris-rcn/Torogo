'use strict';

// Tests for ai/tdsearch2.js.  Run directly: node ai/tdsearch2.test.js

const { Game2, BLACK, WHITE, EMPTY, PASS } = require('../game2.js');
const { makeRng } = require('../xorshift.js');
const Util = require('../util.js');
const { create } = require('./tdsearch2.js');

let failures = 0;
function check(cond, msg) { if (!cond) { failures++; console.error('FAIL [tdsearch2]:', msg); } }

function agent(overrides) { return create(Util.makeCfg(null, overrides)); }

// Random legal play, tracking changed cells, so captures occur.
function playRandom(g, rng, changed) {
  const move = g.randomLegalMove(rng);
  let n = 0;
  if (move !== PASS) {
    const caps = g.captureList(move);
    changed[n++] = move;
    for (let i = 0; i < caps.length; i++) changed[n++] = caps[i];
  }
  g.play(move);
  return n;
}

// ── Incremental score maintenance matches a full recompute ──────────────────
{
  const N = 7, rng = makeRng(11);
  const a = agent({ TD_LAYERS: '1,5,9', TD_TEMP: '1' });
  const g = new Game2(N, true);
  a.getMove(g, 0, { rng });             // sizes the state (budget 0: no sims)
  const st = a._internals();
  // Non-trivial weights so scores differ between points and codes.
  for (let i = 0; i < st.w9.length; i++) st.w9[i] = (rng.random() - 0.5) * 2;
  for (let i = 0; i < st.w5.length; i++) st.w5[i] = (rng.random() - 0.5);
  for (let i = 0; i < st.w1.length; i++) st.w1[i] = (rng.random() - 0.5) * 0.5;
  for (let i = 0; i < st.c9.length; i++) st.c9[i] = (rng.random() - 0.5) * 0.1;
  for (let i = 0; i < st.c4.length; i++) st.c4[i] = (rng.random() - 0.5) * 0.1;
  for (let i = 0; i < st.c1.length; i++) st.c1[i] = (rng.random() - 0.5) * 0.1;
  st.recomputeAll(g.cells, g._nbr, g._dnbr);
  const changed = new Int32Array(N * N + 1);
  let captures = 0, worst = 0;
  for (let step = 0; step < 150 && !g.gameOver; step++) {
    const before = g.emptyCount;
    const n = playRandom(g, rng, changed);
    if (n > 1) captures++;
    if (n > 0) st.recomputeAround(g, changed, n);
    // Reference: a fresh full recompute on a second instance.
    const b = agent({ TD_LAYERS: '1,5,9', TD_TEMP: '1' });
    b._internals().setup(N);
    const sb = b._internals();
    sb.w9.set(st.w9); sb.w5.set(st.w5); sb.w1.set(st.w1);
    sb.c9.set(st.c9); sb.c4.set(st.c4); sb.c1.set(st.c1);
    sb.recomputeAll(g.cells, g._nbr, g._dnbr);
    for (let p = 0; p < N * N; p++) {
      if (st.i1[p] !== sb.i1[p] || st.i4[p] !== sb.i4[p] || st.i9[p] !== sb.i9[p]) worst = 1;
    }
    for (let m = 0; m < 2; m++) {
      const dZ = Math.abs(st.Z[m] - sb.Z[m]);
      if (dZ > worst) worst = dZ;
      for (let p = 0; p < N * N; p++) {
        const d = Math.abs(st.ex[m][p] - sb.ex[m][p]);
        if (d > worst) worst = d;
      }
      const dS = Math.abs(st.S[m] - sb.S[m]) / Math.max(1, sb.S[m]);
      if (dS > worst) worst = dS;
    }
    void before;
  }
  check(captures > 0, `incremental test saw no captures (${captures})`);
  check(worst < 1e-9, `incremental ex/S/Z/indices drifted from full recompute by ${worst}`);
}

// ── REINFORCE update: gradient sums to zero, chosen point moves with the advantage ──
{
  const N = 5;
  const a = agent({ TD_LAYERS: '1', TD_LR: '0.5', TD_TEMP: '2', TD_BASELINE: '0.9', TD_CRITIC_LAYERS: 'none' });
  const g = new Game2(N, true);
  a.getMove(g, 0, { rng: makeRng(3) });
  const st = a._internals();
  // Build one recorded step by hand: run one sim with TD_SIMS-free internals.
  // Simulate records steps then updates; instead exercise update() via simulate
  // on a copied instance and inspect the layer-1 weights.
  st.recomputeAll(g.cells, g._nbr, g._dnbr);
  const before = Float32Array.from(st.w1);
  const steps = st.simulate(g, makeRng(5));
  check(steps > 0, 'simulate produced no steps');
  // Every step's gradient over its distribution sums to lr/T·A·(1 − Σπ) = 0,
  // so the total change on each mover's layer-1 weights is 0 up to rounding.
  const area = N * N;
  for (let m = 0; m < 2; m++) {
    let sum = 0;
    for (let p = 0; p < area; p++) sum += st.w1[m * area + p] - before[m * area + p];
    check(Math.abs(sum) < 1e-4, `mover ${m}: layer-1 weight changes sum to ${sum}, expected 0`);
  }
  check(st.base[0] !== 0.5 || st.base[1] !== 0.5, 'baseline did not move after a sim');
}

// ── Sampling respects legality and eyes; exhausted distribution passes ───────
{
  const N = 5;
  const a = agent({ TD_LAYERS: '1' });
  const g = new Game2(N, false);
  a.getMove(g, 0, { rng: makeRng(7) });
  const st = a._internals();
  st.recomputeAll(g.cells, g._nbr, g._dnbr);
  const rng = makeRng(9);
  for (let i = 0; i < 20; i++) {
    const p = st.sample(g, 0, rng);
    check(p !== PASS && g.isLegal(p) && !g.isTrueEye(p), `sampled ${p} is not a legal non-eye point`);
  }
  st.ex[0].fill(0); st.S[0] = 0;
  check(st.sample(g, 0, rng) === PASS, 'empty distribution should sample PASS');
}

// ── Critic: a lopsided position's value moves toward the outcome ────────────
{
  const N = 7;
  const a = agent({ TD_SIMS: '40' });
  const g = new Game2(N, true);
  // Black builds a big framework while White passes; Black wins these sims.
  const rng = makeRng(21);
  for (let i = 0; i < 12; i++) { g.play(PASS); g.play(g.randomLegalMove(rng)); }
  check(!g.gameOver && g.current === -1, 'setup: expected White to move in a live game');
  const r = a.getMove(g, 1000, { rng });
  const st = a._internals();
  check(st.CRITIC, 'critic should be on by default');
  const m = g.current === BLACK ? 0 : 1;
  const V = st.sigmoid(st.Z[m]);
  check(V > 0.6, `root value should favour Black after 40 sims, got ${V.toFixed(3)}`);
  check(/V=0\.[6-9]/.test(r.info), `info should report the critic value: ${r.info}`);
}
{
  // Critic off: no critic tables, the EMA baseline moves instead.
  const N = 5;
  const a = agent({ TD_CRITIC_LAYERS: 'none', TD_SIMS: '5' });
  const g = new Game2(N, true);
  a.getMove(g, 1000, { rng: makeRng(2) });
  const st = a._internals();
  check(!st.CRITIC && st.c9.length === 0, 'critic tables should be empty when off');
  check(st.base[0] !== 0.5, 'EMA baseline should move when the critic is off');
}

// ── getMove: legal moves, a sims cap, and reset on a new game ───────────────
{
  const N = 7;
  const a = agent({ TD_SIMS: '3' });
  const g = new Game2(N, true);
  const rng = makeRng(13);
  for (let i = 0; i < 6; i++) {
    const r = a.getMove(g, 1000, { rng });
    check(g.isLegal(r.move), `getMove returned illegal move ${r.move}`);
    check(/sims=3\b/.test(r.info), `info should report sims=3: ${r.info}`);
    g.play(r.move);
  }
  const st = a._internals();
  let nonzero = 0;
  for (let i = 0; i < st.w9.length; i++) if (st.w9[i] !== 0) nonzero++;
  check(nonzero > 0, 'no weights learned after six moves');
}
{
  // Time-budgeted instance: learn on one game, then a fresh game with budget 0
  // runs no sims, so the reset must leave every weight at zero.
  const N = 7;
  const a = agent({});
  const g = new Game2(N, true);
  const rng = makeRng(17);
  for (let i = 0; i < 3; i++) g.play(a.getMove(g, 20, { rng }).move);
  const st = a._internals();
  let nonzero = 0;
  for (let i = 0; i < st.w9.length; i++) if (st.w9[i] !== 0) nonzero++;
  check(nonzero > 0, 'time-budgeted instance learned nothing');
  a.getMove(new Game2(N, true), 0, { rng });
  let any = 0;
  for (let i = 0; i < st.w9.length; i++) if (st.w9[i] !== 0) any++;
  for (let i = 0; i < st.c9.length; i++) if (st.c9[i] !== 0) any++;
  check(any === 0 && st.base[0] === 0.5, `new game did not reset (nonzero weights: ${any})`);
}

if (failures) { console.error(`[tdsearch2] ${failures} test(s) failed`); process.exit(1); }
else console.log('[tdsearch2] all tests passed');
