'use strict';

// Tests for ai/dt-reinforce.js's chain layer.  Run directly: node ai/dt-reinforce.test.js

const { Game2, EMPTY, BLACK } = require('../game2.js');
const { makeRng } = require('../xorshift.js');
const Util = require('../util.js');
const { create } = require('./dt-reinforce.js');

let failures = 0;
function check(cond, msg) { if (!cond) { failures++; console.error('FAIL [dt-reinforce]:', msg); } }

// Full-length actor sims, no truncation, no ppat tail, no root filter.
function agent(overrides) {
  return create(Util.makeCfg(null, Object.assign({
    ACTOR_CHAIN_LAYER: '1', ACTOR_DEPTH: '9999', TRUNC_MAX_PHASE: '0', PPAT_MIN_PHASE: '1',
    ROOT_MOVE_FILTER: '0', PLAYOUTS: '1',
  }, overrides)));
}

function randomPosition(N, plies, seed) {
  const g = new Game2(N, true), r = makeRng(seed);
  for (let i = 0; i < plies && !g.gameOver; i++) g.play(g.randomLegalMove(r));
  return g;
}

// Liberties of the chain holding stone s, from the board's bitsets.
function libsOf(g, s) {
  const gid = g._gid[s], W = g._W, b = gid * W, out = [];
  for (let wi = 0; wi < W; wi++) {
    let w = g._lw[b + wi];
    while (w) { out.push(wi * 32 + 31 - Math.clz32(w & -w)); w &= w - 1; }
  }
  return out;
}

// ── Incremental chain sums match a fresh recompute ───────────────────────────
// LR 0 keeps the weights fixed, so the kSum maintained move by move through a
// whole sim must equal chainSum recomputed on the sim's final board.
{
  const N = 7;
  const a = agent({ LR: '0' });
  const root = randomPosition(N, 6, 1);
  a.getMove(root, 0, { rng: makeRng(2) });          // sizes the state
  const st = a._internals();
  const rng = makeRng(9);
  for (let i = 0; i < st.wK.length; i++) st.wK[i] = (rng.random() - 0.5) * 2;
  let worst = 0, checked = 0, qualifying = 0;
  for (let sim = 0; sim < 40; sim++) {
    const g = randomPosition(N, 4 + (sim % 20), 100 + sim);
    if (g.gameOver) continue;
    st.simulate(g, rng);                            // leaves the internal board as the sim's last
    const fin = st.lastBoard();
    for (let p = 0; p < st.area; p++) {
      if (fin.cells[p] !== EMPTY) continue;
      const d = Math.abs(st.kSum[p] - st.chainSum(fin, p));
      if (d > worst) worst = d;
      checked++;
      if (st.chainSum(fin, p) !== 0) qualifying++;
    }
  }
  check(checked > 0 && qualifying > 0, `incremental test checked ${checked} points, ${qualifying} with chain terms`);
  check(worst < 1e-12, `incremental kSum drifted from a fresh recompute by ${worst}`);
}

// ── The chain gradient equals the sum of the point gradients on its liberties ──
// One actor ply per sim: the step's gradient for a chain is Σ over its liberties
// of ([a = chosen] − π(a)), and the (mover, point) table's change at each point
// is exactly that point's term, so each root chain's weight change must equal
// the sum of w1's changes over its liberties.
{
  const N = 9;
  const a = agent({ ACTOR_DEPTH: '1', LR: '0.5' });
  const root = randomPosition(N, 40, 7);
  a.getMove(root, 0, { rng: makeRng(3) });
  const st = a._internals();
  st.reset();
  const m = root.current === BLACK ? 0 : 1;
  const w1Before = Float64Array.from(st.w1), wKBefore = Float64Array.from(st.wK);
  st.simulate(root, makeRng(11));
  let chains = 0, worst = 0;
  const seen = new Set();
  for (let s = 0; s < st.area; s++) {
    if (root.cells[s] === EMPTY) continue;
    const gid = root._gid[s];
    if (seen.has(gid)) continue;
    seen.add(gid);
    if (root._ls[gid] > 3) continue;
    const key = st.chainKey(root, gid);
    let expect = 0;
    for (const p of libsOf(root, s)) expect += st.w1[m * st.area + p] - w1Before[m * st.area + p];
    const got = st.wK[key] - wKBefore[key];
    worst = Math.max(worst, Math.abs(got - expect));
    chains++;
  }
  check(chains > 0, 'gradient test found no qualifying chains at the root');
  check(worst < 1e-6, `chain weight change differs from its liberties' point changes by ${worst}`);
}

// ── Last-move slice: scores stay current as its key moves every ply ──────────
// LR 0 and random response weights: after a whole actor-played sim, every
// empty point's maintained score must equal a fresh score() under the final
// key (a missed rescore leaves the previous ply's row in sc).
{
  const N = 7;
  const a = agent({ LR: '0', ACTOR_CHAIN_LAYER: '0', ACTOR_LASTMOVE_LAYER: '1' });
  const root = randomPosition(N, 6, 1);
  a.getMove(root, 0, { rng: makeRng(2) });
  const st = a._internals();
  const rng = makeRng(5);
  for (let i = 0; i < st.wLM.length; i++) st.wLM[i] = (rng.random() - 0.5) * 2;
  let worst = 0, checked = 0;
  for (let sim = 0; sim < 30; sim++) {
    const g = randomPosition(N, 4 + (sim % 15), 300 + sim);
    if (g.gameOver) continue;
    st.simulate(g, rng);
    const fin = st.lastBoard();
    for (let m = 0; m < 2; m++) for (let p = 0; p < st.area; p++) {
      if (fin.cells[p] !== EMPTY) continue;
      worst = Math.max(worst, Math.abs(st.sc[m][p] - st.score(m, p))); checked++;
    }
  }
  check(checked > 0 && worst < 1e-12, `last-move slice: maintained scores drifted by ${worst} over ${checked} points`);
}

// ── Last-move slice: a step trains exactly its (mover, last move) row ────────
// Two actor plies per sim: ply 0 has no last move, ply 1 keys on ply 0's move
// and is the only step of its mover, so that one row's change must equal the
// (mover, point) table's change for that mover, and every other row stays 0.
{
  const N = 9;
  const a = agent({ LR: '0.5', ACTOR_DEPTH: '2', ACTOR_CHAIN_LAYER: '0', ACTOR_LASTMOVE_LAYER: '1' });
  const root = randomPosition(N, 30, 7);
  a.getMove(root, 0, { rng: makeRng(3) });
  const st = a._internals(), A = st.area;
  let worst = 0, sims = 0, stray = 0;
  for (let k = 0; k < 10; k++) {
    st.reset();
    st.simulate(root, makeRng(20 + k));
    const L = st.chosen()[0], m1 = root.current === BLACK ? 1 : 0;   // ply 1's mover
    if (L === -1) continue;
    sims++;
    for (let m = 0; m < 2; m++) for (let r = 0; r < A; r++) for (let p = 0; p < A; p++) {
      const v = st.wLM[(m * A + r) * A + p];
      if (m === m1 && r === L) worst = Math.max(worst, Math.abs(v - st.w1[m1 * A + p]));
      else if (v !== 0) stray++;
    }
  }
  check(sims > 0 && worst < 1e-7 && stray === 0, `last-move slice: row change off by ${worst}, ${stray} stray weights (${sims} sims)`);
}

if (failures) { console.error(`[dt-reinforce] ${failures} test(s) failed`); process.exit(1); }
else console.log('[dt-reinforce] all tests passed');
