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
  const a = agent({ TD_ACTOR_LAYER5_DEPTH: '9999', TD_ACTOR_LAYER9_DEPTH: '9999', TD_TEMP: '1' });
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
    const b = agent({ TD_ACTOR_LAYER5_DEPTH: '9999', TD_ACTOR_LAYER9_DEPTH: '9999', TD_TEMP: '1' });
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
  const a = agent({ TD_ACTOR_LR: '0.5', TD_TEMP: '2', TD_BASELINE: '0.9', TD_CRITIC_LAYERS: 'none' });
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
  const a = agent({});
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

// ── Depth switch: refreshing without layers 5/9 leaves layer-1 scores only ──
{
  const N = 5, rng = makeRng(31);
  const a = agent({ TD_ACTOR_LAYER5_DEPTH: '3', TD_ACTOR_LAYER9_DEPTH: '3' });
  const g = new Game2(N, true);
  for (let i = 0; i < 6; i++) g.play(g.randomLegalMove(rng));
  a._internals().setup(N);
  const st = a._internals();
  for (let i = 0; i < st.w9.length; i++) st.w9[i] = rng.random();
  for (let i = 0; i < st.w5.length; i++) st.w5[i] = rng.random();
  for (let i = 0; i < st.w1.length; i++) st.w1[i] = rng.random() - 0.5;
  st.setActive(true, true);
  st.recomputeAll(g.cells, g._nbr, g._dnbr);
  const area = N * N;
  let differs = 0;
  for (let p = 0; p < area; p++) if (g.cells[p] === EMPTY && st.sc[0][p] !== st.w1[p]) differs++;
  check(differs > 0, 'with layers on, scores should include layer 5/9 terms');
  st.setActive(false, false);
  st.refreshScores(g.cells);
  let bad = 0, sum = 0;
  for (let p = 0; p < area; p++) {
    if (g.cells[p] !== EMPTY) { if (st.ex[1][p] !== 0) bad++; continue; }
    if (st.sc[1][p] !== st.w1[area + p]) bad++;
    sum += st.ex[1][p];
  }
  check(bad === 0, `after the switch ${bad} points still carry layer 5/9 terms`);
  check(Math.abs(sum - st.S[1]) < 1e-9, 'refreshScores left S inconsistent');
}

// ── Playout tail: the actor plays only the first TD_ACTOR_DEPTH plies ───────
{
  const N = 9;
  const a = agent({ TD_ACTOR_DEPTH: '3', TD_TRUNC_PHASE_DELTA: '0' });   // untruncated: the ppat tail
  const g = new Game2(N, true);
  a.getMove(g, 0, { rng: makeRng(41) });
  const st = a._internals();
  const steps = st.simulate(g, makeRng(43));
  check(steps > 3, `sim should run past the actor depth, got ${steps} steps`);
  check(st.lastActorSteps === 3, `actor should play exactly 3 plies, played ${st.lastActorSteps}`);
  let nonzero = 0;
  for (let i = 0; i < st.w1.length; i++) if (st.w1[i] !== 0) nonzero++;
  check(nonzero > 0, 'actor learned nothing from its plies');
}

// ── Critic tail off: the critic stops at the actor depth, the actor still learns ──
{
  const N = 9;
  const a = agent({ TD_ACTOR_DEPTH: '4', TD_CRITIC_TAIL: '0', TD_TRUNC_PHASE_DELTA: '0' });
  const g = new Game2(N, true);
  a.getMove(g, 0, { rng: makeRng(51) });
  const st = a._internals();
  const steps = st.simulate(g, makeRng(53));
  check(steps > 4, `sim should run past the actor depth, got ${steps}`);
  check(st.lastCriticSteps === 4, `critic should stop at ply 4, recorded ${st.lastCriticSteps}`);
  let c = 0; for (let i = 0; i < st.c4.length; i++) if (st.c4[i] !== 0) c++;
  let w = 0; for (let i = 0; i < st.w1.length; i++) if (st.w1[i] !== 0) w++;
  check(c > 0 && w > 0, `critic (${c}) and actor (${w}) should both have learned`);
}

// ── Truncation: actor plies, a random buffer, then the vpat leaf ────────────
{
  const N = 9;
  const a = agent({ TD_TRUNC_PHASE_DELTA: '0.2', TD_TRUNC_ACTOR_DEPTH: '5', TD_TRUNC_MAX_PHASE: '1' });
  const g = new Game2(N, true);
  const r = a.getMove(g, 0, { rng: makeRng(61) });   // budget 0: sets up, arms truncation
  check(/trunc=22\b/.test(r.info), `info should show truncation at ply 5 + ceil(0.2*81) = 22: ${r.info}`);
  const st = a._internals();
  const steps = st.simulate(g, makeRng(63));
  check(steps === 22, `truncated sim should stop at 5 + 17 = 22 plies, ran ${steps}`);
  check(st.lastActorSteps === 5, `actor should play 5 plies, played ${st.lastActorSteps}`);
  const z = st.lastReturn;
  check(z > 0 && z < 1, `return should be a vpat value strictly inside (0,1), got ${z}`);
  // Above the phase gate the same instance runs full sims with the outcome.
  st.setTrunc(false, 0);
  const full = st.simulate(g, makeRng(65));
  check(full > 22 && (st.lastReturn === 0 || st.lastReturn === 1), `untruncated sim should run to the end with a 0/1 outcome (${full} steps, return ${st.lastReturn})`);
}

// ── Root selection by visits: the played point is the most-sampled first ply ──
{
  const N = 7;
  const a = agent({ TD_ROOT_SELECT: 'visits', TD_SIMS: '50' });
  const g = new Game2(N, true);
  const r = a.getMove(g, 1000, { rng: makeRng(71) });
  const st = a._internals();
  let total = 0, top = 0;
  for (let p = 0; p < N * N; p++) { total += st.rootVisits[p]; if (st.rootVisits[p] > top) top = st.rootVisits[p]; }
  check(total === 50, `every sim's first ply should be counted, got ${total}`);
  check(st.rootVisits[r.move] === top, `played point has ${st.rootVisits[r.move]} visits, max is ${top}`);
  check(new RegExp(`visits=${top}\\b`).test(r.info), `info should report the visits: ${r.info}`);
}

// ── Root selection by alpha-beta over the critic ────────────────────────────
{
  const N = 7;
  const a = agent({ TD_ROOT_SELECT: 'ab', TD_AB_DEPTH: '2', TD_AB_WIDTH: '4', TD_SIMS: '20' });
  const g = new Game2(N, true);
  const rng = makeRng(81);
  for (let i = 0; i < 4; i++) {
    const r = a.getMove(g, 1000, { rng });
    check(g.isLegal(r.move) && r.move !== PASS, `ab root returned ${r.move}`);
    check(/ab=d2w4\b/.test(r.info), `info should name the search: ${r.info}`);
    g.play(r.move);
  }
  let threw = false;
  try { agent({ TD_ROOT_SELECT: 'ab', TD_CRITIC_LAYERS: 'none' }); } catch (e) { threw = true; }
  check(threw, 'ab without a critic should be refused');
}

// ── Priors: enter the score and the logit; distillation moves them toward the residuals ──
{
  const Priors = require('../tdsearch2-priors.js');
  const N = 7, rng = makeRng(91);
  const a = agent({});
  const g = new Game2(N, true);
  for (let i = 0; i < 6; i++) g.play(g.randomLegalMove(rng));
  a._internals().setup(N);
  const pr = Priors.make();
  for (let i = 0; i < pr.actor9.length; i++) pr.actor9[i] = rng.random() - 0.5;
  for (let i = 0; i < pr.critic4.length; i++) pr.critic4[i] = rng.random() - 0.5;
  a.setPriors(pr);
  const st = a._internals();
  st.recomputeAll(g.cells, g._nbr, g._dnbr);
  // With zero tables the score IS the actor prior of the point's code, and the
  // logit IS the sum of the critic prior over active windows.
  const area = N * N;
  let bad = 0, zExpect = [0, 0];
  for (let p = 0; p < area; p++) {
    const b = p * 4;
    const k5 = (g.cells[g._nbr[b]] + 1) + 3 * (g.cells[g._nbr[b + 1]] + 1) + 9 * (g.cells[g._nbr[b + 2]] + 1) + 27 * (g.cells[g._nbr[b + 3]] + 1);
    const k9 = k5 + 81 * ((g.cells[g._dnbr[b]] + 1) + 3 * (g.cells[g._dnbr[b + 1]] + 1) + 9 * (g.cells[g._dnbr[b + 2]] + 1) + 27 * (g.cells[g._dnbr[b + 3]] + 1));
    if (g.cells[p] === EMPTY && Math.abs(st.sc[0][p] - pr.actor9[k9]) > 1e-6) bad++;
    const k4 = (g.cells[p] + 1) + 3 * (g.cells[g._nbr[b + 3]] + 1) + 9 * (g.cells[g._nbr[b + 1]] + 1) + 27 * (g.cells[g._dnbr[b + 3]] + 1);
    if (k4 !== 0) { zExpect[0] += pr.critic4[k4]; zExpect[1] += pr.critic4[81 + k4]; }
  }
  check(bad === 0, `${bad} points' scores differ from the actor prior`);
  check(Math.abs(st.Z[0] - zExpect[0]) < 1e-6 && Math.abs(st.Z[1] - zExpect[1]) < 1e-6, `logits ${st.Z[0]},${st.Z[1]} differ from the critic prior sums ${zExpect}`);
  // Distillation: a residual on one point moves that point's code prior by step × residual.
  const p0 = g._emptyCells ? g._emptyCells[0] : 0;
  let pt = -1; for (let p = 0; p < area; p++) if (g.cells[p] === EMPTY) { pt = p; break; }
  st.w1[pt] = 2.0;   // mover 0 residual at pt
  const b0 = pt * 4;
  const k5 = (g.cells[g._nbr[b0]] + 1) + 3 * (g.cells[g._nbr[b0 + 1]] + 1) + 9 * (g.cells[g._nbr[b0 + 2]] + 1) + 27 * (g.cells[g._nbr[b0 + 3]] + 1);
  const k9 = k5 + 81 * ((g.cells[g._dnbr[b0]] + 1) + 3 * (g.cells[g._dnbr[b0 + 1]] + 1) + 9 * (g.cells[g._dnbr[b0 + 2]] + 1) + 27 * (g.cells[g._dnbr[b0 + 3]] + 1));
  const before = pr.actor9[k9];
  a.distilPriors(g, 0.5);
  check(pr.actor9[k9] >= before + 1.0 - 1e-6, `prior at the point's code should rise by >= 0.5*2 (other points sharing the code have zero residual): ${before} -> ${pr.actor9[k9]}`);
  void p0;
  // Round trip through the file format.
  const S = require('path').join(process.env.TD_TEST_SCRATCH || require('os').tmpdir(), 'tdsearch2-priors-test.js');
  Priors.save(S, pr, 'test');
  const back = Priors.load(S);
  let diff = 0;
  for (let i = 0; i < pr.actor9.length; i++) if (back.actor9[i] !== pr.actor9[i]) diff++;
  for (let i = 0; i < pr.critic4.length; i++) if (back.critic4[i] !== pr.critic4[i]) diff++;
  check(diff === 0 && back.comment === 'test', `priors file round trip changed ${diff} values`);
}

// ── Critic: a lopsided position's value moves toward the outcome ────────────
{
  const N = 7;
  const a = agent({ TD_SIMS: '40', TD_CRITIC_LAYERS: '1,4,9' });   // the layer set this threshold was set on
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
  const a = agent({ TD_SIMS: '3', TD_ACTOR_LAYER9_DEPTH: '9999' });
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
  const a = agent({ TD_ACTOR_LAYER9_DEPTH: '4' });
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
