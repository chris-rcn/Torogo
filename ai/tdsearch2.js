'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

const Util = (typeof require === 'function') ? require('../util.js') : window.Util;
const { BLACK, EMPTY, PASS } = Util.load('./game2.js', 'Game2');
const { makeRng } = Util.load('./xorshift.js', 'XorShift');

// tdsearch2 — online actor-critic learning during think time.
//
// Each move, self-play simulations run from the current position.  The actor
// policy that plays them is updated by policy gradient; the critic value is
// updated by TD; the actor's argmax is played.  Both persist across the moves
// of one game (reset on a new game) so knowledge accumulates.
//
// Features are LOCATION-DEPENDENT by design — no symmetry, the board is in
// the orientation it is in — and indexed combinatorially, no hashing.  All are
// keyed by the side to move as well as the point.
//
// Actor: score(p) for an empty point p is the sum of the layers
//   1: the point itself                    (mover, p)               always on
//   5: its 4 orthogonal neighbours          (mover, p, base-3 code of 4 cells)
//   9: those plus the 4 diagonals           (mover, p, base-3 code of 8 cells)
// and the policy is a softmax over the empty points.  Layers 5 and 9 are each
// active only for the first N plies of a sim (their depth knob; 0 = off): deep
// in a sim the board has diverged from the root, so updates to those exact
// local patterns land where no root will read them, while the first plies
// serve this root and the next move's.  At the ply a layer switches off, the
// scores are refreshed once without it.
//
// Critic: V(s) = σ(z), z = the sum over anchors p of the enabled layers
//   1: the cell at p                        (mover, p, colour)
//   4: the 2×2 window p, right, down, down-right  (mover, p, 4-cell code)
//   9: the 3×3 window centred on p           (mover, p, 9-cell code)
// An all-empty window is inactive (contributes nothing).  With actor layer 9
// and critic layer 9 both on, one 8-cell code serves both.
//
// Everything is maintained incrementally: a move only changes points within
// one cell of the placed or captured stones, and one pass over those points
// refreshes both movers' actor scores and both movers' critic logits.
// Sampling rejects illegal and true-eye points lazily (their weight is zeroed
// until their neighbourhood changes).
//
// Learning:
//   critic: two-ply same-parity TD(0), ONLINE with a two-step lag — at step t
//           the features of step t−2 (the same mover's previous position) are
//           pushed toward V(s_t); the last two steps are pushed toward the
//           outcome when the sim ends.  A three-slot ring of index snapshots
//           supplies step t−2's features; the live logit is corrected for the
//           features the two positions share.
//   actor:  REINFORCE at sim end (it needs the final result), in step order,
//           with advantage A_t = (1−β)·δ_t + β·(R − V_t) from the mover's view,
//           where δ_t = V_{t+2} − V_t is the TD advantage, R the final result
//           and β = TD_ADV_MIX.  Its per-step records are the sampled-from
//           distribution and the point codes.  Without a critic the baseline
//           is a per-mover EMA of sim returns.
//
// No pretrained models, no simulation truncation (planned later).
//
// ── Factory ──
// create(cfg) -> { getMove }.  cfg is a Util.makeCfg reader (P1_/P2_ prefixes in selfplay).
//
// Config:
//   TD_ACTOR_LAYER5_DEPTH  plies of a sim for which actor layer 5 is on; 0 = off (default 0)
//   TD_ACTOR_LAYER9_DEPTH  plies of a sim for which actor layer 9 is on; 0 = off (default 0)
//   TD_ACTOR_LR       actor step size                                  (default 0.1)
//   TD_TEMP           softmax temperature for the simulations          (default 1)
//   TD_CRITIC_LAYERS  critic layers, comma list from 1,4,9; none = off (default 1,4,9)
//   TD_CRITIC_LR      critic step size, per active feature             (default 0.6)
//   TD_ADV_MIX        β: share of the final result in the advantage    (default 0.5)
//   TD_BASELINE       EMA decay of the return baseline, critic off only (default 0.9)
//   TD_SIMS           cap on simulations per move; 0 = time budget only (default 0)
function create(cfg) {
  cfg = cfg || Util.makeCfg();

  const D5 = cfg.int('TD_ACTOR_LAYER5_DEPTH', 0);
  const D9 = cfg.int('TD_ACTOR_LAYER9_DEPTH', 0);
  const USE5 = D5 > 0, USE9 = D9 > 0;        // layer ever used (tables, snapshots)
  let act5 = USE5, act9 = USE9;              // layer active at the current sim ply
  const LR       = cfg.float('TD_ACTOR_LR', 0.1);
  const TEMP     = cfg.float('TD_TEMP', 1);
  const cStr     = cfg.str('TD_CRITIC_LAYERS', '1,4,9');
  const cList    = (cStr === '' || cStr === 'none') ? [] : cStr.split(',').map(s => parseInt(s, 10));
  const C1 = cList.includes(1), C4 = cList.includes(4), C9 = cList.includes(9);
  const CRITIC   = C1 || C4 || C9;
  const CLR      = cfg.float('TD_CRITIC_LR', 0.6);
  const ADV_MIX  = cfg.float('TD_ADV_MIX', 0.5);
  const BASE_EMA = cfg.float('TD_BASELINE', 0.9);
  const SIMS_CAP = cfg.int('TD_SIMS', 0);
  const NEED9    = USE9 || C9;             // the 8-cell code is ever needed

  // ── Per-instance state (sized on first use; rebuilt if the board size changes) ──
  let area = 0;
  let w1 = null, w5 = null, w9 = null;          // actor weights: [mover][p](code)
  let c1 = null, c4 = null, c9 = null;          // critic weights: [mover][p](code)
  let sc = null, ex = null;                     // per-mover score / exp(score/T) over points
  const S = [0, 0];                             // per-mover Σ ex over allowed points
  const Z = [0, 0];                             // per-mover critic logit of the current board
  let k5a = null, k9a = null;                   // per-point current window codes
  let i1 = null, i4 = null, i9 = null;          // per-anchor current critic index (−1 inactive)
  let nAct = 0;                                 // active critic features on the current board
  let mark = null, list = null;                 // dedup scratch for affected points
  let changed = null;                           // cells altered by a move (stone + captures)
  // Per-step records for the actor's update at sim end.
  let exs = null, Ss = null, movers = null, chosen = null, Vs = null, k5s = null, k9s = null;
  // Three-slot ring of critic index snapshots (step t in slot t % 3) for the
  // two-step-lagged critic update, with the active-feature count per slot.
  let i1r = null, i4r = null, i9r = null, nActR = null;
  let maxSteps = 0;
  const base = [0.5, 0.5];                      // per-mover return baseline (critic off)
  let lastMoveCount = -1;

  function setup(N) {
    area = N * N;
    w1 = new Float32Array(2 * area);
    w5 = new Float32Array(USE5 ? 2 * area * 81 : 0);
    w9 = new Float32Array(USE9 ? 2 * area * 6561 : 0);
    c1 = new Float32Array(C1 ? 2 * area * 3 : 0);
    c4 = new Float32Array(C4 ? 2 * area * 81 : 0);
    c9 = new Float32Array(C9 ? 2 * area * 19683 : 0);
    sc = [new Float64Array(area), new Float64Array(area)];
    ex = [new Float64Array(area), new Float64Array(area)];
    k5a = new Int32Array(area); k9a = new Int32Array(area);
    i1 = new Int32Array(area); i4 = new Int32Array(area); i9 = new Int32Array(area);
    mark = new Uint8Array(area);
    list = new Int32Array(area);
    changed = new Int32Array(area + 1);
    maxSteps = 3 * area + 21;
    exs    = new Float64Array(maxSteps * area);
    Ss     = new Float64Array(maxSteps);
    Vs     = new Float64Array(maxSteps);
    movers = new Uint8Array(maxSteps);
    chosen = new Int32Array(maxSteps);
    k5s = new Int32Array(USE5 ? maxSteps * area : 0);
    k9s = new Int32Array(USE9 ? maxSteps * area : 0);
    i1r = new Int32Array(C1 ? 3 * area : 0); i4r = new Int32Array(C4 ? 3 * area : 0); i9r = new Int32Array(C9 ? 3 * area : 0);
    nActR = new Int32Array(3);
  }

  function reset() {
    w1.fill(0); w5.fill(0); w9.fill(0);
    c1.fill(0); c4.fill(0); c9.fill(0);
    base[0] = base[1] = 0.5;
  }

  // Window codes at p from a cell array (base 3, cell + 1 per digit).
  function code5(cells, nbr, p) {
    const b = p * 4;
    return (cells[nbr[b]] + 1) + 3 * (cells[nbr[b + 1]] + 1) + 9 * (cells[nbr[b + 2]] + 1) + 27 * (cells[nbr[b + 3]] + 1);
  }
  function code9(cells, dnbr, p, c5) {
    const b = p * 4;
    return c5 + 81 * ((cells[dnbr[b]] + 1) + 3 * (cells[dnbr[b + 1]] + 1) + 9 * (cells[dnbr[b + 2]] + 1) + 27 * (cells[dnbr[b + 3]] + 1));
  }
  // 2×2 window anchored at p: p, right, down, down-right (nbr 3, nbr 1, dnbr 3).
  function code4(cells, nbr, dnbr, p) {
    const b = p * 4;
    return (cells[p] + 1) + 3 * (cells[nbr[b + 3]] + 1) + 9 * (cells[nbr[b + 1]] + 1) + 27 * (cells[dnbr[b + 3]] + 1);
  }

  // Actor score for mover m at p from precomputed codes (active layers only).
  function scoreFrom(m, p, k5, k9) {
    const mp = m * area + p;
    let s = w1[mp];
    if (act5) s += w5[mp * 81 + k5];
    if (act9) s += w9[mp * 6561 + k9];
    return s;
  }
  function score(cells, nbr, dnbr, m, p) {
    const k5 = (USE5 || NEED9) ? code5(cells, nbr, p) : 0;
    const k9 = NEED9 ? code9(cells, dnbr, p, k5) : 0;
    return scoreFrom(m, p, k5, k9);
  }

  // Swap one critic layer's index at anchor p, keeping both movers' logits
  // and the active-feature count in step.
  function swapCritic(w, idx, size, p, ni) {
    const oi = idx[p];
    if (oi === ni) return;
    for (let m = 0; m < 2; m++) {
      const o = m * size;
      Z[m] += (ni >= 0 ? w[o + ni] : 0) - (oi >= 0 ? w[o + oi] : 0);
    }
    if (oi < 0) nAct++; else if (ni < 0) nAct--;
    idx[p] = ni;
  }

  // Critic indices anchored at p (−1 = inactive), from the codes; shared by
  // the incremental path and the backward pass.
  function critIdx1(v, p)             { return v === EMPTY ? -1 : p * 3 + v + 1; }
  function critIdx4(cells, nbr, dnbr, p) { const k4 = code4(cells, nbr, dnbr, p); return k4 === 0 ? -1 : p * 81 + k4; }
  function critIdx9(v, p, k9)         { const k = k9 + 6561 * (v + 1); return k === 0 ? -1 : p * 19683 + k; }

  function recomputeCritic(cells, nbr, dnbr, p, k9, v) {
    if (C1) swapCritic(c1, i1, area * 3, p, critIdx1(v, p));
    if (C4) swapCritic(c4, i4, area * 81, p, critIdx4(cells, nbr, dnbr, p));
    if (C9) swapCritic(c9, i9, area * 19683, p, critIdx9(v, p, k9));
  }

  // Recompute everything anchored at p: both movers' actor score/ex (S kept
  // in step) and every critic layer's index (Z kept in step).  Kept small so
  // V8 inlines the code and score helpers into it (the hot function).
  function recompute(cells, nbr, dnbr, p) {
    const k5 = (USE5 || NEED9) ? code5(cells, nbr, p) : 0;
    const k9 = NEED9 ? code9(cells, dnbr, p, k5) : 0;
    k5a[p] = k5; k9a[p] = k9;
    const v  = cells[p];
    for (let m = 0; m < 2; m++) {
      const old = ex[m][p];
      let e = 0;
      if (v === EMPTY) { const s = scoreFrom(m, p, k5, k9); sc[m][p] = s; e = Math.exp(s / TEMP); }
      ex[m][p] = e;
      S[m] += e - old;
    }
    if (CRITIC) recomputeCritic(cells, nbr, dnbr, p, k9, v);
  }

  function recomputeAll(cells, nbr, dnbr) {
    ex[0].fill(0); ex[1].fill(0); S[0] = S[1] = 0;
    i1.fill(-1); i4.fill(-1); i9.fill(-1); Z[0] = Z[1] = 0; nAct = 0;
    for (let p = 0; p < area; p++) recompute(cells, nbr, dnbr, p);
  }

  // Refresh only the actor scores from the stored codes, after the active
  // layer set changes mid-sim.  Exclusions are lifted; sampling re-applies them.
  function refreshScores(cells) {
    S[0] = S[1] = 0;
    for (let p = 0; p < area; p++) {
      for (let m = 0; m < 2; m++) {
        let e = 0;
        if (cells[p] === EMPTY) { const s = scoreFrom(m, p, k5a[p], k9a[p]); sc[m][p] = s; e = Math.exp(s / TEMP); }
        ex[m][p] = e;
        S[m] += e;
      }
    }
  }

  // After stone changes at the cells in `changed`, recompute every point
  // whose 3×3 neighbourhood contains one of them (the cell and its 8
  // neighbours).  That covers every actor and critic feature that can move.
  function recomputeAround(g, changed, nChanged) {
    const cells = g.cells, nbr = g._nbr, dnbr = g._dnbr;
    let n = 0;
    for (let i = 0; i < nChanged; i++) {
      const q = changed[i], b = q * 4;
      if (!mark[q]) { mark[q] = 1; list[n++] = q; }
      for (let d = 0; d < 4; d++) {
        const a = nbr[b + d];  if (!mark[a]) { mark[a] = 1; list[n++] = a; }
        const c = dnbr[b + d]; if (!mark[c]) { mark[c] = 1; list[n++] = c; }
      }
    }
    for (let i = 0; i < n; i++) { const p = list[i]; mark[p] = 0; recompute(cells, nbr, dnbr, p); }
  }

  // Sample a point for mover m from ex[m] (softmax), rejecting illegal and
  // true-eye points by zeroing them.  PASS when nothing is left.
  function sample(g, m, rng) {
    const e = ex[m];
    while (S[m] > 1e-300) {
      let u = rng.random() * S[m], p = -1;
      for (let i = 0; i < area; i++) { const v = e[i]; if (v > 0) { p = i; u -= v; if (u < 0) break; } }
      if (p < 0) break;
      if (g.isLegal(p) && !g.isTrueEye(p)) return p;
      S[m] -= e[p]; e[p] = 0;
    }
    S[m] = 0;
    return PASS;
  }

  function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

  // One simulation from `game`; returns the step count.  Records per step the
  // board, the sampled-from distribution, mover, chosen move, V(s_t) and the
  // critic's active feature indices, for the backward pass.
  function simulate(game, rng) {
    const g = game.clone();
    const nbr = g._nbr, dnbr = g._dnbr, cells = g.cells;
    act5 = USE5; act9 = USE9;              // ply 0: every enabled layer is on
    recomputeAll(cells, nbr, dnbr);
    let t = 0;
    while (!g.gameOver && t < maxSteps - 1) {
      const m = g.current === BLACK ? 0 : 1;
      const o = t * area;
      const a5 = t < D5, a9 = t < D9;
      if (a5 !== act5 || a9 !== act9) { act5 = a5; act9 = a9; refreshScores(cells); }
      if (CRITIC) {
        Vs[t] = sigmoid(Z[m]);
        const slot = t % 3, so = slot * area;
        if (C1) i1r.set(i1, so);
        if (C4) i4r.set(i4, so);
        if (C9) i9r.set(i9, so);
        nActR[slot] = nAct;
        // Step t−2 (same mover) is now two plies on: push it toward V(s_t).
        if (t >= 2) criticUpdate((t - 2) % 3, m, Vs[t - 2], Vs[t], true);
      }
      const move = sample(g, m, rng);
      exs.set(ex[m], o);
      if (act5) k5s.set(k5a, o);
      if (act9) k9s.set(k9a, o);
      Ss[t] = S[m]; movers[t] = m; chosen[t] = move;
      const prevKo = g.ko;
      let nChanged = 0;
      if (move !== PASS) {
        const caps = g.captureList(move);
        changed[nChanged++] = move;
        for (let i = 0; i < caps.length; i++) changed[nChanged++] = caps[i];
      }
      g.play(move);
      if (nChanged > 0) recomputeAround(g, changed, nChanged);
      // A ko point turns legal again once any other move is played; its
      // neighbourhood did not change, so lift its exclusion explicitly.
      if (prevKo !== PASS && cells[prevKo] === EMPTY) recompute(cells, nbr, dnbr, prevKo);
      t++;
    }
    const z = g.calcWinner() === BLACK ? 1 : 0;
    // Terminal: the last two steps have no position two plies on.
    if (CRITIC) for (let u = Math.max(0, t - 2); u < t; u++) criticUpdate(u % 3, movers[u], Vs[u], z, false);
    update(t, z);
    return t;
  }

  // Critic TD step for the position in ring slot `slot` (mover m, value v):
  // Δw = lr·(target − v)/n on each of its active features in mover m's tables.
  // A feature index encodes its anchor, so it is still active on the live board
  // iff the live index at that anchor equals it; with `fixZ` the live logit
  // Z[m] is corrected for those shared features.
  function criticUpdate(slot, m, v, target, fixZ) {
    const n = nActR[slot];
    if (n === 0) return;
    const step = CLR * (target - v) / n;
    const so = slot * area;
    let dz = 0;
    if (C1) { const b = m * area * 3;     for (let p = 0; p < area; p++) { const i = i1r[so + p]; if (i >= 0) { c1[b + i] += step; if (i1[p] === i) dz += step; } } }
    if (C4) { const b = m * area * 81;    for (let p = 0; p < area; p++) { const i = i4r[so + p]; if (i >= 0) { c4[b + i] += step; if (i4[p] === i) dz += step; } } }
    if (C9) { const b = m * area * 19683; for (let p = 0; p < area; p++) { const i = i9r[so + p]; if (i >= 0) { c9[b + i] += step; if (i9[p] === i) dz += step; } } }
    if (fixZ) Z[m] += dz;
  }

  // Actor policy-gradient step for the record at t with advantage A (mover's
  // view): for every point a in the sampled-from distribution,
  // Δw(a) = lr/T · A · ([a = chosen] − π(a)) on each layer's weight for (m, a).
  function actorUpdate(t, m, A) {
    const move = chosen[t];
    if (move === PASS) return;             // PASS is outside the softmax
    const o = t * area;
    const e = exs.subarray(o, o + area);
    const k5 = k5s.subarray(o, o + area), k9 = k9s.subarray(o, o + area);
    const u5 = t < D5, u9 = t < D9;        // layers that were active at this ply
    const invS = 1 / Ss[t];
    const k = LR / TEMP * A;
    for (let p = 0; p < area; p++) {
      const v = e[p];
      if (v <= 0) continue;
      const gr = k * ((p === move ? 1 : 0) - v * invS);
      const mp = m * area + p;
      w1[mp] += gr;
      if (u5) w5[mp * 81 + k5[p]] += gr;
      if (u9) w9[mp * 6561 + k9[p]] += gr;
    }
  }

  // Backward pass over one sim's records.  z = outcome, P(BLACK wins) ∈ {0,1}.
  // G[m] carries the same-mover chain: the next value of mover m's positions
  // (V_{t+2}), starting from the outcome.  Values are BLACK's; the mover's view
  // is v for BLACK and 1 − v for WHITE.
  // Actor update over the sim's records, in step order.  z = outcome,
  // P(BLACK wins) ∈ {0,1}; values are BLACK's, the mover's view is v for BLACK
  // and 1 − v for WHITE.
  function update(steps, z) {
    if (!CRITIC) {
      const adv = [z - base[0], (1 - z) - base[1]];
      base[0] += (1 - BASE_EMA) * (z - base[0]);
      base[1] += (1 - BASE_EMA) * ((1 - z) - base[1]);
      for (let t = 0; t < steps; t++) actorUpdate(t, movers[t], adv[movers[t]]);
      return;
    }
    for (let t = 0; t < steps; t++) {
      const m = movers[t];
      const v = Vs[t];
      const next = t + 2 < steps ? Vs[t + 2] : z;   // the same mover's next position, or the outcome
      const sign = m === 0 ? 1 : -1;      // mover's view of a BLACK-view difference
      const delta = sign * (next - v);    // TD advantage
      const mc    = sign * (z - v);       // final-result advantage
      actorUpdate(t, m, (1 - ADV_MIX) * delta + ADV_MIX * mc);
    }
  }

  function getMove(game, budgetMs = 1000, options = {}) {
    if (game.consecutivePasses > 0 && game.calcWinner() === game.current) {
      return { move: PASS, info: 'end the game; ahead' };
    }
    if (area !== game.N * game.N) { setup(game.N); reset(); }
    const d = game.moveCount - lastMoveCount;
    if (d < 0 || d > 2) reset();           // a new game (or an unexpected jump)
    lastMoveCount = game.moveCount;

    const rng = options.rng || makeRng();
    const tStart = Date.now();
    let sims = 0, longest = 0, totalSteps = 0;
    while (true) {
      if (SIMS_CAP > 0 ? sims >= SIMS_CAP : Date.now() - tStart >= budgetMs) break;
      const steps = simulate(game, rng);
      if (steps > longest) longest = steps;
      totalSteps += steps;
      sims++;
    }

    // Play the learned policy's argmax over legal non-eye points.
    const cells = game.cells, nbr = game._nbr, dnbr = game._dnbr;
    const m = game.current === BLACK ? 0 : 1;
    act5 = USE5; act9 = USE9;             // the root is ply 0
    recomputeAll(cells, nbr, dnbr);       // root scores and logits
    let best = PASS, bestS = -Infinity;
    for (let p = 0; p < area; p++) {
      if (cells[p] !== EMPTY || !game.isLegal(p) || game.isTrueEye(p)) continue;
      const s = sc[m][p] + rng.random() * 1e-9;
      if (s > bestS) { bestS = s; best = p; }
    }
    const val = CRITIC ? sigmoid(Z[m]) : base[m];
    return { move: best, info: `sims=${sims} steps=${totalSteps} longest=${longest} V=${val.toFixed(3)} score=${bestS.toFixed(3)}` };
  }

  // Test hook: live views of the internals (state arrays are created by setup).
  function _internals() {
    return { setup, reset, recomputeAll, recomputeAround, refreshScores, score, sample, simulate, update, sigmoid,
             setActive: (a5, a9) => { act5 = a5; act9 = a9; },
             sc, ex, S, Z, i1, i4, i9, k5a, k9a, base, w1, w5, w9, c1, c4, c9, area, CRITIC };
  }

  return { getMove, _internals };
}

let _default = null;
function _def() { return _default || (_default = create(Util.makeCfg())); }
if (typeof module !== 'undefined') module.exports = { create, getMove: (g, b, o) => _def().getMove(g, b, o) };
else window.getMove = (g, b, o) => _def().getMove(g, b, o);

})();
