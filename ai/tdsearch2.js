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
// Actor: score(p) for an empty point p is the sum of the enabled layers
//   1: the point itself                    (mover, p)
//   5: its 4 orthogonal neighbours          (mover, p, base-3 code of 4 cells)
//   9: those plus the 4 diagonals           (mover, p, base-3 code of 8 cells)
// and the policy is a softmax over the empty points.
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
// Learning, once per sim in a backward pass:
//   critic: two-ply same-parity TD(0) — the target for step t is V at step t+2
//           (the same mover's next position), the outcome at the end.
//   actor:  REINFORCE with advantage A_t = (1−β)·δ_t + β·(R − V_t), all from the
//           mover's view, where δ_t = V_{t+2} − V_t is the TD advantage and R
//           the final result; β = TD_ADV_MIX.  Without a critic the baseline
//           is a per-mover EMA of sim returns.
//
// No pretrained models, no simulation truncation (planned later).
//
// ── Factory ──
// create(cfg) -> { getMove }.  cfg is a Util.makeCfg reader (P1_/P2_ prefixes in selfplay).
//
// Config:
//   TD_LAYERS         actor layers, comma list from 1,5,9              (default 1,5,9)
//   TD_LR             actor step size                                  (default 0.1)
//   TD_TEMP           softmax temperature for the simulations          (default 1)
//   TD_CRITIC_LAYERS  critic layers, comma list from 1,4,9; none = off (default 1,4,9)
//   TD_CRITIC_LR      critic step size, per active feature             (default 0.3)
//   TD_ADV_MIX        β: share of the final result in the advantage    (default 0.5)
//   TD_BASELINE       EMA decay of the return baseline, critic off only (default 0.9)
//   TD_SIMS           cap on simulations per move; 0 = time budget only (default 0)
function create(cfg) {
  cfg = cfg || Util.makeCfg();

  const layerList = cfg.str('TD_LAYERS', '1,5,9').split(',').map(s => parseInt(s, 10));
  const USE1 = layerList.includes(1), USE5 = layerList.includes(5), USE9 = layerList.includes(9);
  if (!USE1 && !USE5 && !USE9) throw new Error('tdsearch2: TD_LAYERS must include at least one of 1,5,9');
  const LR       = cfg.float('TD_LR', 0.1);
  const TEMP     = cfg.float('TD_TEMP', 1);
  const cStr     = cfg.str('TD_CRITIC_LAYERS', '1,4,9');
  const cList    = (cStr === '' || cStr === 'none') ? [] : cStr.split(',').map(s => parseInt(s, 10));
  const C1 = cList.includes(1), C4 = cList.includes(4), C9 = cList.includes(9);
  const CRITIC   = C1 || C4 || C9;
  const CLR      = cfg.float('TD_CRITIC_LR', 0.3);
  const ADV_MIX  = cfg.float('TD_ADV_MIX', 0.5);
  const BASE_EMA = cfg.float('TD_BASELINE', 0.9);
  const SIMS_CAP = cfg.int('TD_SIMS', 0);
  const NEED9    = USE9 || C9;

  // ── Per-instance state (sized on first use; rebuilt if the board size changes) ──
  let area = 0;
  let w1 = null, w5 = null, w9 = null;          // actor weights: [mover][p](code)
  let c1 = null, c4 = null, c9 = null;          // critic weights: [mover][p](code)
  let sc = null, ex = null;                     // per-mover score / exp(score/T) over points
  const S = [0, 0];                             // per-mover Σ ex over allowed points
  const Z = [0, 0];                             // per-mover critic logit of the current board
  let i1 = null, i4 = null, i9 = null;          // per-anchor current critic index (−1 inactive)
  let mark = null, list = null;                 // dedup scratch for affected points
  let changed = null;                           // cells altered by a move (stone + captures)
  // Per-step records for the backward pass.
  let boards = null, exs = null, Ss = null, movers = null, chosen = null, Vs = null;
  let i1s = null, i4s = null, i9s = null;
  let maxSteps = 0;
  const base = [0.5, 0.5];                      // per-mover return baseline (critic off)
  let lastMoveCount = -1;

  function setup(N) {
    area = N * N;
    w1 = new Float32Array(2 * area);
    w5 = new Float32Array(2 * area * 81);
    w9 = new Float32Array(2 * area * 6561);
    c1 = new Float32Array(C1 ? 2 * area * 3 : 0);
    c4 = new Float32Array(C4 ? 2 * area * 81 : 0);
    c9 = new Float32Array(C9 ? 2 * area * 19683 : 0);
    sc = [new Float64Array(area), new Float64Array(area)];
    ex = [new Float64Array(area), new Float64Array(area)];
    i1 = new Int32Array(area); i4 = new Int32Array(area); i9 = new Int32Array(area);
    mark = new Uint8Array(area);
    list = new Int32Array(area);
    changed = new Int32Array(area + 1);
    maxSteps = 3 * area + 21;
    boards = new Int8Array(maxSteps * area);
    exs    = new Float64Array(maxSteps * area);
    Ss     = new Float64Array(maxSteps);
    Vs     = new Float64Array(maxSteps);
    movers = new Uint8Array(maxSteps);
    chosen = new Int32Array(maxSteps);
    i1s = new Int32Array(C1 ? maxSteps * area : 0);
    i4s = new Int32Array(C4 ? maxSteps * area : 0);
    i9s = new Int32Array(C9 ? maxSteps * area : 0);
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

  // Actor score for mover m at p from precomputed codes.
  function scoreFrom(m, p, k5, k9) {
    const mp = m * area + p;
    let s = 0;
    if (USE1) s += w1[mp];
    if (USE5) s += w5[mp * 81 + k5];
    if (USE9) s += w9[mp * 6561 + k9];
    return s;
  }
  function score(cells, nbr, dnbr, m, p) {
    const k5 = (USE5 || NEED9) ? code5(cells, nbr, p) : 0;
    const k9 = NEED9 ? code9(cells, dnbr, p, k5) : 0;
    return scoreFrom(m, p, k5, k9);
  }

  // Swap one critic layer's index at anchor p, keeping both movers' logits in step.
  function swapCritic(w, idx, size, p, ni) {
    const oi = idx[p];
    if (oi === ni) return;
    for (let m = 0; m < 2; m++) {
      const o = m * size;
      Z[m] += (ni >= 0 ? w[o + ni] : 0) - (oi >= 0 ? w[o + oi] : 0);
    }
    idx[p] = ni;
  }

  // Recompute everything anchored at p: both movers' actor score/ex (S kept
  // in step) and every critic layer's index (Z kept in step).
  function recompute(cells, nbr, dnbr, p) {
    const k5 = (USE5 || NEED9) ? code5(cells, nbr, p) : 0;
    const k9 = NEED9 ? code9(cells, dnbr, p, k5) : 0;
    const v  = cells[p];
    for (let m = 0; m < 2; m++) {
      const old = ex[m][p];
      let e = 0;
      if (v === EMPTY) { const s = scoreFrom(m, p, k5, k9); sc[m][p] = s; e = Math.exp(s / TEMP); }
      ex[m][p] = e;
      S[m] += e - old;
    }
    if (C1) swapCritic(c1, i1, area * 3, p, v === EMPTY ? -1 : p * 3 + v + 1);
    if (C4) { const k4 = code4(cells, nbr, dnbr, p); swapCritic(c4, i4, area * 81, p, k4 === 0 ? -1 : p * 81 + k4); }
    if (C9) { const k = k9 + 6561 * (v + 1); swapCritic(c9, i9, area * 19683, p, k === 0 ? -1 : p * 19683 + k); }
  }

  function recomputeAll(cells, nbr, dnbr) {
    ex[0].fill(0); ex[1].fill(0); S[0] = S[1] = 0;
    i1.fill(-1); i4.fill(-1); i9.fill(-1); Z[0] = Z[1] = 0;
    for (let p = 0; p < area; p++) recompute(cells, nbr, dnbr, p);
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
    recomputeAll(cells, nbr, dnbr);
    let t = 0;
    while (!g.gameOver && t < maxSteps - 1) {
      const m = g.current === BLACK ? 0 : 1;
      const move = sample(g, m, rng);
      const o = t * area;
      boards.set(cells, o);
      exs.set(ex[m], o);
      Ss[t] = S[m]; movers[t] = m; chosen[t] = move;
      if (CRITIC) {
        Vs[t] = sigmoid(Z[m]);
        if (C1) i1s.set(i1, o);
        if (C4) i4s.set(i4, o);
        if (C9) i9s.set(i9, o);
      }
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
    update(t, z, nbr, dnbr);
    return t;
  }

  // Critic TD step for the record at t: Δw = lr·(target − V)/n on every active
  // feature of mover m's tables.
  function criticUpdate(t, m, target) {
    const o = t * area;
    let n = 0;
    if (C1) for (let p = 0; p < area; p++) if (i1s[o + p] >= 0) n++;
    if (C4) for (let p = 0; p < area; p++) if (i4s[o + p] >= 0) n++;
    if (C9) for (let p = 0; p < area; p++) if (i9s[o + p] >= 0) n++;
    if (n === 0) return;
    const step = CLR * (target - Vs[t]) / n;
    if (C1) { const b = m * area * 3;     for (let p = 0; p < area; p++) { const i = i1s[o + p]; if (i >= 0) c1[b + i] += step; } }
    if (C4) { const b = m * area * 81;    for (let p = 0; p < area; p++) { const i = i4s[o + p]; if (i >= 0) c4[b + i] += step; } }
    if (C9) { const b = m * area * 19683; for (let p = 0; p < area; p++) { const i = i9s[o + p]; if (i >= 0) c9[b + i] += step; } }
  }

  // Actor policy-gradient step for the record at t with advantage A (mover's
  // view): for every point a in the sampled-from distribution,
  // Δw(a) = lr/T · A · ([a = chosen] − π(a)) on each layer's weight for (m, a).
  function actorUpdate(t, m, A, nbr, dnbr) {
    const move = chosen[t];
    if (move === PASS) return;             // PASS is outside the softmax
    const o = t * area;
    const cells = boards.subarray(o, o + area);
    const e = exs.subarray(o, o + area);
    const invS = 1 / Ss[t];
    const k = LR / TEMP * A;
    for (let p = 0; p < area; p++) {
      const v = e[p];
      if (v <= 0) continue;
      const gr = k * ((p === move ? 1 : 0) - v * invS);
      const mp = m * area + p;
      if (USE1) w1[mp] += gr;
      if (USE5 || USE9) {
        const k5 = code5(cells, nbr, p);
        if (USE5) w5[mp * 81 + k5] += gr;
        if (USE9) w9[mp * 6561 + code9(cells, dnbr, p, k5)] += gr;
      }
    }
  }

  // Backward pass over one sim's records.  z = outcome, P(BLACK wins) ∈ {0,1}.
  // G[m] carries the same-mover chain: the next value of mover m's positions
  // (V_{t+2}), starting from the outcome.  Values are BLACK's; the mover's view
  // is v for BLACK and 1 − v for WHITE.
  function update(steps, z, nbr, dnbr) {
    if (!CRITIC) {
      const adv = [z - base[0], (1 - z) - base[1]];
      base[0] += (1 - BASE_EMA) * (z - base[0]);
      base[1] += (1 - BASE_EMA) * ((1 - z) - base[1]);
      for (let t = 0; t < steps; t++) actorUpdate(t, movers[t], adv[movers[t]], nbr, dnbr);
      return;
    }
    const G = [z, z];
    for (let t = steps - 1; t >= 0; t--) {
      const m = movers[t];
      const next = G[m], v = Vs[t];
      const sign = m === 0 ? 1 : -1;      // mover's view of a BLACK-view difference
      const delta = sign * (next - v);    // TD advantage
      const mc    = sign * (z - v);       // final-result advantage
      actorUpdate(t, m, (1 - ADV_MIX) * delta + ADV_MIX * mc, nbr, dnbr);
      criticUpdate(t, m, next);
      G[m] = v;
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
    let sims = 0, longest = 0;
    while (true) {
      if (SIMS_CAP > 0 ? sims >= SIMS_CAP : Date.now() - tStart >= budgetMs) break;
      const steps = simulate(game, rng);
      if (steps > longest) longest = steps;
      sims++;
    }

    // Play the learned policy's argmax over legal non-eye points.
    const cells = game.cells, nbr = game._nbr, dnbr = game._dnbr;
    const m = game.current === BLACK ? 0 : 1;
    recomputeAll(cells, nbr, dnbr);       // root scores and logits
    let best = PASS, bestS = -Infinity;
    for (let p = 0; p < area; p++) {
      if (cells[p] !== EMPTY || !game.isLegal(p) || game.isTrueEye(p)) continue;
      const s = sc[m][p] + rng.random() * 1e-9;
      if (s > bestS) { bestS = s; best = p; }
    }
    const val = CRITIC ? sigmoid(Z[m]) : base[m];
    return { move: best, info: `sims=${sims} longest=${longest} V=${val.toFixed(3)} score=${bestS.toFixed(3)}` };
  }

  // Test hook: live views of the internals (state arrays are created by setup).
  function _internals() {
    return { setup, reset, recomputeAll, recomputeAround, score, sample, simulate, update, sigmoid,
             sc, ex, S, Z, i1, i4, i9, base, w1, w5, w9, c1, c4, c9, area, CRITIC };
  }

  return { getMove, _internals };
}

let _default = null;
function _def() { return _default || (_default = create(Util.makeCfg())); }
if (typeof module !== 'undefined') module.exports = { create, getMove: (g, b, o) => _def().getMove(g, b, o) };
else window.getMove = (g, b, o) => _def().getMove(g, b, o);

})();
