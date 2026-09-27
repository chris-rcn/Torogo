'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

const Util = (typeof require === 'function') ? require('../util.js') : window.Util;
const { BLACK, EMPTY, PASS } = Util.load('./game2.js', 'Game2');
const { makeRng } = Util.load('./xorshift.js', 'XorShift');

// tdsearch2 — online policy learning during think time (actor only, so far).
//
// Each move, self-play simulations run from the current position; the actor
// policy that plays them is updated by REINFORCE from the simulation outcomes,
// and the learned policy's argmax is played.  The policy persists across the
// moves of one game (reset on a new game) so knowledge accumulates.
//
// Features are LOCATION-DEPENDENT by design — no symmetry, the board is in
// the orientation it is in — and indexed combinatorially, no hashing:
//   layer 1: the point itself                   (mover, p)
//   layer 5: the point's 4 orthogonal neighbours (mover, p, base-3 code of 4 cells)
//   layer 9: those plus the 4 diagonals          (mover, p, base-3 code of 8 cells)
// score(p) = the sum of the enabled layers' weights; the policy is a softmax
// over the empty points.  Scores are kept per mover and maintained
// incrementally: a move only changes the scores of points within one cell of
// the placed or captured stones.  Sampling rejects illegal and true-eye points
// lazily (their weight is zeroed until their neighbourhood changes).
//
// No pretrained models, no simulation truncation (planned later).
//
// ── Factory ──
// create(cfg) -> { getMove }.  cfg is a Util.makeCfg reader (P1_/P2_ prefixes in selfplay).
//
// Config:
//   TD_LAYERS      comma list from 1,5,9                       (default 1,5,9)
//   TD_LR          policy-gradient step size                    (default 0.1)
//   TD_TEMP        softmax temperature for the simulations      (default 1)
//   TD_BASELINE    EMA decay of the per-mover return baseline   (default 0.9)
//   TD_SIMS        cap on simulations per move; 0 = time budget only (default 0)
function create(cfg) {
  cfg = cfg || Util.makeCfg();

  const layerList = cfg.str('TD_LAYERS', '1,5,9').split(',').map(s => parseInt(s, 10));
  const USE1 = layerList.includes(1), USE5 = layerList.includes(5), USE9 = layerList.includes(9);
  if (!USE1 && !USE5 && !USE9) throw new Error('tdsearch2: TD_LAYERS must include at least one of 1,5,9');
  const LR       = cfg.float('TD_LR', 0.1);
  const TEMP     = cfg.float('TD_TEMP', 1);
  const BASE_EMA = cfg.float('TD_BASELINE', 0.9);
  const SIMS_CAP = cfg.int('TD_SIMS', 0);

  // ── Per-instance state (sized on first use; rebuilt if the board size changes) ──
  let area = 0;
  let w1 = null, w5 = null, w9 = null;          // weights: [mover][p](code)
  let sc = null, ex = null;                     // per-mover score / exp(score/T) over points
  const S = [0, 0];                             // per-mover Σ ex over allowed points
  let mark = null, list = null;                 // dedup scratch for affected points
  let boards = null, exs = null, Ss = null, movers = null, chosen = null;   // per-step records
  let changed = null;                           // cells altered by a move (stone + captures)
  let maxSteps = 0;
  const base = [0.5, 0.5];                      // per-mover return baseline (EMA)
  let lastMoveCount = -1;

  function setup(N) {
    area = N * N;
    w1 = new Float32Array(2 * area);
    w5 = new Float32Array(2 * area * 81);
    w9 = new Float32Array(2 * area * 6561);
    sc = [new Float64Array(area), new Float64Array(area)];
    ex = [new Float64Array(area), new Float64Array(area)];
    mark = new Uint8Array(area);
    list = new Int32Array(area);
    maxSteps = 3 * area + 21;
    boards = new Int8Array(maxSteps * area);
    exs    = new Float64Array(maxSteps * area);
    Ss     = new Float64Array(maxSteps);
    movers = new Uint8Array(maxSteps);
    chosen = new Int32Array(maxSteps);
    changed = new Int32Array(area + 1);
  }

  function reset() {
    w1.fill(0); w5.fill(0); w9.fill(0);
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

  function score(cells, nbr, dnbr, m, p) {
    const mp = m * area + p;
    let s = 0;
    if (USE1) s += w1[mp];
    if (USE5 || USE9) {
      const c5 = code5(cells, nbr, p);
      if (USE5) s += w5[mp * 81 + c5];
      if (USE9) s += w9[mp * 6561 + code9(cells, dnbr, p, c5)];
    }
    return s;
  }

  // Recompute both movers' score/ex at p, keeping S in step.
  function recompute(cells, nbr, dnbr, p) {
    for (let m = 0; m < 2; m++) {
      const old = ex[m][p];
      let e = 0;
      if (cells[p] === EMPTY) { const s = score(cells, nbr, dnbr, m, p); sc[m][p] = s; e = Math.exp(s / TEMP); }
      ex[m][p] = e;
      S[m] += e - old;
    }
  }

  function recomputeAll(cells, nbr, dnbr) {
    ex[0].fill(0); ex[1].fill(0); S[0] = S[1] = 0;
    for (let p = 0; p < area; p++) recompute(cells, nbr, dnbr, p);
  }

  // After stone changes at the cells in `changed`, recompute every point
  // whose 9-cell window contains one of them (the cell and its 8 neighbours).
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

  // One simulation from `game`; returns the step count.  Records per step
  // (board, the sampled-from distribution, mover, chosen move) for the update.
  function simulate(game, rng) {
    const g = game.clone();
    const nbr = g._nbr, dnbr = g._dnbr, cells = g.cells;
    recomputeAll(cells, nbr, dnbr);
    let t = 0;
    while (!g.gameOver && t < maxSteps - 1) {
      const m = g.current === BLACK ? 0 : 1;
      const move = sample(g, m, rng);
      boards.set(cells, t * area);
      exs.set(ex[m], t * area);
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
    update(t, z, nbr, dnbr);
    return t;
  }

  // REINFORCE with a per-mover baseline: for every point a in the sampled-from
  // distribution, Δw(a) = lr/T · A · ([a = chosen] − π(a)) on each layer's
  // weight for (mover, a).  PASS steps are outside the softmax and get no update.
  function update(steps, z, nbr, dnbr) {
    const adv = [z - base[0], (1 - z) - base[1]];   // the sim's advantage per mover
    base[0] += (1 - BASE_EMA) * (z - base[0]);
    base[1] += (1 - BASE_EMA) * ((1 - z) - base[1]);
    for (let t = 0; t < steps; t++) {
      const m = movers[t];
      const A = adv[m];
      const move = chosen[t];
      if (move === PASS) continue;
      const cells = boards.subarray(t * area, (t + 1) * area);
      const e = exs.subarray(t * area, (t + 1) * area);
      const invS = 1 / Ss[t];
      const k = LR / TEMP * A;
      for (let p = 0; p < area; p++) {
        const v = e[p];
        if (v <= 0) continue;
        const gr = k * ((p === move ? 1 : 0) - v * invS);
        const mp = m * area + p;
        if (USE1) w1[mp] += gr;
        if (USE5 || USE9) {
          const c5 = code5(cells, nbr, p);
          if (USE5) w5[mp * 81 + c5] += gr;
          if (USE9) w9[mp * 6561 + code9(cells, dnbr, p, c5)] += gr;
        }
      }
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
    let best = PASS, bestS = -Infinity;
    for (let p = 0; p < area; p++) {
      if (cells[p] !== EMPTY || !game.isLegal(p) || game.isTrueEye(p)) continue;
      const s = score(cells, nbr, dnbr, m, p) + rng.random() * 1e-9;
      if (s > bestS) { bestS = s; best = p; }
    }
    return { move: best, info: `sims=${sims} longest=${longest} base=${base[m].toFixed(3)} score=${bestS.toFixed(3)}` };
  }

  // Test hook: live views of the internals (state arrays are created by setup).
  function _internals() {
    return { setup, reset, recomputeAll, recomputeAround, score, sample, simulate, update,
             sc, ex, S, base, w1, w5, w9, area };
  }

  return { getMove, _internals };
}

let _default = null;
function _def() { return _default || (_default = create(Util.makeCfg())); }
if (typeof module !== 'undefined') module.exports = { create, getMove: (g, b, o) => _def().getMove(g, b, o) };
else window.getMove = (g, b, o) => _def().getMove(g, b, o);

})();
