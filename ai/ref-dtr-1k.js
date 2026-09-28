'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
const Util = (typeof require === 'function') ? require('../util.js') : window.Util;
const { BLACK, EMPTY, PASS } = Util.load('./game2.js', 'Game2');
const { makeRng } = Util.load('./xorshift.js', 'XorShift');
const PPat = Util.load('./ppat-lib.js', 'PPatterns');
const VPat = Util.load('./vpatterns.js', 'VPatterns');

// ref-dtr-1k — frozen fixed-compute reference: ai/dt-reinforce.js at
// PLAYOUTS=1000, every other knob at its 2026-09-28 default, with the model
// files frozen as ref/ref-dtr-1k-vpat.js (out/vpat-1j9ad1fk.js) and
// ref/ref-dtr-1k-ppat.js (out/ppat-data-233162-best-ref-candidate.js).
// The move budget is ignored: 1000 simulations per move, whatever the clock.
//
// Frozen once fielded: any strength-affecting change gets a new name.
// All parameters are hardcoded.  This script reads no environment variables.
//
// dt-reinforce — decision-time REINFORCE: a policy learned during think time.
//
// Each move, self-play simulations run from the current position.  The actor
// policy that plays them is updated by REINFORCE from the simulations'
// returns, and the actor's argmax is played.  The policy persists across the
// moves of one game (reset on a new game) so knowledge accumulates.  No
// critic, no priors, no search: the actor-only case of dt-actor-critic.
//
// Features are LOCATION-DEPENDENT by design — no symmetry, the board is in
// the orientation it is in — and indexed combinatorially, no hashing.  All are
// keyed by the side to move as well as the point.
//
// Actor: score(p) for an empty point p is the sum of the layers
//   1: the point itself                    (mover, p)               always on
//   5: its 4 orthogonal neighbours          (mover, p, base-3 code of 4 cells)
//   9: those plus the 4 diagonals           (mover, p, base-3 code of 8 cells)
// and the policy is a softmax over the empty points.  The actor plays only
// the first TD_ACTOR_DEPTH plies of a sim; the rest is the standard playout
// (uniform below PPAT_MIN_PHASE, the ppat policy above it).  Silver et al.
// switch to a default policy after ~6 plies, but here the actor's own moves
// are its training data: mdMae improved monotonically out to ~50 plies and
// plateaued 50-100, with unlimited slightly worse (2026-09-27).  Layers 5 and 9 are each
// active only for the first N plies of a sim (their depth knob; 0 = off): deep
// in a sim the board has diverged from the root, so updates to those exact
// local patterns land where no root will read them, while the first plies
// serve this root and the next move's.  At the ply a layer switches off, the
// scores are refreshed once without it.
//
// Everything is maintained incrementally: a move only changes points within
// one cell of the placed or captured stones, and one pass over those points
// refreshes both movers' actor scores.
// Sampling rejects illegal and true-eye points lazily (their weight is zeroed
// until their neighbourhood changes).
//
// Learning: REINFORCE at sim end (it needs the return), in step order over the
// plies the actor played (playout-tail moves carry no gradient), with the step
// TD_ACTOR_TERM_LR·(R − b_m) in the mover's view, where R is the sim's return
// and b_m a per-mover EMA of returns as baseline.  Its per-step records are the
// sampled-from distribution and the point codes.
//
// Truncation (TD_TRUNC_PHASE_DELTA > 0): a sim plays TD_TRUNC_ACTOR_DEPTH actor
// plies, then ceil(delta * area) UNIFORM random plies — the fielded trunc
// agent's rule, here a buffer so the actor cannot steer into the leaf
// model's defects — and stops; the vpat model's value of the truncation point
// stands in for the outcome everywhere the outcome is used.  The model is the
// anchor; no grounding schedule.  Truncation is used only when the truncation
// point's phase (root phase + truncation ply / area, captures ignored) is below
// TD_TRUNC_MAX_PHASE — a property of the model's trusted band, so it stays put
// while the actor depth and delta are swept.
//
// ── Factory ──
// create() -> { getMove }.  Hardcoded configuration (dt-reinforce's names):
const ACTOR_DEPTH       = 35;      // TD_ACTOR_DEPTH: plies of a sim the actor plays; the rest is the standard playout
const D5                = 999;     // TD_ACTOR_LAYER5_DEPTH: plies of a sim for which actor layer 5 is on
const D9                = 999;     // TD_ACTOR_LAYER9_DEPTH: plies of a sim for which actor layer 9 is on
const TERM_LR           = 0.002;   // TD_ACTOR_TERM_LR: actor step size on the return minus the baseline
const TEMP              = 1;       // TD_TEMP: softmax temperature for the simulations
const BASE_EMA          = 0.9;     // TD_ACTOR_RETURN_EMA: decay of the per-mover return baseline
const PLAYOUTS_CAP      = 1000;    // PLAYOUTS: simulations per move (the move budget is ignored)
const TRUNC_DELTA       = 0.2;     // TD_TRUNC_PHASE_DELTA: random buffer after the actor plies, fraction of the area
const TRUNC_ACTOR_DEPTH = 5;       // TD_TRUNC_ACTOR_DEPTH: actor plies in a truncated sim before the buffer
const TRUNC_MAX_PHASE   = 0.52;    // TD_TRUNC_MAX_PHASE: truncate only when the truncation point's phase is below this
const PPAT_MIN_PHASE    = 0.6;     // PPAT_MIN_PHASE: tail moves are uniform below this board fullness
const VPAT_PATH = _isNode ? require('path').join(__dirname, '..', 'ref', 'ref-dtr-1k-vpat.js') : null;   // TRUNC_VPAT_DATA
const PPAT_PATH = _isNode ? require('path').join(__dirname, '..', 'ref', 'ref-dtr-1k-ppat.js') : null;   // PPAT_DATA

function create() {
  const USE5 = D5 > 0, USE9 = D9 > 0;        // layer ever used (tables, snapshots)
  let act5 = USE5, act9 = USE9;              // layer active at the current sim ply
  let actorOn = true;                        // the actor plays the current sim ply (else the tail)
  const raw = _isNode ? require(VPAT_PATH)
                      : (typeof window !== 'undefined' && window.truncVpatModel) || null;
  if (!raw) throw new Error(`ref-dtr-1k: cannot load the truncation vpat model from ${_isNode ? VPAT_PATH : 'window.truncVpatModel'}`);
  const vpatModel = VPat.modelFromRaw(raw);
  let truncActive = false, truncPly = 0;    // per move: truncate this move's sims; the ply the sim stops at

  const need9    = USE9;                   // the 8-cell code is needed

  // Playout tail: the standard ppat playout.  A hard failure, not a fallback.
  const ppatModel = _isNode ? PPat.loadWeights(PPAT_PATH)
                            : PPat.loadWeights((typeof window !== 'undefined' && window.PPATWeights) || null);
  if (!ppatModel) throw new Error(`ref-dtr-1k: cannot load ppat weights from ${_isNode ? PPAT_PATH : 'window.PPATWeights'}`);
  ppatModel.ppatMinPhase = PPAT_MIN_PHASE;
  let ppatState = null;

  // ── Per-instance state (sized on first use; rebuilt if the board size changes) ──
  let area = 0;
  let w1 = null, w5 = null, w9 = null;          // actor weights: [mover][p](code)
  let sc = null, ex = null;                     // per-mover score / exp(score/T) over points
  const S = [0, 0];                             // per-mover Σ ex over allowed points
  let k5a = null, k9a = null;                   // per-point current window codes
  let mark = null, list = null;                 // dedup scratch for affected points
  let changed = null;                           // cells altered by a move (stone + captures)
  // Per-step records for the actor's update at sim end.
  let exs = null, Ss = null, movers = null, chosen = null, k5s = null, k9s = null;
  let fromActor = null;                         // step sampled from the actor (else the tail)
  let lastActorSteps = 0, lastReturn = 0;
  let maxSteps = 0;
  const base = [0.5, 0.5];                      // per-mover return baseline
  let lastMoveCount = -1;

  function setup(N) {
    area = N * N;
    w1 = new Float32Array(2 * area);
    w5 = new Float32Array(USE5 ? 2 * area * 81 : 0);
    w9 = new Float32Array(USE9 ? 2 * area * 6561 : 0);
    sc = [new Float64Array(area), new Float64Array(area)];
    ex = [new Float64Array(area), new Float64Array(area)];
    k5a = new Int32Array(area); k9a = new Int32Array(area);
    mark = new Uint8Array(area);
    list = new Int32Array(area);
    changed = new Int32Array(area + 1);
    maxSteps = 3 * area + 21;
    exs    = new Float64Array(maxSteps * area);
    Ss     = new Float64Array(maxSteps);
    movers = new Uint8Array(maxSteps);
    chosen = new Int32Array(maxSteps);
    fromActor = new Uint8Array(maxSteps);
    ppatState = PPat.createState(N);
    k5s = new Int32Array(USE5 ? maxSteps * area : 0);
    k9s = new Int32Array(USE9 ? maxSteps * area : 0);
  }

  function reset() {
    w1.fill(0); w5.fill(0); w9.fill(0);
    base[0] = base[1] = 0.5;
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
    const k5 = (USE5 || need9) ? code5(cells, nbr, p) : 0;
    const k9 = need9 ? code9(cells, dnbr, p, k5) : 0;
    return scoreFrom(m, p, k5, k9);
  }

  // Recompute everything anchored at p: both movers' actor score/ex (S kept
  // in step; skipped in the playout tail, where the scores are unused).  Kept
  // small so V8 inlines the code and score helpers into it (the hot function).
  function recompute(cells, nbr, dnbr, p) {
    const k5 = (USE5 || need9) ? code5(cells, nbr, p) : 0;
    const k9 = need9 ? code9(cells, dnbr, p, k5) : 0;
    k5a[p] = k5; k9a[p] = k9;
    const v  = cells[p];
    if (actorOn) {
      for (let m = 0; m < 2; m++) {
        const old = ex[m][p];
        let e = 0;
        if (v === EMPTY) { const s = scoreFrom(m, p, k5, k9); sc[m][p] = s; e = Math.exp(s / TEMP); }
        ex[m][p] = e;
        S[m] += e - old;
      }
    }
  }

  function recomputeAll(cells, nbr, dnbr) {
    ex[0].fill(0); ex[1].fill(0); S[0] = S[1] = 0;
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
  // neighbours).  That covers every actor feature that can move.
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

  // Leaf value of a truncated sim: P(BLACK wins) from the vpat model.
  function vpatValueB(g) {
    return VPat.evaluateFeatures(VPat.extractFeatures(g, vpatModel.preparedSpecs, false, undefined, true), vpatModel.weights);
  }

  // One simulation from `game`; returns the step count.  Records per step the
  // sampled-from distribution, the point codes, mover and chosen move, for
  // the update.
  function simulate(game, rng) {
    const g = game.clone();
    const nbr = g._nbr, dnbr = g._dnbr, cells = g.cells;
    act5 = USE5; act9 = USE9;              // ply 0: every enabled layer is on
    actorOn = true;
    recomputeAll(cells, nbr, dnbr);
    let t = 0, actorSteps = 0;
    const actorDepth = truncActive ? TRUNC_ACTOR_DEPTH : ACTOR_DEPTH;
    while (!g.gameOver && t < maxSteps - 1 && !(truncActive && t >= truncPly)) {
      const m = g.current === BLACK ? 0 : 1;
      const o = t * area;
      actorOn = t < actorDepth;
      if (actorOn) {
        const a5 = t < D5, a9 = t < D9;
        if (a5 !== act5 || a9 !== act9) { act5 = a5; act9 = a9; refreshScores(cells); }
      }
      let move;
      if (actorOn) {
        move = sample(g, m, rng);
        exs.set(ex[m], o);
        if (act5) k5s.set(k5a, o);
        if (act9) k9s.set(k9a, o);
        Ss[t] = S[m];
        actorSteps++;
      } else {
        move = truncActive ? g.randomLegalMove(rng) : PPat.ppatMove(g, ppatState, ppatModel, rng);
      }
      fromActor[t] = actorOn ? 1 : 0;
      movers[t] = m; chosen[t] = move;
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
    lastActorSteps = actorSteps;
    // The return: the outcome, or at the truncation point the vpat leaf value.
    const z = g.gameOver ? (g.calcWinner() === BLACK ? 1 : 0)
            : (truncActive && t >= truncPly) ? vpatValueB(g)
            : (g.calcWinner() === BLACK ? 1 : 0);
    lastReturn = z;
    update(t, z);
    return t;
  }

  // Actor policy-gradient step for the record at t with the step-scaled
  // advantage g (mover's view, learning rates already applied): for every
  // point a in the sampled-from distribution,
  // Δw(a) = g/T · ([a = chosen] − π(a)) on each layer's weight for (m, a).
  function actorUpdate(t, m, g) {
    const move = chosen[t];
    if (!fromActor[t] || move === PASS) return;   // tail moves and PASS are outside the softmax
    const o = t * area;
    const e = exs.subarray(o, o + area);
    const k5 = k5s.subarray(o, o + area), k9 = k9s.subarray(o, o + area);
    const u5 = t < D5, u9 = t < D9;        // layers that were active at this ply
    const invS = 1 / Ss[t];
    const k = g / TEMP;
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

  // Actor update over the sim's records, in step order.  z = the return,
  // P(BLACK wins); the mover's view is z for BLACK and 1 − z for WHITE, and
  // the advantage is that minus the mover's baseline as it stood before this
  // sim (the baseline then moves toward the return).
  function update(steps, z) {
    const adv = [z - base[0], (1 - z) - base[1]];
    base[0] += (1 - BASE_EMA) * (z - base[0]);
    base[1] += (1 - BASE_EMA) * ((1 - z) - base[1]);
    for (let t = 0; t < steps; t++) actorUpdate(t, movers[t], TERM_LR * adv[movers[t]]);
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
    truncPly = TRUNC_ACTOR_DEPTH + Math.ceil(TRUNC_DELTA * area);       // actor plies + buffer plies
    truncActive = TRUNC_DELTA > 0 && game.phase() + truncPly / area < TRUNC_MAX_PHASE;
    const tStart = Date.now();
    let sims = 0, longest = 0, totalSteps = 0;
    while (true) {
      if (PLAYOUTS_CAP > 0 ? sims >= PLAYOUTS_CAP : Date.now() - tStart >= budgetMs) break;
      const steps = simulate(game, rng);
      if (steps > longest) longest = steps;
      totalSteps += steps;
      sims++;
    }

    // Play the actor's argmax over legal non-eye points.
    const cells = game.cells, nbr = game._nbr, dnbr = game._dnbr;
    const m = game.current === BLACK ? 0 : 1;
    act5 = USE5; act9 = USE9;             // the root is ply 0
    actorOn = true;
    recomputeAll(cells, nbr, dnbr);       // root scores
    let best = PASS, bestS = -Infinity;
    for (let p = 0; p < area; p++) {
      if (cells[p] !== EMPTY || !game.isLegal(p) || game.isTrueEye(p)) continue;
      const s = sc[m][p] + rng.random() * 1e-9;
      if (s > bestS) { bestS = s; best = p; }
    }
    return { move: best, info: `sims=${sims} steps=${totalSteps} longest=${longest}${truncActive ? ` trunc=${truncPly}` : ''} base=${base[m].toFixed(3)} score=${bestS.toFixed(3)}` };
  }

  // Test hook: live views of the internals (state arrays are created by setup).
  function _internals() {
    return { setup, reset, recomputeAll, recomputeAround, refreshScores, score, sample, simulate, update,
             setActive: (a5, a9) => { act5 = a5; act9 = a9; },
             get lastActorSteps() { return lastActorSteps; },
             get lastReturn() { return lastReturn; },
             setTrunc: (active, plies) => { truncActive = active; truncPly = plies; },
             sc, ex, S, k5a, k9a, base, w1, w5, w9, area };
  }

  return { getMove, _internals };
}

// ── Feature keys ──────────────────────────────────────────────────────────────
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

if (_isNode) console.error(`ref-dtr-1k: dt-reinforce at ${PLAYOUTS_CAP} playouts/move (budget ignored), ` +
  `actor depth ${ACTOR_DEPTH}, trunc delta ${TRUNC_DELTA} / actor ${TRUNC_ACTOR_DEPTH} / max phase ${TRUNC_MAX_PHASE}, ` +
  `models ref/ref-dtr-1k-vpat.js, ref/ref-dtr-1k-ppat.js`);
let _default = null;
function _def() { return _default || (_default = create()); }
if (typeof module !== 'undefined') module.exports = { create, getMove: (g, b, o) => _def().getMove(g, b, o) };
else window.getMove = (g, b, o) => _def().getMove(g, b, o);

})();
