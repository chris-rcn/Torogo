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
// Actor: score(p) for an empty point p is one weight per (mover, p), and the
// policy is a softmax over the empty points.  (Pattern layers over the 4
// orthogonal neighbours and the 8 surrounding cells were tried and removed:
// paired runs at 2 s showed no gain from either.)  The actor plays only
// the first TD_ACTOR_DEPTH plies of a sim; the rest is the standard playout
// (uniform below PPAT_MIN_PHASE, the ppat policy above it).  Silver et al.
// switch to a default policy after ~6 plies, but here the actor's own moves
// are its training data: mdMae improved monotonically out to ~50 plies and
// plateaued 50-100, with unlimited slightly worse (2026-09-27).
//
// Everything is maintained incrementally: a changed cell alters only its own
// score, and its 8 neighbours only their legality or eye status.
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
// create(cfg) -> { getMove }.  cfg is a Util.makeCfg reader (P1_/P2_ prefixes in selfplay).
//
// Config:
//   TD_ACTOR_DEPTH    plies of a sim the actor plays; the rest is the standard playout (default 35)
//   PPAT_DATA         ppat weight file for the playout tail
//                     (default out/ppat-data-233162-best-ref-candidate.js)
//   PPAT_MIN_PHASE    tail moves are uniform below this board fullness      (default 0.6)
//   TD_ACTOR_TERM_LR  actor step size on the return minus the baseline.  Ladder at
//                     2 s: 0.04 -> 0.0189 ... 0.005 -> 0.0108, 0.002 -> 0.0103 (default 0.002)
//   TD_TEMP           softmax temperature for the simulations          (default 1)
//   TD_ACTOR_RETURN_EMA  the actor's baseline is a per-mover EMA of sim returns;
//                     this is its decay                                  (default 0.9)
//   PLAYOUTS          cap on simulations per move; 0 = time budget only (default 0)
//   TD_TRUNC_PHASE_DELTA  length of the random buffer after the actor plies, as a
//                     fraction of the area; 0 = no truncation            (default 0.2)
//   TD_TRUNC_ACTOR_DEPTH  actor plies in a truncated sim before the random buffer (default 5)
//   TD_TRUNC_MAX_PHASE  truncate only when the TRUNCATION POINT's phase would be
//                     below this                                       (default 0.52)
//   TRUNC_VPAT_DATA   the leaf model (default out/vpat-1j9ad1fk.js, the fielded one)
function create(cfg) {
  cfg = cfg || Util.makeCfg();

  const ACTOR_DEPTH = cfg.int('TD_ACTOR_DEPTH', 35);
  let actorOn = true;                        // the actor plays the current sim ply (else the tail)
  const TERM_LR  = cfg.float('TD_ACTOR_TERM_LR', 0.002);
  const TEMP     = cfg.float('TD_TEMP', 1);
  const BASE_EMA = cfg.float('TD_ACTOR_RETURN_EMA', 0.9);
  const PLAYOUTS_CAP = cfg.int('PLAYOUTS', 0);
  const TRUNC_DELTA       = cfg.float('TD_TRUNC_PHASE_DELTA', 0.2);
  const TRUNC_ACTOR_DEPTH = cfg.int('TD_TRUNC_ACTOR_DEPTH', 5);
  const TRUNC_MAX_PHASE   = cfg.float('TD_TRUNC_MAX_PHASE', 0.52);
  let vpatModel = null;
  if (TRUNC_DELTA > 0) {
    const vpatPath = _isNode
      ? cfg.str('TRUNC_VPAT_DATA', require('path').join(__dirname, '..', 'out', 'vpat-1j9ad1fk.js'))
      : null;
    const raw = _isNode ? require(require('path').resolve(vpatPath))
                        : (typeof window !== 'undefined' && window.truncVpatModel) || null;
    if (!raw) throw new Error(`dt-reinforce: cannot load the truncation vpat model from ${_isNode ? vpatPath : 'window.truncVpatModel'}`);
    vpatModel = VPat.modelFromRaw(raw);
  }
  let truncActive = false, truncPly = 0;    // per move: truncate this move's sims; the ply the sim stops at

  // Playout tail: the standard ppat playout.  A hard failure, not a fallback.
  const ppatPath = _isNode
    ? cfg.str('PPAT_DATA', require('path').join(__dirname, '..', 'out', 'ppat-data-233162-best-ref-candidate.js'))
    : null;
  const ppatModel = _isNode ? PPat.loadWeights(ppatPath)
                            : PPat.loadWeights((typeof window !== 'undefined' && window.PPATWeights) || null);
  if (!ppatModel) throw new Error(`dt-reinforce: cannot load ppat weights from ${_isNode ? ppatPath : 'window.PPATWeights'}`);
  ppatModel.ppatMinPhase = cfg.float('PPAT_MIN_PHASE', 0.6);
  let ppatState = null;

  // ── Per-instance state (sized on first use; rebuilt if the board size changes) ──
  let area = 0;
  let w1 = null;                                // actor weights: [mover][p]
  let sc = null, ex = null;                     // per-mover score / exp(score/T) over points
  const S = [0, 0];                             // per-mover Σ ex over allowed points
  let mark = null, list = null;                 // dedup scratch for affected points
  let changed = null;                           // cells altered by a move (stone + captures)
  // Per-step records for the actor's update at sim end.
  let exs = null, Ss = null, movers = null, chosen = null;
  let fromActor = null;                         // step sampled from the actor (else the tail)
  let lastActorSteps = 0, lastReturn = 0;
  let maxSteps = 0;
  const base = [0.5, 0.5];                      // per-mover return baseline
  let lastMoveCount = -1;

  function setup(N) {
    area = N * N;
    w1 = new Float32Array(2 * area);
    sc = [new Float64Array(area), new Float64Array(area)];
    ex = [new Float64Array(area), new Float64Array(area)];
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
  }

  function reset() {
    w1.fill(0);
    base[0] = base[1] = 0.5;
  }

  // Actor score for mover m at p.
  function score(m, p) { return w1[m * area + p]; }

  // Recompute p: both movers' actor score/ex (S kept in step; skipped in the
  // playout tail, where the scores are unused).
  function recompute(cells, p) {
    const v = cells[p];
    if (actorOn) {
      for (let m = 0; m < 2; m++) {
        const old = ex[m][p];
        let e = 0;
        if (v === EMPTY) { const s = score(m, p); sc[m][p] = s; e = Math.exp(s / TEMP); }
        ex[m][p] = e;
        S[m] += e - old;
      }
    }
  }

  function recomputeAll(cells) {
    ex[0].fill(0); ex[1].fill(0); S[0] = S[1] = 0;
    for (let p = 0; p < area; p++) recompute(cells, p);
  }

  // Lift a lazy exclusion at p (sample() zeroes illegal and true-eye points)
  // without recomputing its score: the score is current, only its legality or
  // eye status may have changed.
  function relift(cells, p) {
    if (!actorOn || cells[p] !== EMPTY) return;
    for (let m = 0; m < 2; m++) {
      if (ex[m][p] > 0) continue;
      const e = Math.exp(sc[m][p] / TEMP);
      ex[m][p] = e; S[m] += e;
    }
  }

  // After stone changes at the cells in `changed`: recompute each changed
  // cell (the only points whose score can move) and lift exclusions on its 8
  // neighbours, whose legality or true-eye status can change.  One deduped
  // pass in a fixed order (each changed cell, then its 4 orthogonal
  // neighbours; the diagonals after), so S accumulates deterministically.
  function recomputeAround(g, changed, nChanged) {
    const cells = g.cells, nbr = g._nbr, dnbr = g._dnbr;
    let n = 0;
    for (let i = 0; i < nChanged; i++) {
      const q = changed[i], b = q * 4;
      if (mark[q] !== 2) { if (!mark[q]) list[n++] = q; mark[q] = 2; }
      for (let d = 0; d < 4; d++) { const a = nbr[b + d]; if (!mark[a]) { mark[a] = 1; list[n++] = a; } }
    }
    for (let i = 0; i < n; i++) {
      const p = list[i];
      if (mark[p] === 2) recompute(cells, p); else relift(cells, p);
      mark[p] = 0;
    }
    for (let i = 0; i < nChanged; i++) {
      const b = changed[i] * 4;
      for (let d = 0; d < 4; d++) relift(cells, dnbr[b + d]);
    }
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
  // sampled-from distribution, mover and chosen move, for the update.
  function simulate(game, rng) {
    const g = game.clone();
    const cells = g.cells;
    actorOn = true;
    recomputeAll(cells);
    let t = 0, actorSteps = 0;
    const actorDepth = truncActive ? TRUNC_ACTOR_DEPTH : ACTOR_DEPTH;
    while (!g.gameOver && t < maxSteps - 1 && !(truncActive && t >= truncPly)) {
      const m = g.current === BLACK ? 0 : 1;
      const o = t * area;
      actorOn = t < actorDepth;
      let move;
      if (actorOn) {
        move = sample(g, m, rng);
        exs.set(ex[m], o);
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
      if (prevKo !== PASS && cells[prevKo] === EMPTY) recompute(cells, prevKo);
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
  // Δw(a) = g/T · ([a = chosen] − π(a)) on the weight for (m, a).
  function actorUpdate(t, m, g) {
    const move = chosen[t];
    if (!fromActor[t] || move === PASS) return;   // tail moves and PASS are outside the softmax
    const o = t * area;
    const e = exs.subarray(o, o + area);
    const invS = 1 / Ss[t];
    const k = g / TEMP;
    for (let p = 0; p < area; p++) {
      const v = e[p];
      if (v <= 0) continue;
      w1[m * area + p] += k * ((p === move ? 1 : 0) - v * invS);
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
    const cells = game.cells;
    const m = game.current === BLACK ? 0 : 1;
    actorOn = true;
    recomputeAll(cells);                  // root scores
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
    return { setup, reset, recomputeAll, recomputeAround, score, sample, simulate, update,
             get lastActorSteps() { return lastActorSteps; },
             get lastReturn() { return lastReturn; },
             setTrunc: (active, plies) => { truncActive = active; truncPly = plies; },
             sc, ex, S, base, w1, area };
  }

  return { getMove, _internals };
}

let _default = null;
function _def() { return _default || (_default = create(Util.makeCfg())); }
if (typeof module !== 'undefined') module.exports = { create, getMove: (g, b, o) => _def().getMove(g, b, o) };
else window.getMove = (g, b, o) => _def().getMove(g, b, o);

})();
