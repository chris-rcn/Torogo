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
// Actor: score(p) for an empty point p is one weight per (mover, p), plus
// optional stacked slices keyed by a per-ply integer, whose weights add to
// it and learn the same gradient:
//   phase   (mover, p, phase bucket of the sim's board)  TD_ACTOR_PHASE_BUCKETS
//   root    (mover, p) on ply 0 of a sim only            TD_ACTOR_ROOT_LAYER
// and the policy is a softmax over the empty points.  A slice's key can
// change mid-sim (a bucket edge crossed; ply 1 leaves the root), and then
// every score is refreshed once.  The root slice can be zeroed at the start
// of every getMove (TD_ACTOR_ROOT_RESET), so it holds only what this move's
// sims say about this root, and it has its own step size.  (Pattern layers
// over the 4 orthogonal neighbours and the 8 surrounding cells were tried and
// removed: paired runs at 2 s showed no gain from either.)  The actor plays the whole
// sim by default (TD_ACTOR_DEPTH 999); with a smaller depth the rest is a
// playout tail, uniform random below PPAT_MIN_PHASE and the ppat policy
// above it, and PPAT_MIN_PHASE defaults to 1 (ppat off: at 100 ms on 776
// positions it made no difference, 2026-09-28).  Silver et al. switch to a
// default policy after ~6 plies, but here the actor's own moves are its
// training data.
//
// Everything is maintained incrementally: a changed cell alters only its own
// score, and its 8 neighbours only their legality or eye status.
// Sampling rejects illegal and true-eye points lazily (their weight is zeroed
// until their neighbourhood changes).  Each mover's weights also live in a
// Fenwick (binary indexed) tree of prefix sums, so a draw walks log2(area)
// nodes instead of scanning the board, and a weight change updates as many.
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
//   TD_ACTOR_DEPTH    plies of a sim the actor plays; the rest is the playout tail (default 999)
//   PPAT_MIN_PHASE    tail moves are uniform below this board fullness, ppat above;
//                     1 = ppat off, no model loaded                       (default 1)
//   PPAT_DATA         ppat weight file for the tail, loaded only when PPAT_MIN_PHASE < 1
//                     (default out/ppat-data-233162-best-ref-candidate.js)
//   TD_ACTOR_TERM_LR  actor step size on the return minus the baseline.  Ladder at
//                     2 s: 0.04 -> 0.0189 ... 0.005 -> 0.0108, 0.002 -> 0.0103 (default 0.002)
//   TD_ACTOR_PHASE_BUCKETS  stacked slice keyed by the sim board's phase bucket, this
//                     many equal-width buckets of [0,1]; 0 = off           (default 0)
//   TD_ACTOR_ROOT_LAYER  1 = stacked slice read and trained on ply 0 of a sim only
//                     (and by the root argmax)                            (default 0)
//   TD_ACTOR_ROOT_RESET  1 = zero the root slice at the start of every getMove (default 1)
//   TD_ACTOR_ROOT_LR  step size of the root slice                (default TD_ACTOR_TERM_LR)
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

  const ACTOR_DEPTH = cfg.int('TD_ACTOR_DEPTH', 999);
  let actorOn = true;                        // the actor plays the current sim ply (else the tail)
  const TERM_LR  = cfg.float('TD_ACTOR_TERM_LR', 0.002);
  const PB       = cfg.int('TD_ACTOR_PHASE_BUCKETS', 0);       // phase slice: bucket count, 0 = off
  const USE_R    = cfg.int('TD_ACTOR_ROOT_LAYER', 0) !== 0;    // root slice
  const ROOT_RESET = cfg.int('TD_ACTOR_ROOT_RESET', 1) !== 0;
  const ROOT_LR  = cfg.float('TD_ACTOR_ROOT_LR', TERM_LR);
  let curB = 0, atRoot = false;              // the slices' current keys (sim state)
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

  // Playout tail: uniform random below PPAT_MIN_PHASE, the ppat policy above.
  // The ppat model is loaded only if it can ever play (min phase < 1), and
  // then its absence is a hard failure, not a fallback.
  const PPAT_MIN_PHASE = cfg.float('PPAT_MIN_PHASE', 1);
  let ppatModel = null;
  if (PPAT_MIN_PHASE < 1) {
    const ppatPath = _isNode
      ? cfg.str('PPAT_DATA', require('path').join(__dirname, '..', 'out', 'ppat-data-233162-best-ref-candidate.js'))
      : null;
    ppatModel = _isNode ? PPat.loadWeights(ppatPath)
                        : PPat.loadWeights((typeof window !== 'undefined' && window.PPATWeights) || null);
    if (!ppatModel) throw new Error(`dt-reinforce: cannot load ppat weights from ${_isNode ? ppatPath : 'window.PPATWeights'}`);
    ppatModel.ppatMinPhase = PPAT_MIN_PHASE;
  }
  let ppatState = null;

  // ── Per-instance state (sized on first use; rebuilt if the board size changes) ──
  let area = 0;
  let w1 = null;                                // actor weights: [mover][p]
  let wP = null, wR = null;                     // slice weights: phase [mover][p][bucket], root [mover][p]
  let bs = null;                                // per-step phase bucket, for the update
  let sc = null, ex = null;                     // per-mover score / exp(score/T) over points
  const S = [0, 0];                             // per-mover Σ ex over allowed points
  let tree = null, TOP = 0;                     // per-mover Fenwick tree over ex (1-based); TOP = largest power of 2 <= area
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
    wP = new Float32Array(PB > 0 ? 2 * area * PB : 0);
    wR = new Float32Array(USE_R ? 2 * area : 0);
    sc = [new Float64Array(area), new Float64Array(area)];
    ex = [new Float64Array(area), new Float64Array(area)];
    tree = [new Float64Array(area + 1), new Float64Array(area + 1)];
    TOP = 1; while (TOP * 2 <= area) TOP *= 2;
    mark = new Uint8Array(area);
    list = new Int32Array(area);
    changed = new Int32Array(area + 1);
    maxSteps = 3 * area + 21;
    exs    = new Float64Array(maxSteps * area);
    Ss     = new Float64Array(maxSteps);
    movers = new Uint8Array(maxSteps);
    chosen = new Int32Array(maxSteps);
    fromActor = new Uint8Array(maxSteps);
    bs = new Uint8Array(maxSteps);
    ppatState = ppatModel ? PPat.createState(N) : null;
  }

  function reset() {
    w1.fill(0); wP.fill(0); wR.fill(0);
    base[0] = base[1] = 0.5;
  }

  // Phase bucket of a board: equal-width buckets of its fullness.
  function phaseBucket(g) { const b = Math.floor((1 - g.emptyCount / area) * PB); return b >= PB ? PB - 1 : b; }

  // Actor score for mover m at p under the slices' current keys.
  function score(m, p) {
    const mp = m * area + p;
    let s = w1[mp];
    if (PB > 0) s += wP[mp * PB + curB];
    if (USE_R && atRoot) s += wR[mp];
    return s;
  }

  // ── Fenwick tree over ex[m] ──
  // Add d to point p's weight in mover m's tree.
  function treeAdd(m, p, d) {
    const t = tree[m];
    for (let i = p + 1; i <= area; i += i & -i) t[i] += d;
  }
  // Build mover m's tree from ex[m] in O(area).
  function treeBuild(m) {
    const t = tree[m], e = ex[m];
    for (let i = 1; i <= area; i++) t[i] = e[i - 1];
    for (let i = 1; i <= area; i++) { const j = i + (i & -i); if (j <= area) t[j] += t[i]; }
  }
  // Total weight in mover m's tree (its own arithmetic, so a draw with u
  // below it lands inside the tree).
  function treeTotal(m) {
    const t = tree[m];
    let s = 0;
    for (let i = area; i > 0; i -= i & -i) s += t[i];
    return s;
  }
  // The point whose prefix range holds u: the largest 0-based p with
  // prefix(p) <= u, i.e. prefix(p) <= u < prefix(p + 1).
  function treeDraw(m, u) {
    const t = tree[m];
    let pos = 0;
    for (let bit = TOP; bit > 0; bit >>= 1) {
      const nxt = pos + bit;
      if (nxt <= area && t[nxt] <= u) { pos = nxt; u -= t[nxt]; }
    }
    return pos;
  }

  // Set point p's weight for mover m, keeping S and the tree in step.
  function setWeight(m, p, e) {
    const old = ex[m][p];
    if (e === old) return;
    ex[m][p] = e;
    S[m] += e - old;
    treeAdd(m, p, e - old);
  }

  // Recompute p: both movers' actor score/ex (S and the trees kept in step;
  // skipped in the playout tail, where the scores are unused).
  function recompute(cells, p) {
    const v = cells[p];
    if (actorOn) {
      for (let m = 0; m < 2; m++) {
        let e = 0;
        if (v === EMPTY) { const s = score(m, p); sc[m][p] = s; e = Math.exp(s / TEMP); }
        setWeight(m, p, e);
      }
    }
  }

  function recomputeAll(cells) {
    S[0] = S[1] = 0;
    for (let m = 0; m < 2; m++) {
      const e = ex[m], scm = sc[m];
      for (let p = 0; p < area; p++) {
        let v = 0;
        if (cells[p] === EMPTY) { const s = score(m, p); scm[p] = s; v = Math.exp(s / TEMP); }
        e[p] = v; S[m] += v;
      }
      treeBuild(m);
    }
  }

  // Lift a lazy exclusion at p (sample() zeroes illegal and true-eye points)
  // without recomputing its score: the score is current, only its legality or
  // eye status may have changed.
  function relift(cells, p) {
    if (!actorOn || cells[p] !== EMPTY) return;
    for (let m = 0; m < 2; m++) {
      if (ex[m][p] > 0) continue;
      setWeight(m, p, Math.exp(sc[m][p] / TEMP));
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

  // Sample a point for mover m from ex[m] (softmax) through its tree,
  // rejecting illegal and true-eye points by zeroing them.  PASS when nothing
  // is left.  A draw can land on a zero-weight point only through rounding in
  // the tree's node sums; the next positive weight is taken.
  function sample(g, m, rng) {
    const e = ex[m];
    while (S[m] > 1e-300) {
      const total = treeTotal(m);
      if (!(total > 0)) break;
      let p = treeDraw(m, rng.random() * total);
      if (e[p] <= 0) {
        let q = p + 1; while (q < area && e[q] <= 0) q++;
        if (q >= area) { q = 0; while (q < p && e[q] <= 0) q++; if (q >= p) break; }
        p = q;
      }
      if (g.isLegal(p) && !g.isTrueEye(p)) return p;
      setWeight(m, p, 0);
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
    atRoot = USE_R; curB = PB > 0 ? phaseBucket(g) : 0;
    recomputeAll(cells);
    let t = 0, actorSteps = 0;
    const actorDepth = truncActive ? TRUNC_ACTOR_DEPTH : ACTOR_DEPTH;
    while (!g.gameOver && t < maxSteps - 1 && !(truncActive && t >= truncPly)) {
      const m = g.current === BLACK ? 0 : 1;
      const o = t * area;
      actorOn = t < actorDepth;
      // A slice key changed since the scores were last computed: refresh them.
      if (actorOn) {
        const b = PB > 0 ? phaseBucket(g) : 0, r = USE_R && t === 0;
        if (b !== curB || r !== atRoot) { curB = b; atRoot = r; recomputeAll(cells); }
      }
      let move;
      if (actorOn) {
        move = sample(g, m, rng);
        exs.set(ex[m], o);
        Ss[t] = S[m];
        bs[t] = curB;
        actorSteps++;
      } else {
        move = truncActive || !ppatModel ? g.randomLegalMove(rng) : PPat.ppatMove(g, ppatState, ppatModel, rng);
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

  // Actor policy-gradient step for the record at t with advantage adv
  // (mover's view): for every point a in the sampled-from distribution,
  // Δw(a) = lr·adv/T · ([a = chosen] − π(a)) on the weight for (m, a) and on
  // each active slice's weight, the root slice at its own step size.
  function actorUpdate(t, m, adv) {
    const move = chosen[t];
    if (!fromActor[t] || move === PASS) return;   // tail moves and PASS are outside the softmax
    const o = t * area;
    const e = exs.subarray(o, o + area);
    const invS = 1 / Ss[t];
    const k = TERM_LR * adv / TEMP, kR = ROOT_LR * adv / TEMP;
    const b = bs[t], root = USE_R && t === 0;
    for (let p = 0; p < area; p++) {
      const v = e[p];
      if (v <= 0) continue;
      const gr = (p === move ? 1 : 0) - v * invS;
      const mp = m * area + p;
      w1[mp] += k * gr;
      if (PB > 0) wP[mp * PB + b] += k * gr;
      if (root) wR[mp] += kR * gr;
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
    for (let t = 0; t < steps; t++) actorUpdate(t, movers[t], adv[movers[t]]);
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
    if (USE_R && ROOT_RESET) wR.fill(0);   // the root slice holds only this move's sims
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
    atRoot = USE_R; curB = PB > 0 ? phaseBucket(game) : 0;
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
             sc, ex, S, base, w1, wP, wR, area };
  }

  return { getMove, _internals };
}

let _default = null;
function _def() { return _default || (_default = create(Util.makeCfg())); }
if (typeof module !== 'undefined') module.exports = { create, getMove: (g, b, o) => _def().getMove(g, b, o) };
else window.getMove = (g, b, o) => _def().getMove(g, b, o);

})();
