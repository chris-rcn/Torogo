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
const ABSearch = Util.load('./ab-search.js', 'ABSearch');

// tdsearch — TD search (Silver, Sutton & Müller): a value function learned
// during think time, with simulations played by search over it.
//
// Each move, self-play simulations run from the current position.  Each of
// a sim's first TD_SIM_SEARCH_PLIES plies is the critic's one-ply argmax
// (mover's view) or, with probability TD_SIM_SEARCH_EPSILON, a uniform random
// legal move; the rest of the sim is the standard playout (uniform below
// PPAT_MIN_PHASE, the ppat policy above it).  The critic is updated by TD
// during the sim; the root move is alpha-beta over the learned critic, depth
// 1 being the same one-ply search.  The critic persists across the moves of one game
// (reset on a new game) so knowledge accumulates.
//
// Features are LOCATION-DEPENDENT by design — no symmetry, the board is in
// the orientation it is in — and indexed combinatorially, no hashing.  All are
// keyed by the side to move as well as the anchor.
//
// Critic: V(s) = σ(z), z = the sum over anchors p of the enabled layers
//   1: the cell at p                        (mover, p, colour)
//   4: the 2×2 window p, right, down, down-right  (mover, p, 4-cell code)
// An all-empty window is inactive (contributes nothing).  No priors: every
// table starts at zero and is learned during think time.  (A 3×3 layer was
// tried and removed: Silver et al. found no gain from 3×3 value patterns and
// neither did our ladder.)
//
// Everything is maintained incrementally: a changed cell q is in exactly four
// windows — its own cell and the 2×2 windows anchored at q, its left, its up
// and its up-left neighbour — and one pass over those anchors refreshes both
// movers' critic logits.
//
// Learning: two-ply same-parity TD(0), ONLINE with a two-step lag — at step t
// the features of step t−2 (the same mover's previous position) are pushed
// toward V(s_t); the last two steps are pushed toward the outcome when the
// sim ends.  A three-slot ring of index snapshots supplies step t−2's
// features; the live logit is corrected for the features the two positions
// share.
//
// Truncation (TD_TRUNC_MAX_PHASE > 0): a sim plays its TD_SIM_SEARCH_PLIES
// searched plies, then ceil(delta * area) UNIFORM random plies — the fielded
// trunc agent's rule, here a buffer so the search cannot steer into the leaf
// model's defects — and stops; the vpat model's value of the truncation point
// stands in for the outcome everywhere the outcome is used.  The model is the
// anchor; no grounding schedule.  Truncation is used only when the truncation
// point's phase (root phase + truncation ply / area, captures ignored) is below
// TD_TRUNC_MAX_PHASE — a property of the model's trusted band, so it stays put
// while the search depth and delta are swept.
//
// ── Factory ──
// create(cfg) -> { getMove }.  cfg is a Util.makeCfg reader (P1_/P2_ prefixes in selfplay).
//
// Config:
//   TD_SIM_SEARCH_PLIES  plies of a sim played by the critic's one-ply search; the
//                     rest is the standard playout                       (default 6)
//   TD_SIM_SEARCH_EPSILON  on a searched ply, probability of a UNIFORM random legal
//                     move instead (Silver et al.'s epsilon-greedy)      (default 0.1)
//   PPAT_DATA         ppat weight file for the playout tail
//                     (default out/ppat-data-233162-best-ref-candidate.js)
//   PPAT_MIN_PHASE    tail moves are uniform below this board fullness      (default 0.6)
//   TD_CRITIC_LAYERS  critic layers, comma list from 1,4                (default 1,4)
//   TD_CRITIC_LR      critic step size, per active feature             (default 0.3)
//   TD_CRITIC_TAIL    1 = the critic is maintained and learns through the playout
//                     tail; 0 = it stops at the searched plies, the tail only
//                     delivers the outcome                               (default 0)
//   PLAYOUTS          cap on simulations per move; 0 = time budget only (default 0)
//   TD_AB_DEPTH       root alpha-beta depth over the critic's value, every legal
//                     point a candidate at every node; 1 = the one-ply search (default 1)
//   TD_TRUNC_PHASE_DELTA  length of the random buffer after the searched plies, as a
//                     fraction of the area; 0 = the leaf right after the searched
//                     plies                                            (default 0.2)
//   TD_TRUNC_MAX_PHASE  truncate only when the TRUNCATION POINT's phase would be
//                     below this; 0 = truncation off, no leaf model loaded (default 0.52)
//   TRUNC_VPAT_DATA   the leaf model (default out/vpat-1j9ad1fk.js, the fielded one)
function create(cfg) {
  cfg = cfg || Util.makeCfg();

  const SEARCH_PLIES = cfg.int('TD_SIM_SEARCH_PLIES', 6);
  const SEARCH_EPSILON = cfg.float('TD_SIM_SEARCH_EPSILON', 0.1);
  const cStr     = cfg.str('TD_CRITIC_LAYERS', '1,4');
  const cList    = (cStr === '' || cStr === 'none') ? [] : cStr.split(',').map(s => parseInt(s, 10));
  const C1 = cList.includes(1), C4 = cList.includes(4);
  if (!(C1 || C4) || cList.some(l => l !== 1 && l !== 4)) throw new Error(`tdsearch: TD_CRITIC_LAYERS must be a non-empty subset of 1,4, got '${cStr}'`);
  const CLR      = cfg.float('TD_CRITIC_LR', 0.3);
  const CRITIC_TAIL = cfg.int('TD_CRITIC_TAIL', 0) !== 0;
  let criticOn = true;                     // the critic is maintained at the current sim ply
  const PLAYOUTS_CAP = cfg.int('PLAYOUTS', 0);
  const AB_DEPTH = cfg.int('TD_AB_DEPTH', 1);
  if (AB_DEPTH < 1) throw new Error(`tdsearch: TD_AB_DEPTH must be at least 1, got ${AB_DEPTH}`);
  const TRUNC_DELTA       = cfg.float('TD_TRUNC_PHASE_DELTA', 0.2);
  const TRUNC_MAX_PHASE   = cfg.float('TD_TRUNC_MAX_PHASE', 0.52);
  let vpatModel = null;
  if (TRUNC_MAX_PHASE > 0) {
    const vpatPath = _isNode
      ? cfg.str('TRUNC_VPAT_DATA', require('path').join(__dirname, '..', 'out', 'vpat-1j9ad1fk.js'))
      : null;
    const raw = _isNode ? require(require('path').resolve(vpatPath))
                        : (typeof window !== 'undefined' && window.truncVpatModel) || null;
    if (!raw) throw new Error(`tdsearch: cannot load the truncation vpat model from ${_isNode ? vpatPath : 'window.truncVpatModel'}`);
    vpatModel = VPat.modelFromRaw(raw);
  }
  let truncActive = false, truncPly = 0;    // per move: truncate this move's sims; the ply the sim stops at


  // Playout tail: the standard ppat playout.  A hard failure, not a fallback.
  const ppatPath = _isNode
    ? cfg.str('PPAT_DATA', require('path').join(__dirname, '..', 'out', 'ppat-data-233162-best-ref-candidate.js'))
    : null;
  const ppatModel = _isNode ? PPat.loadWeights(ppatPath)
                            : PPat.loadWeights((typeof window !== 'undefined' && window.PPATWeights) || null);
  if (!ppatModel) throw new Error(`tdsearch: cannot load ppat weights from ${_isNode ? ppatPath : 'window.PPATWeights'}`);
  ppatModel.ppatMinPhase = cfg.float('PPAT_MIN_PHASE', 0.6);
  let ppatState = null;

  // ── Per-instance state (sized on first use; rebuilt if the board size changes) ──
  let area = 0;
  let c1 = null, c4 = null;                     // critic weights: [mover][p](code)
  const Z = [0, 0];                             // per-mover critic logit of the current board
  let i1 = null, i4 = null;                     // per-anchor current critic index (−1 inactive)
  let nAct = 0;                                 // active critic features on the current board
  let mark = null, list = null;                 // dedup scratch for affected anchors
  let changed = null;                           // cells altered by a move (stone + captures)
  let movers = null, Vs = null;                 // per-step mover and V(s_t), for the critic's updates
  let lastCriticSteps = 0, lastReturn = 0, lastSearched = 0, lastRandom = 0;
  // Three-slot ring of critic index snapshots (step t in slot t % 3) for the
  // two-step-lagged critic update, with the active-feature count per slot.
  let i1r = null, i4r = null, nActR = null;
  let maxSteps = 0;

  function setup(N) {
    area = N * N;
    c1 = new Float32Array(C1 ? 2 * area * 3 : 0);
    c4 = new Float32Array(C4 ? 2 * area * 81 : 0);
    i1 = new Int32Array(area); i4 = new Int32Array(area);
    mark = new Uint8Array(area);
    list = new Int32Array(area);
    changed = new Int32Array(area + 1);
    maxSteps = 3 * area + 21;
    Vs     = new Float64Array(maxSteps);
    movers = new Uint8Array(maxSteps);
    ppatState = PPat.createState(N);
    i1r = new Int32Array(C1 ? 3 * area : 0); i4r = new Int32Array(C4 ? 3 * area : 0);
    nActR = new Int32Array(3);
  }

  function reset() {
    c1.fill(0); c4.fill(0);
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

  // Recompute every critic layer's index anchored at p (Z kept in step).
  // Kept small so V8 inlines the code helper into it (the hot function).
  function recompute(cells, nbr, dnbr, p) {
    if (C1) swapCritic(c1, i1, area * 3, p, critIdx1(cells[p], p));
    if (C4) swapCritic(c4, i4, area * 81, p, critIdx4(cells, nbr, dnbr, p));
  }

  function recomputeAll(cells, nbr, dnbr) {
    i1.fill(-1); i4.fill(-1); Z[0] = Z[1] = 0; nAct = 0;
    for (let p = 0; p < area; p++) recompute(cells, nbr, dnbr, p);
  }

  // The anchors whose windows contain cell q: q itself (its cell, and the 2×2
  // window anchored there) and the 2×2 windows anchored at its left (nbr 2),
  // up (nbr 0) and up-left (dnbr 0) neighbours.  Appended to list[] with
  // mark[] dedup; returns the new length.
  function anchorsOf(q, nbr, dnbr, n) {
    const b = q * 4;
    if (!mark[q]) { mark[q] = 1; list[n++] = q; }
    const l = nbr[b + 2];  if (!mark[l])  { mark[l]  = 1; list[n++] = l; }
    const u = nbr[b];      if (!mark[u])  { mark[u]  = 1; list[n++] = u; }
    const ul = dnbr[b];    if (!mark[ul]) { mark[ul] = 1; list[n++] = ul; }
    return n;
  }

  // After stone changes at the cells in `changed`, recompute every anchor
  // whose window contains one of them.
  function recomputeAround(g, changed, nChanged) {
    const cells = g.cells, nbr = g._nbr, dnbr = g._dnbr;
    let n = 0;
    for (let i = 0; i < nChanged; i++) n = anchorsOf(changed[i], nbr, dnbr, n);
    for (let i = 0; i < n; i++) { const p = list[i]; mark[p] = 0; recompute(cells, nbr, dnbr, p); }
  }

  function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

  // ── One-ply search over the critic ─────────────────────────────────────────
  // The critic's value of each legal non-eye candidate's RESULT, in the
  // mover's view; the argmax (dithered) is played.  A move changes only the
  // windows containing a changed cell — the placed stone and any captured
  // stones — so its logit change for the side then to move is a delta over
  // the union of those cells' anchors (anchorsOf: every 1×1 and 2×2 window
  // that can differ), read off the live indices with the changes set
  // temporarily in `cells`.  No clone.
  let srchVal = null, srchKind = null;
  function critVal4(m, i) { return i < 0 || !C4 ? 0 : c4[m * area * 81 + i]; }
  function critVal1(m, i) { return i < 0 || !C1 ? 0 : c1[m * area * 3 + i]; }
  function searchMove(g, m, rng) {
    if (!srchVal) { srchVal = new Float64Array(area); srchKind = new Int8Array(area); }
    const cells = g.cells, nbr = g._nbr, dnbr = g._dnbr, colour = m === 0 ? BLACK : -BLACK, o = 1 - m;
    // Pass 1: the candidates, every legal non-eye point.
    let nCand = 0;
    for (let p = 0; p < area; p++) {
      srchKind[p] = 0;
      if (cells[p] !== EMPTY || !g.isLegal(p) || g.isTrueEye(p)) continue;
      srchKind[p] = 1; nCand++;
    }
    if (nCand === 0) return PASS;
    // Pass 2: each candidate's value by local delta on the live indices.
    const zBase = Z[o];
    for (let p = 0; p < area; p++) {
      if (srchKind[p] !== 1) continue;
      const caps = g.isCapture(p) ? g.captureList(p) : null;
      // Apply the move to the cells; gather every anchor whose window can change.
      cells[p] = colour;
      let n = 0;
      for (let ci = -1; ci < (caps ? caps.length : 0); ci++) {
        const q = ci < 0 ? p : caps[ci];
        if (ci >= 0) cells[q] = EMPTY;
        n = anchorsOf(q, nbr, dnbr, n);
      }
      let dz = 0;
      for (let i = 0; i < n; i++) {
        const a = list[i]; mark[a] = 0;
        const v = cells[a];
        if (C1) dz += critVal1(o, v === EMPTY ? -1 : a * 3 + v + 1) - critVal1(o, i1[a]);
        if (C4) { const k4 = code4(cells, nbr, dnbr, a); dz += critVal4(o, k4 === 0 ? -1 : a * 81 + k4) - critVal4(o, i4[a]); }
      }
      // Restore the cells.
      cells[p] = EMPTY;
      if (caps) for (let ci = 0; ci < caps.length; ci++) cells[caps[ci]] = -colour;
      srchVal[p] = sigmoid(zBase + dz);
    }
    let best = PASS, bestV = -Infinity;
    for (let p = 0; p < area; p++) {
      if (!srchKind[p]) continue;
      const v = (m === 0 ? srchVal[p] : 1 - srchVal[p]) + rng.random() * 1e-9;
      if (v > bestV) { bestV = v; best = p; }
    }
    return best;
  }

  // Leaf value of a truncated sim: P(BLACK wins) from the vpat model.
  function vpatValueB(g) {
    return VPat.evaluateFeatures(VPat.extractFeatures(g, vpatModel.preparedSpecs, false, undefined, true), vpatModel.weights);
  }

  // One simulation from `game`; returns the step count.  Records per step the
  // mover, V(s_t) and the critic's active feature indices, for the TD updates.
  function simulate(game, rng) {
    const g = game.clone();
    const nbr = g._nbr, dnbr = g._dnbr, cells = g.cells;
    criticOn = true;
    recomputeAll(cells, nbr, dnbr);
    let t = 0, criticSteps = 0, searched = 0, randomPlies = 0;
    while (!g.gameOver && t < maxSteps - 1 && !(truncActive && t >= truncPly)) {
      const m = g.current === BLACK ? 0 : 1;
      const searchOn = t < SEARCH_PLIES;
      criticOn = searchOn || CRITIC_TAIL;
      if (criticOn) {
        criticSteps = t + 1;
        Vs[t] = sigmoid(Z[m]);
        const slot = t % 3, so = slot * area;
        if (C1) i1r.set(i1, so);
        if (C4) i4r.set(i4, so);
        nActR[slot] = nAct;
        // Step t−2 (same mover) is now two plies on: push it toward V(s_t).
        if (t >= 2) criticUpdate((t - 2) % 3, m, Vs[t - 2], Vs[t], true);
      }
      let move;
      if (searchOn) {
        if (SEARCH_EPSILON > 0 && rng.random() < SEARCH_EPSILON) { move = g.randomLegalMove(rng); randomPlies++; }
        else { move = searchMove(g, m, rng); searched++; }
      } else {
        move = truncActive ? g.randomLegalMove(rng) : PPat.ppatMove(g, ppatState, ppatModel, rng);
      }
      movers[t] = m;
      let nChanged = 0;
      if (move !== PASS) {
        const caps = g.captureList(move);
        changed[nChanged++] = move;
        for (let i = 0; i < caps.length; i++) changed[nChanged++] = caps[i];
      }
      g.play(move);
      if (nChanged > 0) recomputeAround(g, changed, nChanged);
      t++;
    }
    lastCriticSteps = criticSteps; lastSearched = searched; lastRandom = randomPlies;
    // The return: the outcome, or at the truncation point the vpat leaf value.
    const z = g.gameOver ? (g.calcWinner() === BLACK ? 1 : 0)
            : (truncActive && t >= truncPly) ? vpatValueB(g)
            : (g.calcWinner() === BLACK ? 1 : 0);
    lastReturn = z;
    // Terminal: the last two critic steps have no valued position two plies on.
    for (let u = Math.max(0, criticSteps - 2); u < criticSteps; u++) criticUpdate(u % 3, movers[u], Vs[u], z, false);
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
    if (fixZ) Z[m] += dz;
  }

  // Root alpha-beta over the critic, depth >= 2.  Every node is a clone: a
  // full recompute on its board gives the critic logit (the value); every
  // legal non-eye point is a candidate.  Depth 1 is searchMove's local-delta
  // pass, the same argmax without the clones.
  function abEvaluate(g) {
    recomputeAll(g.cells, g._nbr, g._dnbr);
    return sigmoid(Z[g.current === BLACK ? 0 : 1]);
  }
  function abCandidates(g) {
    const cells = g.cells, pts = [];
    for (let p = 0; p < area; p++) if (cells[p] === EMPTY && g.isLegal(p) && !g.isTrueEye(p)) pts.push(p);
    return pts;
  }

  // Game continuity: the board as it stood after this agent's last move.  The
  // next root continues the same game only if it is that board plus one
  // opponent move (a stone, with any of this agent's stones it captured, or a
  // pass) one ply later; anything else is a new game and the tables reset.
  let afterCells = null, afterCount = -1, afterMover = 0;
  function isContinuation(game) {
    const n = game.N * game.N;
    if (afterCount < 0 || afterCells.length !== n || game.moveCount !== afterCount + 1 || game.current !== afterMover) return false;
    const cells = game.cells, opp = -afterMover;
    let placed = 0, captured = 0;
    for (let p = 0; p < n; p++) {
      const a = afterCells[p], c = cells[p];
      if (a === c) continue;
      if (a === EMPTY && c === opp) placed++;
      else if (a === afterMover && c === EMPTY) captured++;
      else return false;
    }
    return placed === 1 || (placed === 0 && captured === 0);
  }
  function recordAfter(game, move) {
    const g = game.clone(), n = game.N * game.N;
    g.play(move);
    if (!afterCells || afterCells.length !== n) afterCells = new Int8Array(n);
    for (let p = 0; p < n; p++) afterCells[p] = g.cells[p];
    afterCount = g.moveCount; afterMover = game.current;
  }

  function getMove(game, budgetMs = 1000, options = {}) {
    if (game.consecutivePasses > 0 && game.calcWinner() === game.current) {
      recordAfter(game, PASS);
      return { move: PASS, info: 'end the game; ahead' };
    }
    if (area !== game.N * game.N) { setup(game.N); reset(); }
    if (!isContinuation(game)) reset();    // a new game

    const rng = options.rng || makeRng();
    truncPly = SEARCH_PLIES + Math.ceil(TRUNC_DELTA * area);             // searched plies + buffer plies
    truncActive = game.phase() + truncPly / area < TRUNC_MAX_PHASE;
    const tStart = Date.now();
    let sims = 0, longest = 0, totalSteps = 0;
    while (true) {
      if (PLAYOUTS_CAP > 0 ? sims >= PLAYOUTS_CAP : Date.now() - tStart >= budgetMs) break;
      const steps = simulate(game, rng);
      if (steps > longest) longest = steps;
      totalSteps += steps;
      sims++;
    }

    // Root selection: alpha-beta over the critic at AB_DEPTH.
    const m = game.current === BLACK ? 0 : 1;
    criticOn = true;
    let best;
    if (AB_DEPTH > 1) {
      best = ABSearch.search(game, AB_DEPTH, abEvaluate, 1e-9, { getCandidates: abCandidates, rng });
      recomputeAll(game.cells, game._nbr, game._dnbr);     // back to the root's logits
    } else {
      recomputeAll(game.cells, game._nbr, game._dnbr);
      best = searchMove(game, m, rng);
    }
    const val = sigmoid(Z[m]);
    recordAfter(game, best);
    return { move: best, info: `sims=${sims} steps=${totalSteps} longest=${longest}${truncActive ? ` trunc=${truncPly}` : ''} V=${val.toFixed(3)} ab=d${AB_DEPTH}` };
  }

  // Test hook: live views of the internals (state arrays are created by setup).
  function _internals() {
    return { setup, reset, recomputeAll, recomputeAround, simulate, sigmoid, searchMove, isContinuation, recordAfter,
             get lastCriticSteps() { return lastCriticSteps; },
             get lastReturn() { return lastReturn; },
             get lastSearched() { return lastSearched; },
             get lastRandom() { return lastRandom; },
             setTrunc: (active, plies) => { truncActive = active; truncPly = plies; },
             Z, i1, i4, c1, c4, area };
  }

  return { getMove, _internals };
}

// ── Feature key ───────────────────────────────────────────────────────────────
// Window code at p from a cell array (base 3, cell + 1 per digit).
// 2×2 window anchored at p: p, right, down, down-right (nbr 3, nbr 1, dnbr 3).
function code4(cells, nbr, dnbr, p) {
  const b = p * 4;
  return (cells[p] + 1) + 3 * (cells[nbr[b + 3]] + 1) + 9 * (cells[nbr[b + 1]] + 1) + 27 * (cells[dnbr[b + 3]] + 1);
}

let _default = null;
function _def() { return _default || (_default = create(Util.makeCfg())); }
if (typeof module !== 'undefined') module.exports = { create, getMove: (g, b, o) => _def().getMove(g, b, o) };
else window.getMove = (g, b, o) => _def().getMove(g, b, o);

})();
