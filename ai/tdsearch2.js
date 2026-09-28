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
const FeaturePol = Util.load('./featurepol-lib.js', 'FeaturePol');

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
//   actor:  REINFORCE at sim end (it needs the final result), in step order
//           over the plies it played (playout-tail moves carry no gradient),
//           with advantage A_t = ρ·δ_t + (1−ρ)·(R − V_t) from the mover's view,
//           where δ_t = V_{t+2} − V_t is the TD advantage, R the final result
//           and ρ = TD_ADV_RATIO.  Its per-step records are the sampled-from
//           distribution and the point codes.  Without a critic the baseline
//           is a per-mover EMA of sim returns.
//
// Priors: location-independent twins of two tables — actor: (mover, 8-cell
// code around the point); critic: (mover, 3×3 code), used whether or not the
// online 3×3 layer is on — added inside the score and the logit, so the
// online tables learn RESIDUALS on them and the updates are unchanged.  They
// are converted at create time: the actor prior from a featurepol model's
// stones8 space (TD_PRIOR_FPOL_DATA — every 8-cell pattern is placed on a
// small board and its stones8 component read for each mover), the critic
// prior from a vpat 3:1 model (TD_PRIOR_VPAT_DATA — every 3×3 pattern is
// placed on a small board and its size-3 component read; a vpat model has
// no side to move, so both movers share it).  Both default to fielded
// models; an empty path turns a prior off (zero).
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
//   TD_ACTOR_DEPTH    plies of a sim the actor plays; the rest is the standard playout (default 60)
//   PPAT_DATA         ppat weight file for the playout tail
//                     (default out/ppat-data-233162-best-ref-candidate.js)
//   PPAT_MIN_PHASE    tail moves are uniform below this board fullness      (default 0.6)
//   TD_ACTOR_LAYER5_DEPTH  plies of a sim for which actor layer 5 is on; 0 = off (default 0)
//   TD_ACTOR_LAYER9_DEPTH  plies of a sim for which actor layer 9 is on; 0 = off (default 0)
//   TD_ACTOR_LR       actor step size                                  (default 0.05)
//   TD_TEMP           softmax temperature for the simulations          (default 1)
//   TD_CRITIC_LAYERS  critic layers, comma list from 1,4,9; none = off (default 1,4)
//   TD_CRITIC_LR      critic step size, per active feature             (default 0.5)
//   TD_CRITIC_TAIL    1 = the critic is maintained and learns through the playout
//                     tail; 0 = it stops at the actor depth, the tail only
//                     delivers the outcome.  Silver et al. leave this open;
//                     measured 0.0214 vs 0.0228 at 200 ms, 27% faster  (default 0)
//   TD_ADV_RATIO      ρ: share of the TD advantage in the actor's advantage;
//                     the rest is the final result minus V             (default 0.8)
//   TD_RETURN_EMA     with the critic OFF, the actor's baseline is a per-mover EMA of
//                     sim returns; this is its decay                     (default 0.9)
//   TD_SIMS           cap on simulations per move; 0 = time budget only (default 0)
//   TD_SIM_SEARCH_PLIES  on each of a sim's first N actor plies, WHILE THE SIM IS ON
//                     THE PRINCIPAL VARIATION (every ply so far was the actor's
//                     argmax or a searched pick), the move may be the critic's
//                     one-ply argmax (mover's view) instead of the actor's sample;
//                     the first sampled deviation ends searching for that sim.
//                     Needs a critic.                                     (default 1)
//   TD_SIM_SEARCH_RATIO  probability, per such ply, of searching rather than
//                     sampling — the softmax sample is the exploration.  300 ms on
//                     1000 positions: 0 -> 0.0149, 0.1 -> 0.0127, 0.2 -> 0.0132,
//                     0.4 -> 0.0135: an occasional critic pick, not the default ply (default 0.1)
//   TD_SIM_SEARCH_WIDTH  a searched ply values only the actor's top K legal points
//                     (score incl. the prior); 0 = every legal point.  200 ms on 1000
//                     positions: 4 -> 0.0169, 8 -> 0.0164, 16 -> 0.0150, 0 -> 0.0155;
//                     300 ms: 15 -> 0.0141, 25-40 -> 0.0137, 60 -> 0.0143  (default 30)
//   TD_ROOT_SELECT    actor = play the actor's argmax; softmax = sample the actor's
//                     softmax at TD_TEMP (self-play diversity, e.g. for training);
//                     visits = play the point most often sampled as a sim's first
//                     ply, ties by actor score; ab = alpha-beta over the critic's
//                     value with the actor's top-TD_AB_WIDTH points as candidates
//                     at every node                                    (default actor)
//   TD_AB_DEPTH       ab: search depth in plies                        (default 2)
//   TD_AB_WIDTH       ab: candidates per node, the actor's top points   (default 5)
//   TD_TRUNC_PHASE_DELTA  length of the random buffer after the actor plies, as a
//                     fraction of the area; 0 = no truncation            (default 0.2)
//   TD_TRUNC_ACTOR_DEPTH  actor plies in a truncated sim before the random buffer (default 4)
//   TD_TRUNC_MAX_PHASE  truncate only when the TRUNCATION POINT's phase would be
//                     below this                                       (default 0.5)
//   TRUNC_VPAT_DATA   the leaf model (default out/vpat-1j9ad1fk.js, the fielded one)
//   TD_PRIOR_FPOL_DATA  featurepol model whose stones8 space becomes the actor prior;
//                     '' = none                  (default out/featurepol-yp81nwj8.js)
//   TD_PRIOR_VPAT_DATA  vpat 3:1 model whose 3x3 windows become the critic prior;
//                     '' = none                  (default out/vpat-fold-vjjnk618.js)
//   TD_PRIOR_VPAT_WEIGHT  the critic prior's weight in the logit           (default 1)
//   TD_PRIOR_FPOL_WEIGHT  the actor prior's weight in the score; featurepol logits are
//                     large (sd 1-4 on 13x13) and only a light prior helps: 0.1 beat 0
//                     and 0.2 at 300 ms on 1000 positions                 (default 0.1)
function create(cfg) {
  cfg = cfg || Util.makeCfg();

  const ACTOR_DEPTH = cfg.int('TD_ACTOR_DEPTH', 60);
  const D5 = cfg.int('TD_ACTOR_LAYER5_DEPTH', 0);
  const D9 = cfg.int('TD_ACTOR_LAYER9_DEPTH', 0);
  const USE5 = D5 > 0, USE9 = D9 > 0;        // layer ever used (tables, snapshots)
  let act5 = USE5, act9 = USE9;              // layer active at the current sim ply
  let actorOn = true;                        // the actor plays the current sim ply (else the tail)
  const LR       = cfg.float('TD_ACTOR_LR', 0.05);
  const TEMP     = cfg.float('TD_TEMP', 1);
  const cStr     = cfg.str('TD_CRITIC_LAYERS', '1,4');
  const cList    = (cStr === '' || cStr === 'none') ? [] : cStr.split(',').map(s => parseInt(s, 10));
  const C1 = cList.includes(1), C4 = cList.includes(4), C9 = cList.includes(9);
  const CRITIC   = C1 || C4 || C9;
  const CLR      = cfg.float('TD_CRITIC_LR', 0.5);
  const CRITIC_TAIL = cfg.int('TD_CRITIC_TAIL', 0) !== 0;
  let criticOn = CRITIC;                   // the critic is maintained at the current sim ply
  const ADV_RATIO = cfg.float('TD_ADV_RATIO', 0.8);
  const BASE_EMA = cfg.float('TD_RETURN_EMA', 0.9);
  const SIMS_CAP = cfg.int('TD_SIMS', 0);
  const SEARCH_PLIES = cfg.int('TD_SIM_SEARCH_PLIES', 1);
  const SEARCH_RATIO = cfg.float('TD_SIM_SEARCH_RATIO', 0.1);
  const SEARCH_WIDTH = cfg.int('TD_SIM_SEARCH_WIDTH', 30);
  const ROOT_SELECT = cfg.str('TD_ROOT_SELECT', 'actor');
  if (!['actor', 'softmax', 'visits', 'ab'].includes(ROOT_SELECT)) throw new Error(`tdsearch2: TD_ROOT_SELECT must be actor, softmax, visits or ab, got ${ROOT_SELECT}`);
  const AB_DEPTH = cfg.int('TD_AB_DEPTH', 2);
  const AB_WIDTH = cfg.int('TD_AB_WIDTH', 5);
  if (ROOT_SELECT === 'ab' && !CRITIC) throw new Error('tdsearch2: TD_ROOT_SELECT ab needs a critic (TD_CRITIC_LAYERS)');
  if (SEARCH_PLIES > 0 && !CRITIC) throw new Error('tdsearch2: TD_SIM_SEARCH_PLIES needs a critic (TD_CRITIC_LAYERS)');
  const TRUNC_DELTA       = cfg.float('TD_TRUNC_PHASE_DELTA', 0.2);
  const TRUNC_ACTOR_DEPTH = cfg.int('TD_TRUNC_ACTOR_DEPTH', 4);
  const TRUNC_MAX_PHASE   = cfg.float('TD_TRUNC_MAX_PHASE', 0.5);
  let vpatModel = null;
  if (TRUNC_DELTA > 0) {
    const vpatPath = _isNode
      ? cfg.str('TRUNC_VPAT_DATA', require('path').join(__dirname, '..', 'out', 'vpat-1j9ad1fk.js'))
      : null;
    const raw = _isNode ? require(require('path').resolve(vpatPath))
                        : (typeof window !== 'undefined' && window.truncVpatModel) || null;
    if (!raw) throw new Error(`tdsearch2: cannot load the truncation vpat model from ${_isNode ? vpatPath : 'window.truncVpatModel'}`);
    vpatModel = VPat.modelFromRaw(raw);
  }
  let truncActive = false, truncPly = 0;    // per move: truncate this move's sims; the ply the sim stops at

  // Priors: zero arrays without a file, so the lookups are unconditional.
  const fpolPriorPath = _isNode ? cfg.str('TD_PRIOR_FPOL_DATA', require('path').join(__dirname, '..', 'out', 'featurepol-yp81nwj8.js')) : '';
  const pa9 = fpolPriorPath ? actorPriorFromFeaturepol(fpolPriorPath, cfg.float('TD_PRIOR_FPOL_WEIGHT', 0.1)) : new Float32Array(2 * 6561);
  const vpatPriorPath = _isNode ? cfg.str('TD_PRIOR_VPAT_DATA', require('path').join(__dirname, '..', 'out', 'vpat-fold-vjjnk618.js')) : '';
  const pc9 = vpatPriorPath ? criticPriorFromVpat(vpatPriorPath, cfg.float('TD_PRIOR_VPAT_WEIGHT', 1)) : new Float32Array(2 * 19683);
  const HAS_PRIOR = !!vpatPriorPath;       // the critic prior needs the layer-9 index maintained
  // The layer-9 critic index is maintained if its table is on OR a prior is
  // present (the prior needs the window's code either way).
  const K9 = C9 || HAS_PRIOR;
  const need9    = USE9 || C9 || !!fpolPriorPath || !!vpatPriorPath;   // the 8-cell code is needed (layers, or a prior)

  // Playout tail: the standard ppat playout.  A hard failure, not a fallback.
  const ppatPath = _isNode
    ? cfg.str('PPAT_DATA', require('path').join(__dirname, '..', 'out', 'ppat-data-233162-best-ref-candidate.js'))
    : null;
  const ppatModel = _isNode ? PPat.loadWeights(ppatPath)
                            : PPat.loadWeights((typeof window !== 'undefined' && window.PPATWeights) || null);
  if (!ppatModel) throw new Error(`tdsearch2: cannot load ppat weights from ${_isNode ? ppatPath : 'window.PPATWeights'}`);
  ppatModel.ppatMinPhase = cfg.float('PPAT_MIN_PHASE', 0.6);
  let ppatState = null;

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
  let fromActor = null;                         // step sampled from the actor (else the tail)
  let rootVisits = null;                        // per point: sims whose first ply was that point
  let lastActorSteps = 0, lastCriticSteps = 0, lastReturn = 0, lastSearched = 0;
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
    fromActor = new Uint8Array(maxSteps);
    rootVisits = new Int32Array(area);
    ppatState = PPat.createState(N);
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

  // Actor score for mover m at p from precomputed codes (active layers only),
  // on top of the actor prior for the point's 8-cell code.
  function scoreFrom(m, p, k5, k9) {
    const mp = m * area + p;
    let s = w1[mp] + pa9[m * 6561 + k9];
    if (act5) s += w5[mp * 81 + k5];
    if (act9) s += w9[mp * 6561 + k9];
    return s;
  }
  function score(cells, nbr, dnbr, m, p) {
    const k5 = (USE5 || need9) ? code5(cells, nbr, p) : 0;
    const k9 = need9 ? code9(cells, dnbr, p, k5) : 0;
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

  // Layer 9 carries the critic prior: a window's value is its table weight
  // (if the layer is on) plus the prior for its code (index % 19683), for
  // either mover.  Only table features count toward nAct (the lr/n norm).
  function swapCritic9(p, ni) {
    const oi = i9[p];
    if (oi === ni) return;
    const size = area * 19683;
    for (let m = 0; m < 2; m++) {
      const o = m * size, po = m * 19683;
      Z[m] += (ni >= 0 ? (C9 ? c9[o + ni] : 0) + pc9[po + ni % 19683] : 0) - (oi >= 0 ? (C9 ? c9[o + oi] : 0) + pc9[po + oi % 19683] : 0);
    }
    if (C9) { if (oi < 0) nAct++; else if (ni < 0) nAct--; }
    i9[p] = ni;
  }
  function recomputeCritic(cells, nbr, dnbr, p, k9, v) {
    if (C1) swapCritic(c1, i1, area * 3, p, critIdx1(v, p));
    if (C4) swapCritic(c4, i4, area * 81, p, critIdx4(cells, nbr, dnbr, p));
    if (K9) swapCritic9(p, critIdx9(v, p, k9));
  }

  // Recompute everything anchored at p: both movers' actor score/ex (S kept
  // in step; skipped in the playout tail, where the scores are unused) and
  // every critic layer's index (Z kept in step).  Kept small so V8 inlines
  // the code and score helpers into it (the hot function).
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
    if (criticOn) recomputeCritic(cells, nbr, dnbr, p, k9, v);
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

  // ── One-ply search over the critic (sim plies) ──────────────────────────────
  // The critic's value of each legal non-eye candidate's RESULT, in the
  // mover's view; the argmax (dithered) is played.  A move changes only the
  // windows containing a changed cell — the placed stone and any captured
  // stones — so its logit change for the side then to move is a delta over
  // the union of those cells' 3×3 neighbourhoods (every 1×1, 2×2 and 3×3
  // window that can differ), read off the live indices with the changes set
  // temporarily in `cells`.  No clone.  Illegal and true-eye points are
  // excluded from the actor's distribution as sample() does, so the ply's
  // recorded distribution is exact.
  // With SEARCH_WIDTH > 0 only the actor's top K legal points (by score, prior
  // included) are valued: the actor orders, the critic verifies.
  let srchVal = null, srchKind = null, srchTop = null;
  function critVal9(m, i) { return i < 0 ? 0 : (C9 ? c9[m * area * 19683 + i] : 0) + pc9[m * 19683 + i % 19683]; }
  function critVal4(m, i) { return i < 0 || !C4 ? 0 : c4[m * area * 81 + i]; }
  function critVal1(m, i) { return i < 0 || !C1 ? 0 : c1[m * area * 3 + i]; }
  function searchMove(g, m, rng) {
    if (!srchVal) { srchVal = new Float64Array(area); srchKind = new Int8Array(area); srchTop = new Int32Array(area); }
    const cells = g.cells, nbr = g._nbr, dnbr = g._dnbr, colour = m === 0 ? BLACK : -BLACK, o = 1 - m;
    // Pass 1: legality (excluding as sample() does).  Candidates: every legal
    // point, or with a width the top K by actor score (a small insertion list).
    let nCand = 0;
    const K = SEARCH_WIDTH, scm = sc[m];
    for (let p = 0; p < area; p++) {
      srchKind[p] = 0;
      if (cells[p] !== EMPTY) continue;
      if (!g.isLegal(p) || g.isTrueEye(p)) { S[m] -= ex[m][p]; ex[m][p] = 0; continue; }
      if (K <= 0) { srchKind[p] = 1; nCand++; continue; }
      // Keep the K best scores seen so far, descending, in srchTop[0..nCand).
      let j = nCand < K ? nCand : K - 1;
      if (nCand >= K && scm[srchTop[j]] >= scm[p]) continue;
      while (j > 0 && scm[srchTop[j - 1]] < scm[p]) { srchTop[j] = srchTop[j - 1]; j--; }
      srchTop[j] = p;
      if (nCand < K) nCand++;
    }
    if (K > 0) for (let j = 0; j < nCand; j++) srchKind[srchTop[j]] = 1;
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
        const b = q * 4;
        if (!mark[q]) { mark[q] = 1; list[n++] = q; }
        for (let d = 0; d < 4; d++) {
          const a = nbr[b + d];  if (!mark[a]) { mark[a] = 1; list[n++] = a; }
          const c = dnbr[b + d]; if (!mark[c]) { mark[c] = 1; list[n++] = c; }
        }
      }
      let dz = 0;
      for (let i = 0; i < n; i++) {
        const a = list[i]; mark[a] = 0;
        const v = cells[a];
        if (C1) dz += critVal1(o, v === EMPTY ? -1 : a * 3 + v + 1) - critVal1(o, i1[a]);
        if (C4) { const k4 = code4(cells, nbr, dnbr, a); dz += critVal4(o, k4 === 0 ? -1 : a * 81 + k4) - critVal4(o, i4[a]); }
        if (K9) { const k = code9(cells, dnbr, a, code5(cells, nbr, a)) + 6561 * (v + 1); dz += critVal9(o, k === 0 ? -1 : a * 19683 + k) - critVal9(o, i9[a]); }
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
  // board, the sampled-from distribution, mover, chosen move, V(s_t) and the
  // critic's active feature indices, for the backward pass.
  function simulate(game, rng) {
    const g = game.clone();
    const nbr = g._nbr, dnbr = g._dnbr, cells = g.cells;
    act5 = USE5; act9 = USE9;              // ply 0: every enabled layer is on
    actorOn = true; criticOn = CRITIC;
    recomputeAll(cells, nbr, dnbr);
    let t = 0, actorSteps = 0, criticSteps = 0, searched = 0;
    let onPV = true;                       // every ply so far was the actor's argmax or a searched pick
    const actorDepth = truncActive ? TRUNC_ACTOR_DEPTH : ACTOR_DEPTH;
    while (!g.gameOver && t < maxSteps - 1 && !(truncActive && t >= truncPly)) {
      const m = g.current === BLACK ? 0 : 1;
      const o = t * area;
      actorOn = t < actorDepth;
      criticOn = CRITIC && (actorOn || CRITIC_TAIL);
      if (actorOn) {
        const a5 = t < D5, a9 = t < D9;
        if (a5 !== act5 || a9 !== act9) { act5 = a5; act9 = a9; refreshScores(cells); }
      }
      if (criticOn) {
        criticSteps = t + 1;
        Vs[t] = sigmoid(Z[m]);
        const slot = t % 3, so = slot * area;
        if (C1) i1r.set(i1, so);
        if (C4) i4r.set(i4, so);
        if (C9) i9r.set(i9, so);
        nActR[slot] = nAct;
        // Step t−2 (same mover) is now two plies on: push it toward V(s_t).
        if (t >= 2) criticUpdate((t - 2) % 3, m, Vs[t - 2], Vs[t], true);
      }
      let move;
      if (actorOn) {
        const maysearch = onPV && t < SEARCH_PLIES;
        if (maysearch && (SEARCH_RATIO >= 1 || rng.random() < SEARCH_RATIO)) { move = searchMove(g, m, rng); searched++; }
        else {
          move = sample(g, m, rng);
          if (maysearch) {                 // stays on the PV only if the sample was the actor's argmax
            const scm = sc[m], e = ex[m];
            let bestS = -Infinity;
            for (let p = 0; p < area; p++) if (e[p] > 0 && scm[p] > bestS) bestS = scm[p];
            onPV = move !== PASS && scm[move] === bestS;
          }
        }
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
      if (t === 0 && move !== PASS) rootVisits[move]++;
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
    lastActorSteps = actorSteps; lastCriticSteps = criticSteps; lastSearched = searched;
    // The return: the outcome, or at the truncation point the vpat leaf value.
    const z = g.gameOver ? (g.calcWinner() === BLACK ? 1 : 0)
            : (truncActive && t >= truncPly) ? vpatValueB(g)
            : (g.calcWinner() === BLACK ? 1 : 0);
    lastReturn = z;
    // Terminal: the last two critic steps have no valued position two plies on.
    if (CRITIC) for (let u = Math.max(0, criticSteps - 2); u < criticSteps; u++) criticUpdate(u % 3, movers[u], Vs[u], z, false);
    update(t, z, criticSteps);
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
    if (!fromActor[t] || move === PASS) return;   // tail moves and PASS are outside the softmax
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
  function update(steps, z, criticSteps) {
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
      const next = t + 2 < criticSteps ? Vs[t + 2] : z;   // the same mover's next valued position, or the outcome
      const sign = m === 0 ? 1 : -1;      // mover's view of a BLACK-view difference
      const delta = sign * (next - v);    // TD advantage
      const mc    = sign * (z - v);       // final-result advantage
      actorUpdate(t, m, ADV_RATIO * delta + (1 - ADV_RATIO) * mc);
    }
  }

  // Root alpha-beta over the critic.  Every node is a clone: a full recompute
  // on its board gives both the actor scores (candidate ordering) and the
  // critic logit (the value).  ~depth^width nodes per move, negligible.
  function abEvaluate(g) {
    recomputeAll(g.cells, g._nbr, g._dnbr);
    return sigmoid(Z[g.current === BLACK ? 0 : 1]);
  }
  function abCandidates(g) {
    recomputeAll(g.cells, g._nbr, g._dnbr);
    const m = g.current === BLACK ? 0 : 1, cells = g.cells, scm = sc[m];
    const pts = [];
    for (let p = 0; p < area; p++) if (cells[p] === EMPTY && g.isLegal(p) && !g.isTrueEye(p)) pts.push(p);
    pts.sort((a, b) => scm[b] - scm[a]);
    return pts.length > AB_WIDTH ? pts.slice(0, AB_WIDTH) : pts;
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
    rootVisits.fill(0);
    let sims = 0, longest = 0, totalSteps = 0;
    while (true) {
      if (SIMS_CAP > 0 ? sims >= SIMS_CAP : Date.now() - tStart >= budgetMs) break;
      const steps = simulate(game, rng);
      if (steps > longest) longest = steps;
      totalSteps += steps;
      sims++;
    }

    // Root selection over legal non-eye points: the actor's argmax, or the
    // most-visited first ply with the actor score breaking ties (so with no
    // sims at all it is the actor's argmax).
    const cells = game.cells, nbr = game._nbr, dnbr = game._dnbr;
    const m = game.current === BLACK ? 0 : 1;
    act5 = USE5; act9 = USE9;             // the root is ply 0
    actorOn = true; criticOn = CRITIC;
    let best = PASS, bestS = -Infinity, bestV = -1;
    if (ROOT_SELECT === 'ab') {
      best = ABSearch.search(game, AB_DEPTH, abEvaluate, 1e-9, { getCandidates: abCandidates, rng });
      recomputeAll(cells, nbr, dnbr);     // back to the root's scores and logits
      if (best !== PASS) bestS = sc[m][best];
    } else if (ROOT_SELECT === 'softmax') {
      recomputeAll(cells, nbr, dnbr);     // root scores and logits
      // Sample the actor's softmax over legal non-eye points.
      let tot = 0;
      for (let p = 0; p < area; p++) if (cells[p] === EMPTY && game.isLegal(p) && !game.isTrueEye(p)) tot += ex[m][p];
      let u = rng.random() * tot;
      for (let p = 0; p < area; p++) {
        if (cells[p] !== EMPTY || !game.isLegal(p) || game.isTrueEye(p)) continue;
        best = p; bestS = sc[m][p];
        u -= ex[m][p]; if (u < 0) break;
      }
    } else {
      recomputeAll(cells, nbr, dnbr);     // root scores and logits
      const byVisits = ROOT_SELECT === 'visits';
      for (let p = 0; p < area; p++) {
        if (cells[p] !== EMPTY || !game.isLegal(p) || game.isTrueEye(p)) continue;
        const s = sc[m][p] + rng.random() * 1e-9;
        const v = byVisits ? rootVisits[p] : 0;
        if (v > bestV || (v === bestV && s > bestS)) { bestV = v; bestS = s; best = p; }
      }
    }
    const val = CRITIC ? sigmoid(Z[m]) : base[m];
    const how = ROOT_SELECT === 'ab' ? ` ab=d${AB_DEPTH}w${AB_WIDTH}` : ROOT_SELECT === 'visits' ? ` visits=${bestV}` : ROOT_SELECT === 'softmax' ? ' softmax' : '';
    return { move: best, info: `sims=${sims} steps=${totalSteps} longest=${longest}${truncActive ? ` trunc=${truncPly}` : ''} V=${val.toFixed(3)} score=${bestS.toFixed(3)}${how}` };
  }

  // Test hook: live views of the internals (state arrays are created by setup).
  function _internals() {
    return { setup, reset, recomputeAll, recomputeAround, refreshScores, score, sample, simulate, update, sigmoid,
             setActive: (a5, a9) => { act5 = a5; act9 = a9; },
             searchMove,
             get lastActorSteps() { return lastActorSteps; },
             get lastCriticSteps() { return lastCriticSteps; },
             get lastReturn() { return lastReturn; },
             get lastSearched() { return lastSearched; },
             rootVisits,
             setTrunc: (active, plies) => { truncActive = active; truncPly = plies; },
             sc, ex, S, Z, i1, i4, i9, k5a, k9a, base, w1, w5, w9, c1, c4, c9, area, CRITIC };
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

// ── Actor prior from a featurepol model ───────────────────────────────────────
// stones8 is our 8-cell code by another name.  For every code and mover, place
// the ring around the centre of an empty 5×5 board (the ring's outer
// neighbours are empty, so every stone has liberties and legality is exact)
// and read the model's stones8 component at the centre.  A centre the mover
// cannot play (suicide, or a true eye) has no component and stays 0 — it is
// never sampled anyway.
function actorPriorFromFeaturepol(path, weight) {
  const { Game2 } = Util.load('./game2.js', 'Game2');
  const { weights } = FeaturePol.loadModel({ name: 'tdsearch2-actor-prior', path });
  const spaceIdx = weights.spec.spaces.findIndex(sp => sp.str === 'stones8');
  if (spaceIdx < 0) throw new Error(`tdsearch2: TD_PRIOR_FPOL_DATA model ${path} has no stones8 space (spec '${weights.spec.str}')`);
  const N = 5, centre = 2 * N + 2, nSpaces = weights.spec.spaces.length;
  const state = FeaturePol.createState(N, weights.spec, { components: true });
  const ring = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]];   // code digit order
  const pa9 = new Float32Array(2 * 6561);
  for (let k = 0; k < 6561; k++) {
    for (let m = 0; m < 2; m++) {
      const g = new Game2(N, false);
      let r = k;
      for (let i = 0; i < 8; i++) { const d = r % 3; r = (r - d) / 3; if (d !== 1) g._place((2 + ring[i][0]) * N + 2 + ring[i][1], d - 1); }
      g.current = m === 0 ? BLACK : -BLACK;
      const cv = FeaturePol.componentValues(g, state, weights);
      for (let i = 0; i < cv.count; i++) if (cv.moves[i] === centre) { pa9[m * 6561 + k] = weight * cv.values[i * nSpaces + spaceIdx]; break; }
    }
  }
  return pa9;
}

// ── Critic prior from a vpat model ────────────────────────────────────────────
// Our 3×3 critic code is a vpat size-3 window keyed without symmetry.  For
// every code, place the 3×3 pattern at the centre of an empty 5×5 board and
// read the model's size-3 component whose window is centred there (vpat
// anchors a 3×3 at its top-left).  Colour only (maxLibs 1), so stones need no
// liberties.  A vpat model has no side to move: both movers get the value.
function criticPriorFromVpat(path, weight) {
  const { Game2 } = Util.load('./game2.js', 'Game2');
  const model = VPat.loadWeights(path);
  const s3 = model.specs.filter(sp => sp.size === 3 && !sp.turn);
  if (s3.length !== 1 || s3[0].maxLibs !== 1) throw new Error(`tdsearch2: TD_PRIOR_VPAT_DATA model ${path} must have one size-3 maxLibs-1 spec (has ${VPat.specString(model.specs)})`);
  const N = 5, anchor = 1 * N + 1;      // the window whose centre is (2,2)
  const cells9 = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1], [0, 0]];   // code9 digits, then the centre
  const pc9 = new Float32Array(2 * 19683);
  for (let k = 1; k < 19683; k++) {       // 0 = all empty: no feature, no weight
    const g = new Game2(N, false);
    let r = k;
    for (let i = 0; i < 9; i++) { const d = r % 3; r = (r - d) / 3; if (d !== 1) g._place((2 + cells9[i][0]) * N + 2 + cells9[i][1], d - 1); }
    const cv = VPat.componentValues(g, model);
    let w = 0;
    for (let i = 0; i < cv.count; i++) if (cv.sizes[i] === 3 && cv.anchors[i] === anchor) { w = cv.values[i]; break; }
    pc9[k] = pc9[19683 + k] = weight * w;
  }
  return pc9;
}

let _default = null;
function _def() { return _default || (_default = create(Util.makeCfg())); }
if (typeof module !== 'undefined') module.exports = { create, getMove: (g, b, o) => _def().getMove(g, b, o) };
else window.getMove = (g, b, o) => _def().getMove(g, b, o);

})();
