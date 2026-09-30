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
const FeaturePol = Util.load('./featurepol-lib.js', 'FeaturePol');
const { game3FromGame2 } = Util.load('./game3.js', 'Game3');

// dt-reinforce — decision-time REINFORCE: a policy learned during think time.
//
// Each move, self-play simulations run from the current position.  The actor
// policy that plays them is updated by REINFORCE from the simulations'
// returns, and the actor's argmax is played.  The policy persists across the
// moves of one game (reset on a new game) so knowledge accumulates.  No
// critic, no priors, no search: the actor-only case of dt-actor-critic.
//
// Features are LOCATION-DEPENDENT by design — no symmetry, the board is in
// the orientation it is in — and indexed combinatorially, no hashing.  All but
// the colourblind table are keyed by the side to move as well as the point.
//
// Actor: score(p) for an empty point p is the sum of up to two base tables,
//   colour      (mover, p)                          ACTOR_COLOR_LAYER (default on)
//   colourblind (p), shared by both sides: "my best move is my opponent's
//               best move"; it learns from both sides' plies, twice the
//               samples                             ACTOR_COLORBLIND_LAYER
// plus
// optional stacked slices keyed by a per-ply integer, whose weights add to
// it and learn the same gradient:
//   phase   (mover, p, phase bucket of the sim's board)  ACTOR_PHASE_BUCKETS
//   root    (mover, p) on ply 0 of a sim only            ACTOR_ROOT_LAYER
//   local   (mover, p, p one of the 8 points around the last SIM move)  ACTOR_LOCAL_LAYER
//           — sim moves only: ply 0 and the root argmax see no last move, so
//           the root decision never depends on the game's last move, and at
//           ply 0 the key would be the same in every sim anyway
// plus the chain layer (ACTOR_CHAIN_LAYER): one weight per chain STATE,
// keyed by (the chain's lowest-index stone, its stone count, its liberty
// count) for chains of either colour with at most CHAIN_MAX_LIBS (3)
// liberties.  A point's score adds the weight of each such chain it is
// adjacent to, once per chain.  Attacking and extending a chain are one
// weight (colourblind: the point that attacks a chain for one side extends it
// for the other).  Any stone played on or next to a chain changes its stone
// or liberty count, so its key: each tactical state learns on its own.  The
// table is indexed exactly (3 * area * area weights), no hashing.  After each
// move, the liberties of the chains it touched (and of those next to its
// captures) are recomputed when the chain qualified before or after.
// and the policy is a softmax over the empty points.  A slice's key can
// change mid-sim (a bucket edge crossed; ply 1 leaves the root), and then
// every score is refreshed once; the local key moves with each sim move and
// only the old and new last move's neighbours are recomputed.  The root slice can be zeroed at the start
// of every getMove (ACTOR_ROOT_RESET), so it holds only what this move's
// sims say about this root.  (Pattern layers
// over the 4 orthogonal neighbours and the 8 surrounding cells were tried and
// removed: paired runs at 2 s showed no gain from either.)  The actor plays
// the first ACTOR_DEPTH plies of a sim (default 30); the rest is a playout
// tail, uniform random below PPAT_MIN_PHASE and the ppat policy above it,
// and PPAT_MIN_PHASE defaults to 0.6 (in a match from phase 0.5 at 500 ms on
// 12x12 the ppat tail won 62 of 96 against the uniform one, 2026-09-29; the
// MD file had scored it the other way).  Unlimited depth lost 0.0047 mae to depth
// 30 on the untruncated roots at 2 s (2026-09-29).  Silver et al. switch to
// a default policy after ~6 plies, but here the actor's own moves are its
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
// LR·(R − b_m) in the mover's view, where R is the sim's return
// and b_m a per-mover EMA of returns as baseline.  Its per-step records are the
// sampled-from distribution and the point codes.
//
// Truncation (TRUNC_MAX_PHASE > 0): a sim plays TRUNC_ACTOR_DEPTH actor
// plies, then a buffer of UNIFORM random plies — the fielded trunc agent's
// rule, here so the actor cannot steer into the truncation model's defects — and
// stops; the vpat model's value of the truncation point stands in for the
// outcome everywhere the outcome is used.  The model is the anchor; no
// grounding schedule.  The truncation point's phase (root phase + truncation
// ply / area, captures ignored) must stay below TRUNC_MAX_PHASE, the
// model's trusted band.  The buffer is ceil(TRUNC_PHASE_DELTA * area)
// plies when the root allows it, and shortened for roots nearer the band's
// edge as far as ceil(TRUNC_PHASE_DELTA_MIN * area) plies; a root that
// cannot fit even the minimum buffer under the edge runs full sims.  With
// min = max a root either takes the full buffer or does not truncate.
// Defaults 0.12 / 0.09 from the 2 s ladders of 2026-09-29: the buffer's
// own effect flattened by 0.12 on roots below 0.28, and minimums down to
// 0.06 were level with the max on the roots the minimum acts on.
//
// ── Factory ──
// create(cfg) -> { getMove, valueB }.  cfg is a Util.makeCfg reader (P1_/P2_
// prefixes in selfplay).  valueB(game, options) -> P(BLACK wins) is the mean
// return of the same sims getMove would run from the position (the actor
// learning as it goes; truncation, playouts and budget per the settings),
// the value oracle gen-agent-evals labels with.
//
// Config:
//   ACTOR_DEPTH       plies of a sim the actor plays; the rest is the playout tail.
//                     2 s ladder on roots 0.42-0.8: 20 -> 0.0084, 30 -> 0.0080,
//                     40 -> 0.0092, unlimited -> 0.0127                (default 30)
//   PPAT_MIN_PHASE    tail moves are uniform below this board fullness, ppat above;
//                     1 = ppat off, no model loaded                     (default 0.6)
//   PPAT_DATA         ppat weight file for the tail, loaded only when PPAT_MIN_PHASE < 1
//                     (default out/ppat-data-233162-best-ref-candidate.js)
//   FPOL_WEIGHT_0     root move influence: featurepol's logit times a weight is
//   FPOL_WEIGHT_1     added to the actor's score in the root argmax (offline
//                     knowledge at the root only; the sims are untouched).  The
//                     weight is FPOL_WEIGHT_0 at phase 0 and FPOL_WEIGHT_1 at
//                     phase 1, linear in the root's fullness between; equal
//                     values give a flat weight; both 0 = off, no model
//                     loaded                                        (defaults 0, 0)
//   ROOT_MOVE_FILTER  K: the root argmax is taken over featurepol's top K moves
//                     only (the sims are untouched); 0 = off.  Match at 500 ms on
//                     12x12, K=20 vs off: 36 of 47 won (2026-09-29); the MD file
//                     scored it the other way, its labeller underrates
//                     featurepol choices                                (default 20)
//   FPOL_DATA         the featurepol model, loaded only when a weight or
//                     ROOT_MOVE_FILTER is non-zero
//                                              (default ref/ref-fp-heavy-data.js)
//   FPOL_RANK_TOPN    the model's vpat<n> ranking term is computed over its top N
//                     candidates, as ref-fp-heavy does; 0 = off         (default 14)
//   LR                actor step size on the return minus the baseline.  Ladder at
//                     2 s: 0.04 -> 0.0189 ... 0.005 -> 0.0108, 0.002 -> 0.0103 (default 0.002)
//   ACTOR_PHASE_BUCKETS  stacked slice keyed by the sim board's phase bucket, this
//                     many equal-width buckets of [0,1]; 0 = off           (default 0)
//   ACTOR_ROOT_LAYER  1 = stacked slice read and trained on ply 0 of a sim only
//                     (and by the root argmax)                            (default 0)
//   ACTOR_ROOT_RESET  1 = zero the root slice at the start of every getMove (default 1)
//   RESET_EACH_MOVE   1 = start every move from zero actor weights and baseline, as
//                     if each root were a new game (what evalmovedetails measures);
//                     0 = carry them from move to move within a game.  2 s matches
//                     vs fp-heavy: carried won 1 of 18, reset 8 of 13  (default 1)
//   ACTOR_COLOR_LAYER  1 = the (mover, point) table; 0 = drop it     (default 1)
//   ACTOR_COLORBLIND_LAYER  1 = add a (point) table shared by both sides; at
//                     least one of the two base tables must be on.  MD ladder on
//                     md-trunc30k roots >= 0.6, root filter off: better at every
//                     rung 16-4096 sims (4096: mae 0.0777 vs 0.1006), worth
//                     ~4-8x the sims (2026-09-30)                        (default 1)
//   ACTOR_LOCAL_LAYER  1 = stacked slice keyed by "one of the 8 points around the
//                     last sim move" (none at ply 0 and the root)           (default 0)
//   ACTOR_CHAIN_LAYER  1 = add the chain layer: one weight per state of each chain
//                     with at most 3 liberties, added to its adjacent points'
//                     scores, shared by both sides                       (default 0)
//   TEMP              softmax temperature for the simulations          (default 1)
//   ACTOR_RETURN_EMA  the actor's baseline is a per-mover EMA of sim returns;
//                     this is its decay                                  (default 0.9)
//   PLAYOUTS          cap on simulations per move; 0 = time budget only (default 0)
//   TRUNC_PHASE_DELTA  the random buffer after the actor plies, as a fraction of
//                     the area, for roots that fit it under TRUNC_MAX_PHASE;
//                     0 = truncation right after the actor plies          (default 0.12)
//   TRUNC_PHASE_DELTA_MIN  the shortest buffer a root may truncate with: roots
//                     nearer the band's edge get the longest buffer that still
//                     fits, down to this; below it they run full sims
//                                                                  (default 0.09)
//   TRUNC_ACTOR_DEPTH  actor plies in a truncated sim before the random buffer (default 10)
//   TRUNC_MAX_PHASE     truncate only when the TRUNCATION POINT's phase would be
//                     below this; 0 = truncation off, no model loaded     (default 0.57)
//   TRUNC_VPAT_DATA   the truncation model (default out/vpat-1j9ad1fk.js, the fielded one)
const CHAIN_MAX_LIBS = 3;   // the chain layer keys chains with at most this many liberties

function create(cfg) {
  cfg = cfg || Util.makeCfg();

  const ACTOR_DEPTH = cfg.int('ACTOR_DEPTH', 30);
  let actorOn = true;                        // the actor plays the current sim ply (else the tail)
  const TERM_LR  = cfg.float('LR', 0.002);
  const PB       = cfg.int('ACTOR_PHASE_BUCKETS', 0);       // phase slice: bucket count, 0 = off
  const USE_R    = cfg.int('ACTOR_ROOT_LAYER', 0) !== 0;    // root slice
  const ROOT_RESET = cfg.int('ACTOR_ROOT_RESET', 1) !== 0;
  const RESET_EACH_MOVE = cfg.int('RESET_EACH_MOVE', 1) !== 0;
  const USE_L    = cfg.int('ACTOR_LOCAL_LAYER', 0) !== 0;   // local slice
  const USE_M    = cfg.int('ACTOR_COLOR_LAYER', 1) !== 0;   // (mover, point) base table
  const USE_C    = cfg.int('ACTOR_COLORBLIND_LAYER', 1) !== 0;   // (point) base table, both sides
  const USE_K    = cfg.int('ACTOR_CHAIN_LAYER', 0) !== 0;        // chain-state layer
  if (!USE_M && !USE_C) throw new Error('dt-reinforce: ACTOR_COLOR_LAYER and ACTOR_COLORBLIND_LAYER are both off');
  let curB = 0, atRoot = false;              // the slices' current keys (sim state)
  const TEMP     = cfg.float('TEMP', 1);
  const BASE_EMA = cfg.float('ACTOR_RETURN_EMA', 0.9);
  const PLAYOUTS_CAP = cfg.int('PLAYOUTS', 0);
  const TRUNC_DELTA       = cfg.float('TRUNC_PHASE_DELTA', 0.12);
  const TRUNC_DELTA_MIN   = cfg.float('TRUNC_PHASE_DELTA_MIN', 0.09);
  if (TRUNC_DELTA_MIN > TRUNC_DELTA) throw new Error(`dt-reinforce: TRUNC_PHASE_DELTA_MIN ${TRUNC_DELTA_MIN} exceeds TRUNC_PHASE_DELTA ${TRUNC_DELTA}`);
  const TRUNC_ACTOR_DEPTH = cfg.int('TRUNC_ACTOR_DEPTH', 10);
  const TRUNC_MAX_PHASE   = cfg.float('TRUNC_MAX_PHASE', 0.57);
  let vpatModel = null;
  if (TRUNC_MAX_PHASE > 0) {
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
  const PPAT_MIN_PHASE = cfg.float('PPAT_MIN_PHASE', 0.6);
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

  // Root move influence: featurepol's logit, weighted, added to the actor's
  // score in the root argmax.  Offline knowledge at the root only; the sims
  // and their learning are untouched.
  const FPOL_W0 = cfg.float('FPOL_WEIGHT_0', 0), FPOL_W1 = cfg.float('FPOL_WEIGHT_1', 0);
  const fpolWeight = (phase) => FPOL_W0 + (FPOL_W1 - FPOL_W0) * phase;
  const FPOL_TOP_K = cfg.int('ROOT_MOVE_FILTER', 20);
  let fpWeights = null, fpState = null, fpScores = null, fpPt = null, fpAllow = null;
  if (FPOL_W0 !== 0 || FPOL_W1 !== 0 || FPOL_TOP_K > 0) {
    const fpPath = _isNode ? cfg.str('FPOL_DATA', require('path').join(__dirname, '..', 'ref', 'ref-fp-heavy-data.js')) : undefined;
    fpWeights = FeaturePol.loadModel({ name: 'dt-reinforce', path: fpPath }).weights;
    const rankTopN = cfg.int('FPOL_RANK_TOPN', 14);
    if (fpWeights.spec.rankSpaces && fpWeights.spec.rankSpaces.length > 0) fpWeights.rankTopN = rankTopN;
  }
  // Per-point featurepol logit at the root (0 where featurepol lists no
  // move, i.e. illegal or true-eye points), or null when the influence is off.
  // With ROOT_MOVE_FILTER, fpAllow marks featurepol's top K points (the root
  // argmax's candidates).
  function fpRootLogits(game) {
    if (!fpWeights) return null;
    if (!fpState || fpState.N !== game.N) {
      fpState = { N: game.N, state: FeaturePol.createState(game.N, fpWeights.spec) };
      fpScores = new Float64Array(area + 1); fpPt = new Float64Array(area); fpAllow = new Uint8Array(area);
    }
    const { state } = fpState;
    FeaturePol.extractFeatures(game, state, fpWeights, fpWeights.spec.needsLadder ? game3FromGame2(game) : undefined);
    const n = FeaturePol.scoreAll(state, fpWeights, fpScores);
    fpPt.fill(0);
    for (let i = 0; i < n; i++) fpPt[state.moves[i]] = fpScores[i];
    if (FPOL_TOP_K > 0) {
      const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => fpScores[b] - fpScores[a]);
      fpAllow.fill(0);
      for (let j = 0; j < Math.min(FPOL_TOP_K, n); j++) fpAllow[state.moves[order[j]]] = 1;
    }
    return fpPt;
  }

  // ── Per-instance state (sized on first use; rebuilt if the board size changes) ──
  let area = 0;
  let w1 = null, wC = null;                     // actor base weights: mover [mover][p], colourblind [p]
  let wP = null, wR = null, wL = null;          // slice weights: phase [mover][p][bucket], root [mover][p], local [mover][p][0|1]
  let bs = null, lastAt = null;                 // per-step phase bucket and sim last move, for the update
  let loc = null, curLast = PASS;               // local slice: per-point key (1 = one of the 8 points around curLast), the sim's last move
  let locMark = null;                           // scratch: the local points of one recorded step
  // Chain layer: wK[key] over chain states; kSum[p] = Σ wK over the qualifying
  // chains adjacent to empty p (both movers' scores share it); gK = the board
  // being scored.  Per actor step, each qualifying chain's key, its liberties'
  // Σ ex (the sampled-from mass) and whether the chosen move touched it, for
  // the update: kOff[t] .. kOff[t + 1] index kKey / kMass / kHit.
  let wK = null, kSum = null, gK = null;
  let kStamp = null, kStampN = 0;               // per-gid visit stamps (freed gids are not cleared)
  let kOff = null, kKey = null, kMass = null, kHit = null;
  let kEnt = null, kEntQ = null;                // scratch: chains a move affects (a stone of each, qualified before)
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

  function setup(N) {
    area = N * N;
    w1 = new Float32Array(USE_M ? 2 * area : 0);
    wC = new Float32Array(USE_C ? area : 0);
    wP = new Float32Array(PB > 0 ? 2 * area * PB : 0);
    wR = new Float32Array(USE_R ? 2 * area : 0);
    wL = new Float32Array(USE_L ? 2 * area * 2 : 0);
    wK = new Float32Array(USE_K ? CHAIN_MAX_LIBS * area * area : 0);
    kSum = new Float64Array(area);
    kStamp = new Int32Array(area + 4); kStampN = 0;
    kOff = new Int32Array(3 * area + 22);
    kKey = new Int32Array(USE_K ? 1024 : 0); kMass = new Float64Array(USE_K ? 1024 : 0); kHit = new Uint8Array(USE_K ? 1024 : 0);
    kEnt = new Int32Array(4 * area + 8); kEntQ = new Uint8Array(4 * area + 8);
    loc = new Uint8Array(area); locMark = new Uint8Array(area);
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
    lastAt = new Int32Array(maxSteps);
    ppatState = ppatModel ? PPat.createState(N) : null;
  }

  function reset() {
    w1.fill(0); wC.fill(0); wP.fill(0); wR.fill(0); wL.fill(0); wK.fill(0);
    base[0] = base[1] = 0.5;
  }

  // Phase bucket of a board: equal-width buckets of its fullness.
  function phaseBucket(g) { const b = Math.floor((1 - g.emptyCount / area) * PB); return b >= PB ? PB - 1 : b; }

  // Actor score for mover m at p under the slices' current keys.
  function score(m, p) {
    const mp = m * area + p;
    let s = USE_M ? w1[mp] : 0;
    if (USE_C) s += wC[p];
    if (PB > 0) s += wP[mp * PB + curB];
    if (USE_R && atRoot) s += wR[mp];
    if (USE_L) s += wL[mp * 2 + loc[p]];
    if (USE_K) s += kSum[p];
    return s;
  }

  // ── Chain layer ──
  // Key of chain gid on board g: (lowest-index stone, stone count, liberty
  // count), an exact index into wK.  Only for liberty counts 1..CHAIN_MAX_LIBS.
  function chainKey(g, gid) {
    const W = g._W, sw = g._sw, b = gid * W;
    let rep = 0;
    for (let wi = 0; wi < W; wi++) {
      const w = sw[b + wi];
      if (w !== 0) { rep = wi * 32 + 31 - Math.clz32(w & -w); break; }
    }
    return (rep * area + g._ss[gid] - 1) * CHAIN_MAX_LIBS + g._ls[gid] - 1;
  }
  // Σ wK over the distinct qualifying chains adjacent to empty point p.
  function chainSum(g, p) {
    const nbr = g._nbr, gidArr = g._gid, ls = g._ls, cells = g.cells, b = p * 4;
    let s = 0, g0 = -1, g1 = -1, g2 = -1;
    for (let d = 0; d < 4; d++) {
      const a = nbr[b + d];
      if (cells[a] === EMPTY) continue;
      const gid = gidArr[a];
      if (gid === g0 || gid === g1 || gid === g2) continue;
      if (d === 0) g0 = gid; else if (d === 1) g1 = gid; else g2 = gid;
      if (ls[gid] <= CHAIN_MAX_LIBS) s += wK[chainKey(g, gid)];
    }
    return s;
  }
  // Recompute every liberty of chain gid (deduped across calls by mark 3).
  function recomputeLibs(g, gid, touched) {
    const W = g._W, lw = g._lw, b = gid * W, cells = g.cells;
    let n = touched.n;
    for (let wi = 0; wi < W; wi++) {
      let w = lw[b + wi];
      while (w) {
        const p = wi * 32 + 31 - Math.clz32(w & -w);
        w &= w - 1;
        if (mark[p] !== 3) { mark[p] = 3; list[n++] = p; recompute(cells, p); }
      }
    }
    touched.n = n;
  }
  const kTouched = { n: 0 };
  // Before playing `move` (colour `color`) with captures caps: note the chains
  // it will change — its neighbours', and the mover's chains next to each
  // captured stone — by one stone each, with whether each qualified.
  function chainsBefore(g, move, caps, color) {
    const nbr = g._nbr, cells = g.cells, gidArr = g._gid, ls = g._ls;
    let n = 0;
    for (let d = 0; d < 4; d++) {
      const a = nbr[move * 4 + d];
      if (cells[a] !== EMPTY) { kEnt[n] = a; kEntQ[n++] = ls[gidArr[a]] <= CHAIN_MAX_LIBS ? 1 : 0; }
    }
    for (let i = 0; i < caps.length; i++) {
      const c = caps[i];
      for (let d = 0; d < 4; d++) {
        const a = nbr[c * 4 + d];
        if (cells[a] === color) { kEnt[n] = a; kEntQ[n++] = ls[gidArr[a]] <= CHAIN_MAX_LIBS ? 1 : 0; }
      }
    }
    kEnt[n] = move; kEntQ[n++] = 0;       // the mover's (merged or new) chain
    return n;
  }
  // After the move: recompute the liberties of each surviving affected chain
  // that qualified before or qualifies now (a chain's key reaches exactly its
  // liberties' scores).
  function chainsAfter(g, nEnt) {
    const cells = g.cells, gidArr = g._gid, ls = g._ls;
    kStampN++;
    kTouched.n = 0;
    for (let i = 0; i < nEnt; i++) {
      const a = kEnt[i];
      if (cells[a] === EMPTY) continue;             // captured
      const gid = gidArr[a];
      let q = kEntQ[i];
      for (let j = i + 1; j < nEnt; j++) if (cells[kEnt[j]] !== EMPTY && gidArr[kEnt[j]] === gid) q |= kEntQ[j];
      if (kStamp[gid] === kStampN) continue;        // done for this gid
      kStamp[gid] = kStampN;
      if (q || ls[gid] <= CHAIN_MAX_LIBS) recomputeLibs(g, gid, kTouched);
    }
    for (let i = 0; i < kTouched.n; i++) mark[list[i]] = 0;
  }
  // Record step t's qualifying chains for the update: key, the Σ ex[m] over
  // its liberties, and whether `move` is one of them.
  function chainRecord(g, t, m, move) {
    const cells = g.cells, gidArr = g._gid, ls = g._ls, lw = g._lw, W = g._W, e = ex[m];
    let n = kOff[t];
    kStampN++;
    for (let p = 0; p < area; p++) {
      if (cells[p] === EMPTY) continue;
      const gid = gidArr[p];
      if (kStamp[gid] === kStampN) continue;
      kStamp[gid] = kStampN;
      if (ls[gid] > CHAIN_MAX_LIBS) continue;
      if (n === kKey.length) {                      // grow the step records
        const k2 = new Int32Array(2 * n); k2.set(kKey); kKey = k2;
        const m2 = new Float64Array(2 * n); m2.set(kMass); kMass = m2;
        const h2 = new Uint8Array(2 * n); h2.set(kHit); kHit = h2;
      }
      const b = gid * W;
      let mass = 0;
      for (let wi = 0; wi < W; wi++) {
        let w = lw[b + wi];
        while (w) { const q = wi * 32 + 31 - Math.clz32(w & -w); w &= w - 1; mass += e[q]; }
      }
      kKey[n] = chainKey(g, gid);
      kMass[n] = mass;
      kHit[n] = move !== PASS && (lw[b + (move >> 5)] & (1 << (move & 31))) !== 0 ? 1 : 0;
      n++;
    }
    kOff[t + 1] = n;
  }

  // Local slice: make `move` the sim's last move.  The old last move's 8
  // neighbours lose the key, the new one's gain it, and those points are
  // recomputed (deduped; PASS has no neighbours).
  function setLast(cells, nbr, dnbr, move) {
    let n = 0;
    for (const q of [curLast, move]) {
      if (q === PASS) continue;
      const b = q * 4, v = q === move ? 1 : 0;
      for (let d = 0; d < 4; d++) {
        const a = nbr[b + d], c = dnbr[b + d];
        loc[a] = v; loc[c] = v;
        if (!mark[a]) { mark[a] = 1; list[n++] = a; }
        if (!mark[c]) { mark[c] = 1; list[n++] = c; }
      }
    }
    curLast = move;
    for (let i = 0; i < n; i++) { const p = list[i]; mark[p] = 0; recompute(cells, p); }
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
      if (USE_K && v === EMPTY) kSum[p] = chainSum(gK, p);
      for (let m = 0; m < 2; m++) {
        let e = 0;
        if (v === EMPTY) { const s = score(m, p); sc[m][p] = s; e = Math.exp(s / TEMP); }
        setWeight(m, p, e);
      }
    }
  }

  function recomputeAll(cells) {
    S[0] = S[1] = 0;
    if (USE_K) for (let p = 0; p < area; p++) if (cells[p] === EMPTY) kSum[p] = chainSum(gK, p);
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
    gK = g;
    actorOn = true;
    atRoot = USE_R; curB = PB > 0 ? phaseBucket(g) : 0;
    if (USE_L) { loc.fill(0); curLast = PASS; }   // ply 0: no last move
    recomputeAll(cells);
    const nbr = g._nbr, dnbr = g._dnbr;
    nbrT = nbr; dnbrT = dnbr;
    let t = 0, actorSteps = 0;
    if (USE_K) kOff[0] = 0;
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
        bs[t] = curB; lastAt[t] = curLast;
        if (USE_K) chainRecord(g, t, m, move);
        actorSteps++;
      } else {
        move = truncActive || !ppatModel ? g.randomLegalMove(rng) : PPat.ppatMove(g, ppatState, ppatModel, rng);
        if (USE_K) kOff[t + 1] = kOff[t];
      }
      fromActor[t] = actorOn ? 1 : 0;
      movers[t] = m; chosen[t] = move;
      const prevKo = g.ko;
      let nChanged = 0, nEnt = 0;
      if (move !== PASS) {
        const caps = g.captureList(move);
        changed[nChanged++] = move;
        for (let i = 0; i < caps.length; i++) changed[nChanged++] = caps[i];
        if (USE_K && actorOn) nEnt = chainsBefore(g, move, caps, g.current);
      }
      g.play(move);
      if (nChanged > 0) recomputeAround(g, changed, nChanged);
      if (nEnt > 0) chainsAfter(g, nEnt);
      // A ko point turns legal again once any other move is played; its
      // neighbourhood did not change, so lift its exclusion explicitly.
      if (prevKo !== PASS && cells[prevKo] === EMPTY) recompute(cells, prevKo);
      if (USE_L) setLast(cells, nbr, dnbr, move);
      t++;
    }
    lastActorSteps = actorSteps;
    // The return: the outcome, or at the truncation point the vpat model's value.
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
  // each active slice's weight.
  function actorUpdate(t, m, adv) {
    const move = chosen[t];
    if (!fromActor[t] || move === PASS) return;   // tail moves and PASS are outside the softmax
    const o = t * area;
    const e = exs.subarray(o, o + area);
    const invS = 1 / Ss[t];
    const k = TERM_LR * adv / TEMP;
    const b = bs[t], root = USE_R && t === 0;
    // The local key at this step: the 8 neighbours of the step's last move.
    const L = USE_L ? lastAt[t] : PASS;
    if (L !== PASS) { const bb = L * 4; for (let d = 0; d < 4; d++) { locMark[nbrT[bb + d]] = 1; locMark[dnbrT[bb + d]] = 1; } }
    for (let p = 0; p < area; p++) {
      const v = e[p];
      if (v <= 0) continue;
      const gr = (p === move ? 1 : 0) - v * invS;
      const mp = m * area + p;
      if (USE_M) w1[mp] += k * gr;
      if (USE_C) wC[p] += k * gr;
      if (PB > 0) wP[mp * PB + b] += k * gr;
      if (root) wR[mp] += k * gr;
      if (USE_L) wL[mp * 2 + locMark[p]] += k * gr;
    }
    if (L !== PASS) { const bb = L * 4; for (let d = 0; d < 4; d++) { locMark[nbrT[bb + d]] = 0; locMark[dnbrT[bb + d]] = 0; } }
    // Chain layer: Σ over a chain's liberties of ([a = chosen] − π(a)) is
    // [chosen touches it] − its liberties' mass / S.
    if (USE_K) for (let i = kOff[t]; i < kOff[t + 1]; i++) wK[kKey[i]] += k * (kHit[i] - kMass[i] * invS);
  }

  // Actor update over the sim's records, in step order.  z = the return,
  // P(BLACK wins); the mover's view is z for BLACK and 1 − z for WHITE, and
  // the advantage is that minus the mover's baseline as it stood before this
  // sim (the baseline then moves toward the return).
  let nbrT = null, dnbrT = null;                // the board's neighbour tables, kept for the update
  function update(steps, z) {
    const adv = [z - base[0], (1 - z) - base[1]];
    base[0] += (1 - BASE_EMA) * (z - base[0]);
    base[1] += (1 - BASE_EMA) * ((1 - z) - base[1]);
    for (let t = 0; t < steps; t++) actorUpdate(t, movers[t], adv[movers[t]]);
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

  // Prepare for a decision at `game` (new-game detection, the root slice's
  // reset, this move's truncation) and run its sims: PLAYOUTS of them, or as
  // many as budgetMs allows.  Returns the sims' count, step totals and the
  // sum of their returns.
  function runSims(game, budgetMs, rng) {
    if (area !== game.N * game.N) { setup(game.N); reset(); }
    if (RESET_EACH_MOVE || !isContinuation(game)) reset();   // a new game, or every move
    if (USE_R && ROOT_RESET) wR.fill(0);   // the root slice holds only this move's sims
    // The buffer: the full one if its truncation point stays under the band's
    // edge, else the longest that does; the root truncates if that is at least
    // the minimum buffer.
    const phase = game.phase();
    let buffer = Math.ceil(TRUNC_DELTA * area);
    while (buffer > 0 && !(phase + (TRUNC_ACTOR_DEPTH + buffer) / area < TRUNC_MAX_PHASE)) buffer--;
    truncPly = TRUNC_ACTOR_DEPTH + buffer;
    truncActive = buffer >= Math.ceil(TRUNC_DELTA_MIN * area) && phase + truncPly / area < TRUNC_MAX_PHASE;
    const tStart = Date.now();
    let sims = 0, longest = 0, totalSteps = 0, sumZ = 0;
    while (true) {
      if (PLAYOUTS_CAP > 0 ? sims >= PLAYOUTS_CAP : Date.now() - tStart >= budgetMs) break;
      const steps = simulate(game, rng);
      if (steps > longest) longest = steps;
      totalSteps += steps;
      sumZ += lastReturn;
      sims++;
    }
    return { sims, longest, totalSteps, sumZ };
  }

  // Value oracle: P(BLACK wins) as the mean return of this position's sims,
  // under whatever truncation the settings give this position.  A terminal
  // position is scored exactly.
  function valueB(game, options = {}) {
    const g = game.cells ? game : game.toGame2();
    if (g.gameOver) return g.calcWinner() === BLACK ? 1 : 0;
    const budgetMs = options.budgetMs > 0 ? options.budgetMs : 1000;
    const { sims, sumZ } = runSims(g, budgetMs, options.rng || makeRng());
    afterCount = -1;                       // no move was played: the next call is a new game
    if (sims === 0) throw new Error('dt-reinforce: valueB ran no sims (budget too small)');
    return sumZ / sims;
  }

  function getMove(game, budgetMs = 1000, options = {}) {
    if (game.consecutivePasses > 0 && game.calcWinner() === game.current) {
      recordAfter(game, PASS);
      return { move: PASS, info: 'end the game; ahead' };
    }
    const rng = options.rng || makeRng();
    const { sims, longest, totalSteps } = runSims(game, budgetMs, rng);

    // Play the argmax over legal non-eye points of the actor's score plus the
    // weighted featurepol logit when the influence is on.
    const cells = game.cells;
    const m = game.current === BLACK ? 0 : 1;
    const fp = fpRootLogits(game);
    const fpW = fp ? fpolWeight(game.phase()) : 0;
    actorOn = true;
    atRoot = USE_R; curB = PB > 0 ? phaseBucket(game) : 0;
    if (USE_L) { loc.fill(0); curLast = PASS; }   // the root sees no last move
    gK = game;
    recomputeAll(cells);                  // root scores
    let best = PASS, bestS = -Infinity;
    for (let p = 0; p < area; p++) {
      if (cells[p] !== EMPTY || !game.isLegal(p) || game.isTrueEye(p)) continue;
      if (FPOL_TOP_K > 0 && !fpAllow[p]) continue;
      const s = sc[m][p] + (fp ? fpW * fp[p] : 0) + rng.random() * 1e-9;
      if (s > bestS) { bestS = s; best = p; }
    }
    recordAfter(game, best);
    return { move: best, info: `sims=${sims} steps=${totalSteps} longest=${longest}${truncActive ? ` trunc=${truncPly}` : ''} base=${base[m].toFixed(3)} score=${bestS.toFixed(3)}${fp ? ` fpW=${fpW.toFixed(3)}` : ''}${FPOL_TOP_K > 0 ? ` fpK=${FPOL_TOP_K}` : ''}` };
  }

  // Test hook: live views of the internals (state arrays are created by setup).
  function _internals() {
    return { setup, reset, recomputeAll, recomputeAround, score, sample, simulate, update, isContinuation, recordAfter,
             get lastActorSteps() { return lastActorSteps; },
             get lastReturn() { return lastReturn; },
             setTrunc: (active, plies) => { truncActive = active; truncPly = plies; },
             setLast, chosen: () => chosen, sc, ex, S, base, w1, wC, wP, wR, wL, loc, area,
             wK, kSum, chainKey, chainSum, lastBoard: () => gK };
  }

  return { getMove, valueB, _internals };
}

let _default = null;
function _def() { return _default || (_default = create(Util.makeCfg())); }
if (typeof module !== 'undefined') module.exports = { create, getMove: (g, b, o) => _def().getMove(g, b, o), valueB: (g, o) => _def().valueB(g, o) };
else { window.getMove = (g, b, o) => _def().getMove(g, b, o); window.valueB = (g, o) => _def().valueB(g, o); }

})();
