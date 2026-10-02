'use strict';

// puct-trunc: puct-ppat-fp with TRUNCATED playouts.  Truncation is a
// per-decision choice made at the root: when the root phase is below
// TRUNC_ROOT_PHASE, every playout runs for the prefix length set by
// TRUNC_PHASE_DELTA (a move count, or a fullness advance under
// LEGACY_PHASE_DELTA) and then its leaf value is a static vpatterns evaluation
// (TRUNC_VPAT_DATA, a train-vpat-supervised checkpoint: V(s) = P(BLACK wins))
// instead of the terminal score.  Otherwise every playout runs to the end.
// The prefix moves still fill the RAVE trace either way.  (An averaged
// multi-position evaluation window was tried and removed: consecutive-
// position evaluator errors are near-perfectly correlated, so averaging
// bought nothing offline or live — 2026-09-04.)
//
// The truncation check is integer (empty-count drop >= ceil(delta*area)), so
// captures during the prefix delay it correctly: the point is defined by net
// board-filling progress, not move count.
//
// PUCT MCTS with policy-driven priors, top-K candidate pruning at every node
// including the root (TOP_K, 0 = full width), RAVE, and
// ppat-policy full-playout leaf
// evaluation.
//
// Each unexpanded edge is expanded on first contact: a leaf node is created
// (policy extraction + priors) and one playout is run from it.  A ppat playout
// is expensive enough that the node-creation cost is always worth paying, so
// there is no lazy-expansion threshold.
//
// Terminal positions are scored exactly.  Values are backpropagated
// fractionally: each chooser is credited value (BLACK) or 1 − value (WHITE).
//
// Node selection:
//   score = Q + C_PUCT · P(s,a) · √N_total / (1 + N_a)
// where Q is the move's mean backpropagated value (RAVE-blended when RAVE_K
// > 0) and P(s,a) is the policy softmax probability for the move.
//
// ── Factory ──
// The module exports create(cfg) → { getMove, valueB }.  cfg is a
// Util.makeCfg reader (slot-aware env): each instance resolves its own config
// and loads its own weights at construction, so two instances in one process
// (e.g. selfplay p1 vs p2) never collide.  A lazy default instance backs the
// direct module.getMove / module.valueB exports for single-agent callers.

(function () {

const Util = (typeof require === 'function') ? require('../util.js') : window.Util;
const { PASS, BLACK, Game2 } = Util.load('./game2.js', 'Game2');
const { makeRng }            = Util.load('./xorshift.js', 'XorShift');
const FeaturePol            = Util.load('./featurepol-lib.js', 'FeaturePol');
const { game3FromGame2 }     = Util.load('./game3.js', 'Game3');
const _ppat                 = Util.load('./ppat-lib.js', 'PPatterns');
const VPat                  = Util.load('./vpatterns.js', 'VPatterns');
const Symmetry              = Util.load('./symmetry.js', 'Symmetry');
const { createState, ppatMove, loadWeights } = _ppat;

const performance = (typeof window !== 'undefined' && window.performance)
  ? window.performance : require('perf_hooks').performance;

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

// ── Config-free constants & helpers (shared across instances) ──────────────────

const PRIOR_WINS   = 0.001;
const PRIOR_VISITS = 2 * PRIOR_WINS;
const RESIGN_MIN_PLAYOUTS = 20000;

// ── Factory ────────────────────────────────────────────────────────────────────

// Build an independent agent instance whose config and state are bound at
// construction.  cfg is a Util.makeCfg reader (slot-aware env); all per-instance
// state lives in this closure.
function create(cfg) {
  cfg = cfg || Util.makeCfg();

  // Static evaluator: a vpatterns checkpoint (train-vpat-supervised).  Hard
  // failure, not a fallback — this agent's identity IS its truncated playouts,
  // and a silently-missing evaluator would field plain puct-ppat-fp under the
  // wrong name.  Loaded up front (before the truncation knobs) so a 'trunc'
  // block baked into the model file can supply their defaults.
  const _vpatPath = _isNode
    ? cfg.str('TRUNC_VPAT_DATA',
        require('path').join(__dirname, '..', 'out', 'vpat-1j9ad1fk.js'))
    : null;
  if (_isNode && !_vpatPath) {
    throw new Error(`puct-trunc[${cfg.slot != null ? cfg.slot : '-'}]: TRUNC_VPAT_DATA is required`);
  }
  const _vpatRaw = _isNode
    ? require(require('path').resolve(_vpatPath))
    : (typeof window !== 'undefined' && window.truncVpatModel) || null;
  if (!_vpatRaw) {
    throw new Error(`puct-trunc[${cfg.slot != null ? cfg.slot : '-'}]: cannot load vpatterns evaluator from ` +
      (_isNode ? 'TRUNC_VPAT_DATA' : 'window.truncVpatModel'));
  }
  // Truncation default baked into the model file: { delta }.  An env var
  // (TRUNC_PHASE_DELTA) still overrides.
  const _truncMeta = (_vpatRaw && _vpatRaw.trunc) || {};

  // PUCT exploration constant — weight of the prior P(s,a) relative to Q.
  const C_PUCT     = cfg.float('C_PUCT', 0.25);
  // RAVE blend strength: Q mixes rave/real win-rate with weight RAVE_K/(RAVE_K+n).
  const RAVE_K     = cfg.float('RAVE_K', 800);
  // Top-K kept move count, applied at EVERY node including the root (0 = full
  // width).  30 beat 40 by 52.3% over 1427 games (match8, 2026-09-09).  Two
  // things were folded in here: the old separate ROOT_TOP_K (a sweep found the
  // root's best top-K tracks the interior's), and a phase interpolation —
  // TOP_K_A at phase 0 (empty board), TOP_K_B at phase 1 (full board), rounded
  // to nearest — since the best top-K rises with phase (tight opening, wider
  // late).  A == B is a flat top-K; the defaults ramp 10 (opening) -> 40 (late).
  const TOP_K_A   = cfg.int('TOP_K_A', 10);
  const TOP_K_B   = cfg.int('TOP_K_B', 40);
  // Prune symmetry-equivalent root moves.  When the root position has a board
  // symmetry (common in the opening — see symmetry.js), moves in the same orbit
  // lead to positions identical up to that symmetry, so they have equal value;
  // keeping one representative concentrates the search budget on distinct moves
  // at no accuracy cost.  Exact and self-limiting (does nothing once the board
  // is asymmetric).  1 = on (default), 0 = off (for A/B).
  const ROOT_SYMMETRY = cfg.int('ROOT_SYMMETRY', 1) !== 0;
  // Lazy expansion, priced by WORK.  An edge's child node (featurepol extraction
  // + priors) is created once the playout work accrued on it reaches EXPAND_WORK;
  // playouts before that run from the unexpanded position.  A playout's work, in
  // units of one ppat (feature-extraction) move, is
  //   PLAYOUT_OVERHEAD + ppatMoves + uniformMoves*UNIFORM_WEIGHT + [trunc]TRUNC_OVERHEAD
  // so it tracks real cost across regimes with one threshold: a ppat move is a
  // full unit, an early uniform-random move (below PPAT_MIN_PHASE, no extraction)
  // a fraction, and a truncated playout — all uniform prefix moves plus one vpat
  // leaf eval — is correctly cheap.  PLAYOUT_OVERHEAD (full-playout fixed cost:
  // clone + tree machinery + terminal scoring), UNIFORM_WEIGHT (c_uniform/c_ppat)
  // and TRUNC_OVERHEAD (the EXTRA fixed cost of a truncated playout — one vpat
  // leaf eval in place of scoring) are fixed defaults, the medians of 30
  // startup calibrations of the default models on 13x13 (2026-10-01); per-run
  // timing varied the playout overhead 1.2-2.6 and once failed outright, so it
  // was dropped.  Re-measure when the ppat or vpat model changes.  The env can
  // pin each one.  EXPAND_WORK is the tuning dial.
  const EXPAND_WORK      = cfg.float('EXPAND_WORK', 200);
  const _envPlayoutOvh   = cfg.has('PLAYOUT_OVERHEAD');
  const PLAYOUT_OVERHEAD = cfg.float('PLAYOUT_OVERHEAD', 1.8);
  const _envUniformWt    = cfg.has('UNIFORM_WEIGHT');
  const UNIFORM_WEIGHT   = cfg.float('UNIFORM_WEIGHT', 0.09);
  const _envTruncOvh     = cfg.has('TRUNC_OVERHEAD');
  const TRUNC_OVERHEAD   = cfg.float('TRUNC_OVERHEAD', 3.4);
  let _lastPlayoutPpat = 0, _lastPlayoutUniform = 0, _lastPlayoutTrunc = false;   // set by playout, read in runSearch
  // Fixed playout count per decision; when non-zero, overrides the time budget.
  const PLAYOUTS   = cfg.int('PLAYOUTS', 0);
  // Truncation point: net board-fullness advance past the leaf before the
  // playout stops for a static evaluation (board-size invariant).  The
  // default 1 can never be reached, so out of the box every playout runs to
  // the end — plain puct-ppat-fp behaviour until the knobs are set.  Resolution
  // order: env var, else the model file's baked delta, else 0.2 (the champion).
  const _deltaFromModel = !cfg.has('TRUNC_PHASE_DELTA') && _truncMeta.delta != null;
  const TRUNC_PHASE_DELTA = cfg.has('TRUNC_PHASE_DELTA') ? cfg.float('TRUNC_PHASE_DELTA', 0.21)
                          : (_deltaFromModel ? _truncMeta.delta : 0.2);
  // Prefix length in moves, set once per turn from the board size (getMove).
  let _prefixLen = 0;
  // How the truncation prefix is measured.  Default: descend a fixed number of
  // MOVES, ceil(TRUNC_PHASE_DELTA * area), so every truncated playout goes the
  // same distance whatever it captures.  LEGACY_PHASE_DELTA=true restores the
  // pre-2026-09-10 method: descend until the net empty count has dropped by
  // that much, which captures push further away.
  const LEGACY_PHASE_DELTA = cfg.bool('LEGACY_PHASE_DELTA', false);
  // Root-decision truncation: the whole decision truncates (every playout
  // substitutes the vpat value at its prefix endpoint) iff the ROOT phase is
  // below this; otherwise every playout runs full.  Deciding once at the root
  // keeps a search self-consistent — no truncated/full mix within one tree, so
  // no per-leaf gate seam.  Default 0.35 is the champion threshold (its old
  // gate B 0.55 minus delta 0.20 — the point at which the shallowest possible
  // endpoint, root + delta, would reach the trusted-evaluator band edge).
  const TRUNC_ROOT_PHASE = cfg.float('TRUNC_ROOT_PHASE', 0.30);
  // Set per decision in runSearch: rootPhase < TRUNC_ROOT_PHASE.
  let _truncActive = false;

  // Static evaluator (the raw was loaded up top so its baked truncation defaults
  // could feed the knobs above).  modelFromRaw builds a fresh weight table and
  // prepares the specs, using the health model embedded in the vpat file.
  const _vpatModel = VPat.modelFromRaw(_vpatRaw);
  // Name the evaluator file and the truncation knobs in the banner: two slots
  // (P1_/P2_TRUNC_*) otherwise print identical lines, hiding which evaluator
  // and gate each side is actually running.
  const _vpatName = _isNode ? require('path').basename(_vpatPath) : 'window.truncVpatModel';
  console.error(`puct-trunc[${cfg.slot != null ? cfg.slot : '-'}]: ` +
    `${_vpatModel.weights.size} vpat weights (${VPat.specString(_vpatModel.specs)}) from ${_vpatName}, ` +
    `trunc-phase-delta: ${TRUNC_PHASE_DELTA}${_deltaFromModel ? ' (model)' : ''} (${LEGACY_PHASE_DELTA ? 'legacy fullness' : 'moves'}), ` +
    `trunc-root-phase: ${+TRUNC_ROOT_PHASE.toFixed(4)}, ` +
    `expand-work: ${EXPAND_WORK}, ` +
    `root-symmetry: ${ROOT_SYMMETRY ? 'on' : 'off'}`);

  // Static value of `game2`: P(BLACK wins) from the vpatterns evaluator.
  function vpatValueB(game2) {
    return VPat.evaluateFeatures(VPat.extractFeatures(game2, _vpatModel.preparedSpecs, false, undefined, true), _vpatModel.weights);
  }

  // ppat playout policy weights: PPAT_DATA, defaulting to out/ppat-data-233162-best-ref-candidate.js
  // (the current single-phase model, as cascade.js does); window.PPATWeights in
  // the browser.  Hard failure, not a fallback: this agent's strength IS its
  // ppat playouts, so silently running uniform (e.g. a relative PPAT_DATA that
  // misses under a different cwd) fields a wrong engine under the right name.
  const _ppatPath = _isNode
    ? cfg.str('PPAT_DATA', require('path').join(__dirname, '..', 'out', 'ppat-data-233162-best-ref-candidate.js'))
    : null;
  const _model = _isNode
    ? loadWeights(_ppatPath)
    : loadWeights((typeof window !== 'undefined' && window.PPATWeights) || null);
  if (!_model) {
    throw new Error(`puct-trunc[${cfg.slot != null ? cfg.slot : '-'}]: cannot load ppat weights from ` +
      (_isNode ? _ppatPath : 'window.PPATWeights'));
  }
  // Name the ppat file in the banner: two slots (P1_/P2_PPAT_DATA) otherwise
  // print identical lines, hiding which model each side is actually running.
  const _ppatName = _isNode ? require('path').basename(_ppatPath) : 'window.PPATWeights';

  // Use uniform-random playout moves while board fullness < this fraction [0,1]
  // (0 = off).  Skips ppat feature extraction in the early game, where the
  // policy is ≈ uniform.
  _model.ppatMinPhase = cfg.float('PPAT_MIN_PHASE', 0.6);

  console.error(`puct-trunc[${cfg.slot != null ? cfg.slot : '-'}]: ` +
    `expand costs: playout ${PLAYOUT_OVERHEAD}${_envPlayoutOvh ? '(env)' : ''}, ` +
    `uniform-move ${UNIFORM_WEIGHT}${_envUniformWt ? '(env)' : ''}, ` +
    `trunc ${TRUNC_OVERHEAD}${_envTruncOvh ? '(env)' : ''}`);

  // featurepol policy model (priors + top-K pruning).  FPOL_DATA overrides
  // the default checkpoint (browser: window.featurepolModel).
  const _isBrowser  = typeof window !== 'undefined';
  const fpModel     = FeaturePol.loadModel({ name: 'puct-trunc',
    path: _isBrowser ? undefined
                     : cfg.str('FPOL_DATA', require('path').join(__dirname, '..', 'ref', 'ref-fp2-data.js')) });
  const fpWeights   = fpModel.weights;
  // Rank the vpat<n> feature over the best FPOL_RANK_TOPN candidates when the fp
  // spec ranks (0 = off (default), N > 0 = top-N, N < 0 = every candidate).
  // No-op for specs without vpat<n>.
  const FPOL_RANK_TOPN = cfg.int('FPOL_RANK_TOPN', 0);
  if (fpWeights.spec.rankSpaces && fpWeights.spec.rankSpaces.length > 0) fpWeights.rankTopN = FPOL_RANK_TOPN;
  // Softmax temperature of the featurepol priors.  Top-K keeps moves by rank,
  // which temperature does not change; it reshapes the PUCT priors among them.
  // 1 = the policy's own softmax, 0 = all prior on its top move.
  const FPOL_TEMP = cfg.float('FPOL_TEMP', 1);
  if (!(FPOL_TEMP >= 0)) throw new Error(`puct-trunc: FPOL_TEMP must be >= 0, got ${FPOL_TEMP}`);
  console.error(`puct-trunc[${cfg.slot != null ? cfg.slot : '-'}]: ${_model.weights.length} ppat weights from ${_ppatName}, ${fpWeights.size} featurepol weights from ${fpModel.modelName}, fpol-temp ${FPOL_TEMP}${fpWeights.rankTopN > 0 ? `, rank-topn ${fpWeights.rankTopN}` : ''}`);

  let _ppatState = null;
  function _ensurePpatState(N) {
    if (_ppatState === null || _ppatState.moves.length < N * N)
      _ppatState = createState(N);
  }

  const stateByN = new Map();

  // Run featurepol policy extraction + softmax for `game2`.  `game3` is the
  // lockstep mirror (maintained by the search, so extraction skips the
  // game3FromGame2 rebuild).  Returns the shared state (moves/probs
  // populated) or null.
  function _runFp(game2, game3) {
    if (!fpWeights || game2.gameOver) return null;
    const N = game2.N;
    let state = stateByN.get(N);
    if (!state) { state = FeaturePol.createState(N, fpWeights.spec); stateByN.set(N, state); }
    FeaturePol.extractFeatures(game2, state, fpWeights, game3);
    if (state.count === 0) return null;
    FeaturePol.computeSoftmax(state, fpWeights, FPOL_TEMP);
    return state;
  }

  // ppat-policy playout from `game2` (mutates it), truncated: when this decision
  // is truncating (root phase below TRUNC_ROOT_PHASE), after the turn's fixed
  // prefix length in moves it returns the static vpatterns value; otherwise the
  // playout runs to the end.  Fills `played` (pre-zeroed by the caller) with the
  // colour-signed first-occupancy RAVE trace.  Returns P(BLACK wins) — a
  // fraction at a truncation, {0,1} at the end of a full playout.
  function playout(game2, played, rng) {
    const N   = game2.N;
    const cap = N * N;

    _ensurePpatState(N);

    // Truncation trigger, two methods — see LEGACY_PHASE_DELTA.  Both use the
    // same magnitude, ceil(TRUNC_PHASE_DELTA * area); they differ only in what
    // is counted.  MOVES: descend a fixed number of moves, so every truncated
    // playout goes the same distance whatever it captures (a pass counts, as it
    // must for a move count).  FULLNESS: descend until the net empty count has
    // dropped by that much, which captures push further away.
    const truncMoves = _prefixLen;
    const truncEmpty = game2.emptyCount - _prefixLen;
    // Root-decision truncation: whether this decision truncates was decided once
    // from the root phase (runSearch, TRUNC_ROOT_PHASE).  Every playout in a
    // truncating decision substitutes the vpat value at its prefix endpoint,
    // wherever that endpoint's phase lands.
    let truncArmed = _truncActive;

    const ubp = _model.ppatMinPhase;   // moves below this fullness skip extraction (uniform)
    const moveLimit = 3 * game2.emptyCount + 20;
    const weightStep = 1 / cap;
    let moves = 0, uniformMoves = 0;
    let weight = 1.0;

    while (!game2.gameOver && moves < moveLimit) {
      const current = game2.current;
      // Classify BEFORE the move, the same way ppatMove does (fullness < ubp).
      if (ubp > 0 && (cap - game2.emptyCount) / cap < ubp) uniformMoves++;
      const idx = ppatMove(game2, _ppatState, _model, rng);
      if (idx !== PASS && weight > 0 && played[idx] === 0) {
        played[idx] = current === BLACK ? weight : -weight;
      }
      game2.play(idx);
      moves++;
      weight -= weightStep;
      if (truncArmed && (LEGACY_PHASE_DELTA ? game2.emptyCount <= truncEmpty
                                            : moves >= truncMoves)) {
        // The gate itself was decided at playout start (leaf-anchored draw).
        if (!game2.gameOver) { _lastPlayoutPpat = moves - uniformMoves; _lastPlayoutUniform = uniformMoves; _lastPlayoutTrunc = true; return vpatValueB(game2); }
        truncArmed = false;
      }
    }

    _lastPlayoutPpat = moves - uniformMoves; _lastPlayoutUniform = uniformMoves; _lastPlayoutTrunc = false;
    return game2.estimateWinner() === BLACK ? 1 : 0;
  }

  function makeNode(move, parent, ci, game2, N, game3) {
    // Compute the policy softmax once — reused for top-K pruning and the PUCT priors.
    const fpState = _runFp(game2, game3);
    let movesArr = getLegalMoves(game2);
    // Root symmetry pruning (before top-K, so top-K keeps K DISTINCT moves):
    // drop symmetry-equivalent duplicates so the budget lands on distinct moves.
    if (parent === null && ROOT_SYMMETRY) {
      const sym = Symmetry.of(game2);
      if (sym.hasSymmetry()) movesArr = sym.distinctMoves(movesArr);
    }
    // Top-K pruning at every node (root included; symmetry dedup above ran at
    // the root): keep the policy's top K, interpolated by this node's phase
    // between TOP_K_A (phase 0) and TOP_K_B (phase 1).
    const k = Math.round(TOP_K_A + (TOP_K_B - TOP_K_A) * game2.phase());
    if (k > 0) {
      movesArr = _pruneToTopK(movesArr, fpState, k, N);
    }
    const M = movesArr.length;
    const area = N * N;

    const legalMoves = new Int32Array(M);
    for (let i = 0; i < M; i++) legalMoves[i] = movesArr[i];

    const children   = new Array(M).fill(null);
    const wins       = new Float32Array(M).fill(PRIOR_WINS);
    const visits     = new Float32Array(M).fill(PRIOR_VISITS);
    const work       = new Float32Array(M);   // accrued playout work per edge (expansion gate)
    const raveWins   = RAVE_K > 0 ? new Float32Array(area).fill(PRIOR_WINS)   : null;
    const raveVisits = RAVE_K > 0 ? new Float32Array(area).fill(PRIOR_VISITS) : null;

    // PUCT priors per move (renormalised to sum to 1).  PASS gets a 1/area floor;
    // all other entries take the softmax probability directly; then renormalise.
    const priors = new Float32Array(M);
    if (fpState) {
      const probByMove = new Float64Array(area);
      for (let i = 0; i < fpState.count; i++) probByMove[fpState.moves[i]] = fpState.probs[i];
      const floor = 1 / area;
      let sum = 0;
      for (let i = 0; i < M; i++) {
        const m = legalMoves[i];
        const p = (m === PASS) ? floor : (probByMove[m] || floor);
        priors[i] = p;
        sum += p;
      }
      if (sum > 0) {
        const inv = 1 / sum;
        for (let i = 0; i < M; i++) priors[i] *= inv;
      }
    } else {
      const u = 1 / M;
      for (let i = 0; i < M; i++) priors[i] = u;
    }

    const mover = -game2.current;

    return {
      move,
      parent,
      ci,
      mover,
      totalVisits:  0.1,
      selectedChild: -1,

      legalMoves,
      children,
      priors,

      wins,
      visits,
      work,

      raveWins,
      raveVisits
    };
  }

  function nodeScore(moveIdx, node, rng) {
    // Q: mean backpropagated value for this move.
    let Q = node.wins[moveIdx] / node.visits[moveIdx];

    if (RAVE_K > 0) {
      const move   = node.legalMoves[moveIdx];
      const raveWR = (move === PASS) ? 0 : (node.raveWins[move] / node.raveVisits[move]);
      const beta   = RAVE_K / (RAVE_K + node.visits[moveIdx]);
      Q = (1 - beta) * Q + beta * raveWR;
    }

    // PUCT exploration term.  C_PUCT · P(s,a) · √N_total / (1 + N_a).
    const P = node.priors[moveIdx];
    const U = C_PUCT * P * Math.sqrt(node.totalVisits) / (1 + node.visits[moveIdx]);

    return Q + U + 0.001 * rng.random();
  }

  // `game3` is the lockstep mirror of rootGame2's position: every move played on
  // the game2 clone is also played on it, and `depth` (the number of game3 plays)
  // is returned so the caller can undo back to the root position.
  function selectAndExpand(root, rootGame2, N, rng, game3) {
    let node = root;
    const game2 = rootGame2.clone();
    let depth = 0;
    let doPlayout = false;   // true when the simulation ends in a playout visit

    // Simulation path: path[d] is the move chosen by the node at depth d.  PASS
    // entries are kept so depth/colour alignment survives; the RAVE update skips them.
    const path = [];

    while (!game2.gameOver) {
      const M = node.legalMoves.length;
      if (M === 0) break;

      let best = 0, bestScore = -Infinity;
      for (let i = 0; i < M; i++) {
        const s = nodeScore(i, node, rng);
        if (s > bestScore) { bestScore = s; best = i; }
      }

      const move = node.legalMoves[best];
      path.push(move);
      game2.play(move);
      game3.play(move);
      depth++;

      if (!game2.gameOver && game2.consecutivePasses > 0) {
        game2.play(PASS);
        game3.play(PASS);
        depth++;
        node.selectedChild = best;
        break;
      }

      // Expansion: create the child (featurepol extraction + priors) once its
      // edge has accrued EXPAND_WORK of playout work; before that, run the playout
      // from the unexpanded position with stats accumulating on the parent's edge
      // (same backprop shape as the pass-break case above).
      if (node.children[best] === null) {
        if (node.work[best] >= EXPAND_WORK) {
          node.children[best] = makeNode(move, node, best, game2, N, game3);
          node = node.children[best];
          node.selectedChild = -1;
        } else {
          node.selectedChild = best;
        }
        doPlayout = true;
        break;
      }

      node = node.children[best];
      node.selectedChild = -1;
    }

    return { node, game2, path, depth, doPlayout };
  }

  // `played` is the playout's colour-signed RAVE trace, or null for simulations
  // that ended in terminal scoring.
  function backpropagate(node, value, path, played, work) {
    function childMover(n) {
      return -n.mover;
    }

    // RAVE update for node `n` at depth `d`: credit every move its chooser played
    // from depth d onward — the in-tree segment (every second path entry) plus,
    // when a playout ran, its trace (one trace serves all ancestors).
    function updateRave(n, d, won, chooser) {
      const rw = n.raveWins, rv = n.raveVisits;
      for (let j = d; j < path.length; j += 2) {
        const m = path[j];
        if (m === PASS) continue;
        rv[m] += 1;
        rw[m] += won;
      }
      if (played === null) return;
      if (chooser === BLACK) {
        for (let k = 0; k < played.length; k++) {
          const w = played[k];
          if (w > 0) { rv[k] += w; rw[k] += won * w; }
        }
      } else {
        for (let k = 0; k < played.length; k++) {
          const w = played[k];
          if (w < 0) { rv[k] -= w; rw[k] -= won * w; }
        }
      }
    }

    // Depth of `node`: when it has a selected edge it chose path[path.length-1];
    // otherwise the walk descended past the last move before the loop exited.
    let d = path.length - 1;

    const leafIdx = node.selectedChild;
    if (leafIdx !== -1) {
      const chooser = childMover(node);
      const won     = chooser === BLACK ? value : 1 - value;
      node.visits[leafIdx]++;
      node.wins[leafIdx] += won;
      node.work[leafIdx] += work;
      node.totalVisits++;
      if (RAVE_K > 0) updateRave(node, d, won, chooser);
    } else {
      d = path.length;
    }

    while (node.parent !== null) {
      d--;
      const ci      = node.ci;
      const chooser = childMover(node.parent);
      const won     = chooser === BLACK ? value : 1 - value;
      node.parent.visits[ci]++;
      node.parent.wins[ci] += won;
      node.parent.work[ci] += work;
      node.parent.totalVisits++;
      if (RAVE_K > 0) updateRave(node.parent, d, won, chooser);
      node = node.parent;
    }
  }

  // Run the search from `game2` and return the populated root.  Shared by getMove
  // (move selection) and valueB (rootWinRatio).
  function runSearch(game2, N, rng, playoutLimit, timeBudgetMs) {
    // Root-decision truncation: this decision truncates iff the root phase is
    // below TRUNC_ROOT_PHASE.  Decided once here, applied to every playout.
    _truncActive = (1 - game2.emptyCount / (N * N)) < TRUNC_ROOT_PHASE;
    // Lockstep Game3 mirror for featurepol feature extraction — built once per
    // decision, then maintained by play/undo across simulations so extraction
    // never rebuilds it.
    const game3 = game3FromGame2(game2);
    const root = makeNode(null, null, -1, game2, N, game3);

    const played = new Float32Array(N * N);

    const deadline = performance.now() + timeBudgetMs;
    let playouts = 0;

    do {
      playouts++;
      const { node, game2: simGame2, path, depth, doPlayout } = selectAndExpand(root, game2, N, rng, game3);
      let value, trace = null, work = PLAYOUT_OVERHEAD;
      if (doPlayout && !simGame2.gameOver) {
        played.fill(0);
        value = playout(simGame2, played, rng);
        trace = played;
        work = PLAYOUT_OVERHEAD + _lastPlayoutPpat + _lastPlayoutUniform * UNIFORM_WEIGHT + (_lastPlayoutTrunc ? TRUNC_OVERHEAD : 0);
      } else {
        // Simulations that end without a playout are at terminal positions
        // (double pass or descent into a finished game) — score them exactly.
        value = simGame2.calcWinner() === BLACK ? 1 : 0;
      }
      for (let i = 0; i < depth; i++) game3.undo();
      backpropagate(node, value, path, trace, work);
    } while (playoutLimit > 0 ? playouts < playoutLimit : performance.now() < deadline);

    return { root, playouts };
  }

  function getMove(game, timeBudgetMs, options = {}) {
    if (game.gameOver) return { type: 'pass', move: PASS, info: 'game already over' };

    const N          = game.cells ? game.N : game.boardSize;
    const game2      = game.cells ? game.clone() : game.toGame2();
    const rootPlayer = game2.current;

    // Prefix length in MOVES, fixed for the whole tree at the start of the
    // turn.  TRUNC_PHASE_DELTA is expressed as a fullness delta only so that
    // one number carries across board sizes; once the size is known the
    // descent should be a constant number of moves, not a fullness check
    // repeated after every move.  Under the old rule a capture pushed the
    // trigger further away and different playouts descended different
    // distances; now every truncated playout plays exactly this many.
    _prefixLen = Math.ceil(TRUNC_PHASE_DELTA * N * N);

    if (game2.consecutivePasses > 0 && game2.calcWinner() === rootPlayer) {
      return { type: 'pass', move: PASS, info: 'obvious pass: already winning', rootWinRatio: 1 };
    }

    const rng = options.rng || makeRng();
    const playoutLimit = options.playoutLimit || PLAYOUTS;
    const { root, playouts } = runSearch(game2, N, rng, playoutLimit, timeBudgetMs);

    const M = root.legalMoves.length;
    let bestIdx = 0, bestVisits = -1, bestScore = -Infinity;
    for (let i = 0; i < M; i++) {
      const cv = root.visits[i];
      if (cv > bestVisits || (cv === bestVisits && nodeScore(i, root, rng) > bestScore)) {
        bestVisits = cv;
        bestScore  = nodeScore(i, root, rng);
        bestIdx    = i;
      }
    }

    // Root statistics for consumers (recorder, analysis): flat move indices.
    const children = [];
    for (let i = 0; i < M; i++) {
      children.push({
        move:   root.legalMoves[i],
        visits: root.visits[i],
        wins:   root.wins[i],
      });
    }
    children.sort((a, b) => b.visits - a.visits);

    let totalChildWins = 0;
    for (let i = 0; i < M; i++) totalChildWins += root.wins[i];
    const rootWinRatio = totalChildWins / root.totalVisits;

    if (playouts >= RESIGN_MIN_PLAYOUTS && game2.emptyCount <= N * N / 2 && root.wins[bestIdx] <= PRIOR_WINS) {
      return { type: 'pass', move: PASS, info: 'no winning line found', children, rootWinRatio };
    }

    const m = root.legalMoves[bestIdx];
    const cv = root.visits[bestIdx];
    const bestWinRatio = cv > 0 ? root.wins[bestIdx] / cv : 0.5;

    const result = m === PASS ? { type: 'pass', move: PASS, children, rootWinRatio }
                              : { type: 'place', move: m, x: m % N, y: (m / N) | 0, children, rootWinRatio };
    result.info = `value=${(game.current===BLACK?bestWinRatio:(1-bestWinRatio)).toFixed(3)}`;
    return result;
  }

  // Search value of a Game2 position as P(BLACK wins) in [0,1], for use as an SB
  // value oracle (matches ref-vlibpat.valueB / mc-vlib.valueB / puct-hybrid.valueB).
  // Runs a full search (PLAYOUTS playouts, default 1000) and returns the root win
  // ratio mapped from the side-to-move perspective to absolute P(BLACK wins).
  function valueB(game, options = {}) {
    const N     = game.cells ? game.N : game.boardSize;
    const game2 = game.cells ? game.clone() : game.toGame2();
    if (game2.gameOver) return game2.calcWinner() === BLACK ? 1 : 0;

    const r = options.rng || makeRng();
    // Fixed PLAYOUTS wins (as in getMove); else an options.budgetMs time
    // budget; else the historical 1000-playout default.
    const budgetMs = options.budgetMs > 0 ? options.budgetMs : 0;
    const playoutLimit = PLAYOUTS > 0 ? PLAYOUTS : (budgetMs > 0 ? 0 : 1000);
    const { root } = runSearch(game2, N, r, playoutLimit, budgetMs);

    let totalChildWins = 0;
    const M = root.legalMoves.length;
    for (let i = 0; i < M; i++) totalChildWins += root.wins[i];
    const rootWinRatio = totalChildWins / root.totalVisits;     // P(side-to-move wins)
    return game2.current === BLACK ? rootWinRatio : 1 - rootWinRatio;
  }

  return { getMove, valueB };
}

function getLegalMoves(game2) {
  const N     = game2.N;
  const cap   = N * N;
  const cells = game2.cells;
  const moves = [];
  for (let i = 0; i < cap; i++) {
    if (cells[i] !== 0) continue;
    if (game2.isTrueEye(i)) continue;
    if (game2.isLegal(i)) moves.push(i);
  }
  if (moves.length < cap / 3 || game2.consecutivePasses > 0) {
    moves.push(PASS);
  }
  return moves;
}

// When K > 0, keep only the K placements with the highest policy probability.
// PASS (if originally in the move list) is always kept as a fallback.
function _pruneToTopK(allMoves, state, K, N) {
  if (K <= 0 || !state) return allMoves;
  const probByMove = new Float64Array(N * N);
  for (let i = 0; i < state.count; i++) probByMove[state.moves[i]] = state.probs[i];
  const placements = [];
  let hasPass = false;
  for (const m of allMoves) {
    if (m === PASS) hasPass = true;
    else placements.push(m);
  }
  placements.sort((a, b) => probByMove[b] - probByMove[a]);
  const top = placements.slice(0, K);
  if (hasPass) top.push(PASS);
  return top;
}

// ── Default instance (lazy) for direct-require / browser callers ───────────────
// Built from plain env on first use, so single-agent callers (gen-agent-evals,
// bench, record-npats) keep working unchanged.  Two-agent callers (selfplay)
// use create(cfg) directly and never build this default.
let _default = null;
function _def() { return _default || (_default = create(Util.makeCfg())); }

if (typeof module !== 'undefined') {
  module.exports = {
    create,
    getMove: (g, b, o) => _def().getMove(g, b, o),
    valueB:  (g, o)    => _def().valueB(g, o),
  };
} else {
  window.create  = create;
  window.getMove = (g, b, o) => _def().getMove(g, b, o);
  window.valueB  = (g, o)    => _def().valueB(g, o);
}

})();
