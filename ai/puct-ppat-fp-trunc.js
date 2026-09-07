'use strict';

// puct-ppat-fp-trunc: puct-ppat-fp with TRUNCATED playouts.  A playout runs
// until board fullness has advanced by TRUNC_PHASE_DELTA past the leaf; if
// the position's phase is then below TRUNC_MAX_PHASE, the playout stops and
// the leaf value is a static vpatterns evaluation (TRUNC_VPAT_DATA, a
// train-vpat-playout-eval checkpoint: V(s) = P(BLACK wins)) instead of the
// terminal score.  Otherwise the playout continues to the end as usual.
// The prefix moves still fill the RAVE trace either way.  (An averaged
// multi-position evaluation window was tried and removed: consecutive-
// position evaluator errors are near-perfectly correlated, so averaging
// bought nothing offline or live — 2026-09-04.)
//
// The truncation check is integer (empty-count drop >= ceil(delta*area)), so
// captures during the prefix delay it correctly: the point is defined by net
// board-filling progress, not move count.
//
// PUCT MCTS with policy-driven priors, top-K candidate pruning at interior
// nodes (the root searches full width unless ROOT_TOP_K caps it), RAVE, and
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
const { PASS, BLACK } = Util.load('./game2.js', 'Game2');
const { makeRng }            = Util.load('./xorshift.js', 'XorShift');
const FeaturePol            = Util.load('./featurepol-lib.js', 'FeaturePol');
const { game3FromGame2 }     = Util.load('./game3.js', 'Game3');
const _ppat                 = Util.load('./ppat-lib.js', 'PPatterns');
const VPat                  = Util.load('./vpatterns.js', 'VPatterns');
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

  // PUCT exploration constant — weight of the prior P(s,a) relative to Q.
  const C_PUCT     = cfg.float('C_PUCT', 0.5);
  // RAVE blend strength: Q mixes rave/real win-rate with weight RAVE_K/(RAVE_K+n).
  const RAVE_K     = cfg.float('RAVE_K', 400);
  // Top-K kept move count (applies only below root).
  const TOP_K     = cfg.int('TOP_K', 40);
  // Root candidate cap: keep only the policy's top K at the ROOT (0 = all,
  // the classic full-width root).
  const ROOT_TOP_K = cfg.int('ROOT_TOP_K', 0);
  // Lazy expansion: an edge must accumulate this many visits before its child
  // node (featurepol extraction + priors) is created; playouts before that
  // run from the unexpanded position.  1 = expand on first contact (the
  // original economics, tuned for cheap npat extraction).  Featurepol
  // extraction is pricier, so 2 skips the extraction for the many leaves
  // that are only ever visited once.
  const N_EXPAND   = cfg.int('N_EXPAND', 2);
  // Fixed playout count per decision; when non-zero, overrides the time budget.
  const PLAYOUTS   = cfg.int('PLAYOUTS', 0);
  // Playout moves to use the ppat policy before switching to uniform (-1 = all).
  const PPAT_MOVES = cfg.int('PPAT_MOVES', -1);
  // Per-move probability of using ppat (vs uniform) within the PPAT_MOVES window.
  const PPAT_RATIO = cfg.float('PPAT_RATIO', 1);
  // Truncation point: net board-fullness advance past the leaf before the
  // playout stops for a static evaluation (board-size invariant).  The
  // default 1 can never be reached, so out of the box every playout runs to
  // the end — plain puct-ppat-fp behaviour until the knobs are set.
  const TRUNC_PHASE_DELTA = cfg.float('TRUNC_PHASE_DELTA', 1);
  // Truncate only when the position's phase at the truncation point is below
  // this; at or above it the playout runs to the end (late playouts are short
  // and nearly exact, so substitution there is pure downside).
  const TRUNC_MAX_PHASE   = cfg.float('TRUNC_MAX_PHASE', 1);
  // Gate ramp: truncation probability is 1 at or below _A, 0 at or above _B,
  // linear between (drawn once per playout, anchored on the leaf — the
  // endpoint phase is leaf + delta by construction).  Both default to
  // TRUNC_MAX_PHASE, i.e. the hard cliff.  The ramp exists to smooth the
  // truncate/no-truncate currency seam between sibling branches: a k-stone
  // capture shifts the leaf by (k-1)/cap of phase, which across a hard gate
  // flips the subtree's value source outright.
  // TRUNC_MAX_PHASE_A also accepts the sentinel 'auto': the ramp start is
  // then dynamic, the midpoint of (rootPhase + TRUNC_PHASE_DELTA) — the
  // minimum endpoint this decision can produce — and _B.  Shallow leaves
  // (the bulk of the frontier) sit on the p=1 plateau, so the truncation
  // throughput survives; only the deeper half of the reachable range ramps
  // toward full-playout (unbiased) returns.  (The full-span variant, ramp
  // start at root+delta itself, measured a clear loss at budget 500 —
  // 2026-09-06: too many full playouts across the whole zone.)
  const _gateARaw = cfg.str('TRUNC_MAX_PHASE_A', '');
  const GATE_A_AUTO = _gateARaw === 'auto';
  const TRUNC_MAX_PHASE_A = GATE_A_AUTO ? NaN
    : _gateARaw !== '' ? parseFloat(_gateARaw) : TRUNC_MAX_PHASE;
  const TRUNC_MAX_PHASE_B = cfg.float('TRUNC_MAX_PHASE_B', TRUNC_MAX_PHASE);
  if (!GATE_A_AUTO && !(TRUNC_MAX_PHASE_A <= TRUNC_MAX_PHASE_B)) {
    throw new Error(`puct-ppat-fp-trunc[${cfg.slot != null ? cfg.slot : '-'}]: ` +
      `TRUNC_MAX_PHASE_A (${_gateARaw || TRUNC_MAX_PHASE}) must be 'auto' or a number <= TRUNC_MAX_PHASE_B (${TRUNC_MAX_PHASE_B})`);
  }
  // Measured deployment correction for the truncated evals (step 2 of the
  // komi/bias program): the evaluator's lean vs the deployed playout
  // currency, as a linear function of the EVAL-POINT phase — measured per
  // model by measure-trunc-bias / score-bias-curve.  "a,b" means
  // offset(ph) = a + b*ph in win-probability units; it is applied as a
  // LOGIT shift (4*offset, the slope match at v = 0.5), whose natural
  // attenuation at extreme values matches the komi effect shrinking in
  // decided positions.  Empty/off by default (bit-identical).
  const _voRaw = cfg.str('TRUNC_VALUE_OFFSET', '');
  let VO_A = 0, VO_B = 0, VO_ON = false;
  if (_voRaw !== '') {
    const parts = _voRaw.split(',').map(parseFloat);
    if (parts.length < 1 || parts.length > 2 || parts.some(x => !Number.isFinite(x))) {
      throw new Error(`puct-ppat-fp-trunc[${cfg.slot != null ? cfg.slot : '-'}]: ` +
        `TRUNC_VALUE_OFFSET must be "a" or "a,b" (offset = a + b*phase), got "${_voRaw}"`);
    }
    VO_A = parts[0]; VO_B = parts.length === 2 ? parts[1] : 0; VO_ON = true;
  }

  // Cap on the truncation probability: even where the gate would give p = 1,
  // at most this fraction of playouts truncate — the rest run full, keeping
  // an unbiased playout component in every node's value.  The evaluator's
  // variance edge depreciates with budget while its bias doesn't; the cap
  // buys bias anchoring at a throughput price, so it should earn its keep at
  // high budgets if anywhere.  1 = no cap (default, rng stream untouched).
  const TRUNC_MAX_RATIO = cfg.float('TRUNC_MAX_RATIO', 1);
  if (!(TRUNC_MAX_RATIO > 0 && TRUNC_MAX_RATIO <= 1)) {
    throw new Error(`puct-ppat-fp-trunc[${cfg.slot != null ? cfg.slot : '-'}]: ` +
      `TRUNC_MAX_RATIO (${TRUNC_MAX_RATIO}) must be in (0, 1]`);
  }
  // Ramp start used by playout(); in auto mode runSearch refreshes it per
  // decision from the root position.
  let _gateA = TRUNC_MAX_PHASE_A;

  // Static evaluator: a vpatterns checkpoint (train-vpat-playout-eval).
  // Hard failure, not a fallback — this agent's identity IS its truncated
  // playouts, and a silently-missing evaluator would field plain puct-ppat-fp
  // under the wrong name.
  const _vpatPath = _isNode ? cfg.str('TRUNC_VPAT_DATA', '') : null;
  if (_isNode && !_vpatPath) {
    throw new Error(`puct-ppat-fp-trunc[${cfg.slot != null ? cfg.slot : '-'}]: TRUNC_VPAT_DATA is required`);
  }
  const _vpatRaw = _isNode
    ? require(require('path').resolve(_vpatPath))
    : (typeof window !== 'undefined' && window.truncVpatModel) || null;
  if (!_vpatRaw) {
    throw new Error(`puct-ppat-fp-trunc[${cfg.slot != null ? cfg.slot : '-'}]: cannot load vpatterns evaluator from ` +
      (_isNode ? 'TRUNC_VPAT_DATA' : 'window.truncVpatModel'));
  }
  const _vpatWeights = VPat.makeWeights(Math.max(1024, (_vpatRaw.weights.size ?? _vpatRaw.weights.length) * 2));
  for (const [k, v] of _vpatRaw.weights) _vpatWeights.set(k, v);
  const _vpatModel = { specs: _vpatRaw.specs,
                       preparedSpecs: VPat.prepareSpecs(_vpatRaw.specs),
                       weights: _vpatWeights };
  // Name the evaluator file and the truncation knobs in the banner: two slots
  // (P1_/P2_TRUNC_*) otherwise print identical lines, hiding which evaluator
  // and gate each side is actually running.
  const _vpatName = _isNode ? require('path').basename(_vpatPath) : 'window.truncVpatModel';
  console.log(`puct-ppat-fp-trunc[${cfg.slot != null ? cfg.slot : '-'}]: ` +
    `${_vpatModel.weights.size} vpat weights (${_vpatModel.specs.map(sp => `${sp.size}:${sp.maxLibs === 0 ? 'L' : sp.maxLibs}`).join(',')}) from ${_vpatName}, ` +
    `trunc-phase-delta: ${TRUNC_PHASE_DELTA}, trunc-max-phase: ` +
    (GATE_A_AUTO ? `auto(mid(root+delta,B))..${TRUNC_MAX_PHASE_B} (ramp)`
     : TRUNC_MAX_PHASE_A === TRUNC_MAX_PHASE_B ? `${TRUNC_MAX_PHASE_A}`
     : `${TRUNC_MAX_PHASE_A}..${TRUNC_MAX_PHASE_B} (ramp)`) +
    (TRUNC_MAX_RATIO < 1 ? `, trunc-max-ratio: ${TRUNC_MAX_RATIO}` : '') +
    (VO_ON ? `, trunc-value-offset: ${VO_A}${VO_B !== 0 ? `${VO_B >= 0 ? '+' : ''}${VO_B}*ph` : ''}` : ''));

  // Static value of `game2`: P(BLACK wins) from the vpatterns evaluator.
  function vpatValueB(game2) {
    const f = VPat.extractFeatures(game2, _vpatModel.preparedSpecs);
    const v = VPat.evaluateFeatures(f, _vpatModel.weights);
    if (!VO_ON) return v;
    const cap = game2.N * game2.N;
    const ph = (cap - game2.emptyCount) / cap;
    return 1 / (1 + Math.exp(-(f.z - 4 * (VO_A + VO_B * ph))));
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
    throw new Error(`puct-ppat-fp-trunc[${cfg.slot != null ? cfg.slot : '-'}]: cannot load ppat weights from ` +
      (_isNode ? _ppatPath : 'window.PPATWeights'));
  }
  // Name the ppat file in the banner: two slots (P1_/P2_PPAT_DATA) otherwise
  // print identical lines, hiding which model each side is actually running.
  const _ppatName = _isNode ? require('path').basename(_ppatPath) : 'window.PPATWeights';

  // Use uniform-random playout moves while board fullness < this fraction [0,1]
  // (0 = off).  Skips ppat feature extraction in the early game, where the
  // policy is ≈ uniform.
  _model.uniformBelowPhase = cfg.float('PPAT_MIN_PHASE', 0.6);

  // featurepol policy model (priors + top-K pruning).  FPOL_DATA overrides
  // the default checkpoint (browser: window.featurepolModel).
  const _isBrowser  = typeof window !== 'undefined';
  const fpModel     = FeaturePol.loadModel({ name: 'puct-ppat-fp-trunc',
    path: _isBrowser ? undefined
                     : cfg.str('FPOL_DATA', require('path').join(__dirname, '..', 'featurepol-cbk7wa32.js')) });
  const fpWeights   = fpModel.weights;
  console.log(`puct-ppat-fp-trunc[${cfg.slot != null ? cfg.slot : '-'}]: ${_model.weights.length} ppat weights from ${_ppatName}, ${fpWeights.size} featurepol weights from ${fpModel.modelName}`);

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
    FeaturePol.computeSoftmax(state, fpWeights);
    return state;
  }

  // ppat-policy playout from `game2` (mutates it), truncated: once board
  // fullness has advanced by TRUNC_PHASE_DELTA, a position still below
  // TRUNC_MAX_PHASE returns the static vpatterns value; otherwise the playout
  // runs to the end.  Fills `played` (pre-zeroed by the caller) with the
  // colour-signed first-occupancy RAVE trace.  Returns P(BLACK wins) — a
  // fraction at a truncation, {0,1} at the end of a full playout.
  function playout(game2, played, rng) {
    const N   = game2.N;
    const cap = N * N;

    _ensurePpatState(N);

    // Truncation trigger, in integer empties (phase = 1 - empty/cap): fires
    // once, at the first position whose net empty-count drop reaches the
    // delta.  Checked after each move; captures push it further away.
    const truncEmpty = game2.emptyCount - Math.ceil(TRUNC_PHASE_DELTA * cap);
    // Gate, decided up front from the leaf (the endpoint phase is fixed at
    // playout start: the trigger fires at exactly truncEmpty empties).
    // p = 1 at/below _A, 0 at/above _B, linear ramp between; the cliff
    // (_A === _B) takes the no-draw paths, leaving the rng stream untouched.
    const epPhase = (cap - truncEmpty) / cap;
    let truncArmed;
    if (epPhase >= TRUNC_MAX_PHASE_B) truncArmed = false;
    else if (epPhase <= _gateA)       truncArmed = TRUNC_MAX_RATIO >= 1 || rng.random() < TRUNC_MAX_RATIO;
    else truncArmed = rng.random() <
      TRUNC_MAX_RATIO * (TRUNC_MAX_PHASE_B - epPhase) / (TRUNC_MAX_PHASE_B - _gateA);

    const moveLimit = 3 * game2.emptyCount + 20;
    const weightStep = 1 / cap;
    let moves = 0;
    let weight = 1.0;

    while (!game2.gameOver && moves < moveLimit) {
      const current = game2.current;
      // usePolicy: use the ppat policy this move — within the PPAT_MOVES window
      // and (subject to PPAT_RATIO) not a randomly-mixed uniform move.
      const ppatActive = PPAT_MOVES < 0 || moves < PPAT_MOVES;
      const usePolicy  = ppatActive && (PPAT_RATIO >= 1 || rng.random() < PPAT_RATIO);
      const idx = usePolicy ? ppatMove(game2, _ppatState, _model, rng)
                            : game2.randomLegalMove(rng);
      if (idx !== PASS && weight > 0 && played[idx] === 0) {
        played[idx] = current === BLACK ? weight : -weight;
      }
      game2.play(idx);
      moves++;
      weight -= weightStep;
      if (truncArmed && game2.emptyCount <= truncEmpty) {
        // The gate itself was decided at playout start (leaf-anchored draw).
        if (!game2.gameOver) return vpatValueB(game2);
        truncArmed = false;
      }
    }

    return game2.estimateWinner() === BLACK ? 1 : 0;
  }

  function makeNode(move, parent, ci, game2, N, game3) {
    // Compute the policy softmax once — reused for top-K pruning and the PUCT priors.
    const fpState = _runFp(game2, game3);
    let movesArr = getLegalMoves(game2);
    // Top-K pruning: TOP_K below the root; at the root, full width unless
    // ROOT_TOP_K caps the actual decision's candidate set.
    const k = parent !== null ? TOP_K : ROOT_TOP_K;
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
      // edge has N_EXPAND visits; before that, run the playout from the
      // unexpanded position with stats accumulating on the parent's edge
      // (same backprop shape as the pass-break case above).
      if (node.children[best] === null) {
        if (node.visits[best] >= N_EXPAND - 1 + PRIOR_VISITS - 1e-9) {
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
  function backpropagate(node, value, path, played) {
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
      node.parent.totalVisits++;
      if (RAVE_K > 0) updateRave(node.parent, d, won, chooser);
      node = node.parent;
    }
  }

  // Run the search from `game2` and return the populated root.  Shared by getMove
  // (move selection) and valueB (rootWinRatio).
  function runSearch(game2, N, rng, playoutLimit, timeBudgetMs) {
    // Root-anchored ramp start: the minimum endpoint phase this decision can
    // produce (see TRUNC_MAX_PHASE_A='auto').
    if (GATE_A_AUTO) {
      const minEp = (N * N - game2.emptyCount) / (N * N) + TRUNC_PHASE_DELTA;
      _gateA = (minEp + TRUNC_MAX_PHASE_B) / 2;
    }
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
      let value, trace = null;
      if (doPlayout && !simGame2.gameOver) {
        played.fill(0);
        value = playout(simGame2, played, rng);
        trace = played;
      } else {
        // Simulations that end without a playout are at terminal positions
        // (double pass or descent into a finished game) — score them exactly.
        value = simGame2.calcWinner() === BLACK ? 1 : 0;
      }
      for (let i = 0; i < depth; i++) game3.undo();
      backpropagate(node, value, path, trace);
    } while (playoutLimit > 0 ? playouts < playoutLimit : performance.now() < deadline);

    return { root, playouts };
  }

  function getMove(game, timeBudgetMs, options = {}) {
    if (game.gameOver) return { type: 'pass', move: PASS, info: 'game already over' };

    const N          = game.cells ? game.N : game.boardSize;
    const game2      = game.cells ? game.clone() : game.toGame2();
    const rootPlayer = game2.current;

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
