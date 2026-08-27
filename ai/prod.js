'use strict';

// Production agent (self-contained copy of ai/puct-ppat.js, hardcoded
// config — promoted 2026-08-19, beating the previous prod (rave-npat-prune,
// now ai/prodOldC.js) at equal time: 9×9 83%/100g @100ms, 87%/100g @300ms;
// 13×13 78%/37g @1s, margin growing with budget).
//
// PUCT MCTS with policy-driven priors, top-K candidate pruning at interior
// nodes (the root searches full width), RAVE, and ppat-policy full-playout leaf
// evaluation.  Playout policy: single-phase SB-trained ppat checkpoint
// ppat-data.js (= ppat-data-287076-best, band-trained, score 65.35); in the
// browser, load ppat-lib.js and ppat-data.js (sets window.PPATWeights) first.
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
//   score = Q + cPuct · P(s,a) · √N_total / (1 + N_a)
// where Q is the move's mean backpropagated value (RAVE-blended when raveK
// > 0) and P(s,a) is the policy softmax probability for the move.
//
// ── Factory ──
// The module exports create() → { getMove, valueB }.  All parameters are
// hardcoded — no env vars are read, so every instance is identical.  A lazy
// default instance backs the direct module.getMove / module.valueB exports.

(function () {

const Util = (typeof require === 'function') ? require('../util.js') : window.Util;
const { PASS, BLACK } = Util.load('./game2.js', 'Game2');
const { makeRng }            = Util.load('./xorshift.js', 'XorShift');
const NPat                  = Util.load('./npat-lib.js', 'NPatterns');
const { game3FromGame2 }     = Util.load('./game3.js', 'Game3');
const _ppat                 = Util.load('./ppat-lib.js', 'PPatterns');
const { createState, ppatMove, loadWeights } = _ppat;

const performance = (typeof window !== 'undefined' && window.performance)
  ? window.performance : require('perf_hooks').performance;

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

// ── Config-free constants & helpers (shared across instances) ──────────────────

const priorWins   = 0.001;
const priorVisits = 2 * priorWins;
const resignMinPlayouts = 20000;

// ── Factory ────────────────────────────────────────────────────────────────────

// Build an independent agent instance.  Config is hardcoded above the fold;
// all per-instance state lives in this closure.
function create() {
  // ── Hardcoded configuration ─────────────────────────────────────────────────
  // PUCT exploration constant — weight of the prior P(s,a) relative to Q.
  const cPuct     = 0.5;
  // RAVE blend strength: Q mixes rave/real win-rate with weight raveK/(raveK+n).
  const raveK     = 400;
  // Top-K kept move count (applies only below root).
  const topK     = 40;
  // Fixed playout count per decision; 0 = follow the caller's time budget.
  const fixedPlayouts = 0;
  // Playout moves to use the ppat policy before switching to uniform (-1 = all).
  const ppatMoves = -1;
  // Per-move probability of using ppat (vs uniform) within the ppatMoves window.
  const ppatRatio = 1;

  // ppat playout policy weights: the fixed prod checkpoint (window.PPATWeights
  // in the browser).  ppat model { phaseCount, weights }.
  const _model = _isNode
    ? loadWeights(require('path').join(__dirname, '..', 'ppat-data.js'))
    : loadWeights((typeof window !== 'undefined' && window.PPATWeights) || null);

  // Use uniform-random playout moves while board fullness < this fraction [0,1]
  // (0 = off).  Skips ppat feature extraction in the early game, where the
  // policy is ≈ uniform.
  if (_model) _model.uniformBelowPhase = 0;

  // npat policy model (priors + top-K pruning), from the canonical npat-data.js.
  const npatModel   = NPat.loadModel({ name: 'prod' });
  const npatWeights = npatModel.weights;

  let _ppatState = null;
  function _ensurePpatState(N) {
    if (_ppatState === null || _ppatState.moves.length < N * N)
      _ppatState = createState(N);
  }

  const stateByN = new Map();

  // Run npat policy extraction + softmax for `game2`.  `game3` is the lockstep
  // mirror (maintained by the search, so extraction skips the game3FromGame2
  // rebuild).  Returns the shared state (moves/probs populated) or null.
  function _runNpat(game2, game3) {
    if (!npatWeights || game2.gameOver) return null;
    const N = game2.N;
    let state = stateByN.get(N);
    if (!state) { state = NPat.createState(N); stateByN.set(N, state); }
    NPat.extractFeatures(game2, state, undefined, game3, npatWeights);
    if (state.count === 0) return null;
    NPat.computeSoftmax(state, npatWeights);
    return state;
  }

  // ppat-policy playout from `game2` to the end of the game (mutates it).  Fills
  // `played` (pre-zeroed by the caller) with the colour-signed first-occupancy
  // RAVE trace.  Returns the result as P(BLACK wins) ∈ {0,1}.
  function playout(game2, played, rng) {
    const N   = game2.N;
    const cap = N * N;

    _ensurePpatState(N);

    const moveLimit = 3 * game2.emptyCount + 20;
    const weightStep = 1 / cap;
    let moves = 0;
    let weight = 1.0;

    while (!game2.gameOver && moves < moveLimit) {
      const current = game2.current;
      // usePolicy: use the ppat policy this move — within the ppatMoves window
      // and (subject to ppatRatio) not a randomly-mixed uniform move.
      const ppatActive = _model && (ppatMoves < 0 || moves < ppatMoves);
      const usePolicy  = ppatActive && (ppatRatio >= 1 || rng.random() < ppatRatio);
      const idx = usePolicy ? ppatMove(game2, _ppatState, _model, rng)
                            : game2.randomLegalMove(rng);
      if (idx !== PASS && weight > 0 && played[idx] === 0) {
        played[idx] = current === BLACK ? weight : -weight;
      }
      game2.play(idx);
      moves++;
      weight -= weightStep;
    }

    return game2.estimateWinner() === BLACK ? 1 : 0;
  }

  function makeNode(move, parent, ci, game2, N, game3) {
    // Compute the policy softmax once — reused for top-K pruning and the PUCT priors.
    const npatState = _runNpat(game2, game3);
    let movesArr = getLegalMoves(game2);
    // Top-K pruning at interior nodes only: at the root it would constrain the
    // actual decision, and the root gets enough visits to search full width.
    if (topK > 0 && parent !== null) {
      movesArr = _pruneToTopK(movesArr, npatState, topK, N);
    }
    const M = movesArr.length;
    const area = N * N;

    const legalMoves = new Int32Array(M);
    for (let i = 0; i < M; i++) legalMoves[i] = movesArr[i];

    const children   = new Array(M).fill(null);
    const wins       = new Float32Array(M).fill(priorWins);
    const visits     = new Float32Array(M).fill(priorVisits);
    const raveWins   = raveK > 0 ? new Float32Array(area).fill(priorWins)   : null;
    const raveVisits = raveK > 0 ? new Float32Array(area).fill(priorVisits) : null;

    // PUCT priors per move (renormalised to sum to 1).  PASS gets a 1/area floor;
    // all other entries take the softmax probability directly; then renormalise.
    const priors = new Float32Array(M);
    if (npatState) {
      const probByMove = new Float64Array(area);
      for (let i = 0; i < npatState.count; i++) probByMove[npatState.moves[i]] = npatState.probs[i];
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

    if (raveK > 0) {
      const move   = node.legalMoves[moveIdx];
      const raveWR = (move === PASS) ? 0 : (node.raveWins[move] / node.raveVisits[move]);
      const beta   = raveK / (raveK + node.visits[moveIdx]);
      Q = (1 - beta) * Q + beta * raveWR;
    }

    // PUCT exploration term.  cPuct · P(s,a) · √N_total / (1 + N_a).
    const P = node.priors[moveIdx];
    const U = cPuct * P * Math.sqrt(node.totalVisits) / (1 + node.visits[moveIdx]);

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

      // Expand on first contact: create the leaf node, descend into it, and run
      // one playout from it.  No lazy-expansion threshold — a ppat playout costs
      // far more than node creation, so a new node is always worthwhile.
      if (node.children[best] === null) {
        node.children[best] = makeNode(move, node, best, game2, N, game3);
        node = node.children[best];
        node.selectedChild = -1;
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
      if (raveK > 0) updateRave(node, d, won, chooser);
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
      if (raveK > 0) updateRave(node.parent, d, won, chooser);
      node = node.parent;
    }
  }

  // Run the search from `game2` and return the populated root.  Shared by getMove
  // (move selection) and valueB (rootWinRatio).
  function runSearch(game2, N, rng, playoutLimit, timeBudgetMs) {
    // Lockstep Game3 mirror for npat feature extraction — built once per
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
    const playoutLimit = options.playoutLimit || fixedPlayouts;
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

    if (playouts >= resignMinPlayouts && game2.emptyCount <= N * N / 2 && root.wins[bestIdx] <= priorWins) {
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
  // Runs a full search (playouts playouts, default 1000) and returns the root win
  // ratio mapped from the side-to-move perspective to absolute P(BLACK wins).
  function valueB(game, options = {}) {
    const N     = game.cells ? game.N : game.boardSize;
    const game2 = game.cells ? game.clone() : game.toGame2();
    if (game2.gameOver) return game2.calcWinner() === BLACK ? 1 : 0;

    const r = options.rng || makeRng();
    const playoutLimit = fixedPlayouts > 0 ? fixedPlayouts : 1000;
    const { root } = runSearch(game2, N, r, playoutLimit, 0);

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
// create() takes no config, so the default instance and any create()d
// instance are identical.
let _default = null;
function _def() { return _default || (_default = create()); }

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
