'use strict';

// PUCT MCTS with policy-driven priors, top-K candidate pruning at interior
// nodes (the root searches full width), RAVE, and full-playout leaf
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
// > 0) and P(s,a) is the policy softmax probability for the move (PASS and
// out-of-extract moves get a small uniform floor, then priors are
// renormalised to sum to 1).
//
// In the browser, set window.npatsModel to the loaded weights before
// requiring.
//
// The policy is consumed two ways at each new node:
//   1. Top-K candidate pruning — keep only the K highest-probability
//      placements (PASS, if originally legal, is always retained).
//   2. PUCT prior — `node.priors[i]` is the softmax probability that drives
//      exploration in the PUCT score above.

(function () {

const Util = (typeof require === 'function') ? require('../util.js') : window.Util;
const { PASS, BLACK } = Util.load('./game2.js', 'Game2');
const { makeRng }            = Util.load('./xorshift.js', 'XorShift');
const NPats                 = Util.load('./npats-lib.js', 'NPatsLib');
const { game3FromGame2 }     = Util.load('./game3.js', 'Game3');
const _ppat                 = Util.load('./ppat-lib.js', 'PPatterns');
const { createState, ppatMove, loadWeights } = _ppat;

const performance = (typeof window !== 'undefined' && window.performance)
  ? window.performance : require('perf_hooks').performance;

// ppat playout policy weights (PPAT_DATA env, or window.PPATWeights in browser).
// When absent, playouts fall back to uniform-random — same trace/scoring path.
const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
const _weightsArr = _isNode
  ? loadWeights(Util.envStr('PPAT_DATA', ''))
  : loadWeights((typeof window !== 'undefined' && window.PPATWeights) || null);
// Captured AFTER loadWeights so it reflects whether the loaded model carries
// ladder features.  When false, the Game3 mirror is never touched in playouts.
const _LADDER_ENABLED = !!(_weightsArr && _ppat.LADDER_ENABLED);
console.log(`puct-ppat: loaded ${_weightsArr ? _weightsArr.length : 0} ppat weights (ladder: ${_LADDER_ENABLED})`);

let _ppatState = null;
function _ensurePpatState(N) {
  if (_ppatState === null || _ppatState.moves.length < N * N)
    _ppatState = createState(N);
}

// ── Hardcoded configuration ──────────────────────────────────────────────────

// PUCT exploration constant — controls how much weight the prior P(s,a) has
// relative to Q.  Equivalent of EXPLORATION_C from UCB1; tuned higher here
// because PUCT exploration scales with √N (not √log N) and is multiplied by
// per-move priors that sum to ≈ 1.
const C_PUCT        = Util.envFloat('C_PUCT', 0.5);

// RAVE.  When RAVE_K > 0, every node keeps AMAF stats fed from the full
// simulation from that node onward: the in-tree segment is depth-aligned (a
// node at depth d is credited only with its chooser's moves from depth d on,
// every second path entry), and playout simulations additionally contribute
// their colour-signed first-occupancy trace (all playout moves follow every
// tree node, so one trace serves all ancestors).  Q blends rave/real
// win-rate with weight RAVE_K / (RAVE_K + n).  0 disables.
const RAVE_K        = Util.envFloat('RAVE_K', 400);

const NPATS_K = Util.envInt('NPATS_K', 40);  // kept move count (applies only below root).

const PRIOR_WINS   = 0.001;
const PRIOR_VISITS = 2 * PRIOR_WINS;

const RESIGN_MIN_PLAYOUTS = 20000;

// Fixed playout count per decision.  When non-zero, overrides the time budget.
const PLAYOUTS = Util.envInt('PLAYOUTS', 0);

// Number of playout moves to use the ppat policy before switching to uniform.
// -1 = unlimited (ppat for the entire playout).
const PPAT_MOVES = Util.envInt('PPAT_MOVES', -1);

// Per-move probability of using the ppat policy (vs a uniform-random move) during
// playouts, within the PPAT_MOVES window.  1 = always ppat (default).  e.g. 0.5
// mixes in 50% uniform moves: cheaper (skips ppat feature extraction half the
// time) and injects playout variety.
const PPAT_RATIO = Util.envFloat('PPAT_RATIO', 1);

// NPats model: NPATS_WEIGHTS env overrides the canonical npats-data.js.
// Exported (Node) so record-npats.js can train these weights in-process:
// computeProbs reads them live, so updates take effect on the next node.
const npatsModel = NPats.loadModel({
  name: 'puct',
  path: (typeof process !== 'undefined' && process.env.NPATS_WEIGHTS) || undefined,
});
console.log(`puct: loaded ${npatsModel.weights.size} npats weights from ${npatsModel.modelName}`);

const stateByN = new Map();

// Run policy extraction + softmax for `game2`.  `game3` is the lockstep
// mirror of the same position (maintained by the search, so extraction skips
// the per-call game3FromGame2 rebuild).  Returns the shared state (with
// state.moves and state.probs populated) or null if not applicable.
function _runNpats(game2, game3) {
  if (game2.gameOver) return null;
  const N = game2.N;
  let state = stateByN.get(N);
  if (!state) { state = NPats.createState(N); stateByN.set(N, state); }
  const n = NPats.computeProbs(game2, state, game3, npatsModel);
  return n === 0 ? null : state;
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

// ── Playout ───────────────────────────────────────────────────────────────────

// ppat-policy playout from `game2` to the end of the game (mutates it).  Fills
// `played` (pre-zeroed by the caller) with the colour-signed first-occupancy
// RAVE trace: +weight if BLACK played the point first, −weight if WHITE, with
// weight decaying over the playout.  Returns the result as P(BLACK wins) ∈ {0,1}.
//
// `game3` is the persistent lockstep mirror, already in sync with `game2` at the
// leaf.  When the loaded model has ladder features, ppat needs it for ladder
// analysis, so each ppat-window move is mirrored in; those mirror plays are
// undone before return, leaving `game3` exactly as it was at entry (the caller's
// descent-rewind is therefore unaffected).
function playout(game2, played, rng, game3) {
  const N   = game2.N;
  const cap = N * N;

  _ensurePpatState(N);
  const g3 = _LADDER_ENABLED ? game3 : null;   // ppat ladder mirror, or unused

  const moveLimit = 3 * game2.emptyCount + 20;
  const weightStep = 1 / cap;
  let moves = 0;
  let weight = 1.0;
  let g3Plays = 0;

  while (!game2.gameOver && moves < moveLimit) {
    const current = game2.current;
    // ppatActive: ppat is still within its temporal window this move — the g3
    // mirror must stay synced while it is (even on uniform moves), since ppat may
    // be chosen again.  usePolicy: actually use ppat this move (subject to PPAT_RATIO).
    const ppatActive = _weightsArr && (PPAT_MOVES < 0 || moves < PPAT_MOVES);
    const usePolicy  = ppatActive && (PPAT_RATIO >= 1 || rng.random() < PPAT_RATIO);
    const idx = usePolicy ? ppatMove(game2, _ppatState, _weightsArr, g3)
                          : game2.randomLegalMove(rng);
    if (idx !== PASS && weight > 0 && played[idx] === 0) {
      played[idx] = current === BLACK ? weight : -weight;
    }
    game2.play(idx);
    if (g3 && ppatActive) { g3.play(idx); g3Plays++; }
    moves++;
    weight -= weightStep;
  }

  for (let i = 0; i < g3Plays; i++) game3.undo();   // restore mirror to the leaf

  return game2.estimateWinner() === BLACK ? 1 : 0;
}

// ── Tree node ─────────────────────────────────────────────────────────────────

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

function makeNode(move, parent, ci, game2, N, game3) {
  // Compute the policy softmax once — reused for top-K pruning and the PUCT
  // priors.
  const npatsState = _runNpats(game2, game3);
  let movesArr = getLegalMoves(game2);
  // Top-K pruning at interior nodes only: at the root it would constrain the
  // actual decision, and the root gets enough visits to search full width.
  // With an empty (generation-0) model the priors are uniform, so pruning
  // would keep an arbitrary subset — disabled until the model has weights
  // (checked per node: in-process training can populate it mid-run).
  if (NPATS_K > 0 && parent !== null && npatsModel.weights.size > 0) {
    movesArr = _pruneToTopK(movesArr, npatsState, NPATS_K, N);
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

  // PUCT priors per move (renormalised to sum to 1).  The policy softmax
  // covers every legal non-true-eye placement — i.e. all placements in
  // legalMoves — but PASS is never in its output.  PASS gets a 1/area floor;
  // all other entries take the softmax probability directly; then
  // renormalise.
  const priors = new Float32Array(M);
  if (npatsState) {
    const probByMove = new Float64Array(area);
    for (let i = 0; i < npatsState.count; i++) probByMove[npatsState.moves[i]] = npatsState.probs[i];
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

  const node = {
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

  return node;
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

// `game3` is the lockstep mirror of rootGame2's position: every move played
// on the game2 clone is also played on it, and `depth` (the number of game3
// plays) is returned so the caller can undo back to the root position.
function selectAndExpand(root, rootGame2, N, rng, game3) {
  let node = root;
  const game2 = rootGame2.clone();
  let depth = 0;
  let doPlayout = false;   // true when the simulation ends in a playout visit

  // Simulation path: path[d] is the move chosen by the node at depth d.
  // PASS entries are kept so depth/colour alignment survives; the RAVE update
  // skips them.  (The forced second PASS below is not a chosen move and is
  // not recorded.)
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

  // RAVE update for node `n` at depth `d`: credit every move its chooser
  // played from depth d onward — the in-tree segment (every second path
  // entry, colours alternate) plus, when a playout ran, its trace (all
  // playout moves follow every tree node, so the one trace serves all).
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

// ── Public interface ──────────────────────────────────────────────────────────

// Run the search from `game2` and return the populated root.  Shared by
// getMove (move selection) and value (rootWinRatio).
function runSearch(game2, N, rng, playoutLimit, timeBudgetMs) {
  // Lockstep Game3 mirror — built once per decision, then maintained by
  // play/undo across simulations so feature extraction never rebuilds it.
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
      value = playout(simGame2, played, rng, game3);
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
  const playoutLimit = PLAYOUTS > 0 ? PLAYOUTS : 1000;
  const { root } = runSearch(game2, N, r, playoutLimit, 0);

  let totalChildWins = 0;
  const M = root.legalMoves.length;
  for (let i = 0; i < M; i++) totalChildWins += root.wins[i];
  const rootWinRatio = totalChildWins / root.totalVisits;     // P(side-to-move wins)
  return game2.current === BLACK ? rootWinRatio : 1 - rootWinRatio;
}

if (typeof module !== 'undefined') module.exports = { getMove, valueB, npatsModel };
else { window.getMove = getMove; window.valueB = valueB; }

})();
