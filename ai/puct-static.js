'use strict';

// PUCT MCTS with npat-driven priors and top-K candidate pruning at every
// interior node (the root searches full width) — with the random playout
// replaced by a static vlibpat evaluation of the leaf position.  Pure AlphaZero-style search by default; an optional
// tree-RAVE term can be enabled with RAVE_K (see configuration).
//
// Each simulation descends the tree and then values the reached position with
// vlibpat (P(BLACK wins)) instead of rolling out to the end; terminal
// positions are scored exactly.  The value is backpropagated fractionally:
// each chooser is credited value (BLACK) or 1 − value (WHITE).
//
// Nodes expand on first visit (no N_EXPAND delay): the static value is
// deterministic, so revisiting an unexpanded leaf would re-add the same value
// and learn nothing — each simulation grows the tree by exactly one node.
//
// Node selection:
//   score = Q + C_PUCT · P(s,a) · √N_total / (1 + N_a)
// where Q is the move's mean backpropagated value and P(s,a) is the npat
// softmax probability for the move (PASS and out-of-extract moves get a
// small uniform floor, then priors are renormalised to sum to 1).
//
// In the browser, set window.npatModel and window.vlibpatModel to the loaded
// weights before requiring.
//
// npat is consumed two ways at each new node:
//   1. Top-K candidate pruning — keep only the K highest-probability
//      placements (PASS, if originally legal, is always retained).
//   2. PUCT prior — `node.priors[i]` is the softmax probability that drives
//      exploration in the PUCT score above.

(function () {

const Util = (typeof require === 'function') ? require('../util.js') : window.Util;
const { PASS, BLACK } = Util.load('./game2.js', 'Game2');
const { makeRng }            = Util.load('./xorshift.js', 'XorShift');
const NPat                   = Util.load('./npat-lib.js', 'NPatterns');
const { game3FromGame2 }     = Util.load('./game3.js', 'Game3');
const VLib                   = Util.load('./vlibpat.js', 'VlibPat');
const { getAllLadderStatuses } = Util.load('./ladder2.js', 'Ladder2');

const performance = (typeof window !== 'undefined' && window.performance)
  ? window.performance : require('perf_hooks').performance;

// ── Hardcoded configuration ──────────────────────────────────────────────────

// PUCT exploration constant — controls how much weight the prior P(s,a) has
// relative to Q.  Equivalent of EXPLORATION_C from UCB1; tuned higher here
// because PUCT exploration scales with √N (not √log N) and is multiplied by
// per-move priors that sum to ≈ 1.
const C_PUCT        = Util.envFloat('C_PUCT', 0.5);

// Optional tree-RAVE.  When RAVE_K > 0, every node keeps AMAF stats fed from
// the in-tree simulation path — unlike playout RAVE, the update differs per
// node: a node at depth d is credited only with the moves played from depth d
// onward (its chooser's moves, i.e. every second path entry).  Q then blends
// rave/real win-rate with weight RAVE_K / (RAVE_K + n).  0 disables.
const RAVE_K        = Util.envFloat('RAVE_K', 200);

const PRIOR_WINS   = 0.001;
const PRIOR_VISITS = 2 * PRIOR_WINS;

const RESIGN_MIN_PLAYOUTS = 20000;

const NPAT_K = Util.envInt('NPAT_K', 35);  // kept move count

// UNIFORM_PRIOR=1 drops the npat prior entirely: uniform PUCT priors, no
// top-K pruning, npat extraction skipped.  The search is then a pure
// vlibpat product — used to generate teaching data free of npat anchoring
// (visit allocation and the visited set otherwise inherit npat's blind
// spots).  Costs a lot of strength-per-playout; measure before recording.
const UNIFORM_PRIOR = Util.envInt('UNIFORM_PRIOR', 0);

// ROOT_NPAT=0 gives the root uniform priors instead of the npat softmax (the
// root is unpruned regardless), so a near-deterministic npat prior can't pin
// the decision to one move.  Interior nodes are unaffected.  1 = default.
const ROOT_NPAT = Util.envInt('ROOT_NPAT', 0);

// Selection dither — random tie-break noise added to every node score.
// DITHER=0 makes the whole search a deterministic function of the position
// (there are no playouts) — useful for reproducibility experiments, but ties
// then resolve to the lowest cell index, a systematic spatial bias.  Tested
// as a teaching aid 2026-06-12: no measurable effect on distillation (CE
// floor, matches unchanged), so the default stays on.
const DITHER = Util.envFloat('DITHER', 0.001);

// Fixed playout count per decision.  When non-zero, overrides the time budget.
const PLAYOUTS = Util.envInt('PLAYOUTS', 0);

const npatWeights = NPat.loadModel({ name: 'puct-static' }).weights;
const npatStateByN = new Map();

// ── vlibpat static evaluator ──────────────────────────────────────────────────

const VLIBPAT_DATA = Util.envStr  ('VLIBPAT_DATA', './ref/vlibpat-4074.js');
const vpatRaw = Util.load('./' + VLIBPAT_DATA, 'vlibpatModel');
if (!vpatRaw) {
  throw new Error('puct-static: vlibpat model not loaded — set window.vlibpatModel before requiring');
}
const vpatModel = VLib.prepareModel(vpatRaw);
console.error(`puct-static: loaded ${vpatModel.weights.size} vlibpat weights from ${VLIBPAT_DATA}`);

// vlibpat value at `game2` — P(BLACK wins).  `game3` is the lockstep mirror
// of the same position; `ladderStatuses` is the position's shared ladder pass
// (or null to let the extractor run its own).
function _vlibpatEval(game2, game3, ladderStatuses) {
  const f = VLib.extractFeatures(game2, vpatModel.preparedSpecs, false, undefined, game3, ladderStatuses);
  return VLib.evaluateFeatures(f, vpatModel.weights);
}

// Run npat extraction + softmax for `game2`.  `game3` is the lockstep mirror
// of the same position (maintained by the search, so extraction skips the
// per-call game3FromGame2 rebuild); `ladderStatuses` is the position's shared
// ladder pass (or null).  Returns the shared state (with state.moves and
// state.probs populated) or null if not applicable.
function _runNpat(game2, game3, ladderStatuses) {
  if (UNIFORM_PRIOR || !npatWeights || game2.gameOver) return null;
  const N = game2.N;
  let state = npatStateByN.get(N);
  if (!state) { state = NPat.createState(N); npatStateByN.set(N, state); }
  NPat.extractFeatures(game2, state, undefined, game3, npatWeights, ladderStatuses);
  if (state.count === 0) return null;
  NPat.computeSoftmax(state, npatWeights);
  return state;
}

// When K > 0, keep only the K placements with the highest npat probability.
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

// ── Static evaluation ─────────────────────────────────────────────────────────

// Value of `game2` as P(BLACK wins): exact for terminal positions, vlibpat
// otherwise.  `ladderStatuses` is the position's shared ladder pass (or null).
function staticValue(game2, game3, ladderStatuses) {
  if (game2.gameOver) return game2.calcWinner() === BLACK ? 1 : 0;
  return _vlibpatEval(game2, game3, ladderStatuses);
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

function makeNode(move, parent, ci, game2, N, game3, ladderStatuses) {
  // Compute npat softmax once — reused for top-K pruning and the PUCT priors.
  // ROOT_NPAT=0 disables npat at the root only: the root keeps uniform priors
  // (and its already-unpruned full move width), so an overconfident npat prior
  // can't starve the actual decision; interior nodes still use npat.
  const npatState = (parent === null && !ROOT_NPAT) ? null : _runNpat(game2, game3, ladderStatuses);
  let movesArr = getLegalMoves(game2);
  // Top-K pruning at interior nodes only: at the root it would constrain the
  // actual decision, and the root gets enough visits to search full width.
  if (NPAT_K > 0 && parent !== null) movesArr = _pruneToTopK(movesArr, npatState, NPAT_K, N);
  const M = movesArr.length;
  const area = N * N;

  const legalMoves = new Int32Array(M);
  for (let i = 0; i < M; i++) legalMoves[i] = movesArr[i];

  const children   = new Array(M).fill(null);
  const wins       = new Float32Array(M).fill(PRIOR_WINS);
  const visits     = new Float32Array(M).fill(PRIOR_VISITS);
  const raveWins   = RAVE_K > 0 ? new Float32Array(area).fill(PRIOR_WINS)   : null;
  const raveVisits = RAVE_K > 0 ? new Float32Array(area).fill(PRIOR_VISITS) : null;

  // PUCT priors per move (renormalised to sum to 1).  npat's softmax covers
  // every legal non-true-eye placement — i.e. all placements in legalMoves —
  // but PASS is never in npat's output.  PASS gets a 1/area floor; all other
  // entries take npat's softmax probability directly; then renormalise.
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
    // Uniform priors over placements, but PASS keeps only the 1/area floor:
    // passing is almost never correct mid-game and its terminal-scoring value
    // is unreliable, so it must not get a full uniform share (which would let
    // it dominate visits and pass prematurely under uniform priors).
    const floor = 1 / area;
    let sum = 0;
    for (let i = 0; i < M; i++) { priors[i] = (legalMoves[i] === PASS) ? floor : 1; sum += priors[i]; }
    const inv = 1 / sum;
    for (let i = 0; i < M; i++) priors[i] *= inv;
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

  return Q + U + (DITHER > 0 ? DITHER * rng.random() : 0);
}

// `game3` is the lockstep mirror of rootGame2's position: every move played
// on the game2 clone is also played on it, and `depth` (the number of game3
// plays) is returned so the caller can undo back to the root position.
function selectAndExpand(root, rootGame2, N, rng, game3) {
  let node = root;
  const game2 = rootGame2.clone();
  let depth = 0;
  let ladderStatuses = null;   // set when a node is expanded (see below)

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

    // Expand on first visit: the static value is deterministic, so revisiting
    // an unexpanded leaf would re-add the same value and learn nothing.  Each
    // simulation ends at the node it just expanded.
    if (node.children[best] === null) {
      // One ladder pass for this position, shared by the npat extraction in
      // makeNode and the vlibpat evaluation that follows in getMove.
      ladderStatuses = game2.gameOver ? null : getAllLadderStatuses(game3);
      node.children[best] = makeNode(move, node, best, game2, N, game3, ladderStatuses);
      node.selectedChild = best;
      break;
    }

    node = node.children[best];
    node.selectedChild = -1;
  }

  return { node, game2, path, depth, ladderStatuses };
}

function backpropagate(node, value, path) {
  function childMover(n) {
    return -n.mover;
  }

  // Tree-RAVE update for node `n` at depth `d`: credit every move its chooser
  // played from depth d onward (every second path entry, colours alternate).
  function updateRave(n, d, won) {
    const rw = n.raveWins, rv = n.raveVisits;
    for (let j = d; j < path.length; j += 2) {
      const m = path[j];
      if (m === PASS) continue;
      rv[m] += 1;
      rw[m] += won;
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
    if (RAVE_K > 0) updateRave(node, d, won);
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
    if (RAVE_K > 0) updateRave(node.parent, d, won);
    node = node.parent;
  }
}

// ── Public interface ──────────────────────────────────────────────────────────

function getMove(game, timeBudgetMs, options = {}) {
  if (game.gameOver) return { type: 'pass', move: PASS, info: 'game already over' };

  const N          = game.cells ? game.N : game.boardSize;
  const game2      = game.cells ? game.clone() : game.toGame2();
  const rootPlayer = game2.current;

  if (game2.consecutivePasses > 0 && game2.calcWinner() === rootPlayer) {
    return { type: 'pass', move: PASS, info: 'obvious pass: already winning', rootWinRatio: 1,
             value: rootPlayer === BLACK ? 1 : 0 };
  }

  const rng = options.rng || makeRng();
  // Lockstep Game3 mirror — built once per decision, then maintained by
  // play/undo across simulations so feature extraction never rebuilds it.
  const game3 = game3FromGame2(game2);
  const root = makeNode(null, null, -1, game2, N, game3);

  const playoutLimit = options.playoutLimit || PLAYOUTS;
  const deadline = performance.now() + timeBudgetMs;
  let playouts = 0;

  do {
    playouts++;
    const { node, game2: simGame2, path, depth, ladderStatuses } = selectAndExpand(root, game2, N, rng, game3);
    const value = staticValue(simGame2, game3, ladderStatuses);
    for (let i = 0; i < depth; i++) game3.undo();
    backpropagate(node, value, path);
  } while (playoutLimit > 0 ? playouts < playoutLimit : performance.now() < deadline);

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
    return { type: 'pass', move: PASS, info: 'no winning line found', children, rootWinRatio,
             value: rootPlayer === BLACK ? 0 : 1 };
  }

  const m = root.legalMoves[bestIdx];
  const cv = root.visits[bestIdx];
  const bestWinRatio = cv > 0 ? root.wins[bestIdx] / cv : 0.5;

  // Absolute search value of the root position under best play, P(BLACK wins).
  const value = game.current === BLACK ? bestWinRatio : 1 - bestWinRatio;

  const result = m === PASS ? { type: 'pass', move: PASS, children, rootWinRatio, value }
                            : { type: 'place', move: m, x: m % N, y: (m / N) | 0, children, rootWinRatio, value };
  result.info = `value=${value.toFixed(3)}`;
  return result;
}

if (typeof module !== 'undefined') module.exports = { getMove };
else window.getMove = getMove;

})();
