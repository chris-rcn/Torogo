'use strict';

// PUCT MCTS with npat-driven priors and top-K candidate pruning at interior
// nodes (the root searches full width).  Leaves are evaluated by full random
// playouts only — the distillation variant: every playout runs to the end and
// is scored by area, yielding both a win result and a point margin (no
// truncated/vlibpat values).
//
// Each unexpanded edge runs N_FULL full random playouts before the child node
// is expanded (npat extraction + priors) and the descent continues into it.
// Node creation — the most expensive step — only happens on edges the search
// keeps returning to.
//
// Terminal positions are scored exactly.  Values are backpropagated
// fractionally: each chooser is credited value (BLACK) or 1 − value (WHITE).
//
// Node selection:
//   score = Q + C_PUCT · P(s,a) · √N_total / (1 + N_a)
// where Q is the move's mean backpropagated value (RAVE-blended when RAVE_K
// > 0) and P(s,a) is the npat softmax probability for the move (PASS and
// out-of-extract moves get a small uniform floor, then priors are
// renormalised to sum to 1).
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

const performance = (typeof window !== 'undefined' && window.performance)
  ? window.performance : require('perf_hooks').performance;

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

const NPAT_K = Util.envInt('NPAT_K', 40);  // kept move count

// ROOT_NPAT=0 gives the root uniform priors instead of the npat softmax (the
// root is unpruned regardless), so a near-deterministic npat prior can't pin
// the decision to one move.  Interior nodes are unaffected.  1 = default.
const ROOT_NPAT = Util.envInt('ROOT_NPAT', 0);

// Per-edge evaluation schedule: an unexpanded edge runs N_FULL full random
// playouts, then the child node is expanded and the descent continues into it.
const N_FULL      = Util.envInt('N_FULL', 4);

// Playout move policy.  PLAYOUT_NPAT=1 draws playout moves from the npat
// softmax (temperature PLAYOUT_TEMP) instead of uniform random — stronger,
// more realistic rollouts at much higher per-move cost.  0 = random (default).
const PLAYOUT_NPAT = Util.envInt('PLAYOUT_NPAT', 0);
const PLAYOUT_TEMP = Util.envFloat('PLAYOUT_TEMP', 1);

const PRIOR_WINS   = 0.001;
const PRIOR_VISITS = 2 * PRIOR_WINS;

// Root uniform-visiting: PASS is explored this many times as often as a normal
// candidate (its visit count is divided by this when picking the least-visited
// child), giving the pass baseline qp a tighter estimate than one uniform share.
const PASS_EXPLORE = Util.envFloat('PASS_EXPLORE', 3);

const RESIGN_MIN_PLAYOUTS = 20000;

// Fixed playout count per decision.  When non-zero, overrides the time budget.
const PLAYOUTS = Util.envInt('PLAYOUTS', 0);

const npatWeights = NPat.loadModel({ name: 'puct-hybrid' }).weights;
const npatStateByN = new Map();

// ── vlibpat static evaluator ──────────────────────────────────────────────────

const vpatPath = './ref/vlibpat-4074.js';
const vpatRaw = Util.load(vpatPath, 'vlibpatModel');
if (!vpatRaw) {
  throw new Error('puct-hybrid: vlibpat model not loaded — set window.vlibpatModel before requiring');
}
const vpatModel = VLib.prepareModel(vpatRaw);
console.log(`puct-hybrid: loaded ${vpatModel.weights.size} vlibpat weights from ${vpatPath}`);

// vlibpat value at `game2` — P(BLACK wins).  `game3` is the lockstep mirror
// of the same position.  (Unlike puct-static, no shared ladder pass: the
// npat extraction and static eval of a position happen on different visits
// here, so there is no same-simulation pairing to share it between.)
function _vlibpatEval(game2, game3) {
  const f = VLib.extractFeatures(game2, vpatModel.preparedSpecs, false, undefined, game3);
  return VLib.evaluateFeatures(f, vpatModel.weights);
}

// Run npat extraction + softmax for `game2`.  `game3` is the lockstep mirror
// of the same position (maintained by the search, so extraction skips the
// per-call game3FromGame2 rebuild).  Returns the shared state (with
// state.moves and state.probs populated) or null if not applicable.
function _runNpat(game2, game3) {
  if (!npatWeights || game2.gameOver) return null;
  const N = game2.N;
  let state = npatStateByN.get(N);
  if (!state) { state = NPat.createState(N); npatStateByN.set(N, state); }
  NPat.extractFeatures(game2, state, undefined, game3, npatWeights);
  if (state.count === 0) return null;
  NPat.computeSoftmax(state, npatWeights);
  return state;
}

// One npat-policy playout move for `game2` (sampled at PLAYOUT_TEMP).  `game3`
// is the lockstep mirror, kept in sync by the caller so extraction stays
// incremental.  Reuses the per-N npat state (no interleaving with node
// creation within a simulation).
function _npatPlayoutMove(game2, game3, rng) {
  if (game2.gameOver) return PASS;
  const N = game2.N;
  let state = npatStateByN.get(N);
  if (!state) { state = NPat.createState(N); npatStateByN.set(N, state); }
  return NPat.policyMove(game2, state, npatWeights, rng, game3, PLAYOUT_TEMP).move;
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

// ── Leaf evaluation ───────────────────────────────────────────────────────────

// Side-channel point margin (BLACK area − WHITE area − komi) of the most recent
// leaf evaluation, set alongside the win value returned by staticValue/playout.
// NaN when the leaf was a vlibpat win-prob eval (no point estimate); a real margin
// for full playouts and terminal positions.  Read by getMove right after the eval.
let _scoreOut = NaN;

// Value of `game2` as P(BLACK wins): exact for terminal positions, vlibpat
// otherwise.  Also sets _scoreOut (point margin; NaN for the vlibpat case).
function staticValue(game2, game3) {
  if (game2.gameOver) {
    const s = game2.calcScore();
    _scoreOut = s.black - s.white;
    return _scoreOut > 0 ? 1 : 0;
  }
  _scoreOut = NaN;
  return _vlibpatEval(game2, game3);
}

// Full random playout from `game2` (mutates it): runs to the end and scores by
// area.  Fills `played` (pre-zeroed by the caller) with the colour-signed
// first-occupancy RAVE trace: +weight if BLACK played the point first, −weight
// if WHITE, with weight decaying over the playout.  Sets _scoreOut (the point
// margin) and returns the result as P(BLACK wins).
function playout(game2, game3, played, rng) {
  const N   = game2.N;
  const cap = N * N;

  const moveLimit = 3 * game2.emptyCount + 20;
  const weightStep = 1 / cap;
  let moves = 0;
  let weight = 1.0;
  // game3 must follow the playout only when npat picks moves (incremental
  // feature extraction); plain random playouts leave it untouched.
  const needG3 = PLAYOUT_NPAT;

  while (!game2.gameOver && moves < moveLimit) {
    const current = game2.current;
    const idx = PLAYOUT_NPAT ? _npatPlayoutMove(game2, game3, rng)
                             : game2.randomLegalMove(rng);
    if (idx !== PASS && weight > 0 && played[idx] === 0) {
      played[idx] = current === BLACK ? weight : -weight;
    }
    game2.play(idx);
    if (needG3) game3.play(idx);
    moves++;
    weight -= weightStep;
  }

  if (needG3) for (let i = 0; i < moves; i++) game3.undo();   // unwind npat playout moves
  const s = game2.estimateScore();
  _scoreOut = s.black - s.white;                              // point margin for the point channel
  return _scoreOut > 0 ? 1 : 0;
}

// ── Tree node ─────────────────────────────────────────────────────────────────

function getLegalMoves(game2, forcePass) {
  const N     = game2.N;
  const cap   = N * N;
  const cells = game2.cells;
  const moves = [];
  for (let i = 0; i < cap; i++) {
    if (cells[i] !== 0) continue;
    if (game2.isTrueEye(i)) continue;
    if (game2.isLegal(i)) moves.push(i);
  }
  // PASS is a candidate in the endgame / after an opponent pass.  puct-distill also
  // forces it at the root (forcePass) so the actual decision always weighs passing —
  // the distillation baseline (q − qp) needs pass evaluated as a first-class move.
  if (forcePass || moves.length < cap / 3 || game2.consecutivePasses > 0) {
    moves.push(PASS);
  }
  return moves;
}

function makeNode(move, parent, ci, game2, N, game3) {
  // Compute npat softmax once — reused for top-K pruning and the PUCT priors.
  // ROOT_NPAT=0 disables npat at the root only: the root keeps uniform priors
  // (and its already-unpruned full move width), so an overconfident npat prior
  // can't starve the actual decision; interior nodes still use npat.
  const npatState = (parent === null && !ROOT_NPAT) ? null : _runNpat(game2, game3);
  let movesArr = getLegalMoves(game2, parent === null);
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
  // Point channel (root only — the only node whose children are exported): per-child
  // Σ signed point margin and the count of scored (full/terminal) playouts feeding it.
  const pointSum   = parent === null ? new Float32Array(M) : null;
  const pointN     = parent === null ? new Float32Array(M) : null;
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
    // it dominate visits and pass prematurely under uniform root priors).
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
    pointSum,
    pointN,

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

    let best = 0;
    if (node === root) {
      // Root: distribute visits UNIFORMLY across candidates (least-visited first, a
      // round-robin) rather than by PUCT.  PUCT would pour visits into the best move
      // and starve the rest; for distillation we want every candidate evaluated with
      // equal compute so each exported q has comparable, low-variance precision.
      let fewest = Infinity;
      for (let i = 0; i < M; i++) {
        const w = node.legalMoves[i] === PASS ? PASS_EXPLORE : 1;   // explore PASS PASS_EXPLORE× as often
        const v = node.visits[i] / w + 0.001 * rng.random();        // tiny jitter breaks initial ties at random
        if (v < fewest) { fewest = v; best = i; }
      }
    } else {
      let bestScore = -Infinity;
      for (let i = 0; i < M; i++) {
        const s = nodeScore(i, node, rng);
        if (s > bestScore) { bestScore = s; best = i; }
      }
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

    // Staged per-edge evaluation: N_FULL full playouts, then the child is
    // expanded and the descent continues into it (its best edge takes the
    // first playout).  `visits` carries the PRIOR_VISITS offset, so the
    // comparison lands strictly above the threshold.
    if (node.children[best] === null) {
      const v = node.visits[best];
      if (v < N_FULL) {
        node.selectedChild = best;
        doPlayout = true;
        break;
      }
      node.children[best] = makeNode(move, node, best, game2, N, game3);
    }

    node = node.children[best];
    node.selectedChild = -1;
  }

  return { node, game2, path, depth, doPlayout };
}

// `played` is the playout's colour-signed RAVE trace, or null for simulations
// that ended in terminal scoring.  `score` is the BLACK-perspective point margin
// of the leaf (NaN when the leaf had no point estimate); it feeds the root's
// per-child point channel, signed into each chooser's perspective like wins.
function backpropagate(node, value, score, path, played) {
  const hasScore = score === score;   // false for NaN
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
    if (node.pointSum && hasScore) { node.pointSum[leafIdx] += chooser === BLACK ? score : -score; node.pointN[leafIdx]++; }
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
    if (node.parent.pointSum && hasScore) { node.parent.pointSum[ci] += chooser === BLACK ? score : -score; node.parent.pointN[ci]++; }
    if (RAVE_K > 0) updateRave(node.parent, d, won, chooser);
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
    return { type: 'pass', move: PASS, info: 'obvious pass: already winning', rootWinRatio: 1 };
  }

  const rng = options.rng || makeRng();
  // Lockstep Game3 mirror — built once per decision, then maintained by
  // play/undo across simulations so feature extraction never rebuilds it.
  const game3 = game3FromGame2(game2);
  const root = makeNode(null, null, -1, game2, N, game3);

  const played = new Float32Array(N * N);

  // Effort can be specified per root candidate (uniform root visiting spends an equal
  // share on each), scaling the total budget with the root width — or as a fixed total.
  const playoutLimit = options.playoutsPerCand != null
    ? options.playoutsPerCand * root.legalMoves.length
    : (options.playoutLimit || PLAYOUTS);
  const deadline = performance.now() + timeBudgetMs;
  let playouts = 0;

  do {
    playouts++;
    const { node, game2: simGame2, path, depth, doPlayout } = selectAndExpand(root, game2, N, rng, game3);
    let value, trace = null;
    if (doPlayout && !simGame2.gameOver) {
      played.fill(0);
      value = playout(simGame2, game3, played, rng);
      trace = played;
    } else {
      value = staticValue(simGame2, game3);   // terminal positions only — scored exactly
    }
    const score = _scoreOut;   // point margin set by playout/staticValue (NaN if none)
    for (let i = 0; i < depth; i++) game3.undo();
    backpropagate(node, value, score, path, trace);
  } while (playoutLimit > 0 ? playouts < playoutLimit : performance.now() < deadline);

  const M = root.legalMoves.length;
  // Pick the played move by VALUE (mean win ratio wins/visits), not visit count.
  // Under uniform root visiting the visit counts are ~equal and so uninformative;
  // the value is the move-quality signal.  (Irrelevant to distillation, which
  // exports every child's q/n — this only affects the move getMove returns.)
  let bestIdx = 0, bestVal = -Infinity;
  for (let i = 0; i < M; i++) {
    const val = root.wins[i] / root.visits[i] + 0.001 * rng.random();   // jitter breaks ties at random
    if (val > bestVal) { bestVal = val; bestIdx = i; }
  }

  const children = [];
  for (let i = 0; i < M; i++) {
    const m = root.legalMoves[i];
    children.push({
      move: m === PASS ? { type: 'pass' } : { type: 'place', x: m % N, y: (m / N) | 0 },
      visits: root.visits[i],
      wins:   root.wins[i],
      pointSum: root.pointSum[i],   // Σ signed point margin (mover perspective)
      pointN:   root.pointN[i],     // # scored playouts feeding pointSum
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

if (typeof module !== 'undefined') module.exports = { getMove };
else window.getMove = getMove;

})();
