'use strict';

// RAVE with priors seeded from npat softmax probabilities.
//
// Identical to ai/rave.js except: when RAVE_NPAT_VISITS > 0, every newly
// created tree node has its raveVisits/raveWins boosted using npat's softmax
// over the candidate moves at that position.
//
// In the browser, set window.npatModel to the loaded weights before requiring.
//
// Boost formula (per candidate move m with npat softmax probability p):
//   dv = V          dw = V * p
// i.e. uniform virtual visits across all candidates, with a wr=p value claim.
//
// When RAVE_NPAT_K > 0 (default 0 = disabled), each new node's candidate list
// is pruned to the K placements with the highest npat probability (PASS, if
// originally legal, is always retained as a fallback).

(function () {

const Util = (typeof require === 'function') ? require('../util.js') : window.Util;
const { PASS, BLACK, WHITE } = Util.load('./game2.js', 'Game2');
const { makeRng }            = Util.load('./xorshift.js', 'XorShift');
const NPat                   = Util.load('./npat-lib.js', 'NPatterns');
const { game3FromGame2 }     = Util.load('./game3.js', 'Game3');

const performance = (typeof window !== 'undefined' && window.performance)
  ? window.performance : require('perf_hooks').performance;

const EXPLORATION_C = Util.envFloat('EXPLORATION_C', 0.4);
const RAVE_K        = Util.envFloat('RAVE_K', 1200);
const PLAYOUTS      = Util.envInt('PLAYOUTS', 0);
const N_EXPAND      = Util.envInt('N_EXPAND', 3);
const RAVE_INHERIT  = Util.envFloat('RAVE_INHERIT', 0.2);

const PRIOR_WINS   = 0.001;
const PRIOR_VISITS = 2 * PRIOR_WINS;

const RESIGN_MIN_PLAYOUTS = 20000;

// ── npat prior ────────────────────────────────────────────────────────────────

const RAVE_NPAT_VISITS = Util.envFloat('RAVE_NPAT_VISITS', 50);
const RAVE_NPAT_K      = Util.envInt  ('RAVE_NPAT_K', 40);

let npatWeights = null;
const npatStateByN = new Map();

if (RAVE_NPAT_VISITS > 0 || RAVE_NPAT_K > 0) {
  let raw, modelName;
  if (typeof window !== 'undefined') {
    if (!window.npatModel) {
      throw new Error('rave-npat: window.npatModel is not set — load the npat weights script first');
    }
    raw = window.npatModel;
    modelName = 'window.npatModel';
  } else {
    const path = require('path');
    const weightsPath = process.env.NPAT_WEIGHTS
      || path.join(__dirname, '..', 'npat-data.js');
    raw = require(path.resolve(weightsPath));
    modelName = path.basename(weightsPath);
  }
  if (raw.tactStoneLimit !== undefined && raw.tactStoneLimit !== NPat.TACT_STONE_LIMIT) {
    throw new Error(
      `rave-npat: TACT_STONE_LIMIT mismatch — file ${modelName} ` +
      `was trained at ${raw.tactStoneLimit}, runtime is ${NPat.TACT_STONE_LIMIT}. ` +
      `Set NPAT_STONE_LIMIT=${raw.tactStoneLimit} before launching.`
    );
  }
  let has33c = false, hasP12 = false;
  for (const [k] of raw.weights) {
    if (typeof k === 'string') continue;
    if      (k >= NPat.SHAPE33C_RAW_BASE && k < NPat.P12_RAW_BASE) has33c = true;
    else if (k >= NPat.P12_RAW_BASE)                                hasP12 = true;
  }
  npatWeights = NPat.createWeights({
    initialCapacity: Math.max(1024, raw.weights.size | 0),
    use33c: has33c, useP12: hasP12,
  });
  for (const [k, v] of raw.weights) {
    const idx = NPat.internWeight(npatWeights, k);
    npatWeights.vals[idx] = v;
  }
  console.error(`rave-npat-prune: loaded ${npatWeights.size} npat weights from ${modelName} ` +
    `(visits=${RAVE_NPAT_VISITS} k=${RAVE_NPAT_K} 3x3c=${has33c} p12=${hasP12})`);
}

// Run npat extraction + softmax for `game2`.  Returns the shared state (with
// state.moves and state.probs populated) or null if not applicable.
function _runNpat(game2) {
  if (!npatWeights || game2.gameOver) return null;
  const N = game2.N;
  let state = npatStateByN.get(N);
  if (!state) { state = NPat.createState(N); npatStateByN.set(N, state); }
  const game3 = game3FromGame2(game2);
  NPat.extractFeatures(game2, state, undefined, game3, npatWeights);
  if (state.count === 0) return null;
  NPat.computeSoftmax(state, npatWeights);
  return state;
}

function applyNpatPriorFromState(node, state, V) {
  const moves = state.moves;
  const probs = state.probs;
  const rv = node.raveVisits;
  const rw = node.raveWins;
  for (let i = 0; i < state.count; i++) {
    const m = moves[i];
    rv[m] += V;
    rw[m] += V * probs[i];
  }
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

// ── Fast playout helpers ──────────────────────────────────────────────────────

function playTracked(game2, node, played) {
  const wasAlreadyOver = game2.gameOver;
  const N   = game2.N;
  const cap = N * N;

  played.fill(0, 0, cap);

  const moveLimit = 3 * game2.emptyCount + 20;
  const weightStep = 1 / cap;
  let moves = 0;
  let weight = 1.0;

  while (!game2.gameOver && moves < moveLimit) {
    const current = game2.current;
    const idx = game2.randomLegalMove();
    if (idx === PASS) { game2.play(PASS); moves++; continue; }
    if (weight > 0 && played[idx] === 0) played[idx] = current === BLACK ? weight : -weight;
    game2.play(idx);
    moves++;
    weight -= weightStep;
  }

  let winner;
  if (wasAlreadyOver) {
    winner = game2.calcWinner();
  } else {
    winner = game2.estimateWinner();
  }
  return { winner, played };
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

function makeNode(move, parent, ci, game2, N) {
  // Compute npat softmax once — reused for top-K pruning and the RAVE prior.
  const npatState = _runNpat(game2);
  let movesArr = getLegalMoves(game2);
  if (RAVE_NPAT_K > 0) movesArr = _pruneToTopK(movesArr, npatState, RAVE_NPAT_K, N);
  const M = movesArr.length;
  const area = N * N;

  const legalMoves = new Int32Array(M);
  for (let i = 0; i < M; i++) legalMoves[i] = movesArr[i];

  const children   = new Array(M).fill(null);
  const wins       = new Float32Array(M).fill(PRIOR_WINS);
  const visits     = new Float32Array(M).fill(PRIOR_VISITS);
  const raveWins   = new Float32Array(area);
  const raveVisits = new Float32Array(area);

  if (parent === null || parent.parent === null) {
    raveWins.fill(PRIOR_WINS);
    raveVisits.fill(PRIOR_VISITS);
  } else {
    const gparent = parent.parent;
    for (let m = 0; m < area; m++) {
      raveWins[m]   = RAVE_INHERIT * gparent.raveWins[m];
      raveVisits[m] = RAVE_INHERIT * gparent.raveVisits[m];
    }
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

    wins,
    visits,

    raveWins,
    raveVisits
  };

  if (npatState && RAVE_NPAT_VISITS > 0) {
    applyNpatPriorFromState(node, npatState, RAVE_NPAT_VISITS);
  }

  return node;
}

function ucbScore(moveIdx, node, rng) {
  const move  = node.legalMoves[moveIdx];

  const raveWR = (move === PASS) ? 0 : (node.raveWins[move] / node.raveVisits[move]);

  const realW = node.wins[moveIdx];
  const realV = node.visits[moveIdx];
  const realWR = realW / realV;

  const raveWeight = RAVE_K / (RAVE_K + realV);
  const realWeight = 1 - raveWeight;

  const wr = realWeight * realWR + raveWeight * raveWR;

  const scoreBase = wr + 0.001 * rng.random();
  return scoreBase + EXPLORATION_C * Math.sqrt(Math.log(node.totalVisits) / realV);
}

function selectAndExpand(root, rootGame2, N, rng) {
  let node = root;
  const game2 = rootGame2.clone();

  while (!game2.gameOver) {
    const M = node.legalMoves.length;
    if (M === 0) break;

    let best = 0, bestScore = -Infinity;
    for (let i = 0; i < M; i++) {
      const s = ucbScore(i, node, rng);
      if (s > bestScore) { bestScore = s; best = i; }
    }

    game2.play(node.legalMoves[best]);

    if (node.children[best] === null && node.visits[best] >= N_EXPAND) {
      node.children[best] = makeNode(node.legalMoves[best], node, best, game2, N);
    }

    if (!game2.gameOver && game2.consecutivePasses > 0) {
      game2.play(PASS);
      node.selectedChild = best;
      break;
    }

    if (node.children[best] !== null) {
      node = node.children[best];
      node.selectedChild = -1;
      continue;
    }

    node.selectedChild = best;
    break;
  }

  return { node, game2 };
}

function backpropagate(node, winner, played, rootPlayer) {
  function childMover(n) {
    return -n.mover;
  }

  function updateRave(node, won, played, chooser) {
    if (chooser === BLACK) {
      for (let k = 0; k < played.length; k++) {
        const weight = played[k];
        if (weight > 0) {
          node.raveVisits[k] += weight;
          node.raveWins[k]   += won * weight;
        }
      }
    } else {
      for (let k = 0; k < played.length; k++) {
        const weight = played[k];
        if (weight < 0) {
          node.raveVisits[k] -= weight;
          node.raveWins[k]   -= won * weight;
        }
      }
    }
  }

  const leafIdx = node.selectedChild;
  if (leafIdx !== -1) {
    const chooser = childMover(node);
    const won     = winner === chooser ? 1 : 0;
    node.visits[leafIdx]++;
    node.wins[leafIdx] += won;
    node.totalVisits++;
    updateRave(node, won, played, chooser);
  }

  while (node.parent !== null) {
    const ci      = node.ci;
    const chooser = childMover(node.parent);
    const won     = winner === chooser ? 1 : 0;
    node.parent.visits[ci]++;
    node.parent.wins[ci] += won;
    node.parent.totalVisits++;
    updateRave(node.parent, won, played, chooser);
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
  const root = makeNode(null, null, -1, game2, N);

  const played = new Float32Array(N * N);

  const playoutLimit = options.playoutLimit || PLAYOUTS;
  const deadline = performance.now() + timeBudgetMs;
  let playouts = 0;

  do {
    playouts++;
    const { node, game2: simGame2 } = selectAndExpand(root, game2, N, rng);
    const { winner, played: p } = playTracked(simGame2, node, played);
    backpropagate(node, winner, p, rootPlayer);
  } while (playoutLimit > 0 ? playouts < playoutLimit : performance.now() < deadline);

  const M = root.legalMoves.length;
  let bestIdx = 0, bestVisits = -1, bestScore = -Infinity;
  for (let i = 0; i < M; i++) {
    const cv = root.visits[i];
    if (cv > bestVisits || (cv === bestVisits && ucbScore(i, root, rng) > bestScore)) {
      bestVisits = cv;
      bestScore  = ucbScore(i, root, rng);
      bestIdx    = i;
    }
  }

  const children = [];
  for (let i = 0; i < M; i++) {
    const m = root.legalMoves[i];
    children.push({
      move: m === PASS ? { type: 'pass' } : { type: 'place', x: m % N, y: (m / N) | 0 },
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

if (typeof module !== 'undefined') module.exports = { getMove };
else window.RaveNpatPrune = { getMove };

})();
