'use strict';

// RAVE with priors seeded from npat softmax probabilities (Node-only).
//
// Identical to ai/rave.js except: when RAVE_NPAT_VISITS > 0, every newly
// created tree node has its raveVisits/raveWins boosted in proportion to
// npat's softmax over the candidate moves at that position.
//
//   raveVisits[m] += V * p_m
//   raveWins[m]   += wr * V * p_m
//
// where V = RAVE_NPAT_VISITS and wr is set by RAVE_NPAT_MODE:
//   neutral  (default) — wr = 0.5  (npat used purely as policy attention)
//   value              — wr = p_m  (npat softmax read as a value estimate too)

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
if (!_isNode) return;

const path = require('path');
const { performance } = require('perf_hooks');
const { PASS, BLACK, WHITE } = require('../game2.js');
const Util = require('../util.js');
const { makeRng } = require('../xorshift.js');
const NPat = require('../npat-lib.js');
const { game3FromGame2 } = require('../game3.js');

const EXPLORATION_C = Util.envFloat('EXPLORATION_C', 0.4);
const RAVE_K        = Util.envFloat('RAVE_K', 800);
const PLAYOUTS      = Util.envInt('PLAYOUTS', 0);
const N_EXPAND      = Util.envInt('N_EXPAND', 2);
const RAVE_INHERIT  = Util.envFloat('RAVE_INHERIT', 0.2);

const PRIOR_WINS   = 0.001;
const PRIOR_VISITS = 2 * PRIOR_WINS;

const RESIGN_MIN_PLAYOUTS = 20000;

// ── npat prior ────────────────────────────────────────────────────────────────

const RAVE_NPAT_VISITS = Util.envFloat('RAVE_NPAT_VISITS', 0);
const RAVE_NPAT_MODE   = process.env.RAVE_NPAT_MODE || 'neutral';  // 'neutral' | 'value'

let npatWeights = null;
const npatStateByN = new Map();

if (RAVE_NPAT_VISITS > 0) {
  const weightsPath = process.env.NPAT_WEIGHTS
    || path.join(__dirname, '..', 'out', 'npat-9-QD-pat4-6.js');
  const raw = require(path.resolve(weightsPath));
  if (raw.tactStoneLimit !== undefined && raw.tactStoneLimit !== NPat.TACT_STONE_LIMIT) {
    throw new Error(
      `rave-npat: TACT_STONE_LIMIT mismatch — file ${path.basename(weightsPath)} ` +
      `was trained at ${raw.tactStoneLimit}, runtime is ${NPat.TACT_STONE_LIMIT}. ` +
      `Set NPAT_STONE_LIMIT=${raw.tactStoneLimit} before launching.`
    );
  }
  let has33c = false, hasE = false;
  for (const [k] of raw.weights) {
    if (typeof k === 'string') continue;
    if      (k >= NPat.SHAPE33C_RAW_BASE && k < NPat.TYPE_E_RAW_BASE) has33c = true;
    else if (k >= NPat.TYPE_E_RAW_BASE)                                hasE   = true;
  }
  npatWeights = NPat.createWeights({
    initialCapacity: Math.max(1024, raw.weights.size | 0),
    use33c: has33c, useE: hasE,
  });
  for (const [k, v] of raw.weights) {
    const idx = NPat.internWeight(npatWeights, k);
    npatWeights.vals[idx] = v;
  }
  console.error(`rave-npat: loaded ${npatWeights.size} npat weights from ${path.basename(weightsPath)} ` +
    `(visits=${RAVE_NPAT_VISITS} mode=${RAVE_NPAT_MODE} 3x3c=${has33c} E=${hasE})`);
}

function applyNpatPrior(node, game2) {
  if (!npatWeights || RAVE_NPAT_VISITS <= 0 || game2.gameOver) return;
  const N = game2.N;
  let state = npatStateByN.get(N);
  if (!state) { state = NPat.createState(N); npatStateByN.set(N, state); }

  const game3 = game3FromGame2(game2);
  NPat.extractFeatures(game2, state, undefined, game3, npatWeights);
  if (state.count === 0) return;
  NPat.computeSoftmax(state, npatWeights);

  const moves = state.moves;
  const probs = state.probs;
  const rv = node.raveVisits;
  const rw = node.raveWins;
  const useValueMode = RAVE_NPAT_MODE === 'value';

  for (let i = 0; i < state.count; i++) {
    const m = moves[i];
    const p = probs[i];
    const dv = RAVE_NPAT_VISITS * p;
    const wr = useValueMode ? p : 0.5;
    rv[m] += dv;
    rw[m] += wr * dv;
  }
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
    if (played[idx] === 0) played[idx] = current === BLACK ? weight : -weight;
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
  const movesArr = getLegalMoves(game2);
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

  applyNpatPrior(node, game2);

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

module.exports = { getMove };

})();
