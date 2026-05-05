'use strict';

// RAVE with parallel pattern-specific stats blended at selection time.
//
// Identical to ai/rave.js except: alongside the usual aggregate RAVE
// counters, each node also maintains pattern-specific counters that are
// incremented only for rollouts whose local pattern at m (at play time)
// matched the node's own local pattern at m.  Both are updated on every
// playout; no signal is discarded.
//
// Selection blends the two estimates by patternVisits relative to RAVE_T,
// mirroring the existing real/RAVE blend.  RAVE_T = blend knee:
//   patWeight = patVisits / (RAVE_T + patVisits)
//   mixedWR   = (1 - patWeight) * simpleWR + patWeight * patWR
//
// RAVE_T = 0 disables the feature (pure baseline RAVE, zero overhead).

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
if (!_isNode) return;

const { performance } = require('perf_hooks');
const { PASS, BLACK, WHITE } = require('../game2.js');
const Util = require('../util.js');
const { makeRng } = require('../xorshift.js');

const EXPLORATION_C = Util.envFloat('EXPLORATION_C', 0.4);
const RAVE_K        = Util.envFloat('RAVE_K', 800);
const PLAYOUTS      = Util.envInt('PLAYOUTS', 0);
const N_EXPAND      = Util.envInt('N_EXPAND', 2);
const RAVE_INHERIT  = Util.envFloat('RAVE_INHERIT', 0.2);
const RAVE_T        = Util.envFloat('RAVE_T', 0);

const PRIOR_WINS   = 0.001;
const PRIOR_VISITS = 2 * PRIOR_WINS;

const RESIGN_MIN_PLAYOUTS = 20000;

if (RAVE_T > 0) console.error(`rave-t: RAVE_T=${RAVE_T}`);

// ── Local pattern ─────────────────────────────────────────────────────────────

// 4-neighbour pattern code at cell m on a toroidal N×N board.
// Each neighbour: 0=empty, 1=black, 2=white.  Code = L*27 + R*9 + U*3 + D.
function patternAt(cells, m, N) {
  const x = m - ((m / N) | 0) * N;
  const y = (m / N) | 0;
  const xm1 = (x === 0)     ? N - 1 : x - 1;
  const xp1 = (x === N - 1) ? 0     : x + 1;
  const ym1 = (y === 0)     ? N - 1 : y - 1;
  const yp1 = (y === N - 1) ? 0     : y + 1;
  const cL = cells[xm1 + y   * N];
  const cR = cells[xp1 + y   * N];
  const cU = cells[x   + ym1 * N];
  const cD = cells[x   + yp1 * N];
  const eL = cL === 0 ? 0 : (cL === BLACK ? 1 : 2);
  const eR = cR === 0 ? 0 : (cR === BLACK ? 1 : 2);
  const eU = cU === 0 ? 0 : (cU === BLACK ? 1 : 2);
  const eD = cD === 0 ? 0 : (cD === BLACK ? 1 : 2);
  return eL * 27 + eR * 9 + eU * 3 + eD;
}

// ── Fast playout helpers ──────────────────────────────────────────────────────

function playTracked(game2, node, played, playedPattern) {
  const wasAlreadyOver = game2.gameOver;
  const N   = game2.N;
  const cap = N * N;

  played.fill(0, 0, cap);
  if (RAVE_T > 0) playedPattern.fill(-1, 0, cap);

  const moveLimit = 3 * game2.emptyCount + 20;
  const weightStep = 1 / (2 * cap);  // half the baseline rave decay rate
  let moves = 0;
  let weight = 1.0;

  while (!game2.gameOver && moves < moveLimit) {
    const current = game2.current;
    const idx = game2.randomLegalMove();
    if (idx === PASS) { game2.play(PASS); moves++; continue; }
    if (played[idx] === 0) {
      played[idx] = current === BLACK ? weight : -weight;
      if (RAVE_T > 0) playedPattern[idx] = patternAt(game2.cells, idx, N);
    }
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
  return { winner, played, playedPattern };
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

  // Pattern-specific arrays + per-cell pattern lookup, only when feature is on.
  // Pattern stats start empty (no inheritance) — they earn their visits.
  let ravePatWins = null, ravePatVisits = null, nodePattern = null;
  if (RAVE_T > 0) {
    ravePatWins   = new Float32Array(area);
    ravePatVisits = new Float32Array(area);
    nodePattern   = new Int16Array(area);
    const cells = game2.cells;
    for (let m = 0; m < area; m++) nodePattern[m] = patternAt(cells, m, N);
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

    wins,
    visits,

    raveWins,
    raveVisits,
    ravePatWins,
    ravePatVisits,
    nodePattern,
  };
}

function ucbScore(moveIdx, node, rng) {
  const move  = node.legalMoves[moveIdx];

  let raveWR;
  if (move === PASS) {
    raveWR = 0;
  } else {
    const simpleWR = node.raveWins[move] / node.raveVisits[move];
    if (RAVE_T > 0) {
      const pv = node.ravePatVisits[move];
      const patWR     = pv > 0 ? node.ravePatWins[move] / pv : simpleWR;
      const patWeight = pv / (RAVE_T + pv);
      raveWR = (1 - patWeight) * simpleWR + patWeight * patWR;
    } else {
      raveWR = simpleWR;
    }
  }

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

function backpropagate(node, winner, played, playedPattern, rootPlayer) {
  function childMover(n) { return -n.mover; }

  function updateRave(node, won, played, playedPattern, chooser) {
    const useGate = RAVE_T > 0;
    const np  = node.nodePattern;
    const rv  = node.raveVisits;
    const rw  = node.raveWins;
    const prv = node.ravePatVisits;
    const prw = node.ravePatWins;
    if (chooser === BLACK) {
      for (let k = 0; k < played.length; k++) {
        const weight = played[k];
        if (weight > 0) {
          rv[k] += weight;
          rw[k] += won * weight;
          if (useGate && playedPattern[k] === np[k]) {
            prv[k] += weight;
            prw[k] += won * weight;
          }
        }
      }
    } else {
      for (let k = 0; k < played.length; k++) {
        const weight = played[k];
        if (weight < 0) {
          rv[k] -= weight;
          rw[k] -= won * weight;
          if (useGate && playedPattern[k] === np[k]) {
            prv[k] -= weight;
            prw[k] -= won * weight;
          }
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
    updateRave(node, won, played, playedPattern, chooser);
  }

  while (node.parent !== null) {
    const ci      = node.ci;
    const chooser = childMover(node.parent);
    const won     = winner === chooser ? 1 : 0;
    node.parent.visits[ci]++;
    node.parent.wins[ci] += won;
    node.parent.totalVisits++;
    updateRave(node.parent, won, played, playedPattern, chooser);
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

  const played        = new Float32Array(N * N);
  const playedPattern = (RAVE_T > 0) ? new Int16Array(N * N) : null;

  const playoutLimit = options.playoutLimit || PLAYOUTS;
  const deadline = performance.now() + timeBudgetMs;
  let playouts = 0;

  do {
    playouts++;
    const { node, game2: simGame2 } = selectAndExpand(root, game2, N, rng);
    const { winner, played: p, playedPattern: pp } = playTracked(simGame2, node, played, playedPattern);
    backpropagate(node, winner, p, pp, rootPlayer);
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
