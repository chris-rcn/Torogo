'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs (require, process, etc.).
// Wrapped in an IIFE to avoid polluting the global namespace.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

/**
 * RAVE MCTS policy that deliberately plays badly.
 *
 * Identical to rave.js except at the root: the win-ratio component of UCB is
 * inverted (1-wr) so the tree preferentially explores moves it considers bad.
 * All subtrees below the root are evaluated normally.
 */

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

const performance = (typeof window !== 'undefined') ? window.performance
  : require('perf_hooks').performance;

const { PASS, BLACK, WHITE } = _isNode ? require('../game2.js') : window.Game2;
const Util = _isNode ? require('../util.js') : window.Util;

const EXPLORATION_C = Util.envFloat('EXPLORATION_C', 0.3);
const RAVE_EQUIV    = Util.envFloat('RAVE_EQUIV', 2000);
const PLAYOUTS      = Util.envInt('PLAYOUTS', 0);
const N_EXPAND      = Util.envInt('N_EXPAND', 3);
const RAVE_INHERIT  = Util.envFloat('RAVE_INHERIT', 0.2);

// ── Fast playout helpers ──────────────────────────────────────────────────────

function playTracked(game2) {
  const wasAlreadyOver = game2.gameOver;
  const N   = game2.N;
  const cap = N * N;
  const blackPlayed = [];
  const whitePlayed = [];

  const moveLimit = cap + 20;
  let moves = 0;

  while (!game2.gameOver && moves < moveLimit) {
    const current = game2.current;
    const idx = game2.randomLegalMove();
    if (idx === PASS) { game2.play(PASS); moves++; continue; }
    if (current === BLACK) blackPlayed.push(idx);
    else                    whitePlayed.push(idx);
    game2.play(idx);
    moves++;
  }

  let winner;
  if (wasAlreadyOver) {
    winner = game2.calcWinner();
  } else {
    winner = game2.estimateWinner();
  }
  return { winner, blackPlayed, whitePlayed };
}

// ── Tree node ─────────────────────────────────────────────────────────────────

function getLegalMoves(game2, allowPass = true) {
  const N     = game2.N;
  const cap   = N * N;
  const cells = game2.cells;
  const moves = [];
  for (let i = 0; i < cap; i++) {
    if (cells[i] !== 0) continue;
    if (game2.isTrueEye(i)) continue;
    if (game2.isLegal(i)) moves.push(i);
  }
  if (allowPass && (moves.length < cap / 3 || game2.consecutivePasses > 0)) {
    moves.push(PASS);
  }
  return moves;
}

function makeNode(move, parent, ci, mover, game2, N, allowPass = true) {
  const movesArr = getLegalMoves(game2, allowPass);
  const M = movesArr.length;
  const area = N * N;

  const legalMoves = new Int32Array(M);
  for (let i = 0; i < M; i++) legalMoves[i] = movesArr[i];

  const children   = new Array(M).fill(null);
  const wins       = new Float32Array(M).fill(0.001);
  const visits     = new Float32Array(M).fill(0.002);
  const raveWins   = new Float32Array(area);
  const raveVisits = new Float32Array(area);

  if (parent === null || parent.parent === null) {
    raveWins.fill(0.001);
    raveVisits.fill(0.002);
  } else {
    const gparent = parent.parent;
    for (let m = 0; m < area; m++) {
      raveWins[m]   = RAVE_INHERIT * gparent.raveWins[m];
      raveVisits[m] = RAVE_INHERIT * gparent.raveVisits[m];
    }
  }

  return {
    move, parent, ci, mover,
    totalVisits:   1,
    selectedChild: -1,
    legalMoves, children, wins, visits, raveWins, raveVisits
  };
}

function ucbScore(moveIdx, node, polarity) {
  const move   = node.legalMoves[moveIdx];
  const raveWR = (move === PASS) ? 0 : (node.raveWins[move] / node.raveVisits[move]);
  const realW  = node.wins[moveIdx];
  const realV  = node.visits[moveIdx];
  const realWR = realW / realV;
  const wr = (raveWR * RAVE_EQUIV + realWR * realV) / (RAVE_EQUIV + realV);
  return polarity * wr + 0.001 * Math.random()
       + EXPLORATION_C * Math.sqrt(Math.log(node.totalVisits) / (1 + realV));
}

// ── RAVE-MCTS core ────────────────────────────────────────────────────────────

function selectAndExpand(root, rootGame2, N) {
  let node = root;
  const game2 = rootGame2.clone();

  while (!game2.gameOver) {
    const M = node.legalMoves.length;
    if (M === 0) break;

    // At the root, invert win ratio to explore bad moves; elsewhere play normally.
    const polarity = node === root ? -1 : 1;

    let best = 0, bestScore = -Infinity;
    for (let i = 0; i < M; i++) {
      const s = ucbScore(i, node, polarity);
      if (s > bestScore) { bestScore = s; best = i; }
    }

    const mover = game2.current;
    game2.play(node.legalMoves[best]);

    if (node.children[best] === null && node.visits[best] >= N_EXPAND) {
      node.children[best] = makeNode(node.legalMoves[best], node, best, mover, game2, N);
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

function backpropagate(node, winner, blackPlayed, whitePlayed, rootPlayer) {
  function childMover(n) {
    return n.mover === null ? rootPlayer : (n.mover === BLACK ? WHITE : BLACK);
  }

  function updateRave(node, won, played) {
    const invPlayedCount = 1 / (1 + played.length);
    let weight = 1;
    for (let k = 0; k < played.length; k++) {
      const move = played[k];
      node.raveVisits[move] += weight;
      node.raveWins[move] += won * weight;
      weight -= invPlayedCount;
    }
  }

  const leafIdx = node.selectedChild;
  if (leafIdx !== -1) {
    const chooser = childMover(node);
    const won     = winner === chooser ? 1 : 0;
    node.visits[leafIdx]++;
    node.wins[leafIdx] += won;
    node.totalVisits++;
    const played = chooser === BLACK ? blackPlayed : whitePlayed;
    updateRave(node, won, played);
  }

  while (node.parent !== null) {
    const ci  = node.ci;
    const won = winner === node.mover ? 1 : 0;
    node.parent.visits[ci]++;
    node.parent.wins[ci] += won;
    node.parent.totalVisits++;
    const chooser = childMover(node.parent);
    const played  = chooser === BLACK ? blackPlayed : whitePlayed;
    const rWon    = winner === chooser ? 1 : 0;
    updateRave(node.parent, rWon, played);
    node = node.parent;
  }
}

// ── Public interface ──────────────────────────────────────────────────────────

function getMove(game, timeBudgetMs) {
  if (game.gameOver) return { type: 'pass', move: PASS, info: 'game already over' };

  const N          = game.cells ? game.N : game.boardSize;
  const game2      = game.cells ? game.clone() : game.toGame2();
  const rootPlayer = game2.current;


  const root = makeNode(null, null, -1, null, game2, N, false);
  if (root.legalMoves.length === 0) return { type: 'pass', move: PASS, info: 'no non-pass moves' };

  const deadline = performance.now() + timeBudgetMs;
  let playouts = 0;

  do {
    playouts++;
    const { node, game2: simGame2 } = selectAndExpand(root, game2, N);
    const { winner, blackPlayed, whitePlayed } = playTracked(simGame2);
    backpropagate(node, winner, blackPlayed, whitePlayed, rootPlayer);
  } while (PLAYOUTS > 0 ? playouts < PLAYOUTS : performance.now() < deadline);

  // Select the root child with the most visits under the inverted score.
  const M = root.legalMoves.length;
  let bestIdx = 0, bestVisits = -1, bestScore = -Infinity;
  for (let i = 0; i < M; i++) {
    const cv = root.visits[i];
    const sc = ucbScore(i, root, -1);
    if (cv > bestVisits || (cv === bestVisits && sc > bestScore)) {
      bestVisits = cv; bestScore = sc; bestIdx = i;
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
  const rootWinRatio = root.totalVisits > 0 ? totalChildWins / root.totalVisits : 0.5;

  const m = root.legalMoves[bestIdx];
  const cv = root.visits[bestIdx];
  const bestWinRatio = cv > 0 ? root.wins[bestIdx] / cv : 0.5;

  const result = m === PASS ? { type: 'pass', move: PASS, children, rootWinRatio }
                            : { type: 'place', move: m, x: m % N, y: (m / N) | 0, children, rootWinRatio };
  result.info = `win likelihood: ${bestWinRatio.toFixed(3)}`;
return result;
}

if (typeof module !== 'undefined') module.exports = { getMove };
else window.getMove = getMove;

})();
