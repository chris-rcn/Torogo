'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs (require, process, etc.).
// Wrapped in an IIFE to avoid polluting the global namespace.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

/**
 * Monte Carlo Tree Search (MCTS) policy.
 *
 * Builds a search tree one node per iteration.  Each iteration:
 *   1. Select  — walk down the tree picking the child with the highest UCT score
 *   2. Expand  — add one untried child to the selected leaf
 *   3. Simulate — random playout from the new node
 *   4. Backpropagate — update wins/visits up to the root
 *
 * Interface: getMove(game, timeBudgetMs) → { type: 'pass' } | { type: 'place', x, y }
 *   game         - a live Game2 instance (read-only; do not mutate)
 *   timeBudgetMs - milliseconds allowed for this decision (required)
 */

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

const performance = (typeof window !== 'undefined') ? window.performance
  : require('perf_hooks').performance;

const { PASS, BLACK, WHITE } = _isNode ? require('../game2.js') : window.Game2;

const EXPLORATION_C = 1.4;

// ── Playout helper ────────────────────────────────────────────────────────────

function playRandom(game2) {
  const N   = game2.N;
  const cap = N * N;
  const cells = game2.cells;
  const nbr   = game2._nbr;

  const empty = [];
  for (let i = 0; i < cap; i++) if (cells[i] === 0) empty.push(i);

  const moveLimit = empty.length + 20;
  let moves = 0;

  while (!game2.gameOver && moves < moveLimit) {
    let placed = false;
    let end = empty.length;

    while (end > 0) {
      const ri  = Math.floor(Math.random() * end);
      const idx = empty[ri];
      empty[ri] = empty[end - 1];
      empty[end - 1] = idx;
      end--;

      if (game2.isTrueEye(idx)) continue;
      if (!game2.isLegal(idx))  continue;

      const base = idx * 4;
      const n0 = cells[nbr[base]], n1 = cells[nbr[base + 1]],
            n2 = cells[nbr[base + 2]], n3 = cells[nbr[base + 3]];
      game2.play(idx);
      empty[end] = empty[empty.length - 1];
      empty.pop();

      if ((n0 && !cells[nbr[base]])     || (n1 && !cells[nbr[base + 1]]) ||
          (n2 && !cells[nbr[base + 2]]) || (n3 && !cells[nbr[base + 3]])) {
        empty.length = 0;
        for (let i = 0; i < cap; i++) if (cells[i] === 0) empty.push(i);
      }

      placed = true;
      moves++;
      break;
    }

    if (!placed) { game2.play(PASS); moves++; }
  }
}

// ── Tree node ─────────────────────────────────────────────────────────────────

// Returns legal non-true-eye moves as { type, x, y } objects, plus pass when
// the board is half-filled or a consecutive pass has already been played.
function getLegalMoves(game2) {
  const N   = game2.N;
  const cap = N * N;
  const moves = [];
  for (let i = 0; i < cap; i++) {
    if (game2.cells[i] !== 0)  continue;
    if (game2.isTrueEye(i))    continue;
    if (game2.isLegal(i))      moves.push({ type: 'place', x: i % N, y: (i / N) | 0 });
  }
  if (moves.length < cap / 2 || game2.consecutivePasses > 0) moves.push({ type: 'pass' });
  return moves;
}

function makeNode(move, parent, mover) {
  return {
    move,
    parent,
    mover,      // player who played `move` to reach this node (null at root)
    children:   [],
    untried:    null,
    wins:       0,
    visits:     0,
  };
}

function applyMove(game2, move) {
  if (move.type === 'place') game2.play(move.y * game2.N + move.x);
  else game2.play(PASS);
}

function uctScore(child, parentVisits) {
  if (child.visits === 0) return Infinity;
  return child.wins / child.visits +
    EXPLORATION_C * Math.sqrt(Math.log(parentVisits) / child.visits);
}

// ── MCTS core ─────────────────────────────────────────────────────────────────

function selectAndExpand(root, rootGame2) {
  let node = root;
  const game2 = rootGame2.clone();

  while (node.untried !== null && node.untried.length === 0 &&
         node.children.length > 0 && !game2.gameOver) {
    let best = null, bestScore = -1;
    for (const child of node.children) {
      const s = uctScore(child, node.visits);
      if (s > bestScore) { bestScore = s; best = child; }
    }
    node = best;
    applyMove(game2, node.move);
  }

  if (!game2.gameOver) {
    if (node.untried === null) node.untried = getLegalMoves(game2);
    if (node.untried.length > 0) {
      const idx = Math.floor(Math.random() * node.untried.length);
      const move = node.untried[idx];
      node.untried[idx] = node.untried[node.untried.length - 1];
      node.untried.pop();
      const mover = game2.current;
      const child = makeNode(move, node, mover);
      node.children.push(child);
      node = child;
      applyMove(game2, move);

      if (!game2.gameOver && game2.consecutivePasses > 0) {
        game2.play(PASS);
      }
    }
  }

  return { node, game2 };
}

function simulate(game2) {
  const wasAlreadyOver = game2.gameOver;
  playRandom(game2);
  return wasAlreadyOver ? game2.calcWinner() : game2.estimateWinner();
}

function backpropagate(node, winner) {
  while (node !== null) {
    node.visits++;
    if (node.mover !== null && winner === node.mover) node.wins++;
    node = node.parent;
  }
}

// ── Public interface ──────────────────────────────────────────────────────────

function getMove(game, timeBudgetMs) {
  if (game.gameOver) return { type: 'pass', info: 'game already over' };

  const game2      = game.cells ? game.clone() : game.toGame2();
  const rootPlayer = game2.current;

  // Obvious pass: opponent just passed and we're already winning — end the game.
  if (game2.consecutivePasses > 0 && game2.calcWinner() === rootPlayer) {
    return { type: 'pass', info: 'obvious pass: already winning' };
  }

  const root = makeNode(null, null, null);

  const deadline = performance.now() + timeBudgetMs;
  while (performance.now() < deadline) {
    const { node, game2: simGame2 } = selectAndExpand(root, game2);
    const winner = simulate(simGame2);
    backpropagate(node, winner);
  }

  let bestChild = null, bestVisits = -1;
  for (const child of root.children) {
    if (child.visits > bestVisits) { bestVisits = child.visits; bestChild = child; }
  }

  const children = root.children
    .map(c => ({ move: c.move, visits: c.visits, wins: c.wins }))
    .sort((a, b) => b.visits - a.visits);

  if (!bestChild) return { type: 'pass', info: 'no simulations completed', children };
  if (bestChild.wins === 0) return { type: 'pass', info: 'no winning line found', children };
  const result = { ...bestChild.move, children };
  result.info = `win likelihood: ${(bestChild.wins / bestChild.visits).toFixed(3)}`;
  return result;
}

if (typeof module !== 'undefined') module.exports = getMove;
else window.getMove = getMove;

})();
