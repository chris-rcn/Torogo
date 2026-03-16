'use strict';

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
 *   game         - a live Game instance (read-only; do not mutate)
 *   timeBudgetMs - milliseconds allowed for this decision (default: 500)
 */

const performance = (typeof window !== 'undefined') ? window.performance
  : require('perf_hooks').performance;

const DEFAULT_BUDGET_MS = 500;
const EXPLORATION_C = 1.4; // UCT exploration constant

// ── Playout helpers (same as mc.js) ──────────────────────────────────────────

function applyFast(game, x, y) {
  game.board.set(x, y, game.current);
  const cap = game.board.captureGroups(x, y);
  game.consecutivePasses = 0;
  game.current = game.current === 'black' ? 'white' : 'black';
  return cap.black.length + cap.white.length;
}

function playRandom(game) {
  const size = game.boardSize;
  const empty = [];
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++)
      if (game.board.get(x, y) === null) empty.push([x, y]);

  const moveLimit = empty.length + 20;
  let moves = 0;

  while (!game.gameOver && moves < moveLimit) {
    let placed = false;
    let end = empty.length;

    while (end > 0) {
      const i = Math.floor(Math.random() * end);
      const [x, y] = empty[i];
      empty[i] = empty[end - 1];
      empty[end - 1] = [x, y];
      end--;

      const info = game.board.classifyEmpty(x, y, game.current);
      if (info.isTrueEye) continue;

      if (info.hasEmptyNeighbor) {
        const captures = applyFast(game, x, y);
        empty[end] = empty[empty.length - 1];
        empty.pop();
        if (captures > 0) {
          empty.length = 0;
          for (let ey = 0; ey < size; ey++)
            for (let ex = 0; ex < size; ex++)
              if (game.board.get(ex, ey) === null) empty.push([ex, ey]);
        }
        placed = true;
        moves++;
        break;
      }

      const result = game.placeStone(x, y);
      if (result) {
        empty[end] = empty[empty.length - 1];
        empty.pop();
        if (result > 1) {
          empty.length = 0;
          for (let ey = 0; ey < size; ey++)
            for (let ex = 0; ex < size; ex++)
              if (game.board.get(ex, ey) === null) empty.push([ex, ey]);
        }
        placed = true;
        moves++;
        break;
      }
    }

    if (!placed) {
      game.pass();
      moves++;
    }
  }

  if (!game.gameOver) game.endGame();
}

// ── Tree node ────────────────────────────────────────────────────────────────

function legalMoves(game) {
  const moves = [];
  for (let y = 0; y < game.boardSize; y++)
    for (let x = 0; x < game.boardSize; x++) {
      if (game.board.get(x, y) !== null) continue;
      const probe = game.clone();
      if (probe.placeStone(x, y)) moves.push({ type: 'place', x, y });
    }
  const area = game.boardSize * game.boardSize;
  if (game.moveCount >= area / 2 || game.consecutivePasses > 0) moves.push({ type: 'pass' });
  return moves;
}

function makeNode(move, parent, mover) {
  return {
    move,       // move that led to this node (null for root)
    parent,     // parent node (null for root)
    mover,      // the player who played `move` (null for root)
    children:   [],
    untried:    null, // lazily populated array of untried moves
    wins:       0,    // wins from the perspective of `mover`
    visits:     0,
  };
}

function applyMove(game, move) {
  if (move.type === 'place') game.placeStone(move.x, move.y);
  else game.pass();
}

// UCT score: win rate + exploration bonus.  The win rate is from the
// perspective of the player who *made the move into* this node — which is the
// opponent of the player whose turn it is at the parent.  So during selection
// at the parent we want the child whose previous-player win rate is highest.
function uctScore(child, parentVisits) {
  if (child.visits === 0) return Infinity;
  const winRate = child.wins / child.visits;
  return winRate + EXPLORATION_C * Math.sqrt(Math.log(parentVisits) / child.visits);
}

// ── MCTS core ────────────────────────────────────────────────────────────────

function selectAndExpand(root, rootGame) {
  let node = root;
  const game = rootGame.clone();

  // Select: walk down fully-expanded nodes using UCT.
  while (node.untried !== null && node.untried.length === 0 && node.children.length > 0 && !game.gameOver) {
    let best = null;
    let bestScore = -1;
    for (const child of node.children) {
      const s = uctScore(child, node.visits);
      if (s > bestScore) { bestScore = s; best = child; }
    }
    node = best;
    applyMove(game, node.move);
  }

  // Expand: if the node has untried moves, add one child.
  if (!game.gameOver) {
    if (node.untried === null) node.untried = legalMoves(game);
    if (node.untried.length > 0) {
      const idx = Math.floor(Math.random() * node.untried.length);
      const move = node.untried[idx];
      node.untried[idx] = node.untried[node.untried.length - 1];
      node.untried.pop();
      const mover = game.current; // player whose turn it is makes this move
      const child = makeNode(move, node, mover);
      node.children.push(child);
      node = child;
      applyMove(game, move);

      if (!game.gameOver && game.consecutivePasses > 0) {
        const secondPass = makeNode({ type: 'pass' }, node, game.current);
        node.children.push(secondPass);
        node = secondPass;
        applyMove(game, { type: 'pass' });
      }
    }
  }

  return { node, game };
}

function simulate(game) {
  playRandom(game);
  const s = game.scores;
  return s.black.total > s.white.total ? 'black'
       : s.white.total > s.black.total ? 'white'
       : null;
}

function backpropagate(node, winner) {
  while (node !== null) {
    node.visits++;
    if (node.mover !== null && winner === node.mover) node.wins++;
    node = node.parent;
  }
}

// ── Public interface ─────────────────────────────────────────────────────────

function getMove(game, timeBudgetMs) {
  if (game.gameOver) return { type: 'pass', info: 'game already over' };

  const root = makeNode(null, null, null);

  const budgetMs = timeBudgetMs != null ? timeBudgetMs : DEFAULT_BUDGET_MS;
  const deadline = performance.now() + budgetMs;

  while (performance.now() < deadline) {
    const { node, game: simGame } = selectAndExpand(root, game);
    const winner = simulate(simGame);
    backpropagate(node, winner);
  }

  // Pick the root child with the most visits (most robust choice).
  let bestChild = null;
  let bestVisits = -1;
  for (const child of root.children) {
    if (child.visits > bestVisits) {
      bestVisits = child.visits;
      bestChild = child;
    }
  }

  // Root children stats sorted by visits, available to callers that want them.
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
