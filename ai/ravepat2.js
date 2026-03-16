'use strict';

/**
 * RAVE with pattern priors (ravepat2).
 *
 * Identical to rave.js except that each newly expanded tree node is seeded
 * with virtual wins/visits derived from pattern weights rather than starting
 * at zero.  Playouts are the same fast, unbiased random rollouts as rave.js.
 *
 * Prior initialisation:
 *   When a node's untried-move list is first built, the pattern weight is
 *   looked up for every legal move.  The weights are normalised to a
 *   probability distribution p_i over the k moves at that position.  When
 *   child i is later expanded, its node is initialised with
 *
 *     priorVisits = PRIOR_VISITS
 *     priorWins   = PRIOR_VISITS * p_i
 *
 *   The UCT/RAVE score uses (wins + priorWins) / (visits + priorVisits) so
 *   the prior decays naturally as real simulation data accumulates.
 *
 *   The expansion-candidate selection also uses p_i as a fallback when no
 *   RAVE data is available yet, replacing the uniform 0.5 default in rave.js.
 *
 * Interface: getMove(game, timeBudgetMs) → { type: 'pass' } | { type: 'place', x, y }
 *   game         - a live Game instance (read-only; do not mutate)
 *   timeBudgetMs - milliseconds allowed for this decision (default: 500)
 */

const performance = (typeof window !== 'undefined') ? window.performance
  : require('perf_hooks').performance;

const { weight: patternWeight } = require('./pattern.js');

const DEFAULT_BUDGET_MS = 500;
const EXPLORATION_C = 1.4;
// Equivalence parameter.  Override with RAVE_EQUIV=<n>.
const RAVE_EQUIV = (typeof process !== 'undefined' && process.env.RAVE_EQUIV !== undefined)
  ? parseFloat(process.env.RAVE_EQUIV)
  : 300;

// Number of untried moves to sample when expanding a node.
const EXPANSION_CANDIDATES = 2;

// Fixed playout count per decision.  When non-zero, overrides the time budget.
const PLAYOUTS = (typeof process !== 'undefined' && process.env.PLAYOUTS)
  ? parseInt(process.env.PLAYOUTS, 10) : 0;

// Virtual visits injected into each new child node.  Larger values trust the
// pattern prior longer before deferring to simulation results.
const PRIOR_VISITS = 20;

// ── Fast playout helpers ──────────────────────────────────────────────────────

function applyFast(game, x, y) {
  game.board.set(x, y, game.current);
  const cap = game.board.captureGroups(x, y);
  game.consecutivePasses = 0;
  game.current = game.current === 'black' ? 'white' : 'black';
  return cap.black.length + cap.white.length;
}

// Unbiased random playout (identical to rave.js).
// Records cell indices (y*N+x) of every move for RAVE backpropagation.
// Returns { winner, blackPlayed, whitePlayed }.
function playTracked(game) {
  const size = game.boardSize;
  const board = game.board;
  const grid  = board.grid;
  const blackPlayed = [];
  const whitePlayed = [];

  const empty = [];
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++)
      if (grid[y][x] === null) empty.push(y * size + x);

  const moveLimit = empty.length + 20;
  let moves = 0;

  while (!game.gameOver && moves < moveLimit) {
    let placed = false;
    let end = empty.length;
    const current = game.current;

    while (end > 0) {
      const i = Math.floor(Math.random() * end);
      const cellIdx = empty[i];
      const x = cellIdx % size;
      const y = (cellIdx / size) | 0;
      empty[i] = empty[end - 1];
      empty[end - 1] = cellIdx;
      end--;

      const info = board.classifyEmpty(x, y, current);
      if (info.isTrueEye) continue;

      if (info.hasEmptyNeighbor) {
        if (current === 'black') blackPlayed.push(cellIdx);
        else                     whitePlayed.push(cellIdx);
        const captures = applyFast(game, x, y);
        empty[end] = empty[empty.length - 1];
        empty.pop();
        if (captures > 0) {
          empty.length = 0;
          for (let ey = 0; ey < size; ey++)
            for (let ex = 0; ex < size; ex++)
              if (grid[ey][ex] === null) empty.push(ey * size + ex);
        }
        placed = true;
        moves++;
        break;
      }

      const result = game.placeStone(x, y);
      if (result) {
        if (current === 'black') blackPlayed.push(cellIdx);
        else                     whitePlayed.push(cellIdx);
        empty[end] = empty[empty.length - 1];
        empty.pop();
        if (result > 1) {
          empty.length = 0;
          for (let ey = 0; ey < size; ey++)
            for (let ex = 0; ex < size; ex++)
              if (grid[ey][ex] === null) empty.push(ey * size + ex);
        }
        placed = true;
        moves++;
        break;
      }
    }

    if (!placed) { game.pass(); moves++; }
  }

  if (!game.gameOver) game.endGame();
  const s = game.scores;
  const winner = s.black.total > s.white.total ? 'black'
               : s.white.total > s.black.total ? 'white'
               : null;
  return { winner, blackPlayed, whitePlayed };
}

// ── Tree node ─────────────────────────────────────────────────────────────────

// Returns legal moves annotated with a normalised pattern prior (.prior field).
// Moves not yet seen in training data get DEFAULT_WEIGHT via patternWeight.
function legalMovesWithPriors(game) {
  const moves = [];
  const N = game.boardSize;
  let totalWeight = 0;

  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      if (game.board.get(x, y) !== null) continue;
      if (game.board.classifyEmpty(x, y, game.current).isTrueEye) continue;
      const probe = game.clone();
      if (probe.placeStone(x, y)) {
        const w = patternWeight(game, x, y);
        moves.push({ type: 'place', x, y, w });
        totalWeight += w;
      }
    }
  }

  const area = N * N;
  if (game.moveCount >= area / 2 || game.consecutivePasses > 0) {
    moves.push({ type: 'pass', w: 1 });
    totalWeight += 1;
  }

  // Normalise to probabilities so priorWins = PRIOR_VISITS * p is a valid
  // expected win count under the pattern distribution.
  const inv = totalWeight > 0 ? 1 / totalWeight : 0;
  for (const m of moves) m.prior = m.w * inv;

  return moves;
}

// priorWins / priorVisits: virtual wins/visits seeded from the pattern prior.
// Root and sentinel pass nodes are created with 0 priors.
function makeNode(move, parent, mover, N, priorWins, priorVisits) {
  const len = N * N + 1;
  return {
    move,
    parent,
    mover,
    children:     [],
    untried:      null,
    wins:         0,
    visits:       0,
    priorWins:    priorWins    || 0,
    priorVisits:  priorVisits  || 0,
    raveWins:     new Float64Array(len),
    raveVisits:   new Float64Array(len),
  };
}

function moveIndex(move, N) {
  return move.type === 'pass' ? N * N : move.y * N + move.x;
}

function applyMove(game, move) {
  if (move.type === 'place') game.placeStone(move.x, move.y);
  else game.pass();
}

// RAVE-blended UCT score using (wins + priorWins) / (visits + priorVisits).
function raveScore(child, parentVisits, parentRaveWins, parentRaveVisits, midx) {
  const ev = child.visits + child.priorVisits;   // effective visits
  if (ev === 0) return Infinity;
  const mcWR   = (child.wins + child.priorWins) / ev;
  const rv     = parentRaveVisits[midx];
  const raveWR = rv > 0 ? parentRaveWins[midx] / rv : mcWR;
  const beta   = Math.sqrt(RAVE_EQUIV / (3 * ev + RAVE_EQUIV));
  return (1 - beta) * mcWR + beta * raveWR
       + EXPLORATION_C * Math.sqrt(Math.log(parentVisits) / ev);
}

// ── RAVE-MCTS core ────────────────────────────────────────────────────────────

function selectAndExpand(root, rootGame, N) {
  let node = root;
  const game = rootGame.clone();

  // Select: descend through fully-expanded nodes using the RAVE-blended score.
  while (node.untried !== null && node.untried.length === 0
         && node.children.length > 0 && !game.gameOver) {
    let best = null, bestScore = -Infinity;
    for (const child of node.children) {
      const s = raveScore(child, node.visits,
                          node.raveWins, node.raveVisits,
                          moveIndex(child.move, N));
      if (s > bestScore) { bestScore = s; best = child; }
    }
    node = best;
    applyMove(game, node.move);
  }

  // Expand: attach one untried child.
  if (!game.gameOver) {
    if (node.untried === null) node.untried = legalMovesWithPriors(game);
    if (node.untried.length > 0) {
      // Sample up to EXPANSION_CANDIDATES distinct moves and pick the one
      // with the best parent RAVE win rate, falling back to the pattern prior
      // when no RAVE data is available.
      const k = Math.min(EXPANSION_CANDIDATES, node.untried.length);
      let bestIdx = 0;
      let bestScore = -1;
      for (let s = 0; s < k; s++) {
        const pick = s + Math.floor(Math.random() * (node.untried.length - s));
        const tmp = node.untried[s]; node.untried[s] = node.untried[pick]; node.untried[pick] = tmp;
        const midx = moveIndex(node.untried[s], N);
        const rv   = node.raveVisits[midx];
        const score = rv > 0 ? node.raveWins[midx] / rv : (node.untried[s].prior || 0.5);
        if (score > bestScore) { bestScore = score; bestIdx = s; }
      }
      // Move winner to the last position for pop().
      const winnerMove = node.untried[bestIdx];
      node.untried[bestIdx] = node.untried[node.untried.length - 1];
      node.untried[node.untried.length - 1] = winnerMove;
      const move = node.untried.pop();
      // Seed the child with pattern-derived virtual wins/visits.
      const p = move.prior || 0;
      const child = makeNode(move, node, game.current, N,
                             PRIOR_VISITS * p, PRIOR_VISITS);
      node.children.push(child);
      node = child;
      applyMove(game, move);

      // After a pass, force a second pass as a terminal tree node.
      if (!game.gameOver && game.consecutivePasses > 0) {
        const secondPass = makeNode({ type: 'pass' }, node, game.current, N);
        node.children.push(secondPass);
        node = secondPass;
        applyMove(game, { type: 'pass' });
      }
    }
  }

  return { node, game };
}

// Backpropagate MCTS wins/visits and update each ancestor's RAVE arrays.
function backpropagate(node, winner, blackPlayed, whitePlayed, rootPlayer) {
  while (node !== null) {
    node.visits++;
    if (node.mover !== null && winner === node.mover) node.wins++;

    const chooser = node.mover === null ? rootPlayer
      : (node.mover === 'black' ? 'white' : 'black');
    const played = chooser === 'black' ? blackPlayed : whitePlayed;
    const won    = winner === chooser ? 1 : 0;
    for (const cellIdx of played) {
      node.raveVisits[cellIdx]++;
      node.raveWins[cellIdx] += won;
    }

    node = node.parent;
  }
}

// ── Public interface ──────────────────────────────────────────────────────────

function getMove(game, timeBudgetMs) {
  if (game.gameOver) return { type: 'pass', info: 'game already over' };

  const N          = game.boardSize;
  const rootPlayer = game.current;
  const root       = makeNode(null, null, null, N);

  const budgetMs = timeBudgetMs != null ? timeBudgetMs : DEFAULT_BUDGET_MS;
  const deadline = performance.now() + budgetMs;
  let playouts = 0;

  do {
    playouts++;
    const { node, game: simGame } = selectAndExpand(root, game, N);
    const { winner, blackPlayed, whitePlayed } = playTracked(simGame);
    backpropagate(node, winner, blackPlayed, whitePlayed, rootPlayer);
  } while (PLAYOUTS > 0 ? playouts < PLAYOUTS : performance.now() < deadline);

  // Pick the root child with the most visits (most robust criterion).
  let bestChild = null, bestVisits = -1;
  for (const child of root.children) {
    if (child.visits > bestVisits) { bestVisits = child.visits; bestChild = child; }
  }

  const children = root.children
    .map(c => ({ move: c.move, visits: c.visits, wins: c.wins }))
    .sort((a, b) => b.visits - a.visits);

  if (bestChild.wins === 0 && game.moveCount >= N * N / 2) return { type: 'pass', info: 'no winning line found', children };
  const result = { ...bestChild.move, children };
  result.info = `win likelihood: ${(bestChild.wins / bestChild.visits).toFixed(3)}`;
  return result;
}

if (typeof module !== 'undefined') module.exports = getMove;
