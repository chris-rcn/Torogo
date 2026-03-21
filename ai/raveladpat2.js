'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs (require, process, etc.).
// Loaded as a plain <script> tag; do not add require/module/process at top level.

/**
 * RAVE (Rapid Action Value Estimation) MCTS policy with ladder + pattern priors.
 *
 * Node structure: all stats kept in compact child-indexed arrays on the parent.
 * Child nodes are promoted lazily after N_EXPAND playout visits.
 * Priors (ladder + pattern) are computed once at node creation time.
 *
 * Interface: getMove(game, timeBudgetMs) → { type: 'pass' } | { type: 'place', x, y }
 *   game         - a live Game instance (read-only; do not mutate)
 *   timeBudgetMs - milliseconds allowed for this decision (default: 500)
 */

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

const performance = (typeof window !== 'undefined') ? window.performance
  : require('perf_hooks').performance;

const { getLadderStatus2: getLadderStatus } = _isNode ? require('../ladder2.js') : window;
const { PASS, BLACK, WHITE } = _isNode ? require('../game2.js') : window;

const DEFAULT_BUDGET_MS = 500;
const EXPLORATION_C = 1.4;
// Equivalence parameter.  Override with RAVE_EQUIV=<n>.
const RAVE_EQUIV = (typeof process !== 'undefined' && process.env.RAVE_EQUIV !== undefined)
  ? parseFloat(process.env.RAVE_EQUIV)
  : 300;

// Fixed playout count per decision.  When non-zero, overrides the time budget.
const PLAYOUTS = (typeof process !== 'undefined' && process.env.PLAYOUTS)
  ? parseInt(process.env.PLAYOUTS, 10) : 0;

// Total virtual visits contributed by the pattern prior across all children.
const PAT_PRIOR_WEIGHT = (typeof process !== 'undefined' && process.env.PAT_PRIOR_WEIGHT)
  ? parseFloat(process.env.PAT_PRIOR_WEIGHT) : 0;

// Default weight for patterns absent from the training data.
const DEFAULT_WEIGHT = 0.01;

// Minimum playout visits before a child node is promoted (allocated).
const N_EXPAND = (typeof process !== 'undefined' && process.env.N_EXPAND)
  ? parseInt(process.env.N_EXPAND, 10) : 0;

let patternSelectionRatio;

if (_isNode) {
  ({ weight: patternSelectionRatio } = require('../patternValue.js'));
} else {
  // Browser: patternHash2 is a global from patterns2.js loaded as a <script>.
  // Load patterns.csv via fetch and build the ratio table.
  const _patternHash2 = window.patternHash2;
  const _table = new Map();
  fetch('patterns.csv')
    .then(r => r.text())
    .then(text => {
      for (const line of text.trim().split('\n')) {
        if (!line.trim()) continue;
        const parts = line.split(',');
        const hash  = parseInt(parts[0], 10);
        const ratio = parseFloat(parts[1]);
        if (!Number.isNaN(hash) && !Number.isNaN(ratio)) _table.set(hash, ratio);
      }
    });
  patternSelectionRatio = function(game2, idx) {
    const hash = _patternHash2(game2, idx, game2.current);
    return _table.has(hash) ? _table.get(hash) : DEFAULT_WEIGHT;
  };
}

// ── Fast playout helpers ──────────────────────────────────────────────────────

// Random playout using Game2.  Records the flat cell indices of every non-pass
// move made by each player.  Returns { winner, blackPlayed, whitePlayed }.
function playTracked(game2) {
  const wasAlreadyOver = game2.gameOver;
  const N     = game2.N;
  const cap   = N * N;
  const cells = game2.cells;
  const nbr   = game2._nbr;
  const blackPlayed = [];
  const whitePlayed = [];

  const empty = [];
  for (let i = 0; i < cap; i++) {
    if (cells[i] === 0) empty.push(i);
  }

  const moveLimit = empty.length + 20;
  let moves = 0;

  while (!game2.gameOver && moves < moveLimit) {
    let placed = false;
    let end = empty.length;
    const current = game2.current;

    while (end > 0) {
      const ri  = Math.floor(Math.random() * end);
      const idx = empty[ri];
      empty[ri] = empty[end - 1];
      empty[end - 1] = idx;
      end--;

      if (game2.isTrueEye(idx)) continue;
      if (!game2.isLegal(idx))  continue;

      if (current === BLACK) blackPlayed.push(idx);
      else                    whitePlayed.push(idx);

      // Snapshot neighbour occupancy to detect captures after play.
      const base = idx * 4;
      const n0 = cells[nbr[base]], n1 = cells[nbr[base + 1]],
            n2 = cells[nbr[base + 2]], n3 = cells[nbr[base + 3]];
      game2.play(idx);
      empty[end] = empty[empty.length - 1];
      empty.pop();

      // If any previously occupied neighbour became empty, captures occurred.
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

  let winner;
  if (wasAlreadyOver) {
    winner = game2.calcWinner();
  } else {
    winner = game2.estimateWinner();
  }
  return { winner, blackPlayed, whitePlayed };
}

// ── Tree node ─────────────────────────────────────────────────────────────────

// Enumerate legal non-true-eye moves from a Game2 state as integers.
// Place moves are y*N+x; pass is PASS (-1).
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
  if (game2.moveCount >= cap / 2 || game2.consecutivePasses > 0) {
    moves.push(PASS);
  }
  return moves;
}

// Map an integer move to its childIndex array position.
// Place moves (0..N*N-1) are their own index; PASS (-1) maps to N*N.
function moveIndex(move, N) {
  return move === PASS ? N * N : move;
}

// Create a node for the position reached by `move`.
// `game2` is the game state AFTER `move` was played (or initial state for root).
// `mover` is the player who played `move` (null at root).
// All priors are computed eagerly from the current game state.
function makeNode(move, parent, mover, game2, N) {
  const movesArr = getLegalMoves(game2);
  const M = movesArr.length;

  const legalMoves = new Int32Array(M);
  for (let i = 0; i < M; i++) legalMoves[i] = movesArr[i];

  // childIndex[moveIndex(m, N)] = child array index, or -1 if not a legal move.
  const childIndex = new Int16Array(N * N + 1).fill(-1);
  for (let i = 0; i < M; i++) {
    childIndex[moveIndex(legalMoves[i], N)] = i;
  }

  const children    = new Array(M).fill(null);
  const childWins   = new Float32Array(M);
  const childVisits = new Float32Array(M);
  const raveWins    = new Float32Array(M);
  const raveVisits  = new Float32Array(M);
  const priorWins   = new Float32Array(M);
  const priorVisits = new Float32Array(M);

  // Ladder priors — same logic as the former applyPriors().
  // `chooser` = player to move at this node = game2.current after the play.
  const chooser = game2.current;
  const cap = N * N;
  const cells = game2.cells;
  const visitedGids = new Set();

  for (let i = 0; i < cap; i++) {
    const color = cells[i];
    if (color === 0) continue;
    const gid = game2._gid[i];
    if (visitedGids.has(gid)) continue;
    if (game2._ss[gid] < 2) continue;   // only groups of size ≥ 2
    visitedGids.add(gid);

    if (game2._ls[gid] > 2) continue;   // skip groups with >2 liberties

    const statusEntries = getLadderStatus(game2, i);
    if (!statusEntries) continue;

    const groupSize = game2._ss[gid];

    for (const entry of statusEntries) {
      const { x: lx, y: ly } = entry.liberty;
      const li = ly * N + lx;
      const ci = childIndex[li];
      if (ci === -1) continue;

      if (color === chooser && !entry.canEscape) {
        priorVisits[ci] += 2 * groupSize;              // Don't extend doomed group.
      } else if (color !== chooser && entry.canEscape) {
        priorVisits[ci] += 10;                          // Don't chase escaping group.
      } else if (color === chooser && entry.canEscape && !entry.canEscapeAfterPass) {
        priorWins[ci]   += 2 * groupSize;              // Do escape (when urgent).
        priorVisits[ci] += 2 * groupSize;
      } else if (color !== chooser && !entry.canEscape && entry.canEscapeAfterPass) {
        priorWins[ci]   += 2 * groupSize;              // Do chase doomed group (when urgent).
        priorVisits[ci] += 2 * groupSize;
      }
    }
  }

  // Pattern priors — max-normalised, applied to non-PASS moves only.
  if (PAT_PRIOR_WEIGHT > 0) {
    const nonPassIndices = [];
    for (let i = 0; i < M; i++) {
      if (legalMoves[i] !== PASS) nonPassIndices.push(i);
    }
    if (nonPassIndices.length > 0) {
      const ratios = nonPassIndices.map(ci => patternSelectionRatio(game2, legalMoves[ci]));
      const maxR   = ratios.reduce((m, r) => r > m ? r : m, 0);
      const norm   = maxR > 0 ? 1 / maxR : 0;
      const vpp    = PAT_PRIOR_WEIGHT / nonPassIndices.length;
      for (let k = 0; k < nonPassIndices.length; k++) {
        const ci = nonPassIndices[k];
        priorWins[ci]   += ratios[k] * norm * vpp;
        priorVisits[ci] += vpp;
      }
    }
  }

  // totalVisits starts as sum(priorVisits) and is incremented on each playout.
  let totalVisits = 0;
  for (let i = 0; i < M; i++) totalVisits += priorVisits[i];

  return {
    move,
    parent,
    mover,        // player who made `move` to reach this node (null at root)
    totalVisits,  // cached sum of priorVisits + all childVisits

    legalMoves,   // Int32Array(M)
    childIndex,   // Int16Array(N*N+1) — move → child index, or -1
    children,     // Array(M) — promoted child node or null

    childWins,    // Float32Array(M) — playout wins per child
    childVisits,  // Float32Array(M) — playout visits per child
    priorWins,    // Float32Array(M) — virtual wins from priors
    priorVisits,  // Float32Array(M) — virtual visits from priors

    raveWins,     // Float32Array(M) — RAVE wins, indexed by child index
    raveVisits,   // Float32Array(M) — RAVE visits, indexed by child index
  };
}

// RAVE-blended UCT score for child index i of node.
function raveScore(i, node) {
  const totalV = node.priorVisits[i] + node.childVisits[i];
  if (totalV === 0) return Infinity;
  const mcWR   = (node.priorWins[i] + node.childWins[i]) / totalV;
  const rv     = node.raveVisits[i];
  const raveWR = rv > 0 ? node.raveWins[i] / rv : mcWR;
  const beta   = Math.sqrt(RAVE_EQUIV / (3 * totalV + RAVE_EQUIV));
  return (1 - beta) * mcWR + beta * raveWR
       + EXPLORATION_C * Math.sqrt(Math.log(node.totalVisits + 1) / totalV);
}

// ── RAVE-MCTS core ────────────────────────────────────────────────────────────

function selectAndExpand(root, rootGame2, N) {
  let node = root;
  const game2 = rootGame2.clone();
  let leafChildIdx = -1;

  while (!game2.gameOver) {
    const M = node.legalMoves.length;
    if (M === 0) break;

    // Select best child by RAVE-blended score.
    // Use reservoir sampling to break ties uniformly at random.
    let best = 0, bestScore = -Infinity, numBest = 0;
    for (let i = 0; i < M; i++) {
      const s = raveScore(i, node);
      if (s > bestScore) { bestScore = s; best = i; numBest = 1; }
      else if (s === bestScore) { numBest++; if (Math.random() * numBest < 1) best = i; }
    }

    leafChildIdx = best;
    const mover = game2.current;   // player making this move
    game2.play(node.legalMoves[best]);

    // Promote child to a full node once it has accumulated enough visits.
    if (node.children[best] === null && node.childVisits[best] >= N_EXPAND) {
      node.children[best] = makeNode(node.legalMoves[best], node, mover, game2, N);
    }

    // After a pass, always force a second pass so the playout scores the current
    // board position (consecutive passes end the game).  This prevents rollouts
    // from a single-pass state playing on for many random moves and inflating
    // the pass move's apparent win rate.
    if (!game2.gameOver && game2.consecutivePasses > 0) {
      game2.play(PASS);
      break;
    }

    // Descend into the promoted child, if available.
    if (node.children[best] !== null) {
      node = node.children[best];
      leafChildIdx = -1;  // fully descended; backprop walks from this node up
      continue;
    }

    break;  // unpromoted leaf — run playout from here
  }

  return { node, leafChildIdx, game2 };
}

// Backpropagate playout result and update RAVE statistics.
//
// `leafChildIdx` is the index within node.legalMoves of the unpromoted child
// that was played last (or -1 if we descended all the way to a promoted node).
//
// childMover(n): the player choosing the next move from node n.
//   - rootPlayer if n is the root (n.mover === null)
//   - opponent(n.mover) otherwise
function backpropagate(node, leafChildIdx, winner, blackPlayed, whitePlayed, rootPlayer, N) {
  function childMover(n) {
    return n.mover === null ? rootPlayer : (n.mover === BLACK ? WHITE : BLACK);
  }

  // Update the unpromoted leaf child stats (if we stopped before descending).
  if (leafChildIdx !== -1) {
    const won = winner === childMover(node) ? 1 : 0;
    node.childVisits[leafChildIdx]++;
    node.childWins[leafChildIdx] += won;
    node.totalVisits++;
  }

  // Walk up the tree, updating each parent's child arrays and RAVE arrays.
  while (node.parent !== null) {
    const ci  = node.parent.childIndex[moveIndex(node.move, N)];
    const won = winner === node.mover ? 1 : 0;
    node.parent.childVisits[ci]++;
    node.parent.childWins[ci] += won;
    node.parent.totalVisits++;

    // RAVE update at node.parent: credit the parent's chooser's playout moves.
    const chooser = childMover(node.parent);
    const played  = chooser === BLACK ? blackPlayed : whitePlayed;
    const rWon    = winner === chooser ? 1 : 0;
    for (const move of played) {
      // played contains raw cell indices (never PASS), so childIndex[move] works directly.
      const ri = node.parent.childIndex[move];
      if (ri !== -1) {
        node.parent.raveVisits[ri]++;
        node.parent.raveWins[ri] += rWon;
      }
    }

    node = node.parent;
  }
}

// ── Public interface ──────────────────────────────────────────────────────────

function getMove(game, timeBudgetMs) {
  if (game.gameOver) return { type: 'pass', info: 'game already over' };

  const N          = game.cells ? game.N : game.boardSize;
  const game2      = game.cells ? game.clone() : game.toGame2();
  const rootPlayer = game2.current;

  const root = makeNode(null, null, null, game2, N);

  const budgetMs = timeBudgetMs != null ? timeBudgetMs : DEFAULT_BUDGET_MS;
  const deadline = performance.now() + budgetMs;
  let playouts = 0;

  do {
    playouts++;
    const { node, leafChildIdx, game2: simGame2 } = selectAndExpand(root, game2, N);
    const { winner, blackPlayed, whitePlayed } = playTracked(simGame2);
    backpropagate(node, leafChildIdx, winner, blackPlayed, whitePlayed, rootPlayer, N);
  } while (PLAYOUTS > 0 ? playouts < PLAYOUTS : performance.now() < deadline);

  // Best child: most playout visits.
  const M = root.legalMoves.length;
  let bestIdx = 0, bestVisits = -1;
  for (let i = 0; i < M; i++) {
    if (root.childVisits[i] > bestVisits) { bestVisits = root.childVisits[i]; bestIdx = i; }
  }

  const children = [];
  for (let i = 0; i < M; i++) {
    const m = root.legalMoves[i];
    children.push({
      move: m === PASS ? { type: 'pass' } : { type: 'place', x: m % N, y: (m / N) | 0 },
      visits: root.childVisits[i],
      wins:   root.childWins[i],
    });
  }
  children.sort((a, b) => b.visits - a.visits);

  let totalChildWins = 0;
  for (let i = 0; i < M; i++) totalChildWins += root.childWins[i];
  const rootWinRatio = root.totalVisits > 0 ? totalChildWins / root.totalVisits : 0.5;

  if (root.childWins[bestIdx] === 0 && game.moveCount >= N * N / 2) {
    return { type: 'pass', info: 'no winning line found', children, rootWinRatio };
  }

  const m = root.legalMoves[bestIdx];
  const totalV = root.priorVisits[bestIdx] + root.childVisits[bestIdx];
  const bestWinRatio = totalV > 0
    ? (root.priorWins[bestIdx] + root.childWins[bestIdx]) / totalV
    : 0.5;

  const result = m === PASS ? { type: 'pass', children, rootWinRatio }
                            : { type: 'place', x: m % N, y: (m / N) | 0, children, rootWinRatio };
  result.info = `win likelihood: ${bestWinRatio.toFixed(3)}`;
  return result;
}

if (typeof module !== 'undefined') module.exports = getMove;
