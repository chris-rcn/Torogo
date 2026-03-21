'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs (require, process, etc.).
// Loaded as a plain <script> tag; do not add require/module/process at top level.

/**
 * RAVE (Rapid Action Value Estimation) MCTS policy.
 *
 * Like MCTS, but each tree node also maintains AMAF (All-Moves-As-First)
 * statistics for every possible child move.  The UCT selection criterion
 * blends the ordinary MC win rate with the AMAF win rate using a β that
 * decays from 1 (pure RAVE) toward 0 (pure MCTS) as visit count grows:
 *
 *   β = sqrt(EQUIV / (3·n + EQUIV))
 *
 * where n = child.visits and EQUIV is the equivalence parameter — the visit
 * count at which MCTS and RAVE estimates are weighted equally.  A larger
 * EQUIV trusts RAVE longer; a smaller one switches to MCTS sooner.
 *
 * RAVE stats at a node answer: "across all simulations that passed through
 * this node, whenever move X was played by the player to move here (at any
 * point during the playout), did they win?"  This gives every candidate far
 * more data than MCTS alone, especially early in the search.
 *
 * Interface: getMove(game, timeBudgetMs) → { type: 'pass' } | { type: 'place', x, y }
 *   game         - a live Game instance (read-only; do not mutate)
 *   timeBudgetMs - milliseconds allowed for this decision (default: 500)
 *
 * Internally the search converts the input Game to a Game2 instance via
 * game.toGame2() and uses Game2 throughout for speed.
 */

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

const performance = (typeof window !== 'undefined') ? window.performance
  : require('perf_hooks').performance;

const { getLadderStatus2: getLadderStatus } = _isNode ? require('./ladder2.js') : window;
const { PASS, BLACK, WHITE } = _isNode ? require('../game2.js') : window;

const DEFAULT_BUDGET_MS = 500;
const EXPLORATION_C = 1.4;
// Equivalence parameter.  Override with RAVE_EQUIV=<n>.
const RAVE_EQUIV = (typeof process !== 'undefined' && process.env.RAVE_EQUIV !== undefined)
  ? parseFloat(process.env.RAVE_EQUIV)
  : 300;

// Number of untried moves to sample when expanding a node.  The candidate
// with the best parent RAVE win rate is expanded first.  Set to 1 to revert
// to uniform-random expansion.
const EXPANSION_CANDIDATES = 2;

// Fixed playout count per decision.  When non-zero, overrides the time budget.
const PLAYOUTS = (typeof process !== 'undefined' && process.env.PLAYOUTS)
  ? parseInt(process.env.PLAYOUTS, 10) : 0;

// A node must reach this many visits before its ladder priors are applied.
const PAT_PRIOR_VISITS = (typeof process !== 'undefined' && process.env.PAT_PRIOR_VISITS)
  ? parseInt(process.env.PAT_PRIOR_VISITS, 10) : 50;

// Default weight for patterns absent from the training data.
const DEFAULT_WEIGHT = 0.01;

let patternSelectionRatio;

if (_isNode) {
  ({ weight: patternSelectionRatio } = require('./pattern.js'));
} else {
  // Browser: patternHash is a global from patterns.js loaded as a <script>.
  // Load patterns.csv via fetch and build the ratio table.
  const _patternHash = window.patternHash;
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
  patternSelectionRatio = function(game, x, y) {
    const hash = _patternHash(game, x, y, game.current);
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
function legalMoves(game2) {
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

// Each node owns two Float64Arrays of length N*N+1 that store AMAF statistics
// for child moves indexed by cell (y*N+x), with pass at index N*N.
// These answer: "for the player whose turn it is at this node, what is the
// AMAF win rate of playing cell idx from here?"
function makeNode(move, parent, mover, N) {
  const len = N * N + 1;
  return {
    move,
    parent,
    mover,        // player who made `move` to reach this node (null at root)
    children:  [],
    untried:   null,
    wins:         0,
    visits:       0,
    ladderSeeded: false,
    raveWins:     new Float64Array(len),
    raveVisits:   new Float64Array(len),
  };
}

// Map an integer move to its RAVE array index.
// Place moves (0..N*N-1) are their own index; PASS (-1) maps to N*N.
function moveIndex(move, N) {
  return move === PASS ? N * N : move;
}

// RAVE-blended UCT score.  The AMAF win rate is read from the *parent* node's
// arrays because those accumulate stats over all simulations through the parent
// that happened to play this child's move.
function raveScore(child, parentVisits, parentRaveWins, parentRaveVisits, midx) {
  if (child.visits === 0) return Infinity;
  const mcWR   = child.wins / child.visits;
  const rv     = parentRaveVisits[midx];
  const raveWR = rv > 0 ? parentRaveWins[midx] / rv : mcWR;
  const beta   = Math.sqrt(RAVE_EQUIV / (3 * child.visits + RAVE_EQUIV));
  return (1 - beta) * mcWR + beta * raveWR
       + EXPLORATION_C * Math.sqrt(Math.log(parentVisits) / child.visits);
}

// ── RAVE-MCTS core ────────────────────────────────────────────────────────────

function selectAndExpand(root, rootGame2, N) {
  let node = root;
  const game2 = rootGame2.clone();

  if (!node.ladderSeeded && node.visits >= LADDER_VISITS) {
    applyLadderPriors(node, game2, N);
    node.ladderSeeded = true;
  }

  // Select: descend through fully-expanded nodes using the RAVE-blended score.
  while (node.untried !== null && node.untried.length === 0
         && node.children.length > 0 && !game2.gameOver) {
    let best = null, bestScore = -Infinity;
    for (const child of node.children) {
      const s = raveScore(child, node.visits,
                          node.raveWins, node.raveVisits,
                          moveIndex(child.move, N));
      if (s > bestScore) { bestScore = s; best = child; }
    }
    node = best;
    game2.play(node.move);
    if (!node.ladderSeeded && node.visits >= LADDER_VISITS) {
      applyLadderPriors(node, game2, N);
      node.ladderSeeded = true;
    }
  }

  // Expand: attach one untried child.
  if (!game2.gameOver) {
    if (node.untried === null) node.untried = legalMoves(game2);
    if (node.untried.length > 0) {
      // Sample up to EXPANSION_CANDIDATES distinct moves and pick the one
      // with the best parent RAVE win rate (default 0.5 when unvisited).
      const k = Math.min(EXPANSION_CANDIDATES, node.untried.length);
      let bestIdx = 0;
      let bestScore = -1;
      for (let s = 0; s < k; s++) {
        const pick = s + Math.floor(Math.random() * (node.untried.length - s));
        // Swap candidate into slot s so we don't re-sample it.
        const tmp = node.untried[s]; node.untried[s] = node.untried[pick]; node.untried[pick] = tmp;
        const midx = moveIndex(node.untried[s], N);
        const rv   = node.raveVisits[midx];
        const score = rv > 0 ? node.raveWins[midx] / rv : 0.5;
        if (score > bestScore) { bestScore = score; bestIdx = s; }
      }
      // Move winner to the last position for pop().
      const winnerMove = node.untried[bestIdx];
      node.untried[bestIdx] = node.untried[node.untried.length - 1];
      node.untried[node.untried.length - 1] = winnerMove;
      const move = node.untried.pop();
      const child = makeNode(move, node, game2.current, N);
      node.children.push(child);
      node = child;
      game2.play(move);

      // After a pass, always force a second pass as a terminal tree node so the
      // simulation scores the current board position rather than rolling out
      // randomly.  Without this, rollouts from a "one pass" state play on for
      // many more random moves, inflating the pass move's apparent win rate.
      if (!game2.gameOver && game2.consecutivePasses > 0) {
        const secondPass = makeNode(PASS, node, game2.current, N);
        node.children.push(secondPass);
        node = secondPass;
        game2.play(PASS); // game2.gameOver becomes true
      }
    }
  }

  return { node, game2 };
}

// Backpropagate MCTS wins/visits and update each ancestor's RAVE arrays.
//
// RAVE update at node P: the player choosing the *next* move from P is
//   - rootPlayer         if P is the root (mover === null)
//   - opponent(P.mover)  otherwise
// We credit that player's playout moves to P's RAVE arrays so that when
// the same player selects among P's children, both sources of information
// are available.
function backpropagate(node, winner, blackPlayed, whitePlayed, rootPlayer) {
  while (node !== null) {
    node.visits++;
    if (node.mover !== null && winner === node.mover) node.wins++;

    const chooser = node.mover === null ? rootPlayer
      : (node.mover === BLACK ? WHITE : BLACK);
    const played = chooser === BLACK ? blackPlayed : whitePlayed;
    const won    = winner === chooser ? 1 : 0;
    for (const cellIdx of played) {
      node.raveVisits[cellIdx]++;
      node.raveWins[cellIdx] += won;
    }

    node = node.parent;
  }
}

// A node must reach this many visits before its ladder priors are applied.
const LADDER_VISITS = (typeof process !== 'undefined' && process.env.LADDER_VISITS)
  ? parseInt(process.env.LADDER_VISITS, 10) : 5;


// ── Ladder priors ────────────────────────────────────────────────────────────
function applyLadderPriors(node, game2, N) {
  const mover = game2.current;  // BLACK or WHITE

  // Promote a legal move to a pre-created child seeded with virtual wins/visits.
  // If the child already exists (two groups share a liberty), accumulate into it.
  // node.visits is kept in sync so the UCB exploration term is well-defined.
  function seedChild(li, wins, visits) {
    let child = node.children.find(c => c.move === li);
    if (!child) {
      if (node.untried === null) node.untried = legalMoves(game2);
      const idx = node.untried.indexOf(li);
      if (idx === -1) return;
      node.untried.splice(idx, 1);
      child = makeNode(li, node, mover, N);
      node.children.push(child);
    }
    child.wins   += wins;
    child.visits += visits;
    node.visits  += visits;
  }

  const cap        = N * N;
  const cells      = game2.cells;
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
      if (color === mover && !entry.canEscape) {
        seedChild(li, 0, 2 * groupSize);              // Don't extend doomed group.
      } else if (color !== mover && entry.canEscape) {
        seedChild(li, 0, 10);                             // Don't chase escaping group.
      } else if (color === mover && entry.canEscape && !entry.canEscapeAfterPass) {
        seedChild(li, 2 * groupSize, 2 * groupSize);      // Do escape (when urgent).
      } else if (color !== mover && !entry.canEscape && entry.canEscapeAfterPass) {
        seedChild(li, 2 * groupSize, 2 * groupSize);      // Do chase doomed group (when urgent).
      }
    }
  }
}

// ── Public interface ──────────────────────────────────────────────────────────

function getMove(game, timeBudgetMs) {
  if (game.gameOver) return { type: 'pass', info: 'game already over' };

  const N          = game.boardSize;
  const root       = makeNode(null, null, null, N);
  const game2      = game.toGame2();
  const rootPlayer = game2.current;

  const budgetMs = timeBudgetMs != null ? timeBudgetMs : DEFAULT_BUDGET_MS;
  const deadline = performance.now() + budgetMs;
  let playouts = 0;

  do {
    playouts++;
    const { node, game2: simGame2 } = selectAndExpand(root, game2, N);
    const { winner, blackPlayed, whitePlayed } = playTracked(simGame2);
    backpropagate(node, winner, blackPlayed, whitePlayed, rootPlayer);
  } while (PLAYOUTS > 0 ? playouts < PLAYOUTS : performance.now() < deadline);

  // Pick the root child with the most visits.
  let bestChild = null, bestVisits = -1;
  for (const child of root.children) {
    if (child.visits > bestVisits) { bestVisits = child.visits; bestChild = child; }
  }

  const children = root.children
    .map(c => ({
      move: c.move === PASS ? { type: 'pass' }
                            : { type: 'place', x: c.move % N, y: (c.move / N) | 0 },
      visits: c.visits, wins: c.wins,
    }))
    .sort((a, b) => b.visits - a.visits);

  const rootWins = root.children.reduce((s, c) => s + c.wins, 0);
  const rootWinRatio = root.visits > 0 ? rootWins / root.visits : 0.5;
  if (bestChild.wins === 0 && game.moveCount >= N * N / 2) return { type: 'pass', info: 'no winning line found', children, rootWinRatio };
  const m = bestChild.move;
  const result = m === PASS ? { type: 'pass', children, rootWinRatio }
                            : { type: 'place', x: m % N, y: (m / N) | 0, children, rootWinRatio };
  result.info = `win likelihood: ${(bestChild.wins / bestChild.visits).toFixed(3)}`;
  return result;
}

if (typeof module !== 'undefined') module.exports = getMove;
