'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs (require, process, etc.).
// Loaded as a plain <script> tag; do not add require/module/process at top level.
// ladder.js must be loaded before this file.

/**
 * Like ravelad, but adds pattern priors.
 */

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

const performance = (typeof window !== 'undefined') ? window.performance
  : require('perf_hooks').performance;

const { getLadderStatus } = _isNode ? require('../ladder.js') : window;

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
const LADDER_VISITS = (typeof process !== 'undefined' && process.env.LADDER_VISITS)
  ? parseInt(process.env.LADDER_VISITS, 10) : 5;

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

// Random playout that records the cell indices (y*N+x) of every move made by
// each player.  Pass moves carry no cell index and are not recorded.
// Returns { winner, blackPlayed, whitePlayed }.
function playTracked(game) {
  const wasAlreadyOver = game.gameOver;
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

      if (board.classifyEmpty(x, y, current).isTrueEye) continue;

      const result = game.placeStone(x, y);
      if (result) {
        if (current === 'black') blackPlayed.push(cellIdx);
        else                     whitePlayed.push(cellIdx);
        empty[end] = empty[empty.length - 1];
        empty.pop();
        if (result !== true) {
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

  let winner;
  if (wasAlreadyOver) {
    winner = game.calcWinner();
  } else {
    winner = game.estimateWinner();
  }
  return { winner, blackPlayed, whitePlayed };
}

// ── Tree node ─────────────────────────────────────────────────────────────────

function legalMoves(game) {
  const moves = [];
  for (let y = 0; y < game.boardSize; y++)
    for (let x = 0; x < game.boardSize; x++) {
      if (game.board.get(x, y) !== null) continue;
      if (game.board.classifyEmpty(x, y, game.current).isTrueEye) continue;
      if (game.isLegal(x, y)) moves.push({ type: 'place', x, y });
    }
  const area = game.boardSize * game.boardSize;
  if (game.moveCount >= area / 2 || game.consecutivePasses > 0) moves.push({ type: 'pass' });
  return moves;
}

// Returns legal moves annotated with a normalised pattern prior (.prior field).
function legalMovesWithPriors(game) {
  const moves = [];
  const N = game.boardSize;

  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      if (game.board.get(x, y) !== null) continue;
      if (game.board.classifyEmpty(x, y, game.current).isTrueEye) continue;
      if (game.isLegal(x, y)) moves.push({ type: 'place', x, y });
    }
  }

  const area = N * N;
  if (game.moveCount >= area / 2 || game.consecutivePasses > 0) {
    moves.push({ type: 'pass' });
  }

  let totalWeight = 0;

  for (const m of moves) {
    m.w = m.type === 'place' ? patternSelectionRatio(game, m.x, m.y) : 1;
    totalWeight += m.w;
  }

  const inv = totalWeight > 0 ? 1 / totalWeight : 0;
  for (const m of moves) m.prior = m.w * inv;

  return moves;
}

// Each node owns two Float64Arrays of length N*N+1 that store AMAF statistics
// for child moves indexed by cell (y*N+x), with pass at index N*N.
// These answer: "for the player whose turn it is at this node, what is the
// AMAF win rate of playing cell idx from here?"
function makeNode(move, parent, mover, N, wins, visits) {
  const len = N * N + 1;
  return {
    move,
    parent,
    mover,        // player who made `move` to reach this node (null at root)
    children:  [],
    untried:   null,
    wins:         wins,
    visits:       visits,
    ladderSeeded: false,
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

function selectAndExpand(root, rootGame, N) {
  let node = root;
  const game = rootGame.clone();

  if (!node.ladderSeeded && node.visits >= LADDER_VISITS) {
    applyLadderPriors(node, game, N);
    node.ladderSeeded = true;
  }

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
    if (!node.ladderSeeded && node.visits >= LADDER_VISITS) {
      applyLadderPriors(node, game, N);
      node.ladderSeeded = true;
    }
  }

  // Expand: attach one untried child.
  if (!game.gameOver) {
    if (node.untried === null) {
      //node.untried = legalMoves(game);
      node.untried = legalMovesWithPriors(game);
    }
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
      const p = move.prior || 0;
      const child = makeNode(move, node, game.current, N, p * PAT_PRIOR_VISITS, PAT_PRIOR_VISITS);
      node.children.push(child);
      node = child;
      applyMove(game, move);

      // After a pass, always force a second pass as a terminal tree node so the
      // simulation scores the current board position rather than rolling out
      // randomly.  Without this, rollouts from a "one pass" state play on for
      // many more random moves, inflating the pass move's apparent win rate.
      if (!game.gameOver && game.consecutivePasses > 0) {
        const secondPass = makeNode({ type: 'pass' }, node, game.current, N, 0, 0);
        node.children.push(secondPass);
        node = secondPass;
        applyMove(game, { type: 'pass' }); // game.gameOver becomes true
      }
    }
  }

  return { node, game };
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

// ── Ladder priors ────────────────────────────────────────────────────────────
function applyLadderPriors(node, game, N) {
  //return;  // For now, we skip ladder priors.

  const mover = game.current;

  // Promote a legal move to a pre-created child seeded with virtual wins/visits.
  // If the child already exists (two groups share a liberty), accumulate into it.
  // node.visits is kept in sync so the UCB exploration term is well-defined.
  function seedChild(lx, ly, wins, visits) {
    let child = node.children.find(c => c.move.type === 'place' && c.move.x === lx && c.move.y === ly);
    if (!child) {
      if (node.untried === null) node.untried = legalMoves(game);
      const idx = node.untried.findIndex(m => m.type === 'place' && m.x === lx && m.y === ly);
      if (idx === -1) return;
      const [move] = node.untried.splice(idx, 1);
      const p = move.prior || 0;
      child = makeNode(move, node, mover, N, p * PAT_PRIOR_VISITS, PAT_PRIOR_VISITS);
      node.children.push(child);
    }
    child.wins   += wins;
    child.visits += visits;
    node.visits  += visits;
  }

  const visited = new Set();
  for (let py = 0; py < N; py++) {
    for (let px = 0; px < N; px++) {
      const group = game.board.getGroup(px, py);
      if (group.length < 2) continue;
      const groupColor = game.board.get(px, py);
      const gid = game.board._gid[game.board._idx(px, py)];
      if (visited.has(gid)) continue;
      visited.add(gid);
      const libs = game.board.getLiberties(group);
      if (libs.size > 2) continue;

      const statusEntries = getLadderStatus(game, px, py);
      if (!statusEntries) continue;

      for (const entry of statusEntries) {
        const { liberty: { x: lx, y: ly } } = entry;
        if (groupColor === mover && !entry.canEscape) {  // Don't extend doomed group.
          seedChild(lx, ly, 0, 3 * group.length + 1);
        } else if (groupColor !== mover && entry.canEscape) {  // Don't chase escaping group.
          seedChild(lx, ly, 0, 45);  // The importance of this prior does not depend on the group size.
        } else if (groupColor === mover && entry.canEscape && !entry.canEscapeAfterPass) {  // Do escape (when urgent).
          seedChild(lx, ly, 2 * group.length, 2 * group.length);
        } else if (groupColor !== mover && !entry.canEscape && entry.canEscapeAfterPass) {  // Do chase doomed group (when urgent).
          seedChild(lx, ly, 2 * group.length, 2 * group.length);
        }
      }
    }
  }

//  const legals = legalMoves(game);
//  for (const move of legals) {
//    if (move.type !== 'place') continue;
//    const ratio = patternSelectionRatio(game, move.x, move.y);
//    seedChild(move.x, move.y, PAT_PRIOR_VISITS * ratio, PAT_PRIOR_VISITS);
//  }
}

// ── Public interface ──────────────────────────────────────────────────────────

function getMove(game, timeBudgetMs) {
  if (game.gameOver) return { type: 'pass', info: 'game already over' };

  const N          = game.boardSize;
  const rootPlayer = game.current;
  const root       = makeNode(null, null, null, N, 0, 0);

  const budgetMs = timeBudgetMs != null ? timeBudgetMs : DEFAULT_BUDGET_MS;
  const deadline = performance.now() + budgetMs;
  let playouts = 0;

  do {
    playouts++;
    const { node, game: simGame } = selectAndExpand(root, game, N);
    const { winner, blackPlayed, whitePlayed } = playTracked(simGame);
    backpropagate(node, winner, blackPlayed, whitePlayed, rootPlayer);
  } while (PLAYOUTS > 0 ? playouts < PLAYOUTS : performance.now() < deadline);

  // Pick the root child with the most visits.
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
