'use strict';

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
 */

const performance = (typeof window !== 'undefined') ? window.performance
  : require('perf_hooks').performance;

const { isLadderCaptured } = require('../ladder.js');

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


// ── Fast playout helpers ──────────────────────────────────────────────────────

function applyFast(game, x, y) {
  game.board.set(x, y, game.current);
  const cap = game.board.captureGroups(x, y);
  game.consecutivePasses = 0;
  game.current = game.current === 'black' ? 'white' : 'black';
  return cap.black.length + cap.white.length;
}

// Random playout that records the cell indices (y*N+x) of every move made by
// each player.  Pass moves carry no cell index and are not recorded.
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

function legalMoves(game) {
  const moves = [];
  for (let y = 0; y < game.boardSize; y++)
    for (let x = 0; x < game.boardSize; x++) {
      if (game.board.get(x, y) !== null) continue;
      if (game.board.classifyEmpty(x, y, game.current).isTrueEye) continue;
      const probe = game.clone();
      if (probe.placeStone(x, y)) moves.push({ type: 'place', x, y });
    }
  const area = game.boardSize * game.boardSize;
  if (game.moveCount >= area / 2 || game.consecutivePasses > 0) moves.push({ type: 'pass' });
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
    if (node.untried === null) node.untried = legalMoves(game);
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
      const child = makeNode(move, node, game.current, N);
      node.children.push(child);
      node = child;
      applyMove(game, move);

      // After a pass, always force a second pass as a terminal tree node so the
      // simulation scores the current board position rather than rolling out
      // randomly.  Without this, rollouts from a "one pass" state play on for
      // many more random moves, inflating the pass move's apparent win rate.
      if (!game.gameOver && game.consecutivePasses > 0) {
        const secondPass = makeNode({ type: 'pass' }, node, game.current, N);
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

// Virtual visit weight for each ladder-derived prior seeded into child nodes.
const LADDER_PRIOR = (typeof process !== 'undefined' && process.env.LADDER_PRIOR)
  ? parseInt(process.env.LADDER_PRIOR, 10) : 10;


// A node must reach this many visits before its ladder priors are applied.
const LADDER_VISITS = (typeof process !== 'undefined' && process.env.LADDER_VISITS)
  ? parseInt(process.env.LADDER_VISITS, 10) : 3;


// Returns true if the current player playing (lx, ly) puts the group at
// (gx, gy) into a losing ladder — either capturing it immediately or via
// isLadderCaptured after it is put in atari.
function attackStartsCapture(game, gx, gy, lx, ly) {
  const g2 = game.clone();
  if (g2.placeStone(lx, ly) === false) return false;
  const afterGroup = g2.board.getGroup(gx, gy);
  if (afterGroup.length === 0) return true;
  return isLadderCaptured(g2, gx, gy).captured;
}

// ── Ladder priors ────────────────────────────────────────────────────────────
// For every group on the board, run the ladder check twice: once with the
// group's owner (defender) moving first, and once with the opponent
// (attacker) moving first.  Groups whose outcome differs are critical ladders;
// their relevant liberties are seeded into the node's child wins/visits.
//
// 1-liberty group — critical when defFirst=false (attacker first always wins):
//   The liberty is correct for whoever plays it first, so it gets win=1
//   regardless of which side the mover is.
//
// 2-liberty group — critical when atkFirst=true for a liberty L:
//   mover is the attacker → L gets win=1 (initiate the winning ladder)
//   mover is the defender → L gets win=0 (avoid self-atari)
function applyLadderPriors(node, game, N) {
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
      child = makeNode(move, node, mover, N);
      node.children.push(child);
    }
    child.wins   += wins;
    child.visits += visits;
    node.visits  += visits;
  }

  const visited = new Set();
  for (let py = 0; py < N; py++) {
    for (let px = 0; px < N; px++) {
      const group    = game.board.getGroup(px, py);
      if (group.length < 2) continue;
      const groupColor = game.board.get(px, py);
      const gid = game.board._gid[game.board._idx(px, py)];
      if (visited.has(gid)) continue;
      visited.add(gid);
      const libs     = game.board.getLiberties(group);
      const opponent = mover === 'black' ? 'white' : 'black';
      
      if (libs.size <= 2) {
        const ladderResult = isLadderCaptured(game, px, py);
        if (ladderResult.captured && mover == groupColor) {
          // Adding stones to a dead group.
          for (const lstr of libs) {
            const [lx, ly] = lstr.split(',').map(Number);
            seedChild(lx, ly, 0, LADDER_PRIOR);
          }
        } else {
          game.current = opponent;
          const ladderResultOpp = isLadderCaptured(game, px, py);
          game.current = mover;  // restore
          if (ladderResult.captured != ladderResultOpp.captured) {
            // Urgent ladder
            for (const lstr of libs) {
              const [lx, ly] = lstr.split(',').map(Number);
              const wins = ladderResult.moves.includes(lstr) ? LADDER_PRIOR : 0
              //seedChild(lx, ly, wins, LADDER_PRIOR);
            }
          } else {
            // Non-urgent ladder
            for (const lstr of libs) {
              const [lx, ly] = lstr.split(',').map(Number);
              //seedChild(lx, ly, 0, LADDER_PRIOR);
            }
          }
        }
      }
    }
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
