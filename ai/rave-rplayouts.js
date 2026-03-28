'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs (require, process, etc.).
// Wrapped in an IIFE to avoid polluting the global namespace.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

/**
 * RAVE (Rapid Action Value Estimation) MCTS policy.
 *
 * Node structure: all stats kept in compact child-indexed arrays on the parent.
 * Child nodes are promoted lazily after N_EXPAND playout visits.
 */

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

const performance = (typeof window !== 'undefined') ? window.performance
  : require('perf_hooks').performance;

const { PASS, BLACK, WHITE } = _isNode ? require('../game2.js') : window.Game2;
const Util = _isNode ? require('../util.js') : window.Util;

const EXPLORATION_C = Util.envFloat('EXPLORATION_C', 0.3);

// Equivalence parameter.  Override with RAVE_EQUIV=<n>.
const RAVE_EQUIV = Util.envFloat('RAVE_EQUIV', 2000);

// Fixed playout count per decision.  When non-zero, overrides the time budget.
const PLAYOUTS = Util.envInt('PLAYOUTS', 0);

// Minimum playout visits before a child node is promoted (allocated).
const N_EXPAND = Util.envInt('N_EXPAND', 3);

// Fraction of parent RAVE stats inherited by a newly created child node.
// Must be < 1 to prevent unbounded growth down the tree.
const RAVE_INHERIT = Util.envFloat('RAVE_INHERIT', 0.2);

// Fraction of playouts guided by RAVE stats instead of uniform random.
// Guided playouts do not update RAVE (to avoid circularity).
const RAVE_GUIDED = Util.envFloat('RAVE_GUIDED', 0);

// ── Fast playout helpers ──────────────────────────────────────────────────────

// Select a legal non-eye move weighted by RAVE win rates from `node`.
// forChooser: true  → high wr = good (current player is node's chooser)
//             false → low wr = good  (current player is node's opponent)
function guidedMove(game2, node, forChooser) {
  const cap      = game2.N * game2.N;
  const cells    = game2.cells;
  const raveWins = node.raveWins;
  const raveVis  = node.raveVisits;

  let chosen = PASS, totalWeight = 0;
  for (let i = 0; i < cap; i++) {
    if (!game2.isLegal(i)) continue;
    if (game2.isTrueEye(i)) continue;
    const wr = raveWins[i] / raveVis[i];
    const w  = forChooser ? wr : (1 - wr);
    totalWeight += w;
    if (Math.random() * totalWeight < w) chosen = i;
  }
  return chosen !== PASS ? chosen : game2.randomLegalMove();
}

// Playout using Game2.
// mode=false: uniform random; records moves for RAVE update.
// mode=true:  guided by RAVE stats from `node`; does not update RAVE.
// Returns { winner, blackPlayed, whitePlayed }.
function playTracked(game2, node, nodeChooser, guided) {
  const wasAlreadyOver = game2.gameOver;
  const N   = game2.N;
  const cap = N * N;
  const blackPlayed = [];
  const whitePlayed = [];

  const moveLimit = cap + 20;
  let moves = 0;

  while (!game2.gameOver && moves < moveLimit) {
    const current = game2.current;
    const idx = guided ? guidedMove(game2, node, current === nodeChooser)
                       : game2.randomLegalMove();
    if (idx === PASS) { game2.play(PASS); moves++; continue; }
    if (!guided) {
      if (current === BLACK) blackPlayed.push(idx);
      else                    whitePlayed.push(idx);
    }
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
  // Pass move must be at the end (if present).
  if (moves.length < cap / 3 || game2.consecutivePasses > 0) {
    moves.push(PASS);
  }
  return moves;
}

// Create a node for the position reached by `move`.
// `game2` is the game state AFTER `move` was played (or initial state for root).
// `ci`    is this node's index in parent.children / parent.child* arrays (-1 for root).
// `mover` is the player who played `move` (null at root).
// Priors (ladder + pattern) are computed eagerly from the current game state.
function makeNode(move, parent, ci, mover, game2, N) {
  const movesArr = getLegalMoves(game2);
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
    move,
    parent,
    ci,           // this node's index in parent.children / parent.child* arrays (-1 for root)
    mover,        // player who made `move` to reach this node (null at root)
    totalVisits:  1,  // sum of visits; incremented each playout
    selectedChild: -1,  // set by selectAndExpand; read by backpropagate

    legalMoves,   // Int32Array(M)
    children,     // Array(M) — promoted child node or null

    wins,    // Float32Array(M) — playout wins per child
    visits,  // Float32Array(M) — playout visits per child

    raveWins,     // Float32Array(N*N) — RAVE wins indexed by cell; updated by rollouts
    raveVisits
  };
}

// RAVE-blended UCT score for child index i of node.                                                                                                                                                         
// Children with no real playout visits (cv === 0) get a large bonus so they                                                                                                                                 
// are always preferred over visited children.  RAVE (seeded with pattern                                                                                                                                    
// priors) ranks them within the unvisited tier.                                                                                                                                                             
// A separate priorBonus term decays as bonus/(1+realV), so ladder priors                                                                                                                                    
// guide early exploration without ever diluting the RAVE statistics.                                                                

function ucbScore(moveIdx, node) {
  const move  = node.legalMoves[moveIdx];

  // RAVE
  const raveWR = (move === PASS) ? 0 : (node.raveWins[move] / node.raveVisits[move]);

  // Real
  const realW = node.wins[moveIdx];
  const realV = node.visits[moveIdx];
  const realWR = realW / realV;

  // Combined win ratio
  const wr = (raveWR * RAVE_EQUIV + realWR * realV)
           / (       + RAVE_EQUIV +          realV);

  const scoreBase = wr + 0.001 * Math.random();
//  if (realV < 1) {
//    return scoreBase + 10;
//  }
  return scoreBase + EXPLORATION_C * Math.sqrt(Math.log(node.totalVisits) / (1 + realV));
}

// ── RAVE-MCTS core ────────────────────────────────────────────────────────────

function selectAndExpand(root, rootGame2, N) {
  let node = root;
  const game2 = rootGame2.clone();

  while (!game2.gameOver) {
    const M = node.legalMoves.length;
    if (M === 0) break;

    // Select best child by RAVE-blended score.
    let best = 0, bestScore = -Infinity;
    for (let i = 0; i < M; i++) {
      const s = ucbScore(i, node);
      if (s > bestScore) { bestScore = s; best = i; }
    }

    const mover = game2.current;   // player making this move
    game2.play(node.legalMoves[best]);

    // Promote child to a full node once it has accumulated enough visits.
    // Fall through to the descent check so the loop continues into the new node;
    // its children all have cv=0 < N_EXPAND, so the leaf case fires next iteration
    // (exactly one makeNode per playout, same as rave2's one-expansion-per-playout).
    if (node.children[best] === null && node.visits[best] >= N_EXPAND) {
      node.children[best] = makeNode(node.legalMoves[best], node, best, mover, game2, N);
    }

    // After a pass, always force a second pass so the playout scores the current
    // board position (consecutive passes end the game).  This prevents rollouts
    // from a single-pass state playing on for many random moves and inflating
    // the pass move's apparent win rate.
    if (!game2.gameOver && game2.consecutivePasses > 0) {
      game2.play(PASS);
      node.selectedChild = best;
      break;
    }

    // Descend into the promoted child, if available.
    if (node.children[best] !== null) {
      node = node.children[best];
      node.selectedChild = -1;  // reset in case game ends before we select below
      continue;
    }

    // Unpromoted leaf — run playout from here.
    node.selectedChild = best;
    break;
  }

  return { node, game2 };
}

// Backpropagate playout result and update RAVE statistics.
//
// node.selectedChild holds the unpromoted child index that was played last,
// or -1 if we descended all the way to a promoted node (game already over).
//
// childMover(n): the player choosing the next move from node n.
//   - rootPlayer if n is the root (n.mover === null)
//   - opponent(n.mover) otherwise
function backpropagate(node, winner, blackPlayed, whitePlayed, rootPlayer) {
  function childMover(n) {
    return n.mover === null ? rootPlayer : (n.mover === BLACK ? WHITE : BLACK);
  }

  function updateRave(node, won, played) {
    const invPlayedCount = 1 / (1 + played.length)
    let weight = 1;
    for (let k = 0; k < played.length; k++) {
      const move = played[k];
      node.raveVisits[move] += weight;
      node.raveWins[move] += won * weight;
      weight -= invPlayedCount;
    }
  }

  // Update the unpromoted leaf child stats (if we stopped before descending).
  // Also update RAVE at this node so root's RAVE is populated even when no
  // deeper promoted nodes exist (e.g. N_EXPAND=9999).
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

  // Walk up the tree, updating each parent's child arrays and RAVE arrays.
  while (node.parent !== null) {
    const ci  = node.ci;   // stored at promotion time — no lookup needed
    const won = winner === node.mover ? 1 : 0;
    node.parent.visits[ci]++;
    node.parent.wins[ci] += won;
    node.parent.totalVisits++;

    // RAVE update at node.parent: credit the parent's chooser's playout moves.
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

  // Obvious pass: opponent just passed and we're already winning — end the game.
  if (game2.consecutivePasses > 0 && game2.calcWinner() === rootPlayer) {
    return { type: 'pass', move: PASS, info: 'obvious pass: already winning', rootWinRatio: 1 };
  }

  const root = makeNode(null, null, -1, null, game2, N);

  const deadline = performance.now() + timeBudgetMs;
  let playouts = 0;

  do {
    playouts++;
    const { node, game2: simGame2 } = selectAndExpand(root, game2, N);
    const nodeChooser = node.mover === null ? rootPlayer : (node.mover === BLACK ? WHITE : BLACK);
    const guided = RAVE_GUIDED > 0 && Math.random() < RAVE_GUIDED;
    const { winner, blackPlayed, whitePlayed } = playTracked(simGame2, node, nodeChooser, guided);
    backpropagate(node, winner, blackPlayed, whitePlayed, rootPlayer);
  } while (PLAYOUTS > 0 ? playouts < PLAYOUTS : performance.now() < deadline);

  // Best child: most playout visits; ties broken by RAVE-blended score.
  const M = root.legalMoves.length;
  let bestIdx = 0, bestVisits = -1, bestScore = -Infinity;
  for (let i = 0; i < M; i++) {
    const cv = root.visits[i];
    if (cv > bestVisits || (cv === bestVisits && ucbScore(i, root) > bestScore)) {
      bestVisits = cv;
      bestScore  = ucbScore(i, root);
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
  const rootWinRatio = root.totalVisits > 0 ? totalChildWins / root.totalVisits : 0.5;

  if (root.wins[bestIdx] === 0 && game.moveCount >= N * N / 2) {
    return { type: 'pass', move: PASS, info: 'no winning line found', children, rootWinRatio };
  }

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
