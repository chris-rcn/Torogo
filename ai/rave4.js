'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs (require, process, etc.).
// Wrapped in an IIFE to avoid polluting the global namespace.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

/**
 * RAVE + RAVE4 MCTS policy.
 *
 * Extends plain RAVE with context-sensitive RAVE4 stats: each (cell, ctx)
 * pair tracks wins/visits separately, where ctx is the 4-neighbor occupancy
 * at the time the move was played (0–80, encoded as base-3 with values
 * empty=0 / self=1 / opponent=2 for each of the 4 orthogonal neighbours).
 *
 * The UCB formula blends real visits, plain RAVE, and RAVE4:
 *   rave4Equiv = RAVE4_EQUIV_MAX * rave4V / (rave4V + RAVE4_EQUIV_MAX)
 *   wr = (realW + raveWR*RAVE_EQUIV + rave4WR*rave4Equiv)
 *      / (realV + RAVE_EQUIV + rave4Equiv)
 * rave4Equiv starts near 0 and asymptotes to RAVE4_EQUIV_MAX as rave4V grows.
 *
 * RAVE4 stats accumulate more slowly (they are context-specific, so each
 * bucket receives ~1/81 of RAVE's updates), but carry more signal when they
 * do accumulate.
 *
 * Interface: getMove(game, timeBudgetMs) → { type: 'pass' } | { type: 'place', x, y }
 *   game         - a live Game2 instance (read-only; do not mutate)
 *   timeBudgetMs - milliseconds allowed for this decision (required)
 */

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

const performance = (typeof window !== 'undefined') ? window.performance
  : require('perf_hooks').performance;

const { PASS, BLACK, WHITE } = _isNode ? require('../game2.js') : window.Game2;
const Util = _isNode ? require('../util.js') : window.Util;

const EXPLORATION_C = Util.envFloat('EXPLORATION_C', 1.4);
const RAVE_EQUIV    = Util.envFloat('RAVE_EQUIV', 2000);
const RAVE4_EQUIV_MAX = Util.envFloat('RAVE4_EQUIV_MAX', 2000);
const RAVE4_EQUIV_GROWTH = Util.envFloat('RAVE4_EQUIV_GROWTH', 0.0);
const PLAYOUTS      = Util.envInt('PLAYOUTS', 0);
const N_EXPAND      = Util.envInt('N_EXPAND', 3);

// ── Context helper ────────────────────────────────────────────────────────────

// Encode the 4-neighbour occupancy of cell `idx` relative to `mover` as an
// integer in [0, 80].  Each neighbour contributes one base-3 digit:
//   0 = empty,  1 = self (same colour as mover),  2 = opponent.
function cellCtx(idx, mover, cells, nbr) {
  const base = idx * 4;
  const v = (c) => c === 0 ? 0 : c === mover ? 1 : 2;
  return v(cells[nbr[base]])
       + v(cells[nbr[base + 1]]) * 3
       + v(cells[nbr[base + 2]]) * 9
       + v(cells[nbr[base + 3]]) * 27;
}

// ── Fast playout ──────────────────────────────────────────────────────────────

// Random playout using Game2.  Records flat cell indices and their 4-neighbour
// contexts at time of play, separately for each player.
// Returns { winner, blackPlayed, blackCtx, whitePlayed, whiteCtx }.
function playTracked(game2) {
  const wasAlreadyOver = game2.gameOver;
  const N     = game2.N;
  const cap   = N * N;
  const cells = game2.cells;
  const nbr   = game2._nbr;

  const blackPlayed = [], blackCtx = [];
  const whitePlayed = [], whiteCtx = [];

  const empty = [];
  for (let i = 0; i < cap; i++) if (cells[i] === 0) empty.push(i);

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

      // Capture context before the board changes.
      const ctx = cellCtx(idx, current, cells, nbr);

      if (current === BLACK) { blackPlayed.push(idx); blackCtx.push(ctx); }
      else                   { whitePlayed.push(idx); whiteCtx.push(ctx); }

      // Snapshot neighbours to detect captures after play.
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

  const winner = wasAlreadyOver ? game2.calcWinner() : game2.estimateWinner();
  return { winner, blackPlayed, blackCtx, whitePlayed, whiteCtx };
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
  if (moves.length < cap / 3 || game2.consecutivePasses > 0) moves.push(PASS);
  return moves;
}

function makeNode(move, parent, ci, mover, game2, N) {
  const movesArr = getLegalMoves(game2);
  const M = movesArr.length;

  const legalMoves = new Int32Array(M);
  for (let i = 0; i < M; i++) legalMoves[i] = movesArr[i];

  // Precompute 4-neighbour context for each legal move in this position.
  const cells = game2.cells;
  const nbr   = game2._nbr;
  const moverNow = game2.current;
  const ctxs  = new Int32Array(M);
  for (let i = 0; i < M; i++) {
    const mv = movesArr[i];
    ctxs[i] = mv === PASS ? 0 : cellCtx(mv, moverNow, cells, nbr);
  }

  const cap81 = N * N * 81;

  return {
    move,
    parent,
    ci,
    mover,
    totalVisits:   1,
    selectedChild: -1,

    legalMoves,
    ctxs,
    children:    new Array(M).fill(null),

    wins:         new Float32Array(M).fill(0.001),
    visits:       new Float32Array(M).fill(0.002),

    raveWins:     new Float32Array(N * N).fill(0.001),
    raveVisits:   new Float32Array(N * N).fill(0.002),

    rave4Wins:    new Float32Array(cap81).fill(0.001),
    rave4Visits:  new Float32Array(cap81).fill(0.002),
  };
}

// ── UCB score ─────────────────────────────────────────────────────────────────

function ucbScore(moveIdx, node) {
  // RAVE
  const move  = node.legalMoves[moveIdx];
  const raveW  = move === PASS ? 0 : node.raveWins[move];
  const raveV  = move === PASS ? 1 : node.raveVisits[move];
  const raveWR = raveW / raveV;

  // RAVE4
  const r4key   = move === PASS ? 0 : move * 81 + node.ctxs[moveIdx];
  const rave4W  = move === PASS ? 0 : node.rave4Wins[r4key];
  const rave4V  = move === PASS ? 1 : node.rave4Visits[r4key];
  const rave4WR = rave4W / rave4V;
  const rave4Equiv = RAVE4_EQUIV_MAX * (1 - 1 / (1 + RAVE4_EQUIV_GROWTH * Math.sqrt(rave4V)));

  // Real
  const realW = node.wins[moveIdx];
  const realV = node.visits[moveIdx];
  const realWR = realW / realV;

  // Combined win ratio
  const wr = (raveWR * RAVE_EQUIV + rave4WR * rave4Equiv + realWR * realV)
           / (       + RAVE_EQUIV +           rave4Equiv +          realV);

  const scoreBase = wr + 0.001 * Math.random();
  if (realV < 1) {
    return scoreBase + 10;
  }
  return scoreBase + EXPLORATION_C * Math.sqrt(Math.log(node.totalVisits) / realV);
}

// ── RAVE-MCTS core ────────────────────────────────────────────────────────────

function selectAndExpand(root, rootGame2, N) {
  let node = root;
  const game2 = rootGame2.clone();

  while (!game2.gameOver) {
    const M = node.legalMoves.length;
    if (M === 0) break;

    let best = 0, bestScore = -Infinity;
    for (let i = 0; i < M; i++) {
      const s = ucbScore(i, node);
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

function backpropagate(node, winner, blackPlayed, blackCtx, whitePlayed, whiteCtx, rootPlayer) {
  function childMover(n) {
    return n.mover === null ? rootPlayer : (n.mover === BLACK ? WHITE : BLACK);
  }

  function updateRave(node, won, played) {
    const inv = 1 / (1 + played.length);
    let weight = 1;
    for (let k = 0; k < played.length; k++) {
      node.raveVisits[played[k]] += weight;
      node.raveWins[played[k]]   += won * weight;
      weight -= inv;
    }
  }

  function updateRave4(node, won, played, ctxArr) {
    const inv = 1 / (1 + played.length);
    let weight = 1;
    for (let k = 0; k < played.length; k++) {
      const key = played[k] * 81 + ctxArr[k];
      node.rave4Visits[key] += weight;
      node.rave4Wins[key]   += won * weight;
      weight -= inv;
    }
  }

  const leafIdx = node.selectedChild;
  if (leafIdx !== -1) {
    const chooser = childMover(node);
    const won     = winner === chooser ? 1 : 0;
    node.visits[leafIdx]++;
    node.wins[leafIdx] += won;
    node.totalVisits++;

    const played  = chooser === BLACK ? blackPlayed : whitePlayed;
    const ctxArr  = chooser === BLACK ? blackCtx    : whiteCtx;
    updateRave(node, won, played);
    updateRave4(node, won, played, ctxArr);
  }

  while (node.parent !== null) {
    const ci  = node.ci;
    const won = winner === node.mover ? 1 : 0;
    node.parent.visits[ci]++;
    node.parent.wins[ci] += won;
    node.parent.totalVisits++;

    const chooser = childMover(node.parent);
    const played  = chooser === BLACK ? blackPlayed : whitePlayed;
    const ctxArr  = chooser === BLACK ? blackCtx    : whiteCtx;
    const rWon    = winner === chooser ? 1 : 0;
    updateRave(node.parent, rWon, played);
    updateRave4(node.parent, rWon, played, ctxArr);

    node = node.parent;
  }
}

// ── Public interface ──────────────────────────────────────────────────────────

function getMove(game, timeBudgetMs) {
  if (game.gameOver) return { type: 'pass', info: 'game already over' };

  const N          = game.cells ? game.N : game.boardSize;
  const game2      = game.cells ? game.clone() : game.toGame2();
  const rootPlayer = game2.current;

  if (game2.consecutivePasses > 0 && game2.calcWinner() === rootPlayer) {
    return { type: 'pass', info: 'obvious pass: already winning', rootWinRatio: 1 };
  }

  const root = makeNode(null, null, -1, null, game2, N);

  const deadline = performance.now() + timeBudgetMs;
  let playouts = 0;

  do {
    playouts++;
    const { node, game2: simGame2 } = selectAndExpand(root, game2, N);
    const { winner, blackPlayed, blackCtx, whitePlayed, whiteCtx } = playTracked(simGame2);
    backpropagate(node, winner, blackPlayed, blackCtx, whitePlayed, whiteCtx, rootPlayer);
  } while (PLAYOUTS > 0 ? playouts < PLAYOUTS : performance.now() < deadline);

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
    return { type: 'pass', info: 'no winning line found', children, rootWinRatio };
  }

  const m = root.legalMoves[bestIdx];
  const cv = root.visits[bestIdx];
  const bestWinRatio = cv > 0 ? root.wins[bestIdx] / cv : 0.5;

  const result = m === PASS ? { type: 'pass', children, rootWinRatio }
                            : { type: 'place', x: m % N, y: (m / N) | 0, children, rootWinRatio };
  result.info = `win likelihood: ${bestWinRatio.toFixed(3)}`;
  return result;
}

if (typeof module !== 'undefined') module.exports = { getMove };
else window.getMove = getMove;

})();
