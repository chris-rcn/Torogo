'use strict';

/**
 * Influence policy.
 *
 * Extends shore-up by replacing the random fallback (Tier 5) with a
 * Voronoi influence score.  When no tactical move is available, instead
 * of picking randomly among legal non-eye moves the policy simulates each
 * candidate and scores the resulting board state by the number of empty
 * cells that are closer (by BFS distance through empty cells, toroidally)
 * to own stones than to opponent stones.  The candidate that maximises
 * this score is chosen.
 *
 * Priority order each turn:
 *   1. Capture   — capture the largest opponent group in atari
 *   2. Escape    — save the largest own group in atari
 *   3. Shore up  — extend own 2-liberty groups to ≥ 3 liberties (largest first)
 *   4. Threaten  — put the largest opponent group into atari
 *   5. Influence — maximise Voronoi territorial influence
 *
 * Interface: getMove(game) → { type: 'pass' } | { type: 'place', x, y }
 *   game  - a live Game instance (read-only; do not mutate)
 */

const { Game } = require('../game.js');

// ── helpers ────────────────────────────────────────────────────────────────

function cloneGame(game) {
  const g = new Game(game.boardSize);
  g.board             = game.board.clone();
  g.current           = game.current;
  g.captured          = { ...game.captured };
  g.prevHash          = game.prevHash;
  g.consecutivePasses = game.consecutivePasses;
  g.gameOver          = game.gameOver;
  return g;
}

function isTrueEye(board, x, y, color) {
  const N = board.size;
  const ortho = board.getNeighbors(x, y);
  if (!ortho.every(([nx, ny]) => board.get(nx, ny) === color)) return false;
  const diags = [
    [(x + 1) % N,     (y + 1) % N],
    [(x - 1 + N) % N, (y + 1) % N],
    [(x + 1) % N,     (y - 1 + N) % N],
    [(x - 1 + N) % N, (y - 1 + N) % N],
  ];
  return diags.filter(([dx, dy]) => board.get(dx, dy) === color).length >= 3;
}

function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

// Return all groups of `color`, each as { group, libs }, sorted by group size
// descending (largest first).
function groupsByColor(board, color) {
  const visited = new Set();
  const results = [];
  const N = board.size;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const key = `${x},${y}`;
      if (board.get(x, y) !== color || visited.has(key)) continue;
      const group = board.getGroup(x, y);
      group.forEach(([gx, gy]) => visited.add(`${gx},${gy}`));
      const libs = board.getLiberties(group);
      results.push({ group, libs });
    }
  }
  results.sort((a, b) => b.group.length - a.group.length);
  return results;
}

// True if any own group in `clone` has exactly 1 liberty (we'd be in atari).
function leavesOwnGroupAtari(clone, color) {
  const visited = new Set();
  const N = clone.board.size;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const key = `${x},${y}`;
      if (clone.board.get(x, y) !== color || visited.has(key)) continue;
      const group = clone.board.getGroup(x, y);
      group.forEach(([gx, gy]) => visited.add(`${gx},${gy}`));
      if (clone.board.getLiberties(group).size === 1) return true;
    }
  }
  return false;
}

// ── tiers ──────────────────────────────────────────────────────────────────

// Tier 1: capture an opponent group in atari — largest first.
function findCapture(game) {
  const opp = game.current === 'black' ? 'white' : 'black';
  const atari = groupsByColor(game.board, opp).filter(({ libs }) => libs.size === 1);
  for (const { libs } of atari) {
    const [lx, ly] = [...libs][0].split(',').map(Number);
    const clone = cloneGame(game);
    if (clone.placeStone(lx, ly)) return { type: 'place', x: lx, y: ly };
  }
  return null;
}

// Tier 2: save own group in atari — largest first.
function findEscape(game) {
  const color = game.current;
  const atari = groupsByColor(game.board, color).filter(({ libs }) => libs.size === 1);
  for (const { libs } of atari) {
    const [lx, ly] = [...libs][0].split(',').map(Number);
    const clone = cloneGame(game);
    if (clone.placeStone(lx, ly)) return { type: 'place', x: lx, y: ly };
  }
  return null;
}

// Tier 3: extend own groups at exactly 2 liberties to ≥ 3 — largest first.
function findShoreUp(game) {
  const color = game.current;
  const vulnerable = groupsByColor(game.board, color).filter(({ libs }) => libs.size === 2);
  for (const { libs } of vulnerable) {
    for (const libStr of libs) {
      const [x, y] = libStr.split(',').map(Number);
      const clone = cloneGame(game);
      if (!clone.placeStone(x, y)) continue;
      const afterGroup = clone.board.getGroup(x, y);
      if (clone.board.getLiberties(afterGroup).size >= 3)
        return { type: 'place', x, y };
    }
  }
  return null;
}

// Tier 4: put the largest opponent group into atari.
function findThreat(game, candidates) {
  const opp = game.current === 'black' ? 'white' : 'black';
  let bestMove = null;
  let bestSize = -1;

  for (const [x, y] of candidates) {
    const clone = cloneGame(game);
    if (!clone.placeStone(x, y)) continue;

    const visited = new Set();
    const N = clone.board.size;
    for (let gy = 0; gy < N; gy++) {
      for (let gx = 0; gx < N; gx++) {
        const key = `${gx},${gy}`;
        if (clone.board.get(gx, gy) !== opp || visited.has(key)) continue;
        const group = clone.board.getGroup(gx, gy);
        group.forEach(([ax, ay]) => visited.add(`${ax},${ay}`));
        if (clone.board.getLiberties(group).size === 1 && group.length > bestSize) {
          bestSize = group.length;
          bestMove = { type: 'place', x, y };
        }
      }
    }
  }
  return bestMove;
}

// Multi-source BFS through empty cells; returns a 2-D distance array.
function bfsDistances(board, N, seeds) {
  const dist = Array.from({ length: N }, () => new Float32Array(N).fill(Infinity));
  const queue = [];
  for (const [x, y] of seeds) {
    dist[y][x] = 0;
    queue.push([x, y]);
  }
  let head = 0;
  while (head < queue.length) {
    const [cx, cy] = queue[head++];
    for (const [nx, ny] of board.getNeighbors(cx, cy)) {
      if (board.get(nx, ny) === null && dist[ny][nx] === Infinity) {
        dist[ny][nx] = dist[cy][cx] + 1;
        queue.push([nx, ny]);
      }
    }
  }
  return dist;
}

// Count empty cells whose BFS distance from myColor stones is strictly less
// than from opponent stones (Voronoi partition on the toroidal board).
function voronoiScore(board, N, myColor) {
  const opp = myColor === 'black' ? 'white' : 'black';
  const mySeeds = [], oppSeeds = [];
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      if (board.get(x, y) === myColor) mySeeds.push([x, y]);
      else if (board.get(x, y) === opp)  oppSeeds.push([x, y]);
    }
  }
  const myDist  = bfsDistances(board, N, mySeeds);
  const oppDist = bfsDistances(board, N, oppSeeds);
  let score = 0;
  for (let y = 0; y < N; y++)
    for (let x = 0; x < N; x++)
      if (board.get(x, y) === null && myDist[y][x] < oppDist[y][x]) score++;
  return score;
}

// Tier 5: pick the candidate that maximises Voronoi territorial influence.
function findInfluence(game, candidates) {
  const color = game.current;
  const N = game.boardSize;
  let bestMove  = null;
  let bestScore = -1;
  for (const [x, y] of candidates) {
    const clone = cloneGame(game);
    if (!clone.placeStone(x, y)) continue;
    if (leavesOwnGroupAtari(clone, color)) continue;
    const score = voronoiScore(clone.board, N, color);
    if (score > bestScore) {
      bestScore = score;
      bestMove  = { type: 'place', x, y };
    }
  }
  return bestMove;
}

// ── main ───────────────────────────────────────────────────────────────────

module.exports = function getMove(game) {
  if (game.gameOver) return { type: 'pass' };

  const N = game.boardSize;
  const color = game.current;

  const capture = findCapture(game);
  if (capture) return capture;

  const escape = findEscape(game);
  if (escape) return escape;

  const shoreUp = findShoreUp(game);
  if (shoreUp) return shoreUp;

  // Build shuffled candidate list for tiers 4 & 5.
  // Shuffling ensures ties in Tier 5 are broken randomly.
  const candidates = [];
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      if (game.board.get(x, y) !== null) continue;
      if (isTrueEye(game.board, x, y, color)) continue;
      candidates.push([x, y]);
    }
  }
  shuffle(candidates);

  return findThreat(game, candidates)
      || findInfluence(game, candidates)
      || { type: 'pass' };
};
