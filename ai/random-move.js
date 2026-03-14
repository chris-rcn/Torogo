'use strict';

/**
 * Random-move policy.
 *
 * Picks a uniformly random legal move, skipping moves that would fill a
 * single-point true eye for the current player.  Falls back to pass when no
 * such move exists.
 *
 * Interface: getMove(game) → { type: 'pass' } | { type: 'place', x, y }
 *   game  - a live Game instance (read-only; do not mutate)
 *
 * Single-point true eye (toroidal board — no corners/edges, every cell has
 * exactly 4 orthogonal and 4 diagonal neighbours):
 *   An empty cell (x, y) is a true eye for `color` when:
 *     1. All 4 orthogonal neighbours are occupied by `color`.
 *     2. At least 3 of the 4 diagonal neighbours are occupied by `color`.
 */

const { Game } = require('../game.js');

function isTrueEye(board, x, y, color) {
  const N = board.size;
  const ortho = board.getNeighbors(x, y);
  if (!ortho.every(([nx, ny]) => board.get(nx, ny) === color)) return false;
  const diags = [
    [(x + 1) % N,       (y + 1) % N],
    [(x - 1 + N) % N,   (y + 1) % N],
    [(x + 1) % N,       (y - 1 + N) % N],
    [(x - 1 + N) % N,   (y - 1 + N) % N],
  ];
  const friendly = diags.filter(([dx, dy]) => board.get(dx, dy) === color).length;
  return friendly >= 3;
}

function cloneGame(game) {
  const g = new Game(game.boardSize);
  g.board = game.board.clone();
  g.current = game.current;
  g.captured = { ...game.captured };
  g.prevHash = game.prevHash;
  g.consecutivePasses = game.consecutivePasses;
  g.gameOver = game.gameOver;
  return g;
}

module.exports = function getMove(game) {
  if (game.gameOver) return { type: 'pass' };

  const N = game.boardSize;
  const color = game.current;

  // Collect empty cells that are not true eyes for the current player.
  const candidates = [];
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      if (game.board.get(x, y) !== null) continue;
      if (isTrueEye(game.board, x, y, color)) continue;
      candidates.push([x, y]);
    }
  }

  // Fisher-Yates shuffle.
  for (let i = candidates.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [candidates[i], candidates[j]] = [candidates[j], candidates[i]];
  }

  // Return the first candidate that is legal (rules out suicide and Ko).
  for (const [x, y] of candidates) {
    const clone = cloneGame(game);
    if (clone.placeStone(x, y)) return { type: 'place', x, y };
  }

  return { type: 'pass' };
};
