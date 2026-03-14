'use strict';

/**
 * Monte Carlo policy.
 *
 * For each candidate move, performs random playouts to the end of the game
 * and tracks the win ratio.  After at least N total playouts the move with
 * the highest win ratio is returned.
 *
 * Interface: getMove(game) → { type: 'pass' } | { type: 'place', x, y }
 *   game  - a live Game instance (read-only; do not mutate)
 */

const { Game } = require('../game.js');
const randomAgent = require('./random.js');

const N = 10; // number of playouts per candidate move

// Single-point true eye: all 4 orthogonal neighbours are `color`, and at
// least 3 of 4 diagonal neighbours are `color`.  On a toroidal board every
// cell has exactly 4 orthogonal and 4 diagonal neighbours.
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

function playRandom(game) {
  const size = game.boardSize;

  // Build the initial list of empty cells once.
  const empty = [];
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++)
      if (game.board.get(x, y) === null) empty.push([x, y]);

  while (!game.gameOver) {
    let placed = false;

    // Scan candidates in random order without replacement using a partition
    // index `end`.  Elements in [0, end) are untried this turn; elements in
    // [end, empty.length) were tried but were illegal (Ko/suicide) and stay
    // in the list for future turns.
    let end = empty.length;

    while (end > 0) {
      const i = Math.floor(Math.random() * end);
      const [x, y] = empty[i];

      // Move candidate to the boundary so we can remove it cheaply if needed.
      empty[i] = empty[end - 1];
      empty[end - 1] = [x, y];
      end--;

      // Stale entry: cell was filled by a prior move in this playout.
      if (game.board.get(x, y) !== null) {
        // Permanently remove (it is sitting at index `end`).
        empty[end] = empty[empty.length - 1];
        empty.pop();
        continue;
      }

      // Skip single-point true eyes for the current player — filling them is
      // always suicide.  Leave the cell in the list so the opponent can use it
      // and so captures can later dissolve the eye.
      if (isTrueEye(game.board, x, y, game.current)) continue;

      const capBefore = game.captured.black + game.captured.white;

      if (game.placeStone(x, y)) {
        // Permanently remove the now-occupied cell (sitting at index `end`).
        empty[end] = empty[empty.length - 1];
        empty.pop();

        // If any stones were captured they are now empty again.  Since the
        // Game API does not expose which cells were freed, rebuild the list.
        if (game.captured.black + game.captured.white > capBefore) {
          empty.length = 0;
          for (let ey = 0; ey < size; ey++)
            for (let ex = 0; ex < size; ex++)
              if (game.board.get(ex, ey) === null) empty.push([ex, ey]);
        }

        placed = true;
        break;
      }
      // Illegal move (Ko or suicide): element stays at index `end` and will
      // be reconsidered in a future turn.
    }

    if (!placed) game.pass();
  }
}

module.exports = function getMove(game) {
  if (game.gameOver) return { type: 'pass' };

  // All positions are equivalent on a toroidal board — skip MC on move one.
  if (game.lastMove === null && game.consecutivePasses === 0) {
    return randomAgent(game);
  }

  const player = game.current;

  // Build list of legal candidate moves.
  const candidates = [];
  for (let y = 0; y < game.boardSize; y++) {
    for (let x = 0; x < game.boardSize; x++) {
      if (game.board.get(x, y) !== null) continue;
      const probe = cloneGame(game);
      if (probe.placeStone(x, y)) candidates.push({ type: 'place', x, y });
    }
  }
  // Pass is always legal.
  candidates.push({ type: 'pass' });

  // Per-candidate playout statistics.
  const stats = candidates.map(() => ({ wins: 0, plays: 0 }));

  // N playouts per candidate move.
  for (let idx = 0; idx < candidates.length; idx++) {
    const move = candidates[idx];
    for (let t = 0; t < N; t++) {
      const clone = cloneGame(game);
      if (move.type === 'place') {
        clone.placeStone(move.x, move.y);
      } else {
        clone.pass();
      }
      playRandom(clone);

      const s = clone.scores;
      const winner = s.black.total > s.white.total ? 'black'
                   : s.white.total > s.black.total ? 'white'
                   : null;

      stats[idx].plays++;
      if (winner === player) stats[idx].wins++;
    }
  }

  // Select the candidate with the highest win ratio.
  let bestIdx = 0;
  let bestRatio = -1;
  for (let i = 0; i < candidates.length; i++) {
    const ratio = stats[i].plays > 0 ? stats[i].wins / stats[i].plays : 0;
    if (ratio > bestRatio) {
      bestRatio = ratio;
      bestIdx = i;
    }
  }

  return candidates[bestIdx];
};
