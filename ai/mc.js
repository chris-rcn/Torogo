'use strict';

/**
 * Monte Carlo policy.
 *
 * For each candidate move, performs random playouts to the end of the game
 * and tracks the win ratio.  After candidate_playouts per move the move with
 * the highest win ratio is returned.
 *
 * Interface: getMove(game) → { type: 'pass' } | { type: 'place', x, y }
 *   game  - a live Game instance (read-only; do not mutate)
 */

const randomAgent = require('./random.js');

const candidate_playouts = 50; // number of playouts per candidate move


// Lightweight move application for use inside playouts.
// Precondition: (x, y) has at least one empty orthogonal neighbour, which
// guarantees the move is not suicide and makes Ko effectively impossible.
// Skips the two O(n²) board-hash computations that placeStone always does.
// Returns the total number of stones captured (0 in the common case).
function applyFast(game, x, y) {
  game.board.set(x, y, game.current);
  const cap = game.board.captureGroups(x, y);
  game.consecutivePasses = 0;
  game.current = game.current === 'black' ? 'white' : 'black';
  return cap.black.length + cap.white.length;
}


function playRandom(game) {
  const size = game.boardSize;

  // Build the initial list of empty cells once.
  const empty = [];
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++)
      if (game.board.get(x, y) === null) empty.push([x, y]);

  const moveLimit = empty.length + 20;
  let moves = 0;

  while (!game.gameOver && moves < moveLimit) {
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

      // Skip single-point true eyes for the current player — filling them is
      // always suicide.  Leave the cell in the list so the opponent can use it
      // and so captures can later dissolve the eye.
      if (game.board.isTrueEye(x, y, game.current)) continue;

      // Fast path: at least one empty neighbour means the move cannot be
      // suicide and Ko is effectively impossible.  Use the lightweight helper
      // to avoid the two O(n²) board-hash computations inside placeStone.
      const neighbors = game.board.getNeighbors(x, y);
      if (neighbors.some(([nx, ny]) => game.board.get(nx, ny) === null)) {
        const captures = applyFast(game, x, y);
        empty[end] = empty[empty.length - 1];
        empty.pop();
        if (captures > 0) {
          empty.length = 0;
          for (let ey = 0; ey < size; ey++)
            for (let ex = 0; ex < size; ex++)
              if (game.board.get(ex, ey) === null) empty.push([ex, ey]);
        }
        placed = true;
        moves++;
        break;
      }

      // Slow path: all four neighbours are occupied — suicide or Ko possible.
      const result = game.placeStone(x, y);
      if (result) {
        empty[end] = empty[empty.length - 1];
        empty.pop();
        if (result > 1) {
          empty.length = 0;
          for (let ey = 0; ey < size; ey++)
            for (let ex = 0; ex < size; ex++)
              if (game.board.get(ex, ey) === null) empty.push([ex, ey]);
        }
        placed = true;
        moves++;
        break;
      }
      // Illegal move (Ko or suicide): element stays at index `end` and will
      // be reconsidered in a future turn.
    }

    if (!placed) {
      game.pass();
      moves++;
    }
  }

  if (!game.gameOver) game.endGame();
}

module.exports = function getMove(game) {
  if (game.gameOver) return { type: 'pass' };

  const player = game.current;

  // Build list of legal candidate moves.
  const candidates = [];
  for (let y = 0; y < game.boardSize; y++) {
    for (let x = 0; x < game.boardSize; x++) {
      if (game.board.get(x, y) !== null) continue;
      const probe = game.clone();
      if (probe.placeStone(x, y)) candidates.push({ type: 'place', x, y });
    }
  }
  // Pass is always legal.
  candidates.push({ type: 'pass' });

  // Per-candidate playout statistics.
  const stats = candidates.map(() => ({ wins: 0, plays: 0 }));

  // candidate_playouts playouts per candidate move.
  for (let idx = 0; idx < candidates.length; idx++) {
    const move = candidates[idx];
    for (let t = 0; t < candidate_playouts; t++) {
      const clone = game.clone();
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
