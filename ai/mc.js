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
  while (!game.gameOver) {
    const move = randomAgent(game);
    if (move.type === 'place') {
      game.placeStone(move.x, move.y);
    } else {
      game.pass();
    }
  }
}

module.exports = function getMove(game) {
  if (game.gameOver) return { type: 'pass' };

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
