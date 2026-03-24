'use strict';

/**
 * Monte Carlo policy.
 *
 * For each candidate move, performs random playouts to the end of the game
 * and tracks the win ratio.  After candidate_playouts per move the move with
 * the highest win ratio is returned.
 *
 * Interface: getMove(game, timeBudgetMs) → { type: 'pass' } | { type: 'place', x, y }
 *   game         - a live Game instance (read-only; do not mutate)
 *   timeBudgetMs - milliseconds allowed for this decision (required)
 */

const performance = (typeof window !== 'undefined') ? window.performance
  : require('perf_hooks').performance;


// Lightweight move application for use inside playouts.
// Precondition: (x, y) has at least one empty orthogonal neighbour, which
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
    let end = empty.length;

    while (end > 0) {
      const i = Math.floor(Math.random() * end);
      const [x, y] = empty[i];

      empty[i] = empty[end - 1];
      empty[end - 1] = [x, y];
      end--;

      if (game.board.classifyEmpty(x, y, game.current).isTrueEye) continue;

      const result = game.placeStone(x, y);
      if (result) {
        empty[end] = empty[empty.length - 1];
        empty.pop();
        if (result !== true) {
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

}

function getMove(game, timeBudgetMs) {
  if (game.gameOver) return { type: 'pass' };

  const player = game.current;

  // Build list of legal candidate moves.
  const candidates = [];
  for (let y = 0; y < game.boardSize; y++) {
    for (let x = 0; x < game.boardSize; x++) {
      if (game.board.get(x, y) !== null) continue;
      if (game.board.classifyEmpty(x, y, game.current).isTrueEye) continue;
      if (game.isLegal(x, y)) candidates.push({ type: 'place', x, y });
    }
  }
  // Pass is always legal.
  candidates.push({ type: 'pass' });

  // Per-candidate playout statistics.
  const stats = candidates.map(() => ({ wins: 0, plays: 0 }));

  // Round-robin playouts across candidates until the time budget is spent.
  const deadline = performance.now() + timeBudgetMs;
  let cidx = 0;
  while (performance.now() < deadline) {
    const move = candidates[cidx];
    const clone = game.clone();
    if (move.type === 'place') {
      clone.placeStone(move.x, move.y);
    } else {
      clone.pass();
    }
    playRandom(clone);

    const winner = clone.estimateWinner();

    stats[cidx].plays++;
    if (winner === player) stats[cidx].wins++;

    cidx = (cidx + 1) % candidates.length;
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
}

if (typeof module !== 'undefined') module.exports = getMove;
