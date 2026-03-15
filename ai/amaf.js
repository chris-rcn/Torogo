'use strict';

/**
 * AMAF (All-Moves-As-First) Monte Carlo policy.
 *
 * For each candidate, run random playouts starting with that move in
 * round-robin order until a total work budget is exhausted (where work per
 * playout = moves played + 10 for per-playout overhead).  Unlike mc.js, every move the current
 * player makes during the playout is credited — not just the opening move.
 * Moves later in the playout receive less credit via an exponential discount,
 * since they are less representative of what playing there "first" would mean.
 *
 * This means a single playout of candidate A also updates the estimates for
 * candidates B, C, … that happen to appear later in the same playout, giving
 * all candidates far more data than mc.js provides.
 *
 * Interface: getMove(game, timeBudgetMs) → { type: 'pass' } | { type: 'place', x, y }
 *   game         - a live Game instance (read-only; do not mutate)
 *   timeBudgetMs - milliseconds allowed for this decision (default: 500)
 */

const { performance } = require('perf_hooks');
const randomAgent = require('./random.js');

const DEFAULT_BUDGET_MS = 500;
// Weight decay per subsequent player move.  Override with AMAF_DISCOUNT=<n>.
const DISCOUNT = process.env.AMAF_DISCOUNT !== undefined
  ? parseFloat(process.env.AMAF_DISCOUNT)
  : 0.5;
// Weight multiplier for opponent moves.  Override with AMAF_OPP_WEIGHT=<n>.
const OPP_MOVE_WEIGHT = process.env.AMAF_OPP_WEIGHT !== undefined
  ? parseFloat(process.env.AMAF_OPP_WEIGHT)
  : 0;

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

// Like playRandom in mc.js, but collects the cell indices (y*size+x) of
// every subsequent move made by each side, in order (earliest first).
// Returns { winner, played, oppPlayed } where winner is 'black'|'white'|null.
function playTracked(game, trackColor) {
  const size = game.boardSize;
  const board = game.board;
  const grid = board.grid;
  const played = []; // ordered list of cell indices for trackColor's moves
  const oppPlayed = []; // ordered list of cell indices for opponent's moves

  // Build the initial list of empty cell indices (flat).
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
        // Fast path: at least one empty neighbour → no suicide/Ko possible
        if (current === trackColor) played.push(cellIdx);
        else oppPlayed.push(cellIdx);
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

      // Slow path: all four neighbours are occupied — suicide or Ko possible.
      const color = current;
      const result = game.placeStone(x, y);
      if (result) {
        if (color === trackColor) played.push(cellIdx);
        else oppPlayed.push(cellIdx);
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

    if (!placed) {
      game.pass();
      moves++;
    }
  }

  if (!game.gameOver) game.endGame();

  const s = game.scores;
  const winner = s.black.total > s.white.total ? 'black'
               : s.white.total > s.black.total ? 'white'
               : null;

  return { winner, played, oppPlayed, moves };
}

module.exports = function getMove(game, timeBudgetMs) {
  if (game.gameOver) return { type: 'pass' };

  const player = game.current;
  const N = game.boardSize;

  // Build list of legal candidate moves.
  const candidates = [];
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      if (game.board.get(x, y) !== null) continue;
      const probe = game.clone();
      if (!probe.placeStone(x, y)) continue;
      candidates.push({ type: 'place', x, y });
    }
  }
  if (game.moveCount > N * N / 2) {
    candidates.push({ type: 'pass' });
  }

  // AMAF stats indexed by cell (y*N + x); pass stored at N*N.
  // Float64 to accommodate fractional discount weights.
  const wins  = new Float64Array(N * N + 1);
  const plays = new Float64Array(N * N + 1);
  const PASS_IDX = N * N;

  // Round-robin playouts across candidates until the time budget is spent.
  const budgetMs = timeBudgetMs != null ? timeBudgetMs : DEFAULT_BUDGET_MS;
  const deadline = performance.now() + budgetMs;
  let cidx = 0;
  while (performance.now() < deadline) {
    const move = candidates[cidx];
    const clone = game.clone();
    if (move.type === 'place') {
      clone.placeStone(move.x, move.y);
    } else {
      clone.pass();
    }
    const { winner, played, oppPlayed, moves } = playTracked(clone, player);
    const won = winner === player ? 1 : 0;

    // Credit the opening move at full weight (it was played "first").
    if (move.type === 'place') {
      const firstIdx = move.y * N + move.x;
      plays[firstIdx] += 1.0;
      wins[firstIdx]  += won;
    } else {
      plays[PASS_IDX] += 1.0;
      wins[PASS_IDX]  += won;
    }

    // Credit subsequent player moves with exponential discount: the i-th
    // subsequent move gets weight DISCOUNT^(i+1).  Moves near the end of
    // the game are the least representative of a "first move" choice.
    let weight = DISCOUNT;
    for (const idx of played) {
      plays[idx] += weight;
      wins[idx]  += won * weight;
      weight *= DISCOUNT;
    }

    // Credit opponent moves with inverted outcome, scaled by OPP_MOVE_WEIGHT.
    // If the opponent played at X and the opponent won, X is probably an
    // important move — credit it as a win for us too (at reduced weight).
    if (OPP_MOVE_WEIGHT > 0) {
      let oppWeight = DISCOUNT * OPP_MOVE_WEIGHT;
      for (const idx of oppPlayed) {
        plays[idx] += oppWeight;
        wins[idx]  += (1 - won) * oppWeight;
        oppWeight *= DISCOUNT;
      }
    }

    cidx = (cidx + 1) % candidates.length;
  }

  // Select the candidate with the highest AMAF win ratio.  If pass is
  // tied for best it wins outright; otherwise ties are broken randomly.
  let bestRatio = -1;
  let bestCount = 0;
  let bestIdx = 0;
  for (let i = 0; i < candidates.length; i++) {
    const c = candidates[i];
    const idx = c.type === 'pass' ? PASS_IDX : c.y * N + c.x;
    if (plays[idx] === 0) continue;
    const ratio = wins[idx] / plays[idx];
    if (ratio > bestRatio) {
      bestRatio = ratio;
      bestIdx = i;
      bestCount = 1;
    } else if (ratio === bestRatio) {
      bestCount++;
      if (Math.random() * bestCount < 1) bestIdx = i;
    }
  }

  // If every playout is a loss, pass — no move can help.
  if (bestRatio === 0) return { type: 'pass' };

  // Prefer pass when it ties for best ratio.
  if (plays[PASS_IDX] > 0 && wins[PASS_IDX] / plays[PASS_IDX] === bestRatio) {
    const passCandIdx = candidates.findIndex(c => c.type === 'pass');
    if (passCandIdx !== -1) bestIdx = passCandIdx;
  }

  // Before committing to a pass, verify that the actual winner if the game
  // ends now matches the winner in every single pass playout.  If there is
  // any disagreement, fall back to the best non-pass candidate.
  if (candidates[bestIdx].type === 'pass' && plays[PASS_IDX] > 0) {
    const territory = game.calcTerritory();
    const blackTotal = territory.black;
    const whiteTotal = territory.white + game.komi;
    const actualWinner = blackTotal > whiteTotal ? 'black'
                       : whiteTotal > blackTotal ? 'white'
                       : null;
    const allPlayoutsAgree = actualWinner === player
      ? wins[PASS_IDX] === plays[PASS_IDX]
      : true; // losing by territory — always allow passing
    if (!allPlayoutsAgree) {
      // Pick best non-pass candidate instead.
      let altRatio = -1;
      let altCount = 0;
      let altIdx = -1;
      for (let i = 0; i < candidates.length; i++) {
        if (candidates[i].type === 'pass') continue;
        const idx = candidates[i].y * N + candidates[i].x;
        if (plays[idx] === 0) continue;
        const ratio = wins[idx] / plays[idx];
        if (ratio > altRatio) {
          altRatio = ratio;
          altIdx = i;
          altCount = 1;
        } else if (ratio === altRatio) {
          altCount++;
          if (Math.random() * altCount < 1) altIdx = i;
        }
      }
      if (altIdx !== -1) bestIdx = altIdx;
    }
  }

  return candidates[bestIdx];
};
