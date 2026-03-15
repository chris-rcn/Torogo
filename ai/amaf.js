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
 * Interface: getMove(game) → { type: 'pass' } | { type: 'place', x, y }
 *   game  - a live Game instance (read-only; do not mutate)
 */

const randomAgent = require('./random.js');

const WORK_BUDGET = 500_000; // total playout moves per turn
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
  game.captured.black += cap.black.length;
  game.captured.white += cap.white.length;
  game.consecutivePasses = 0;
  game.current = game.current === 'black' ? 'white' : 'black';
  return cap.black + cap.white;
}

// Like playRandom in mc.js, but collects the cell indices (y*size+x) of
// every subsequent move made by each side, in order (earliest first).
// Returns { winner, played, oppPlayed } where winner is 'black'|'white'|null.
function playTracked(game, trackColor) {
  const size = game.boardSize;
  const played = []; // ordered list of cell indices for trackColor's moves
  const oppPlayed = []; // ordered list of cell indices for opponent's moves

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
        if (game.current === trackColor) played.push(y * size + x);
        else oppPlayed.push(y * size + x);
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
      const capBefore = game.captured.black + game.captured.white;
      const color = game.current;
      if (game.placeStone(x, y)) {
        if (color === trackColor) played.push(y * size + x);
        else oppPlayed.push(y * size + x);
        empty[end] = empty[empty.length - 1];
        empty.pop();
        if (game.captured.black + game.captured.white > capBefore) {
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

  const s = game.scores;
  const winner = s.black.total > s.white.total ? 'black'
               : s.white.total > s.black.total ? 'white'
               : null;

  return { winner, played, oppPlayed, moves };
}

module.exports = function getMove(game) {
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

  // Round-robin playouts across candidates until the work budget is spent.
  // Work per playout = moves played + 10 (per-playout overhead).
  let totalWork = 0;
  let cidx = 0;
  while (totalWork < WORK_BUDGET) {
    const move = candidates[cidx];
    const clone = game.clone();
    if (move.type === 'place') {
      clone.placeStone(move.x, move.y);
    } else {
      clone.pass();
    }
    const { winner, played, oppPlayed, moves } = playTracked(clone, player);
    totalWork += moves + 10; // +10 per playout for clone/setup/scoring overhead
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
    const blackTotal = territory.black + game.captured.white;
    const whiteTotal = territory.white + game.captured.black + game.komi;
    const actualWinner = blackTotal > whiteTotal ? 'black'
                       : whiteTotal > blackTotal ? 'white'
                       : null;
    const allPlayoutsAgree = actualWinner === player
      ? wins[PASS_IDX] === plays[PASS_IDX]
      : wins[PASS_IDX] === 0;
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
