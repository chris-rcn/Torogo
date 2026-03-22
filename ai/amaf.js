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
const { PASS, BLACK, WHITE } = require('../game2.js');

const DEFAULT_BUDGET_MS = 500;
const PLAYOUTS = process.env.PLAYOUTS ? parseInt(process.env.PLAYOUTS, 10) : 0;
// Weight decay per subsequent player move.  Override with AMAF_DISCOUNT=<n>.
const DISCOUNT = process.env.AMAF_DISCOUNT !== undefined
  ? parseFloat(process.env.AMAF_DISCOUNT)
  : 0.5;
// Weight multiplier for opponent moves.  Override with AMAF_OPP_WEIGHT=<n>.
const OPP_MOVE_WEIGHT = process.env.AMAF_OPP_WEIGHT !== undefined
  ? parseFloat(process.env.AMAF_OPP_WEIGHT)
  : 0;

// Random playout using Game2.  Returns the ordered lists of cell indices
// played by the current player (played) and opponent (oppPlayed) during the
// playout.  winner is BLACK or WHITE.
function playTracked(game2, trackColor) {
  const N     = game2.N;
  const cap   = N * N;
  const cells = game2.cells;
  const nbr   = game2._nbr;
  const played    = [];  // ordered cell indices for trackColor
  const oppPlayed = [];  // ordered cell indices for the other color

  const empty = [];
  for (let i = 0; i < cap; i++) {
    if (cells[i] === 0) empty.push(i);
  }

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

      if (current === trackColor) played.push(idx);
      else                        oppPlayed.push(idx);

      // Snapshot neighbour occupancy to detect captures after play.
      const base = idx * 4;
      const n0 = cells[nbr[base]], n1 = cells[nbr[base + 1]],
            n2 = cells[nbr[base + 2]], n3 = cells[nbr[base + 3]];
      game2.play(idx);
      empty[end] = empty[empty.length - 1];
      empty.pop();

      // If any previously occupied neighbour became empty, captures occurred.
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

  return { winner: game2.estimateWinner(), played, oppPlayed };
}

module.exports = function getMove(game, timeBudgetMs) {
  if (game.gameOver) return { type: 'pass' };

  const game2  = game.cells ? game.clone() : game.toGame2();
  const player = game2.current;
  const N      = game2.N;
  const cap    = N * N;

  // Build list of legal candidate moves (flat cell indices; pass at cap).
  const candidates = [];
  const cells = game2.cells;
  for (let i = 0; i < cap; i++) {
    if (cells[i] !== 0) continue;
    if (game2.isTrueEye(i)) continue;
    if (game2.isLegal(i)) candidates.push(i);
  }
  if (game2.moveCount >= cap / 2 || game2.consecutivePasses > 0) {
    candidates.push(PASS);
  }

  // AMAF stats indexed by cell (0..N*N-1); pass stored at N*N.
  // Float64 to accommodate fractional discount weights.
  const wins  = new Float64Array(cap + 1);
  const plays = new Float64Array(cap + 1);
  const PASS_IDX = cap;

  // Round-robin playouts across candidates until the budget is exhausted.
  const budgetMs = timeBudgetMs != null ? timeBudgetMs : DEFAULT_BUDGET_MS;
  const deadline = performance.now() + budgetMs;
  let cidx = 0, playoutCount = 0;
  while (PLAYOUTS > 0 ? playoutCount < PLAYOUTS : performance.now() < deadline) {
    playoutCount++;
    const move   = candidates[cidx];
    const clone  = game2.clone();
    clone.play(move);  // works for both cell indices and PASS (-1)

    const { winner, played, oppPlayed } = playTracked(clone, player);
    const won = winner === player ? 1 : 0;

    // Credit the opening move at full weight (it was played "first").
    const firstIdx = move === PASS ? PASS_IDX : move;
    plays[firstIdx] += 1.0;
    wins[firstIdx]  += won;

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

  // Select the candidate with the highest AMAF win ratio; ties broken randomly.
  let bestRatio = -1;
  let bestCount = 0;
  let bestIdx   = 0;
  for (let i = 0; i < candidates.length; i++) {
    const idx   = candidates[i] === PASS ? PASS_IDX : candidates[i];
    if (plays[idx] === 0) continue;
    const ratio = wins[idx] / plays[idx];
    if (ratio > bestRatio) {
      bestRatio = ratio;
      bestIdx   = i;
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
    const passCandIdx = candidates.indexOf(PASS);
    if (passCandIdx !== -1) bestIdx = passCandIdx;
  }

  // Before committing to a pass, verify that the position is actually won.
  // If calcWinner disagrees with playout winners, fall back to the best
  // non-pass candidate.
  if (candidates[bestIdx] === PASS && plays[PASS_IDX] > 0) {
    const actualWinner = game2.calcWinner();
    const allPlayoutsAgree = actualWinner === player
      ? wins[PASS_IDX] === plays[PASS_IDX]
      : true;
    if (!allPlayoutsAgree) {
      let altRatio = -1;
      let altCount = 0;
      let altIdx   = -1;
      for (let i = 0; i < candidates.length; i++) {
        if (candidates[i] === PASS) continue;
        const idx = candidates[i];
        if (plays[idx] === 0) continue;
        const ratio = wins[idx] / plays[idx];
        if (ratio > altRatio) {
          altRatio = ratio;
          altIdx   = i;
          altCount = 1;
        } else if (ratio === altRatio) {
          altCount++;
          if (Math.random() * altCount < 1) altIdx = i;
        }
      }
      if (altIdx !== -1) bestIdx = altIdx;
    }
  }

  const best = candidates[bestIdx];
  return best === PASS
    ? { type: 'pass' }
    : { type: 'place', x: best % N, y: (best / N) | 0 };
};
