'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

/**
 * AMAF (All-Moves-As-First) Monte Carlo policy.
 *
 * For each candidate, run random playouts starting with that move in
 * complete round-robin rounds.  Every move made during a playout is credited
 * to the corresponding cell with a linearly decaying weight (1.0 → 0), so a
 * single playout of candidate A also updates estimates for candidates B, C, …
 * that appear later, giving all candidates far more data than mc.js provides.
 *
 * Interface: getMove(game, timeBudgetMs) → { type: 'pass' } | { type: 'place', x, y }
 *   game         - a live Game instance (read-only; do not mutate)
 *   timeBudgetMs - milliseconds allowed for this decision (required)
 */

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

const performance = (typeof window !== 'undefined') ? window.performance
  : require('perf_hooks').performance;

const { PASS, BLACK, WHITE } = _isNode ? require('../game2.js') : window.Game2;
const Util = _isNode ? require('../util.js') : window.Util;

const PLAYOUTS        = Util.envInt  ('PLAYOUTS', 0);
// Weight multiplier for opponent moves.  Override with AMAF_OPP_WEIGHT=<n>.
const OPP_MOVE_WEIGHT = Util.envFloat('AMAF_OPP_WEIGHT', 0);

// Random playout.  Returns { winner, played } where played is a Float32Array
// of length cap: positive value = played by BLACK, negative = played by WHITE,
// zero = not played.  Weight decreases linearly from 1.0; first play wins.
function playTracked(game2) {
  const cap        = game2.N * game2.N;
  const played     = new Float32Array(cap);
  const moveLimit  = cap + 20;
  const weightStep = 1 / cap;
  let moves = 0, weight = 1.0;

  while (!game2.gameOver && moves < moveLimit) {
    const current = game2.current;
    const idx     = game2.randomLegalMove();
    if (idx === PASS) { game2.play(PASS); moves++; continue; }
    if (played[idx] === 0) played[idx] = current === BLACK ? weight : -weight;
    game2.play(idx);
    moves++;
    weight -= weightStep;
  }

  return { winner: game2.estimateWinner(), played };
}

function getMove(game, timeBudgetMs, options = {}) {
  if (game.gameOver) return { type: 'pass', move: PASS };

  const playoutLimit = options.playoutLimit || PLAYOUTS;

  const game2  = game.cells ? game.clone() : game.toGame2();
  const player = game2.current;
  const N      = game2.N;
  const cap    = N * N;

  // Build list of legal candidate moves (flat cell indices; pass at cap).
  const candidates = [];
  const cells = game2.cells;
  for (let i = 0; i < cap; i++) {
    if (game2.isLegal(i) && !game2.isTrueEye(i)) candidates.push(i);
  }
  if (game2.moveCount >= cap / 2 || game2.consecutivePasses > 0) {
    candidates.push(PASS);
  }
  Util.shuffle(candidates);

  // AMAF stats indexed by cell (0..N*N-1); pass stored at N*N.
  const wins  = new Float32Array(cap + 1);
  const plays = new Float32Array(cap + 1);
  const PASS_IDX = cap;

  // Run complete rounds (one playout per candidate per round) so every
  // candidate always has exactly the same number of direct playouts.
  // Always complete at least one full round before checking the budget.
  const deadline = performance.now() + timeBudgetMs;
  let round = 0, playoutCount = 0;
  while (true) {
    if (round > 0 && (playoutLimit > 0 ? playoutCount >= playoutLimit : performance.now() >= deadline)) break;

    for (let cidx = 0; cidx < candidates.length; cidx++) {
      playoutCount++;
      const move  = candidates[cidx];
      const clone = game2.clone();
      clone.play(move);

      const { winner, played } = playTracked(clone);
      const won = winner === player ? 1 : 0;

      // Credit the opening move at full weight (it was played "first").
      const firstIdx = move === PASS ? PASS_IDX : move;
      plays[firstIdx] += 1.0;
      wins[firstIdx]  += won;

      // Credit playout moves using signed weights from played[].
      const playerSign = player === BLACK ? 1 : -1;
      for (let k = 0; k < cap; k++) {
        const w = played[k];
        if (w === 0) continue;
        if (w * playerSign > 0) {
          const wt = Math.abs(w);
          plays[k] += wt;
          wins[k]  += won * wt;
        } else if (OPP_MOVE_WEIGHT > 0) {
          const wt = Math.abs(w) * OPP_MOVE_WEIGHT;
          plays[k] += wt;
          wins[k]  += (1 - won) * wt;
        }
      }
      if (playoutLimit > 0 && playoutCount >= playoutLimit) break;
    }
    round++;
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
  if (bestRatio === 0) return { type: 'pass', move: PASS };

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
    ? { type: 'pass', move: PASS }
    : { type: 'place', move: best, x: best % N, y: (best / N) | 0 };
}

if (typeof module !== 'undefined') module.exports = { getMove };
else window.Amaf = { getMove };

})();
