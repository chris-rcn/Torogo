'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
const { BLACK, PASS } = _isNode ? require('../game2.js') : window.Game2;
const Util = _isNode ? require('../util.js') : window.Util;

const TD_SIMS  = Util.envInt  ('TD_SIMS', 0);
const TD_DEPTH = Util.envInt  ('TD_DEPTH', 8);
const LR       = Util.envFloat('TD_LR', 0.3);
const EPSILON  = Util.envFloat('TD_EPSILON', 0.1);

// ── Features ──────────────────────────────────────────────────────────────────

// Returns [{ key, polarity }, ...] for the current board position.
//
// Type 1 — per stone: key = cell index, polarity = +1 (BLACK) / -1 (WHITE).
//
// Type 2 — 2×2 window anchored at TL = idx:
//   Raw cell values: +1 BLACK, -1 WHITE, 0 empty.
//   Color-normalized: if net sum < 0, negate all cells; polarity follows.
//   Windows with net sum = 0 (all-empty or color-balanced) are skipped.
//   key = cap + idx*81 + ternary(v0,v1,v2,v3)   (no overlap with type-1 range)
function findFeatures(game) {
  const cap   = game.N * game.N;
  const cells = game.cells;
  const nbr   = game._nbr;
  const dnbr  = game._dnbr;
  const out   = [];

  for (let idx = 0; idx < cap; idx++) {
    const c = cells[idx];

    // 1x1
    if (c !== 0) out.push({ key: idx, polarity: c === BLACK ? 1 : -1 });

    // 2×2
    const tl = idx;
    const ri = nbr [idx * 4 + 3];  // right
    const di = nbr [idx * 4 + 1];  // down
    const dr = dnbr[idx * 4 + 3];  // down-right

    let v0 = cells[tl] === BLACK ? 1 : cells[tl] !== 0 ? -1 : 0;
    let v1 = cells[ri] === BLACK ? 1 : cells[ri] !== 0 ? -1 : 0;
    let v2 = cells[di] === BLACK ? 1 : cells[di] !== 0 ? -1 : 0;
    let v3 = cells[dr] === BLACK ? 1 : cells[dr] !== 0 ? -1 : 0;

    const sum = v0 + v1 + v2 + v3;
    if (sum === 0) continue;

    let polarity = 1;
    if (sum < 0) {
      v0 = -v0; v1 = -v1; v2 = -v2; v3 = -v3;
      polarity = -1;
    }

    const t   = (v0 + 1) * 27 + (v1 + 1) * 9 + (v2 + 1) * 3 + (v3 + 1);
    out.push({ key: cap + idx * 81 + t, polarity });
  }

  return out;
}

// ── Value function ─────────────────────────────────────────────────────────────

// V(s) = P(BLACK wins)
function evaluate(features, weights) {
  let z = 0;
  for (const f of features) {
    const w = weights.get(f.key) ?? 0;
    z += f.polarity * w;
  }
  return z;  // linear
//  return 1 / (1 + Math.exp(-z));  // logistic
}

// Δw_k = (LR / n) · (target − v) · polarity_k
function tdUpdate(features, v, target, weights) {
  const n = features.length;
  if (n === 0) return;
  const step = (LR / n) * (target - v);  // logistic step
//  const step = LR * (target - v);  // linear step?
  for (const f of features) {
    const w = weights.get(f.key) || 0;
    weights.set(f.key, w + step * f.polarity); // logistic update
//    weights.set(f.key, w + step * f.polarity * w); // linear update?
  }
}

// ── 1-ply search ──────────────────────────────────────────────────────────────

function bestMove(game, weights) {
  const cap     = game.N * game.N;
  const isBlack = game.current === BLACK;
  let bestIdx   = PASS;
  let bestScore = isBlack ? -Infinity : Infinity;

  for (let i = 0; i < cap; i++) {
    if (game.cells[i] !== 0) continue;
    if (game.isTrueEye(i))   continue;
    if (!game.isLegal(i))    continue;
    const g = game.clone();
    g.play(i);
    const v = evaluate(findFeatures(g), weights);
    if (isBlack ? v > bestScore : v < bestScore) { bestScore = v; bestIdx = i; }
  }

  return bestIdx;
}

// ── getMove ───────────────────────────────────────────────────────────────────

function getMove(game, budgetMs = 1000) {
  const weights  = new Map();
  const maxMoves = game.N * game.N * 4;
  const deadline = TD_SIMS <= 0 ? Date.now() + budgetMs : Infinity;
  let sim = 0;

  while (TD_SIMS > 0 ? sim < TD_SIMS : Date.now() < deadline) {
    sim++;
    const g = game.clone();
    let prev2 = null, prev1 = null;

    // TD phase: play TD_DEPTH moves with policy (ε-greedy), applying 2-step TD.
    for (let step = 0; step < TD_DEPTH && !g.gameOver; step++) {
      const features = findFeatures(g);
      const v        = evaluate(features, weights);

      if (prev2 !== null) tdUpdate(prev2, evaluate(prev2, weights), v, weights);
      prev2 = prev1;
      prev1 = features;

      g.play(Math.random() < EPSILON ? g.randomLegalMove() : bestMove(g, weights));
    }

    // Random playout to terminal.
    let moves = 0;
    while (!g.gameOver && moves++ < maxMoves) g.play(g.randomLegalMove());

    // Terminal TD updates for the last two positions in the TD phase.
    const outcome = g.calcWinner() === BLACK ? 1 : 0;
    for (const features of [prev2, prev1]) {
      if (features !== null) tdUpdate(features, evaluate(features, weights), outcome, weights);
    }
  }

  return { move: bestMove(game, weights) };
}

// ── Exports ───────────────────────────────────────────────────────────────────

const TDSearch = { getMove };

if (typeof module !== 'undefined') module.exports = TDSearch;
else window.TDSearch = TDSearch;

})();
