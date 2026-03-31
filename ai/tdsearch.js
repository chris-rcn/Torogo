'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
const { BLACK, WHITE, PASS } = _isNode ? require('../game2.js') : window.Game2;
const Util = _isNode ? require('../util.js') : window.Util;

const TD_SIMS       = Util.envInt  ('TD_SIMS', 0);
const WIDE_DEPTH    = Util.envInt  ('TD_WIDE_DEPTH', 0);
const NARROW_DEPTH  = Util.envInt  ('TD_NARROW_DEPTH', 0);
const LR            = Util.envFloat('TD_LR', 0.3);
const EPSILON       = Util.envFloat('TD_EPSILON', 0.1);

// ── Feature buffer ────────────────────────────────────────────────────────────

// Reusable container: avoids allocating a new array + objects on every call.
// maxLen must be >= the maximum number of features any position can produce.
// Upper bound: one 1×1 + one 2×2 + one 1×2 + one 2×1 entry per cell = 4 * cap.
function makeBuf(cap) {
  return { keys: new Int32Array(4 * cap), n: 0 };
}

// ── Features ──────────────────────────────────────────────────────────────────

// Fills buf with features for the current board position.
//
// Type 1 — 1×1 per stone: key = idx*2 + (BLACK ? 0 : 1)
//   range [0, 2*cap)
// Type 2 — 2×2 anchored at TL=idx:  key = 2*cap  + idx*81 + ternary4
//   range [2*cap, 83*cap)
// Type 3 — 1×2 anchored at L=idx:   key = 83*cap + idx*9  + ternary2
//   range [83*cap, 92*cap)
// Type 4 — 2×1 anchored at T=idx:   key = 92*cap + idx*9  + ternary2
//   range [92*cap, 101*cap)
//
// Multi-cell ternary: v ∈ {-1,0,+1} encoded as v+1 ∈ {0,1,2}.
// Windows where all cells are empty (sum = 0 and all v = 0) are skipped.
function findFeatures(game, buf) {
  const cap   = game.N * game.N;
  const cells = game.cells;
  const nbr   = game._nbr;
  const dnbr  = game._dnbr;
  buf.n = 0;

  for (let idx = 0; idx < cap; idx++) {
    const c = cells[idx];

    // 1×1
    if (c !== 0) {
      buf.keys[buf.n] = idx * 2 + (c === BLACK ? 0 : 1);
      buf.n++;
    }

    const ri = nbr [idx * 4 + 3];  // right
    const di = nbr [idx * 4 + 1];  // down
    const dr = dnbr[idx * 4 + 3];  // down-right
    const v0 = cells[idx] === BLACK ? 1 : cells[idx] !== 0 ? -1 : 0;
    const v1 = cells[ri]  === BLACK ? 1 : cells[ri]  !== 0 ? -1 : 0;
    const v2 = cells[di]  === BLACK ? 1 : cells[di]  !== 0 ? -1 : 0;
    const v3 = cells[dr]  === BLACK ? 1 : cells[dr]  !== 0 ? -1 : 0;

    // 1×2 (horizontal)
    if (v0 !== 0 || v1 !== 0) {
      buf.keys[buf.n] = 83 * cap + idx * 9 + (v0 + 1) * 3 + (v1 + 1);
      buf.n++;
    }

    // 2×2
    if (v0 !== 0 || v1 !== 0 || v2 !== 0 || v3 !== 0) {
      buf.keys[buf.n] = 2 * cap + idx * 81 + (v0 + 1) * 27 + (v1 + 1) * 9 + (v2 + 1) * 3 + (v3 + 1);
      buf.n++;
    }
  }
}

// Like findFeatures but treats moveIdx as occupied by moveColor without modifying game.
function findFeaturesWithMove(game, moveIdx, moveColor, buf) {
  const cap   = game.N * game.N;
  const cells = game.cells;
  const nbr   = game._nbr;
  const dnbr  = game._dnbr;
  const mv    = moveColor === BLACK ? 1 : -1;
  buf.n = 0;

  // Inline helper: cell value with override at moveIdx.
  function cv(i) {
    if (i === moveIdx) return mv;
    const c = cells[i];
    return c === BLACK ? 1 : c !== 0 ? -1 : 0;
  }

  for (let idx = 0; idx < cap; idx++) {
    // 1×1
    const v = cv(idx);
    if (v !== 0) {
      buf.keys[buf.n] = idx * 2 + (v > 0 ? 0 : 1);
      buf.n++;
    }

    const ri = nbr [idx * 4 + 3];
    const di = nbr [idx * 4 + 1];
    const dr = dnbr[idx * 4 + 3];
    const v0 = cv(idx), v1 = cv(ri);
    const v2 = cv(di), v3 = cv(dr);

    // 1×2 (horizontal)
    if (v0 !== 0 || v1 !== 0) {
      buf.keys[buf.n] = 83 * cap + idx * 9 + (v0 + 1) * 3 + (v1 + 1);
      buf.n++;
    }

    // 2×2
    if (v0 !== 0 || v1 !== 0 || v2 !== 0 || v3 !== 0) {
      buf.keys[buf.n] = 2 * cap + idx * 81 + (v0 + 1) * 27 + (v1 + 1) * 9 + (v2 + 1) * 3 + (v3 + 1);
      buf.n++;
    }
  }
}

// ── Value function ─────────────────────────────────────────────────────────────

// V(s) = P(BLACK wins)
function evaluate(buf, weights) {
  let v = 0;
  const { keys, n } = buf;
  for (let i = 0; i < n; i++) {
    v += weights.get(keys[i]) ?? 0;
  }
  return v;
}

// Δw_k = (LR / n) · (target − v)
function tdUpdate(buf, v, target, weights) {
  const { keys, n } = buf;
  if (n === 0) return;
  const step = (LR / n) * (target - v);
  for (let i = 0; i < n; i++) {
    const k = keys[i];
    const w = weights.get(k) ?? 0;
    weights.set(k, w + step);
  }
}

// Copy buf contents into snap (pre-allocated buffer of same maxLen).
function snapBuf(buf, snap) {
  const n = buf.n;
  snap.keys.set(buf.keys.subarray(0, n));
  snap.n = n;
}

// ── 1-ply search ──────────────────────────────────────────────────────────────

function search1ply(game, weights, width = 0) {
  const cap     = game.N * game.N;
  const isBlack = game.current === BLACK;
  const buf     = makeBuf(cap);
  let bestIdx   = PASS;
  let bestScore = isBlack ? -Infinity : Infinity;
  const candidates = [];
  for (let i = 0; i < cap; i++) {
    if (game.isLegal(i) && !game.isTrueEye(i)) candidates.push(i);
  }
  const count = width > 0 ? Math.min(width, candidates.length) : candidates.length;
  for (let c = 0; c < count; c++) {
    const j = c + Math.floor(Math.random() * (candidates.length - c));
    const move = candidates[j];
    candidates[j] = candidates[c];
    if (!game.isCapture(move)) {
      findFeaturesWithMove(game, move, game.current, buf);
    } else {
      const g = game.clone();
      g.play(move);
      findFeatures(g, buf);
    }
    const v = evaluate(buf, weights);
    if (isBlack ? v > bestScore : v < bestScore) { bestScore = v; bestIdx = move; }
  }

  return bestIdx;
}

// ── getMove ───────────────────────────────────────────────────────────────────

function getMove(game, budgetMs = 1000) {
  const weights  = new Map();
  const cap      = game.N * game.N;
  const maxMoves = 2 * cap;
  const deadline = TD_SIMS <= 0 ? Date.now() + budgetMs : Infinity;
  let sim = 0;
  const tdUpdateDepth = 100;

  const buf   = makeBuf(cap);
  const snap1 = makeBuf(cap);
  const snap2 = makeBuf(cap);

  while (TD_SIMS > 0 ? sim < TD_SIMS : Date.now() < deadline) {
    sim++;
    const g = game.clone();
    let prev1 = null, prev2 = null;  // point at snap1 / snap2, alternating
    let step = 0;

    // Unified loop: policy (ε-greedy) for TD_DEPTH moves, then AMAF for
    // AMAF_MOVES moves, then random — TD updates applied throughout.
    while (!g.gameOver && step < maxMoves) {

      if (step < tdUpdateDepth) {
        findFeatures(g, buf);
        const val = evaluate(buf, weights);

        if (prev2 !== null) tdUpdate(prev2, prev2.val, val, weights);
        const snap = prev1 === snap1 ? snap2 : snap1;
        snapBuf(buf, snap);
        snap.val = val;
        prev2 = prev1;
        prev1 = snap;
      }

      let move;
      if (Math.random() < EPSILON) {
        move = g.randomLegalMove();
      } else if (step < WIDE_DEPTH) {
        move = search1ply(g, weights);
      } else if (step < NARROW_DEPTH) {
        move = search1ply(g, weights, 2);
      } else {
        move = g.randomLegalMove();
      }
      g.play(move);
      step++;
    }

    // Terminal TD updates for the last two tracked positions.
    const outcome = g.calcWinner() === BLACK ? 1 : 0;
    if (prev2 !== null) tdUpdate(prev2, prev2.val, outcome, weights);
    if (prev1 !== null) tdUpdate(prev1, prev1.val, outcome, weights);
  }

  return { move: search1ply(game, weights) };
}

// ── Exports ───────────────────────────────────────────────────────────────────

const TDSearch = { getMove };

if (typeof module !== 'undefined') module.exports = TDSearch;
else window.TDSearch = TDSearch;

})();
