'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
const { BLACK, WHITE, PASS } = _isNode ? require('../game2.js') : window.Game2;
const Util = _isNode ? require('../util.js') : window.Util;

const TD_SIMS       = Util.envInt  ('TD_SIMS', 0);
const WIDE_DEPTH    = 1000;
const NARROW_DEPTH  = 0;
const LR            = 0.3;
const EPSILON       = 0.1;

// ── Feature buffer ────────────────────────────────────────────────────────────

// Reusable container storing weight indices (resolved from feature keys).
// maxLen upper bound: one 1×1 + one 2×2 + one 1×2 + one 2×1 entry per cell = 4 * cap.
function makeBuf(cap) {
  return { idxs: new Int32Array(4 * cap), n: 0 };
}

// ── Weight store ──────────────────────────────────────────────────────────────

// keyToIdx: Map<featureKey, weightIndex>
// weightsArr: number[] — weight values indexed by weightIndex
//
// resolveKey returns the weight index for a feature key, inserting weight 0 if new.
function resolveKey(key, keyToIdx, weightsArr) {
  let wi = keyToIdx.get(key);
  if (wi === undefined) {
    wi = weightsArr.length;
    keyToIdx.set(key, wi);
    weightsArr.push(0);
  }
  return wi;
}

// ── Features ──────────────────────────────────────────────────────────────────

// Fills buf.idxs with weight indices for the current board position.
//
// Type 1 — 1×1 per stone: key = idx*3 + c + 1  (BLACK=1→2, WHITE=-1→0)
//   range [0, 3*cap)
// Type 2 — 2×2 anchored at TL=idx:  key = 3*cap  + idx*81 + ternary4
//   range [3*cap, 84*cap)
// Type 3 — 1×2 anchored at L=idx:   key = 84*cap + idx*9  + ternary2
//   range [84*cap, 93*cap)
// Type 4 — 2×1 anchored at T=idx:   key = 93*cap + idx*9  + ternary2
//   range [93*cap, 102*cap)
//
// Multi-cell ternary: v ∈ {-1,0,+1} encoded as v+1 ∈ {0,1,2}.
// Windows where all cells are empty are skipped.
function findFeatures(game, buf, keyToIdx, weightsArr) {
  const cap   = game.N * game.N;
  const cells = game.cells;
  const nbr   = game._nbr;
  const dnbr  = game._dnbr;
  buf.n = 0;

  for (let idx = 0; idx < cap; idx++) {
    const v0 = cells[idx];

    // 1×1
    if (v0 !== 0) {
      buf.idxs[buf.n++] = resolveKey(idx * 3 + v0 + 1, keyToIdx, weightsArr);
    }

  }
}

// Like findFeatures but treats moveIdx as occupied by moveColor without modifying game.
function findFeaturesWithMove(game, moveIdx, moveColor, buf, keyToIdx, weightsArr) {
  const cap   = game.N * game.N;
  const cells = game.cells;
  const nbr   = game._nbr;
  const dnbr  = game._dnbr;
  const mv    = moveColor;
  buf.n = 0;

  // Inline helper: cell value with override at moveIdx.
  function cv(i) {
    return i === moveIdx ? mv : cells[i];
  }

  for (let idx = 0; idx < cap; idx++) {
    // 1×1
    const v0 = cv(idx);
    if (v0 !== 0) {
      buf.idxs[buf.n++] = resolveKey(idx * 3 + v0 + 1, keyToIdx, weightsArr);
    }

  }
}

// ── Value function ─────────────────────────────────────────────────────────────

// V(s) = σ(Σ w_k) = P(BLACK wins)
function evaluate(buf, weightsArr) {
  let z = 0;
  const { idxs, n } = buf;
  for (let i = 0; i < n; i++) z += weightsArr[idxs[i]];
  return 1 / (1 + Math.exp(-z));
}

// Δw_k = (LR / n) · (target − v)
function tdUpdate(buf, v, target, weightsArr) {
  const { idxs, n } = buf;
  if (n === 0) return;
  const step = (LR / n) * (target - v);
  for (let i = 0; i < n; i++) weightsArr[idxs[i]] += step;
}

// Copy buf contents into snap (pre-allocated buffer of same maxLen).
function snapBuf(buf, snap) {
  const n = buf.n;
  snap.idxs.set(buf.idxs.subarray(0, n));
  snap.n = n;
}

// ── 1-ply search ──────────────────────────────────────────────────────────────

function search1ply(game, keyToIdx, weightsArr, width = 0) {
  const cap     = game.N * game.N;
  const isBlack = game.current === BLACK;
  const buf     = makeBuf(cap);
  let bestIdx   = PASS;
  let bestScore = isBlack ? -Infinity : Infinity;
  const candidates = []; // TODO: iterative shuffle over all cells (using Game2.allCells).
  for (let i = 0; i < cap; i++) {
    if (game.isLegal(i) && !game.isTrueEye(i)) candidates.push(i);
  }
  const count = width > 0 ? Math.min(width, candidates.length) : candidates.length;
  for (let c = 0; c < count; c++) {
    const j = c + Math.floor(Math.random() * (candidates.length - c));
    const move = candidates[j];
    candidates[j] = candidates[c];
    if (game.isCapture(move)) {
      const g = game.clone(); g.play(move);
      findFeatures(g, buf, keyToIdx, weightsArr);
    } else {
      findFeaturesWithMove(game, move, game.current, buf, keyToIdx, weightsArr);
    }
    const v = evaluate(buf, weightsArr);
    if (isBlack === (v > bestScore)) { bestScore = v; bestIdx = move; }
  }

  return { move: bestIdx, moveScore: bestScore };
}

// ── getMove ───────────────────────────────────────────────────────────────────

function selectTrainingMove(game, step, keyToIdx, weightsArr) {
  if (Math.random() < EPSILON) 
    return game.randomLegalMove();

  if (step < WIDE_DEPTH) 
    return search1ply(game, keyToIdx, weightsArr).move;

  if (step < NARROW_DEPTH) 
    return search1ply(game, keyToIdx, weightsArr, 2).move;

  return game.randomLegalMove();
}

function getMove(game, budgetMs = 1000) {
  if (game.consecutivePasses > 0 && game.calcWinner() === game.current) {
    return { move: PASS, type: 'pass', info: 'End the game; ahead in points.' };
  }

  const keyToIdx = new Map();
  const weightsArr = [];
  const area = game.N * game.N;
  const maxMoves = 2 * area;
  const deadline = TD_SIMS <= 0 ? Date.now() + budgetMs : Infinity;
  let sims = 0;

  const buf   = makeBuf(area);
  const snap1 = makeBuf(area);
  const snap2 = makeBuf(area);

  while (TD_SIMS > 0 ? sims < TD_SIMS : Date.now() < deadline) {
    sims++;
    const g = game.clone();
    let prev1 = null, prev2 = null;  // point at snap1 / snap2, alternating
    let step = 0;

    let outcome = 0.5;
    let move = selectTrainingMove(g, step, keyToIdx, weightsArr);
    while (step < maxMoves) {

      g.play(move);
      if (g.gameOver) {
        outcome = g.calcWinner() === BLACK ? 1 : 0;
        break;
      }
      step++;

      move = selectTrainingMove(g, step, keyToIdx, weightsArr);

      findFeatures(g, buf, keyToIdx, weightsArr);
      const val = evaluate(buf, weightsArr);

      if (prev2 !== null) tdUpdate(prev2, prev2.val, val, weightsArr);
      const snap = prev1 === snap1 ? snap2 : snap1;
      snapBuf(buf, snap);
      snap.val = val;
      prev2 = prev1;
      prev1 = snap;
    }

    // Terminal TD updates for the last two tracked positions.
    if (prev2 !== null) tdUpdate(prev2, prev2.val, outcome, weightsArr);
    if (prev1 !== null) tdUpdate(prev1, prev1.val, outcome, weightsArr);
  }

  const searchResult = search1ply(game, keyToIdx, weightsArr);
  const myScore = game.current === BLACK ? searchResult.moveScore : 1 - searchResult.moveScore
  const move = myScore < 0.01 ? PASS : searchResult.move;
  const result = { move, keyToIdx, weightsArr, sims }
  result.info = `value=${myScore.toFixed(3)}`;
  if (move === PASS) {
    result.type = 'pass';
  } else {
    result.type = 'place';
    result.x = move % game.N;
    result.y = (move / game.N) | 0;
  }
  return result;
}

// ── Exports ───────────────────────────────────────────────────────────────────

if (typeof module !== 'undefined') module.exports = { getMove };
else window.getMove = getMove;

})();
