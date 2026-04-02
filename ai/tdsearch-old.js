'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
const { BLACK, WHITE, EMPTY, PASS } = _isNode ? require('../game2.js') : window.Game2;
const Util = _isNode ? require('../util.js') : window.Util;

const Playout = require('./playout.js');

const TD_SIMS       = Util.envInt  ('TD_SIMS', 0);
const WIDE_DEPTH    = Util.envInt  ('TD_WIDE_DEPTH', 0);
const NARROW_DEPTH  = Util.envInt  ('TD_NARROW_DEPTH', 0);
const LR            = Util.envFloat('TD_LR', 0.3);
const EPSILON       = Util.envFloat('TD_EPSILON', 0.1);
const USE_1x2       = Util.envInt  ('TD_USE_1x2', 0);
const USE_2x2       = Util.envInt  ('TD_USE_2x2', 1);

// Reusable container storing weight indices (resolved from feature keys).
// maxLen upper bound: one 1×1 + one 2×2 + one 1×2 entry per cell = 3 * cap.
function makeBuf(area) {
  return { idxs: new Int32Array((1 + USE_1x2 + USE_2x2) * area), n: 0 };
}

// keyToIdx: Map<featureKey, weightIndex>
// weightsArr: number[] — weight values indexed by weightIndex
//
// resolveKey returns the weight index for a feature key, inserting weight 0 if new.
function resolveKey(key, ctx) {
  let wi = ctx.keyToIdx.get(key);
  if (wi === undefined) {
    wi = ctx.weightsArr.length;
    ctx.keyToIdx.set(key, wi);
    ctx.weightsArr.push(0);
  }
  return wi;
}

function encodeKey11(idx, v0) {
  return idx * 3 + v0 + 1;
}
function encodeKey12(area, idx, v0, v1) {
  return 84 * area + idx * 9 + (v0 + 1) * 3 + (v1 + 1);
}
function encodeKey22(area, idx, v0, v1, v2, v3) {
  return 3 * area + idx * 81 + (v0 + 1) * 27 + (v1 + 1) * 9 + (v2 + 1) * 3 + (v3 + 1);
}

// Fills buf.idxs with weight indices for the current board position.
// Patterns where all cells are empty are skipped.
function findFeatures(game, buf, ctx, doSetNext, nextStoneIdx) {
  const area  = game.N * game.N;
  const cells = game.cells;
  const nbr   = game._nbr;
  const dnbr  = game._dnbr;
  buf.n = 0;

  if (doSetNext) {
    cells[nextStoneIdx] = game.current;
  }
  for (let idx = 0; idx < area; idx++) {
    const v0 = cells[idx];

    // 1×1
    if (v0 !== 0) {
      buf.idxs[buf.n++] = resolveKey(encodeKey11(idx, v0), ctx);
    }

    const v1 = cells[nbr [idx * 4 + 3]];
    const v0v1 = v0*v0 + v1*v1;

    // 1×2 (horizontal)
    if (USE_1x2 && v0v1 > 0) {
      buf.idxs[buf.n++] = resolveKey(encodeKey12(area, idx, v0, v1), ctx);
    }

    // 2×2
    if (USE_2x2) {
      const v2 = cells[nbr[idx * 4 + 1]];
      const v3 = cells[dnbr[idx * 4 + 3]];
      if (v0v1 + v2*v2 + v3*v3 > 0) {
        buf.idxs[buf.n++] = resolveKey(encodeKey22(area, idx, v0, v1, v2, v3), ctx);
      }
    }
  }
  if (doSetNext) {
    cells[nextStoneIdx] = EMPTY;
  }
}

// V(s) = σ(Σ w_k) = P(BLACK wins)
function evaluate(buf, weightsArr) {
  let z = 0;
  const { idxs, n } = buf;
  for (let i = 0; i < n; i++) z += weightsArr[idxs[i]];
  buf.val = 1 / (1 + Math.exp(-z));
}

// Δw_k = (LR / n) · (target − buf.val)
function tdUpdate(buf, target, weightsArr) {
  const { idxs, n } = buf;
  if (n === 0) return;
  const step = LR * (target - buf.val) / n;
  
  for (let i = 0; i < n; i++) {
    weightsArr[idxs[i]] += step;
  }
}

function search1ply(game, ctx, width = 0) {
  const area = game.N * game.N;
  const searchFeats = ctx.searchFeats;
  const isBlack = game.current === BLACK;
  let bestIdx = PASS;
  let bestScore = isBlack ? -Infinity : Infinity;
  const candidates = [];
  for (let i = 0; i < area; i++) {
    if (game.isLegal(i) && !game.isTrueEye(i)) candidates.push(i);
  }
  const count = width > 0 ? Math.min(width, candidates.length) : candidates.length;
  for (let c = 0; c < count; c++) {
    const j = c + Math.floor(Math.random() * (candidates.length - c));
    const move = candidates[j];
    candidates[j] = candidates[c];
    if (game.isCapture(move)) {
      const g = game.clone();
      g.play(move);
      findFeatures(g, searchFeats, ctx);
    } else {
      findFeatures(game, searchFeats, ctx, true, move);
    }
    evaluate(searchFeats, ctx.weightsArr);
    if (isBlack === (searchFeats.val > bestScore)) { 
      bestScore = searchFeats.val;
      bestIdx = move;
    }
  }
  return { move: bestIdx, moveScore: bestScore };
}

function selectTrainingMove(game, step, ctx) {
  if (Math.random() < EPSILON) 
    return { move: game.randomLegalMove() };

  if (step < WIDE_DEPTH) {
    if (false && Math.random() < ctx.lastSearchMoveSame[step]) {
      return { move: ctx.lastSearchMove[step] };
    } else {
      const move = search1ply(game, ctx).move;
      const isSame = ctx.lastSearchMove[step] === move ? 1 : 0;
      ctx.lastSearchMoveSame[step] += 0.1 * (isSame - ctx.lastSearchMoveSame[step]);
      ctx.lastSearchMove[step] = move;
      return { move };
    }
  }

  if (step < NARROW_DEPTH) 
    return search1ply(game, ctx, 2);

//  return { move: game.randomLegalMove() };
  return Playout.getMove(game);
}

function getMove(game, budgetMs = 1000) {
  if (game.consecutivePasses > 0 && game.calcWinner() === game.current) {
    return { move: PASS, type: 'pass', info: 'End the game; ahead in points.' };
  }

  const area = game.N * game.N;
  const maxMoves = 3 * game.emptyCount + 20;
  const deadline = TD_SIMS <= 0 ? Date.now() + budgetMs : Infinity;
  let sims = 0;
  let maxSteps = 0;

  const keyToIdx = new Map();
  const weightsArr = [];
  const lastSearchMove = new Int32Array(WIDE_DEPTH); // indexed by step
  const lastSearchMoveSame = new Float32Array(WIDE_DEPTH); // indexed by step
  const ctx = { keyToIdx, weightsArr, lastSearchMove, lastSearchMoveSame, searchFeats: makeBuf(area) };

  let prev2 = makeBuf(area);
  let prev1 = makeBuf(area);
  let feats = makeBuf(area);

  while (TD_SIMS > 0 ? sims < TD_SIMS : Date.now() < deadline) {
    sims++;
    const g = game.clone();
    let step = 0;

    let outcome;
    let moveSelection = selectTrainingMove(g, step, ctx);
    while (true) {

      g.play(moveSelection.move);
      if (g.gameOver || step > maxMoves) {
        outcome = (g.estimateWinner() === BLACK) ? 1 : 0;
        break;
      }
      step++;

      moveSelection = selectTrainingMove(g, step, ctx);

      findFeatures(g, feats, ctx);
      evaluate(feats, weightsArr);

      if (prev2.n > 0) {
        tdUpdate(prev2, feats.val, weightsArr);
      }

      const temp = prev2;
      prev2 = prev1;
      prev1 = feats;
      feats = temp;
    }

    // Terminal TD updates for the last two tracked positions.
    if (prev2.n > 0) tdUpdate(prev2, outcome, weightsArr);
    if (prev1.n > 0) tdUpdate(prev1, outcome, weightsArr);

    maxSteps = Math.max(maxSteps, step);
  }

  const searchResult = search1ply(game, ctx);
  const myScore = game.current === BLACK ? searchResult.moveScore : 1 - searchResult.moveScore
  const move = myScore < 0.01 ? PASS : searchResult.move;
  const result = { move, ctx, sims, maxSteps }
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

if (typeof module !== 'undefined') {
  module.exports = { getMove };
  require('./tdsearch.test.js').runTests(
    { makeBuf, resolveKey, findFeatures, evaluate, tdUpdate, getMove },
    require('../game2.js')
  );
} else {
  window.getMove = getMove;
}

})();

