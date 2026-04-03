'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
const { BLACK, WHITE, EMPTY, PASS } = _isNode ? require('../game2.js') : window.Game2;
const Util = _isNode ? require('../util.js') : window.Util;

const Playout = require('./playout.js');
const { search: abSearch } = _isNode ? require('../ab-search.js') : window.ABSearch;
const { evaluate: vpEvaluate, loadWeights } = _isNode ? require('../vpatterns.js') : window.VPatterns;

const TD_SIMS       = Util.envInt  ('TD_SIMS', 0);
const WIDE_DEPTH    = Util.envInt  ('TD_WIDE_DEPTH', 0);
const NARROW_DEPTH  = Util.envInt  ('TD_NARROW_DEPTH', 0);
const EPSILON       = Util.envFloat('TD_EPSILON', 0.1);
const USE_1x2       = Util.envInt  ('TD_USE_1x2', 0);
const USE_2x2       = Util.envInt  ('TD_USE_2x2', 1);
const USE_2x3       = Util.envInt  ('TD_USE_2x3', 0);
const LR0           = Util.envFloat('TD_LR0', 0.6);
const LR1           = Util.envFloat('TD_LR1', 0.3);
const AB_DEPTH      = Util.envInt  ('TD_AB_DEPTH', 2);
const EVAL_DEPTH    = Util.envInt  ('TD_EVAL_DEPTH', 0);
const PAT_DATA      = Util.envStr  ('TD_PAT_DATA', '');

let evalModel = null;
if (_isNode && PAT_DATA) evalModel = loadWeights(PAT_DATA);

// Reusable container storing weight indices (resolved from feature keys).
// maxLen upper bound: one 1×1 + one 2×2 + one 1×2 entry per cell = 3 * cap.
function makeBuf(area) {
  return { 
    scratchA: new Int32Array(area),
    scratchB: new Int32Array(area),
    idxs: new Int32Array((1 + USE_1x2 + USE_2x2 + USE_2x3) * area), 
    n: 0,
  };
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

// Fills buf.idxs with weight indices for the current board position.
// Patterns where all cells are empty are skipped.
function findFeatures(game, buf, ctx, doSetNext, nextMove) {
  const area  = game.N * game.N;
  const cells = game.cells;
  const nbr   = game._nbr;
  const dnbr  = game._dnbr;
  buf.n = 0;

  if (nextMove === PASS) doSetNext = false;
  let captures;
  if (doSetNext) {
    captures = game.captureList(nextMove);
    for (let c = 0; c < captures.length; c++) {
      cells[captures[c]] = EMPTY;
    }
    cells[nextMove] = game.current;
  }
  for (let idx = 0; idx < area; idx++) {
    const v0 = cells[idx];
    // 1×1
    if (v0) {
      buf.idxs[buf.n++] = resolveKey(Math.imul(835 + idx, 691 + v0), ctx);
    }

    const v1 = cells[(idx+1)%area];
    const v0v1 = (v0*v0 + v1*v1) | 0;

    // Set scratchA to 2x1 keys.
    buf.scratchA[idx] = v0v1 * Math.imul(389 + v0, 593 + v1);
    if (USE_1x2 && v0v1) {
      buf.idxs[buf.n++] = resolveKey(Math.imul(129 + idx, 972 + buf.scratchA[idx]), ctx);
    }
  }
  // Now combine 1x2 keys into 2x2 keys.
  // Set scratchB to 2x2 keys.
  if (USE_2x2 || USE_2x3) {
    for (let idx = 0; idx < area; idx++) {
      const v0 = buf.scratchA[idx];
      const v1 = buf.scratchA[(idx+game.N)%area];
      const v0v1 = (v0*v0 + v1*v1) | 0;
      buf.scratchB[idx] = v0v1 * Math.imul(843 + v0, 670 + v1);
      if (USE_2x2 && v0v1) {
        buf.idxs[buf.n++] = resolveKey(Math.imul(381 + idx, 463 + buf.scratchB[idx]), ctx);
      }
    }
    if (USE_2x3) {
      for (let idx = 0; idx < area; idx++) {
        const v0 = buf.scratchB[idx];
        const v1 = buf.scratchB[(idx+1)%area];
        if (v0*v0 + v1*v1) {
          const hash = Math.imul(427 + v0, 595 + v1);
          buf.idxs[buf.n++] = resolveKey(Math.imul(743 + idx, 603 + hash), ctx);
        }
      }
    }
  }
  if (doSetNext) {
    for (let c = 0; c < captures.length; c++) {
      cells[captures[c]] = -game.current;
    }
    cells[nextMove] = EMPTY;
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
function tdUpdate(buf, target, weightsArr, lr) {
  const { idxs, n } = buf;
  if (n === 0) return;
  const step = lr * (target - buf.val) / n;
  
  for (let i = 0; i < n; i++) {
    weightsArr[idxs[i]] += step;
  }
}

// Custom 1-ply search which bypasses non-capture moves.
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
  if (candidates.length === 0) {
    return { move: PASS };
  }
  if (game.consecutivePasses > 0 || game.emptyCount < area/2) {
    candidates.push(PASS);
  }
  const count = width > 0 ? Math.min(width, candidates.length) : candidates.length;
  for (let c = 0; c < count; c++) {
    const j = c + Math.floor(Math.random() * (candidates.length - c));
    const move = candidates[j];
    candidates[j] = candidates[c];
    findFeatures(game, searchFeats, ctx, true, move);
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

  return { move: game.randomLegalMove() };
//  return Playout.getMove(game);
}

// These two store the state of the model at the start of the game.
let keyToIdx_start = new Map();
let weightsArr_start = [];

// These two store the current state of the model.
let keyToIdx = new Map();
let weightsArr = [];

let lastMoveCount = 0;

function getMove(game, budgetMs = 1000) {
  if (game.consecutivePasses > 0 && game.calcWinner() === game.current) {
    return { move: PASS, type: 'pass', info: 'End the game; ahead in points.' };
  }

  const moveCountDiff = game.moveCount - lastMoveCount;
  const moveCountUnexpected = moveCountDiff < 0 || moveCountDiff > 2;
  
  if (game.moveCount === 1) {
    keyToIdx   = keyToIdx_start;
    weightsArr = weightsArr_start;
  } else if (moveCountUnexpected) {
    keyToIdx = new Map();
    weightsArr = [];
  }
  lastMoveCount = game.moveCount;

  const area = game.N * game.N;
  const maxMoves = 3 * game.emptyCount + 20;
  const tStart = Date.now();
  let sims = 0;
  let maxSteps = 0;

  const lastSearchMove = new Int32Array(WIDE_DEPTH); // indexed by step
  const lastSearchMoveSame = new Float32Array(WIDE_DEPTH); // indexed by step
  const ctx = { keyToIdx, weightsArr, lastSearchMove, lastSearchMoveSame, searchFeats: makeBuf(area) };

  let prev2 = makeBuf(area);
  let prev1 = makeBuf(area);
  let feats = makeBuf(area);

  let lr = LR1;

  while (true) {

    let progress;
    if (TD_SIMS > 0) {
      progress = sims / TD_SIMS;
    } else {
      progress = (Date.now() - tStart) / budgetMs;
    }
    if (progress >= 1) break;
    if (game.moveCount > 1) {  // Hacky guard to keep start model stable.
      lr = LR1 - progress * (LR0 - LR1);
    }

    sims++;
    const g = game.clone();
    const evalDepth = (EVAL_DEPTH > 0 && evalModel) ? EVAL_DEPTH + (sims % 2) : 10000;
    let step = 0;
    let outcome;
    let moveSelection = selectTrainingMove(g, step, ctx);
    while (true) {

      g.play(moveSelection.move);
      step++;
      if (g.gameOver || step > maxMoves) {
        outcome = (g.estimateWinner() === BLACK) ? 1 : 0;
        break;
      }
      if (step >= evalDepth) {
        outcome = vpEvaluate(g, evalModel);
        break;
      }

      moveSelection = selectTrainingMove(g, step, ctx);

      findFeatures(g, feats, ctx);
      evaluate(feats, weightsArr);

      if (prev2.n > 0) {
        tdUpdate(prev2, feats.val, weightsArr, lr);
      }

      const temp = prev2;
      prev2 = prev1;
      prev1 = feats;
      feats = temp;
    }

    // Terminal TD updates for the last two tracked positions.
    if (prev2.n > 0) tdUpdate(prev2, outcome, weightsArr, lr);
    if (prev1.n > 0) tdUpdate(prev1, outcome, weightsArr, lr);

    maxSteps = Math.max(maxSteps, step);
  }

  if (game.moveCount === 1) {
    keyToIdx_start = new Map(keyToIdx);
    weightsArr_start = weightsArr.slice();
  }

  const evalFn = g => { findFeatures(g, ctx.searchFeats, ctx); evaluate(ctx.searchFeats, ctx.weightsArr); return ctx.searchFeats.val; };
  const searchResult = { move: abSearch(game, AB_DEPTH, evalFn), moveScore: evalFn(game) };

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

