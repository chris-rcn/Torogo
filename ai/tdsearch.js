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
const { makeIntMap } = _isNode ? require('../int-map.js') : window.IntMap;

const SIMS          = Util.envInt  ('TD_SIMS', 0);
const WIDE_DEPTH    = Util.envInt  ('TD_WIDE_DEPTH', 0);
const NARROW_DEPTH  = Util.envInt  ('TD_NARROW_DEPTH', 0);
const EPSILON0      = Util.envFloat('TD_EPSILON0', 0.1);
const EPSILON1      = Util.envFloat('TD_EPSILON1', 0.1);
const USE_1x2       = Util.envInt  ('TD_USE_1x2', 0);
const USE_2x2       = Util.envInt  ('TD_USE_2x2', 1);
const LR0           = Util.envFloat('TD_LR0', 0.6);
const LR1           = Util.envFloat('TD_LR1', 0.3);
const AB_DEPTH      = Util.envInt  ('TD_AB_DEPTH', 1);
const EVAL_DEPTH    = Util.envInt  ('TD_EVAL_DEPTH', 0);
const EVAL_DATA     = Util.envStr  ('TD_EVAL_DATA', '');
const USE_PLAYOUT   = Util.envInt  ('TD_USE_PLAYOUT', '0');

let evalModel = null;
if (_isNode && EVAL_DATA) evalModel = loadWeights(EVAL_DATA);

// Reusable container storing weight indices (resolved from feature keys).
// maxLen upper bound: one 1×1 + one 2×2 + one 1×2 entry per cell = 3 * cap.
function makeBuf(area) {
  const maxN = (1 + USE_1x2 + USE_2x2) * area;
  return {
    scratchA:  new Int32Array(area),
    scratchB:  new Int32Array(area),
    idxs:      new Int32Array(maxN),
    n:         0,
    sum:       0,
    // Scratch for evaluateDelta — avoids per-call allocation in the hot path.
    deltaMark: new Uint8Array(area),   // dedup; reset by unmarking (not fill)
    deltasSA:  new Int32Array(64),     // affected scratchA slot indices
    deltasW:   new Int32Array(64),     // affected 2×2 window indices
    savedSA:   new Int32Array(64),     // saved scratchA values for restore
    // Slot tracking for incremental maintenance (used only by primary buffer).
    slotA:     new Int32Array(area).fill(-1),  // slot in idxs for 1×1 feature at j
    slotB:     new Int32Array(area).fill(-1),  // slot in idxs for 1×2 feature at j
    slotC:     new Int32Array(area).fill(-1),  // slot in idxs for 2×2 feature at j
    posTypeOf: new Int32Array(maxN),           // (j << 2) | type for each slot
  };
}

// keyToIdx: IntMap<featureKey, weightIndex>
// weightsArr: number[] — weight values indexed by weightIndex
//
// resolveKey returns the weight index for a feature key, inserting weight 0 if new.
function resolveKey(key, ctx) {
  let wi = ctx.keyToIdx.get(key);
  if (wi < 0) {
    wi = ctx.weightsArr.length;
    ctx.keyToIdx.set(key, wi);
    ctx.weightsArr.push(0);
  }
  return wi;
}

// Fills buf.idxs with weight indices for the current board position.
// Patterns where all cells are empty are skipped.
function findFeatures(game, buf, ctx, nextMove) {
  const area  = game.N * game.N;
  const cells = game.cells;
  const nbr   = game._nbr;
  const dnbr  = game._dnbr;
  buf.n = 0;

  let captures;
  if (nextMove >= 0) {
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
  if (USE_2x2) {
    for (let idx = 0; idx < area; idx++) {
      const v0 = buf.scratchA[idx];
      const v1 = buf.scratchA[(idx+game.N)%area];
      const v0v1 = (v0*v0 + v1*v1) | 0;
      if (v0v1) {
        const hash = v0v1 * Math.imul(843 + v0, 670 + v1);
        buf.idxs[buf.n++] = resolveKey(Math.imul(381 + idx, 463 + hash), ctx);
      }
    }
  }
  if (nextMove >= 0) {
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
  buf.sum = z;
  buf.val = 1 / (1 + Math.exp(-z));
}

// Returns the change in logit sum when nextMove is played from the position
// whose scratchA is already stored in buf. Temporarily mutates cells and
// scratchA, restoring both before returning.
// Reuses buf.deltaMark/deltasSA/deltasW/savedSA to avoid GC pressure.
function evaluateDelta(game, buf, ctx, nextMove) {
  const N     = game.N;
  const area  = N * N;
  const cells = game.cells;
  const sA    = buf.scratchA;
  const wts   = ctx.weightsArr;

  const captures = game.captureList(nextMove);
  const nCap     = captures.length;

  if (nCap === 0) {
    // ── Fast path: single stone placed, no captures ──────────────────────────
    // Affected scratchA slots: p, p1=(p-1).  Affected windows: p, p1, pN, pN1.
    // sA[(pN +N)%area]=sA[p] and sA[(pN1+N)%area]=sA[p1] by construction.
    const p   = nextMove;
    const p1  = p  > 0     ? p  - 1 : area - 1;
    const pN  = p  >= N    ? p  - N : p  - N + area;
    const pN1 = pN > 0     ? pN - 1 : area - 1;
    const pS  = p  + N < area ? p  + N : p  + N - area;
    const p1S = p1 + N < area ? p1 + N : p1 + N - area;

    // Old sum — cells[p]=EMPTY so no 1×1 feature to subtract.
    let oldSum = 0;
    if (USE_1x2) {
      if (sA[p])  { const wi = ctx.keyToIdx.get(Math.imul(129 + p,  972 + sA[p]));  if (wi >= 0) oldSum += wts[wi]; }
      if (sA[p1]) { const wi = ctx.keyToIdx.get(Math.imul(129 + p1, 972 + sA[p1])); if (wi >= 0) oldSum += wts[wi]; }
    }
    if (USE_2x2) {
      let sa0, sa1, v0v1, h, wi;
      sa0 = sA[p];   sa1 = sA[pS];  v0v1 = (sa0*sa0 + sa1*sa1)|0; if (v0v1) { h = v0v1*Math.imul(843+sa0, 670+sa1); wi = ctx.keyToIdx.get(Math.imul(381+p,   463+h)); if (wi >= 0) oldSum += wts[wi]; }
      sa0 = sA[p1];  sa1 = sA[p1S]; v0v1 = (sa0*sa0 + sa1*sa1)|0; if (v0v1) { h = v0v1*Math.imul(843+sa0, 670+sa1); wi = ctx.keyToIdx.get(Math.imul(381+p1,  463+h)); if (wi >= 0) oldSum += wts[wi]; }
      sa0 = sA[pN];  sa1 = sA[p];   v0v1 = (sa0*sa0 + sa1*sa1)|0; if (v0v1) { h = v0v1*Math.imul(843+sa0, 670+sa1); wi = ctx.keyToIdx.get(Math.imul(381+pN,  463+h)); if (wi >= 0) oldSum += wts[wi]; }
      sa0 = sA[pN1]; sa1 = sA[p1];  v0v1 = (sa0*sa0 + sa1*sa1)|0; if (v0v1) { h = v0v1*Math.imul(843+sa0, 670+sa1); wi = ctx.keyToIdx.get(Math.imul(381+pN1, 463+h)); if (wi >= 0) oldSum += wts[wi]; }
    }

    // Apply move, update scratchA, compute new sum, restore.
    cells[p] = game.current;
    const sv0 = sA[p], sv1 = sA[p1];
    { const v0 = game.current,  v1 = cells[p + 1 < area ? p + 1 : 0]; const vv = (v0*v0+v1*v1)|0; sA[p]  = vv * Math.imul(389+v0, 593+v1); }
    { const v0 = cells[p1],     v1 = game.current;       const vv = (v0*v0+v1*v1)|0; sA[p1] = vv * Math.imul(389+v0, 593+v1); }

    let newSum = wts[resolveKey(Math.imul(835 + p, 691 + game.current), ctx)];
    if (USE_1x2) {
      if (sA[p])  newSum += wts[resolveKey(Math.imul(129 + p,  972 + sA[p]),  ctx)];
      if (sA[p1]) newSum += wts[resolveKey(Math.imul(129 + p1, 972 + sA[p1]), ctx)];
    }
    if (USE_2x2) {
      let sa0, sa1, v0v1, h;
      sa0 = sA[p];   sa1 = sA[pS];  v0v1 = (sa0*sa0 + sa1*sa1)|0; if (v0v1) { h = v0v1*Math.imul(843+sa0, 670+sa1); newSum += wts[resolveKey(Math.imul(381+p,   463+h), ctx)]; }
      sa0 = sA[p1];  sa1 = sA[p1S]; v0v1 = (sa0*sa0 + sa1*sa1)|0; if (v0v1) { h = v0v1*Math.imul(843+sa0, 670+sa1); newSum += wts[resolveKey(Math.imul(381+p1,  463+h), ctx)]; }
      sa0 = sA[pN];  sa1 = sA[p];   v0v1 = (sa0*sa0 + sa1*sa1)|0; if (v0v1) { h = v0v1*Math.imul(843+sa0, 670+sa1); newSum += wts[resolveKey(Math.imul(381+pN,  463+h), ctx)]; }
      sa0 = sA[pN1]; sa1 = sA[p1];  v0v1 = (sa0*sa0 + sa1*sa1)|0; if (v0v1) { h = v0v1*Math.imul(843+sa0, 670+sa1); newSum += wts[resolveKey(Math.imul(381+pN1, 463+h), ctx)]; }
    }

    cells[p] = EMPTY;
    sA[p] = sv0; sA[p1] = sv1;
    return newSum - oldSum;
  }

  // ── Slow path: captures present ───────────────────────────────────────────
  const mark = buf.deltaMark;
  const sArr = buf.deltasSA;
  const wArr = buf.deltasW;
  const svd  = buf.savedSA;

  let nsA = 0;
  for (let ci = -1; ci < nCap; ci++) {
    const p = ci < 0 ? nextMove : captures[ci];
    for (let d = 0; d < 2; d++) {
      const j = d === 0 ? p : (p - 1 + area) % area;
      if (!mark[j]) { mark[j] = 1; sArr[nsA++] = j; }
    }
  }
  for (let i = 0; i < nsA; i++) mark[sArr[i]] = 0;

  let nW = 0;
  for (let ci = -1; ci < nCap; ci++) {
    const p = ci < 0 ? nextMove : captures[ci];
    const pN = (p - N + area) % area;
    for (let d = 0; d < 4; d++) {
      const j = d === 0 ? p : d === 1 ? (p - 1 + area) % area :
                d === 2 ? pN          : (pN - 1 + area) % area;
      if (!mark[j]) { mark[j] = 1; wArr[nW++] = j; }
    }
  }
  for (let i = 0; i < nW; i++) mark[wArr[i]] = 0;

  let oldSum = 0;
  for (let ci = -1; ci < nCap; ci++) {
    const p = ci < 0 ? nextMove : captures[ci];
    const v0 = cells[p];
    if (v0) { const wi = ctx.keyToIdx.get(Math.imul(835 + p, 691 + v0)); if (wi >= 0) oldSum += wts[wi]; }
  }
  if (USE_1x2) {
    for (let i = 0; i < nsA; i++) {
      const j = sArr[i];
      if (sA[j]) { const wi = ctx.keyToIdx.get(Math.imul(129 + j, 972 + sA[j])); if (wi >= 0) oldSum += wts[wi]; }
    }
  }
  if (USE_2x2) {
    for (let i = 0; i < nW; i++) {
      const j = wArr[i], sa0 = sA[j], sa1 = sA[(j + N) % area], v0v1 = (sa0*sa0 + sa1*sa1)|0;
      if (v0v1) { const h = v0v1*Math.imul(843+sa0, 670+sa1); const wi = ctx.keyToIdx.get(Math.imul(381+j, 463+h)); if (wi >= 0) oldSum += wts[wi]; }
    }
  }

  for (let ci = 0; ci < nCap; ci++) cells[captures[ci]] = EMPTY;
  cells[nextMove] = game.current;

  for (let i = 0; i < nsA; i++) {
    const j = sArr[i]; svd[i] = sA[j];
    const v0 = cells[j], v1 = cells[(j + 1) % area], vv = (v0*v0 + v1*v1)|0;
    sA[j] = vv * Math.imul(389 + v0, 593 + v1);
  }

  let newSum = 0;
  for (let ci = -1; ci < nCap; ci++) {
    const p = ci < 0 ? nextMove : captures[ci];
    const v0 = cells[p];
    if (v0) newSum += wts[resolveKey(Math.imul(835 + p, 691 + v0), ctx)];
  }
  if (USE_1x2) {
    for (let i = 0; i < nsA; i++) {
      const j = sArr[i];
      if (sA[j]) newSum += wts[resolveKey(Math.imul(129 + j, 972 + sA[j]), ctx)];
    }
  }
  if (USE_2x2) {
    for (let i = 0; i < nW; i++) {
      const j = wArr[i], sa0 = sA[j], sa1 = sA[(j + N) % area], v0v1 = (sa0*sa0 + sa1*sa1)|0;
      if (v0v1) { const h = v0v1*Math.imul(843+sa0, 670+sa1); newSum += wts[resolveKey(Math.imul(381+j, 463+h), ctx)]; }
    }
  }

  for (let ci = 0; ci < nCap; ci++) cells[captures[ci]] = -game.current;
  cells[nextMove] = EMPTY;
  for (let i = 0; i < nsA; i++) sA[sArr[i]] = svd[i];

  return newSum - oldSum;
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

// ── Incremental feature maintenance ──────────────────────────────────────────
//
// Maintains a "primary" buffer whose idxs/scratchA are kept in sync with the
// live board after each move, at O(k) cost per move instead of O(N²).
// Used exclusively in the training loop; search paths use findFeatures directly.
//
// posTypeOf encodes (j << 2 | type) so a swap-with-last removal can locate and
// update the right slot array for the displaced entry.
// type 0 → slotA (1×1), type 1 → slotB (1×2), type 2 → slotC (2×2).

function _removeSlot(buf, s, slotForJ, j) {
  const last   = --buf.n;
  slotForJ[j]  = -1;
  if (s === last) return;
  buf.idxs[s]      = buf.idxs[last];
  const pt         = buf.posTypeOf[last];
  buf.posTypeOf[s] = pt;
  const lastJ = pt >> 2, lastT = pt & 3;
  if      (lastT === 0) buf.slotA[lastJ] = s;
  else if (lastT === 1) buf.slotB[lastJ] = s;
  else                  buf.slotC[lastJ] = s;
}

// Full extraction with slot tracking. Call once per simulation to initialise
// the primary buffer; applyMoveIncremental maintains it thereafter.
function findFeaturesInit(game, buf, ctx) {
  const N    = game.N;
  const area = N * N;
  const cells = game.cells;
  buf.n = 0;
  buf.slotA.fill(-1, 0, area);
  buf.slotB.fill(-1, 0, area);
  buf.slotC.fill(-1, 0, area);

  for (let idx = 0; idx < area; idx++) {
    const v0 = cells[idx];
    if (v0) {
      const s = buf.n++;
      buf.idxs[s]      = resolveKey(Math.imul(835 + idx, 691 + v0), ctx);
      buf.slotA[idx]   = s;
      buf.posTypeOf[s] = (idx << 2) | 0;
    }
    const v1   = cells[(idx + 1) % area];
    const v0v1 = (v0*v0 + v1*v1) | 0;
    buf.scratchA[idx] = v0v1 * Math.imul(389 + v0, 593 + v1);
    if (USE_1x2 && v0v1) {
      const s = buf.n++;
      buf.idxs[s]      = resolveKey(Math.imul(129 + idx, 972 + buf.scratchA[idx]), ctx);
      buf.slotB[idx]   = s;
      buf.posTypeOf[s] = (idx << 2) | 1;
    }
  }
  if (USE_2x2) {
    for (let idx = 0; idx < area; idx++) {
      const sa0  = buf.scratchA[idx];
      const sa1  = buf.scratchA[(idx + N) % area];
      const v0v1 = (sa0*sa0 + sa1*sa1) | 0;
      if (v0v1) {
        const hash = v0v1 * Math.imul(843 + sa0, 670 + sa1);
        const s = buf.n++;
        buf.idxs[s]      = resolveKey(Math.imul(381 + idx, 463 + hash), ctx);
        buf.slotC[idx]   = s;
        buf.posTypeOf[s] = (idx << 2) | 2;
      }
    }
  }
}

// Update the primary buffer incrementally after nextMove is played.
// Must be called BEFORE game.play(nextMove); restores cells afterward.
// primary.scratchA is permanently updated to match the post-move board.
function applyMoveIncremental(game, buf, ctx, nextMove, captures) {
  const N    = game.N;
  const area = N * N;
  const cells = game.cells;
  const sA   = buf.scratchA;
  const mark = buf.deltaMark;
  const sArr = buf.deltasSA;
  const wArr = buf.deltasW;
  const nCap = captures.length;

  let nsA = 0, nW = 0;
  for (let ci = -1; ci < nCap; ci++) {
    const p = ci < 0 ? nextMove : captures[ci];
    for (let d = 0; d < 2; d++) {
      const j = d === 0 ? p : (p - 1 + area) % area;
      if (!mark[j]) { mark[j] = 1; sArr[nsA++] = j; }
    }
  }
  for (let i = 0; i < nsA; i++) mark[sArr[i]] = 0;

  for (let ci = -1; ci < nCap; ci++) {
    const p  = ci < 0 ? nextMove : captures[ci];
    const pN = (p - N + area) % area;
    for (let d = 0; d < 4; d++) {
      const j = d === 0 ? p : d === 1 ? (p - 1 + area) % area :
                d === 2 ? pN          : (pN - 1 + area) % area;
      if (!mark[j]) { mark[j] = 1; wArr[nW++] = j; }
    }
  }
  for (let i = 0; i < nW; i++) mark[wArr[i]] = 0;

  // Remove old features for affected positions.
  for (let ci = -1; ci < nCap; ci++) {
    const p = ci < 0 ? nextMove : captures[ci];
    if (buf.slotA[p] >= 0) _removeSlot(buf, buf.slotA[p], buf.slotA, p);
  }
  if (USE_1x2) {
    for (let i = 0; i < nsA; i++) {
      const j = sArr[i];
      if (buf.slotB[j] >= 0) _removeSlot(buf, buf.slotB[j], buf.slotB, j);
    }
  }
  if (USE_2x2) {
    for (let i = 0; i < nW; i++) {
      const j = wArr[i];
      if (buf.slotC[j] >= 0) _removeSlot(buf, buf.slotC[j], buf.slotC, j);
    }
  }

  // Apply changes temporarily so scratchA and new features can be computed.
  for (let ci = 0; ci < nCap; ci++) cells[captures[ci]] = EMPTY;
  cells[nextMove] = game.current;

  // Permanently update scratchA for affected positions.
  for (let i = 0; i < nsA; i++) {
    const j  = sArr[i];
    const v0 = cells[j];
    const v1 = cells[(j + 1) % area];
    const vv = (v0*v0 + v1*v1) | 0;
    sA[j]    = vv * Math.imul(389 + v0, 593 + v1);
  }

  // Add new features for affected positions.
  for (let ci = -1; ci < nCap; ci++) {
    const p  = ci < 0 ? nextMove : captures[ci];
    const v0 = cells[p];
    if (v0) {
      const s = buf.n++;
      buf.idxs[s]      = resolveKey(Math.imul(835 + p, 691 + v0), ctx);
      buf.slotA[p]     = s;
      buf.posTypeOf[s] = (p << 2) | 0;
    }
  }
  if (USE_1x2) {
    for (let i = 0; i < nsA; i++) {
      const j = sArr[i];
      if (sA[j]) {
        const s = buf.n++;
        buf.idxs[s]      = resolveKey(Math.imul(129 + j, 972 + sA[j]), ctx);
        buf.slotB[j]     = s;
        buf.posTypeOf[s] = (j << 2) | 1;
      }
    }
  }
  if (USE_2x2) {
    for (let i = 0; i < nW; i++) {
      const j    = wArr[i];
      const sa0  = sA[j];
      const sa1  = sA[(j + N) % area];
      const v0v1 = (sa0*sa0 + sa1*sa1) | 0;
      if (v0v1) {
        const hash = v0v1 * Math.imul(843 + sa0, 670 + sa1);
        const s = buf.n++;
        buf.idxs[s]      = resolveKey(Math.imul(381 + j, 463 + hash), ctx);
        buf.slotC[j]     = s;
        buf.posTypeOf[s] = (j << 2) | 2;
      }
    }
  }

  // Restore cells; game.play will apply these changes permanently.
  for (let ci = 0; ci < nCap; ci++) cells[captures[ci]] = -game.current;
  cells[nextMove] = EMPTY;
}

// Custom 1-ply search with incremental moves and feature extraction.
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

  // Base evaluation: full feature extraction once for the current position.
  findFeatures(game, searchFeats, ctx);
  evaluate(searchFeats, ctx.weightsArr);
  const baseSum = searchFeats.sum;

  const count = width > 0 ? Math.min(width, candidates.length) : candidates.length;
  for (let c = 0; c < count; c++) {
    const j = c + Math.floor(Math.random() * (candidates.length - c));
    const move = candidates[j];
    candidates[j] = candidates[c];
    // PASS leaves the board unchanged; all other moves use the incremental delta.
    const val = move === PASS
      ? searchFeats.val
      : 1 / (1 + Math.exp(-(baseSum + evaluateDelta(game, searchFeats, ctx, move))));
    if (isBlack === (val > bestScore)) {
      bestScore = val;
      bestIdx = move;
    }
  }
  return { move: bestIdx, moveScore: bestScore };
}

function selectTrainingMove(game, step, ctx, epsilon) {
  if (Math.random() < epsilon) 
    return { move: game.randomLegalMove() };

  if (step < WIDE_DEPTH) {
    return search1ply(game, ctx);
  }

  if (step < NARROW_DEPTH) 
    return search1ply(game, ctx, 2);

  if (USE_PLAYOUT) {
    return Playout.getMove(game);
  }
  return { move: game.randomLegalMove() };
}

// These two store the state of the model at the start of the game.
let keyToIdx_start = makeIntMap();
let weightsArr_start = [];

// These two store the current state of the model.
let keyToIdx = makeIntMap();
let weightsArr = [];

let lastMoveCount = 0;

// Runs TD self-play simulations to train the model. Returns { ctx, sims, maxSteps }.
function ponder(game, budgetMs, ctx) {
  const area     = game.N * game.N;
  const maxMoves = 3 * game.emptyCount + 20;
  const tStart   = Date.now();
  const weightsArr = ctx.weightsArr;
  let sims     = 0;
  let maxSteps = 0;

  let prev2   = makeBuf(area);
  let prev1   = makeBuf(area);
  let feats   = makeBuf(area);
  let primary = makeBuf(area);  // always tracks the current sim board state

  // Initialize to the stable values, maybe update later.
  let lr = LR1;
  let epsilon = EPSILON1;

  while (true) {
    let progress;
    if (SIMS > 0) {
      progress = sims / SIMS;
    } else {
      progress = (Date.now() - tStart) / budgetMs;
    }
    if (progress >= 1) break;
    if (game.moveCount > 1) {
      // We are not operating on the preserved game-start model, so ramp these.
      lr = LR1 - progress * (LR0 - LR1);
      epsilon = EPSILON1 - progress * (EPSILON0 - EPSILON1);
    }

    sims++;
    const g = game.clone();
    const evalDepth = (EVAL_DEPTH > 0 && evalModel) ? EVAL_DEPTH + (sims % 2) : 10000;
    let step = 0;
    let outcome;

    // Initialise primary for the real board state; maintained incrementally.
    findFeaturesInit(g, primary, ctx);
    prev1.n = prev2.n = 0;

    let moveSelection = selectTrainingMove(g, step, ctx, epsilon);
    while (true) {
      const nextMove = moveSelection.move;

      // Update primary incrementally before playing (cells still in pre-move state).
      if (nextMove !== PASS) {
        const captures = g.captureList(nextMove);
        applyMoveIncremental(g, primary, ctx, nextMove, captures);
      }

      g.play(nextMove);
      step++;
      if (g.gameOver || step > maxMoves) {
        outcome = (g.estimateWinner() === BLACK) ? 1 : 0;
        break;
      }
      if (step >= evalDepth) {
        outcome = vpEvaluate(g, evalModel);
        break;
      }

      moveSelection = selectTrainingMove(g, step, ctx, epsilon);

      // Copy primary.idxs into feats (O(n) native copy) then evaluate.
      feats.n = primary.n;
      feats.idxs.set(primary.idxs.subarray(0, primary.n));
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

  return { sims, maxSteps };
}

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
    keyToIdx = makeIntMap();
    weightsArr = [];
  }
  lastMoveCount = game.moveCount;

  const area = game.N * game.N;
  const ctx  = { keyToIdx, weightsArr, searchFeats: makeBuf(area) };

  ///////////////////////////////////////////////////////
  const { sims, maxSteps } = ponder(game, budgetMs, ctx);
  ///////////////////////////////////////////////////////

  if (game.moveCount === 1) {
    keyToIdx_start = keyToIdx.clone();
    weightsArr_start = weightsArr.slice();
  }

  const evalFn = g => { findFeatures(g, ctx.searchFeats, ctx); evaluate(ctx.searchFeats, ctx.weightsArr); return ctx.searchFeats.val; };
  const searchResult = { move: abSearch(game, AB_DEPTH, evalFn), moveScore: evalFn(game) };

  const myScore = game.current === BLACK ? searchResult.moveScore : 1 - searchResult.moveScore;
  const move = myScore < 0.01 ? PASS : searchResult.move;
  const result = { move, ctx, sims, maxSteps };
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
    { makeBuf, resolveKey, findFeatures, findFeaturesInit, applyMoveIncremental,
      evaluate, evaluateDelta, search1ply, tdUpdate, getMove },
    require('../game2.js')
  );
} else {
  window.getMove = getMove;
}

})();

