'use strict';

// npat-lib.js — pattern-based policy features with stone-indexed tactical bits.
//
// For each candidate move we extract up to three shape windows (centered 3×3,
// Type A, Type B — each optional via cfg flags) plus four tactical feature
// types expanded into per-stone-index sub-features.  The move's logit is the
// sum of active-shape weights and the tactical contributions; moves are
// sampled by softmax over logits.
//
// Cell encoding (mover-relative, 3 values — pure shape, no tactical bits):
//   0 = EMPTY     empty point
//   1 = FRIEND    own stone
//   2 = FOE       opponent stone
//
// The board is toroidal (game2's neighbour tables wrap), so there is no
// off-board state to encode.
//
// Tactical features (per candidate move, stacking booleans — one firing per
// qualifying chain the move applies to; in a linear model, total contribution
// is count × weight):
//   URGENT_KILL    move is a liberty of a foe chain with libs ≤ 2 that the
//                  mover (as attacker) can capture.
//   URGENT_SAVE    move is a liberty of a friend chain with libs ≤ 2 that the
//                  mover (as defender) can save.
//   WASTED_EXTEND  move is a liberty of a friend chain with libs ≤ 2 that
//                  cannot be saved (moverSucceeds === false, defending).
//   WASTED_ATTACK  move is a liberty of a foe chain with libs ≤ 2 that
//                  cannot be killed (moverSucceeds === false, attacking).
// All four can coexist and stack on the same move.
//
// Raw pattern index for a single shape window:
//   raw = relPos * 3^9 + Σ_i cells[i] * 3^i                       (i = 0..8)
// where relPos ∈ 0..8 is the candidate's position within the window.
// Range: [0, 9 * 19683) = [0, 177147).  Fits in 32 bits.
//
// Tactical raw ids live just above the shape range:
//   TACT_RAW_BASE = 9 * 3^9 = 177147
//   tactical[k]   = TACT_RAW_BASE + k   for k ∈ 0..3
//
// Canonical key: min over 8 D4 symmetries of the transformed raw.  Under σ,
// new_cells[σ(i)] = cells[i] and new_relPos = σ(relPos).  Tactical features
// have no spatial position and are not canonicalised.  Sparse weights live in
// a dense Float64Array indexed by the intern map's dense slot.
//
// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
const { PASS }          = _isNode ? require('./game2.js')  : window.Game2;
const { game3FromGame2 } = _isNode ? require('./game3.js') : window.Game3;
const { getAllLadderStatuses } = _isNode ? require('./ladder2.js') : window.Ladder2;

// ── D4 position permutations on a 3×3 grid ────────────────────────────────────
// Positions linearised as i = row*3 + col, row,col ∈ {0,1,2}.
// perm[i] is the new position of the value originally at position i.

function _mkPerm(fn) {
  const p = new Int32Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
    const [nr, nc] = fn(r, c);
    p[r * 3 + c] = nr * 3 + nc;
  }
  return p;
}

const _D4 = [
  _mkPerm((r, c) => [r, c]),         // 0: Identity
  _mkPerm((r, c) => [c, 2 - r]),     // 1: Rot90CW
  _mkPerm((r, c) => [2 - r, 2 - c]), // 2: Rot180
  _mkPerm((r, c) => [2 - c, r]),     // 3: Rot270CW
  _mkPerm((r, c) => [r, 2 - c]),     // 4: FlipH
  _mkPerm((r, c) => [2 - r, c]),     // 5: FlipV
  _mkPerm((r, c) => [c, r]),         // 6: TransposeMD
  _mkPerm((r, c) => [2 - c, 2 - r]), // 7: TransposeAD
];

// ── Encoding constants ───────────────────────────────────────────────────────

const CELL_BASE    = 3;
const CELLS_BASE   = 19683; // 3^9

// Cell-state labels (exported for readability in other modules / tests).
const CELL_EMPTY   = 0;
const CELL_FRIEND  = 1;
const CELL_FOE     = 2;

// Tactical feature indices (also their offset above TACT_RAW_BASE).
// Each type is expanded into TACT_STONE_LIMIT numbered sub-features indexed
// by stone-index within the chain: a qualifying chain of size s emits the
// sub-features 0..min(s, LIMIT)-1 at every relevant cell.  In a linear
// softmax policy this lets the model learn a size-dependent preference
// (longer chains activate more weights, so the logit scales with chain size
// in a per-type weighted fashion).
const N_TACT             = 4;
// TACT_STONE_LIMIT controls per-stone-index expansion; env-configurable so
// A/B tests can sweep without recompiling (default 8).
const TACT_STONE_LIMIT   = (typeof process !== 'undefined' && process.env && process.env.NPAT_STONE_LIMIT)
  ? parseInt(process.env.NPAT_STONE_LIMIT, 10) : 8;
const N_TACT_SLOTS       = N_TACT * TACT_STONE_LIMIT;  // 32 at default
const TACT_URGENT_KILL   = 0;
const TACT_URGENT_SAVE   = 1;
const TACT_WASTED_EXTEND = 2;
const TACT_WASTED_ATTACK = 3;
const TACT_RAW_BASE      = 9 * CELLS_BASE; // 177147; above the shape-key range

// Centered 3×3 shape window (9 cells, candidate always at relPos=4).  Uses
// canonKey(4, cells) for canonicalisation — raw id lives just above the
// tactical block.
const SHAPE33C_RAW_BASE  = TACT_RAW_BASE + N_TACT_SLOTS;

// Type E shape window (12 cells, 3-wide × 4-tall block, hflip-symmetric):
//
//     x    x    x    (-1, -1) (-1, 0) (-1, +1)
//     x    *    x    (0, -1)  (0, 0)  (0, +1)
//     x    x    x    (+1, -1) (+1, 0) (+1, +1)
//     x    x    x    (+2, -1) (+2, 0) (+2, +1)
//
// 3^12 = 531k raw slots; flat Int32Array cache.
//
// TYPE_E_RAW_BASE preserves the offset from when types A, B, D, T existed,
// so existing E-trained checkpoints continue to load.  The legacy chain was
//   SHAPE33C + 9*CELLS_BASE (was A) + 59049 (A_BASE) + 177147 (B_BASE)
//                                   + 531441 (D_BASE) + 81 (T_BASE).
const TYPE_E_CELLS       = 12;
const TYPE_E_BASE        = 531441; // 3^12
const TYPE_E_RAW_BASE    = SHAPE33C_RAW_BASE + 9 * CELLS_BASE
                         + 59049 + 177147 + 531441 + 81;

// ── Ladder-status annotation ─────────────────────────────────────────────────
//
// Build a per-cell tactical-feature count array for the current game position.
// tactCount is a Uint8Array of length N*N*N_TACT, laid out as
// tactCount[cellIdx * N_TACT + k] = how many qualifying chains make feature k
// apply at cell cellIdx.  Counts stay tiny (≤ number of distinct 1-2-lib
// chains sharing that liberty; in practice 1-2).

function annotateLadders(game, out, game3) {
  const N   = game.N;
  const cap = N * N;

  if (!out) out = { tactCount: new Uint8Array(cap * N_TACT_SLOTS) };
  out.tactCount.fill(0);

  if (game.emptyCount === cap) return out;

  const g3 = game3 || game3FromGame2(game);
  const infos = getAllLadderStatuses(g3);
  const tc = out.tactCount;
  const cur = game.current;

  for (const info of infos) {
    if (!info.status) continue;
    const { libs, moverSucceeds, urgentLibs } = info.status;
    const defending = (info.color === cur);
    const size = g3.groupSize(info.gid);
    const emit = size < TACT_STONE_LIMIT ? size : TACT_STONE_LIMIT;
    let fi, targets;
    if (urgentLibs.length > 0) {
      fi = defending ? TACT_URGENT_SAVE : TACT_URGENT_KILL;
      targets = urgentLibs;
    } else if (!moverSucceeds) {
      fi = defending ? TACT_WASTED_EXTEND : TACT_WASTED_ATTACK;
      targets = libs;
    } else {
      continue;
    }
    const base = fi * TACT_STONE_LIMIT;
    for (const lib of targets) {
      const off = lib * N_TACT_SLOTS + base;
      for (let j = 0; j < emit; j++) tc[off + j]++;
    }
  }

  return out;
}

// ── State ─────────────────────────────────────────────────────────────────────
//
// moves     [count]                       flat board index of move i
// patIds33c [count]                        canonical 3×3c key per move
// tact      [count * N_TACT_SLOTS]         tactical feature counts per move (uint8)
// logits    [count]                        scratch buffer for softmax
// probs     [count]                        scratch buffer for softmax
// ladder    { tactCount }                  reusable ladder-annotation buffer

function createState(N) {
  const cap = N * N;
  // Precomputed toroidal 5×5 neighbour table (rows pr ∈ [-2, 2], cols
  // pc ∈ [-2, 2]).  For each board index idx and each offset, patchNbr[idx*25
  // + (pr+2)*5 + (pc+2)] gives the flat board index of (r+pr mod N, c+pc mod N).
  // This replaces 25 `_wrap` calls per candidate in extractFeatures with one
  // indexed load.  All active shapes (3×3c, A, B) and their D4 orbits fit
  // within rows [-2, +2] × cols [-2, +2].
  const patchNbr = new Int32Array(cap * 25);
  for (let idx = 0; idx < cap; idx++) {
    const r = (idx / N) | 0;
    const c = idx - r * N;
    const base = idx * 25;
    for (let pr = -2; pr <= 2; pr++) {
      const br = _wrap(r + pr, N);
      const rowBase = br * N;
      const prBase  = (pr + 2) * 5;
      for (let pc = -2; pc <= 2; pc++) {
        patchNbr[base + prBase + (pc + 2)] = rowBase + _wrap(c + pc, N);
      }
    }
  }


  return {
    N,
    moves:     new Int32Array(cap),
    patIds33c: new Int32Array(cap),
    patIdsE:   new Int32Array(cap),
    tact:      new Uint8Array(cap * N_TACT_SLOTS),
    logits:    new Float64Array(cap),
    probs:     new Float64Array(cap),
    ladder:    { tactCount: new Uint8Array(cap * N_TACT_SLOTS) },
    patchNbr,
    // Reusable touched-index scratch for reinforceUpdate.  Two shape types
    // (3×3c + E), one dense idx per (chosen + each move).
    touched:   new Int32Array(2 * (cap + 1)),
    count:     0,
  };
}

// ── Weights store ─────────────────────────────────────────────────────────────
//
// Canonical pattern keys (the raw ints from canonKey) can range up to ~90M,
// so we can't index a flat array by them directly.  Instead we intern each
// raw key to a dense 0-based index the first time it is seen.  Weights and
// the reusable reinforce-delta buffer are plain Float64Arrays indexed by the
// dense index; the raw → dense map is a Map<number, number>.
//
// `extractFeatures` performs the intern during extraction, so `state.patIds`
// then holds dense indices, and scoring / softmax / REINFORCE update paths
// are all pure typed-array access.  No Map operations in the inner loops.
//
// The 4 tactical features are interned up front (dense ids 0..3) at
// createWeights time, so weights.tactIds is available before any extraction.

function createWeights(opts) {
  // opts may be a number (initialCapacity, legacy) or an options object
  //   { initialCapacity, use33, use34, useL }.
  // Feature-gating flags default to false — tactical features are always on.
  let initialCapacity = 1024;
  let useTactical = true;
  let use33c = false, useE = false;
  if (typeof opts === 'number') {
    initialCapacity = opts;
  } else if (opts && typeof opts === 'object') {
    if (opts.initialCapacity) initialCapacity = opts.initialCapacity;
    if (opts.useTactical === false) useTactical = false;
    if (opts.use33c) use33c = true;
    if (opts.useE)   useE   = true;
  }
  const w = {
    map:   new Map(),                              // raw canonKey → dense idx
    vals:  new Float64Array(initialCapacity),      // weights[dense idx]
    delta: new Float64Array(initialCapacity),      // reusable reinforce buffer
    size:  0,                                      // next dense idx to assign
    tactIds: new Int32Array(N_TACT_SLOTS),
    cfg:   { useTactical, use33c, useE },
  };
  if (useTactical) {
    for (let k = 0; k < N_TACT_SLOTS; k++) {
      w.tactIds[k] = _internWeight(w, TACT_RAW_BASE + k);
    }
  }
  return w;
}

function _internWeight(w, rawPid) {
  const map = w.map;
  const existing = map.get(rawPid);
  if (existing !== undefined) return existing;
  const idx = w.size;
  if (idx >= w.vals.length) {
    const cap = w.vals.length * 2;
    const nv = new Float64Array(cap); nv.set(w.vals); w.vals = nv;
    const nd = new Float64Array(cap); nd.set(w.delta); w.delta = nd;
  }
  map.set(rawPid, idx);
  w.size = idx + 1;
  return idx;
}

// ── Runtime D4 canonicalisation ──────────────────────────────────────────────
//
// canonKey(relPos, cells) returns the minimum raw-int over 8 D4 symmetries.
//
// For each D4 permutation σ we have tCells[σ(i)] = cells[i], so
//   enc_σ = Σ_j tCells[j] · 3^j = Σ_i cells[i] · 3^{σ(i)}
// and the full raw is σ(relPos) · CELLS_BASE + enc_σ.  Precompute:
//   _D4cw[σ*9 + i]      = 3^{σ(i)}               (cell weight)
//   _D4rp[σ*9 + relPos] = σ(relPos) · CELLS_BASE (relPos offset)
// so each permutation becomes a flat 9-term dot product plus one add.

const _D4cw = new Int32Array(72);
const _D4rp = new Int32Array(72);
(function () {
  const pow = new Int32Array(10);
  pow[0] = 1;
  for (let k = 1; k < 10; k++) pow[k] = pow[k - 1] * CELL_BASE;
  for (let di = 0; di < 8; di++) {
    const perm = _D4[di];
    for (let i = 0; i < 9; i++) {
      _D4cw[di * 9 + i] = pow[perm[i]];
      _D4rp[di * 9 + i] = perm[i] * CELLS_BASE;
    }
  }
})();

// Lazy flat caches for canonKey* functions — filled on first lookup.  Each
// is an Int32Array sized to the shape's raw-input space with -1 as the
// "not yet computed" sentinel.  Cache keys are always the s=0 (identity)
// raw encoding, so a single lookup replaces the 8-way D4 minimum loop.
//
// Shared cache for canonKey (covers both the 9-window 3×3 family and the
// centered 3×3c, which both call into canonKey(relPos, cells[9])).

const _canonKeyCache = new Int32Array(9 * CELLS_BASE).fill(-1); // 708KB

function canonKey(relPos, cells) {
  const c0 = cells[0], c1 = cells[1], c2 = cells[2];
  const c3 = cells[3], c4 = cells[4], c5 = cells[5];
  const c6 = cells[6], c7 = cells[7], c8 = cells[8];
  // Lazy flat cache: key = s=0 raw = relPos·3^9 + Σ cells[i]·3^i,
  // range [0, 9·19683) = [0, 177147).  The cache is shared across the
  // 9-window 3×3 extraction and the centered-3×3c extraction.
  const cacheKey = relPos * CELLS_BASE
    + c0 + c1*3 + c2*9 + c3*27 + c4*81 + c5*243 + c6*729 + c7*2187 + c8*6561;
  const cached = _canonKeyCache[cacheKey];
  if (cached !== -1) return cached;
  const cw = _D4cw, rp = _D4rp;
  let best = 0x7fffffff;
  for (let di = 0; di < 8; di++) {
    const o = di * 9;
    const raw = rp[o + relPos]
      + c0 * cw[o]     + c1 * cw[o + 1] + c2 * cw[o + 2]
      + c3 * cw[o + 3] + c4 * cw[o + 4] + c5 * cw[o + 5]
      + c6 * cw[o + 6] + c7 * cw[o + 7] + c8 * cw[o + 8];
    if (raw < best) best = raw;
  }
  _canonKeyCache[cacheKey] = best;
  return best;
}

const _windowCells  = new Int32Array(9);
const _patch5       = new Int32Array(25); // 5×5 (rows [-2,+2] × cols [-2,+2])

// Type E shape (12 cells, full 3×4 block including the candidate cell,
// hflip-symmetric).  Row-major layout, candidate at index 4.
const _EShape = [
  [-1, -1], [-1,  0], [-1, +1],
  [ 0, -1], [ 0,  0], [ 0, +1],
  [+1, -1], [+1,  0], [+1, +1],
  [+2, -1], [+2,  0], [+2, +1],
];

const _EPatchIdx    = new Int32Array(8 * TYPE_E_CELLS);
const _ECellWeights = new Int32Array(TYPE_E_CELLS);
(function () {
  const d4 = [
    (r, c) => [ r,  c], (r, c) => [ c, -r], (r, c) => [-r, -c], (r, c) => [-c,  r],
    (r, c) => [ r, -c], (r, c) => [-r,  c], (r, c) => [ c,  r], (r, c) => [-c, -r],
  ];
  for (let s = 0; s < 8; s++) {
    for (let i = 0; i < TYPE_E_CELLS; i++) {
      const [r, c] = d4[s](_EShape[i][0], _EShape[i][1]);
      _EPatchIdx[s * TYPE_E_CELLS + i] = (r + 2) * 5 + (c + 2);
    }
  }
  let w = 1;
  for (let i = 0; i < TYPE_E_CELLS; i++) { _ECellWeights[i] = w; w *= CELL_BASE; }
})();

const _canonKeyECache = new Int32Array(TYPE_E_BASE).fill(-1); // 2.1MB

function canonKeyE(patch) {
  const idx = _EPatchIdx;
  const cw  = _ECellWeights;
  let cacheKey = 0;
  for (let i = 0; i < TYPE_E_CELLS; i++) cacheKey += patch[idx[i]] * cw[i];
  const cached = _canonKeyECache[cacheKey];
  if (cached !== -1) return cached;
  let best = cacheKey;
  for (let s = 1; s < 8; s++) {
    const o = s * TYPE_E_CELLS;
    let raw = 0;
    for (let i = 0; i < TYPE_E_CELLS; i++) raw += patch[idx[o + i]] * cw[i];
    if (raw < best) best = raw;
  }
  _canonKeyECache[cacheKey] = best;
  return best;
}

// ── Core feature extraction ───────────────────────────────────────────────────
//
// Toroidal wrap — game2's _nbr only covers ±1 offsets, but we need ±2.

function _wrap(x, N) { x %= N; return x < 0 ? x + N : x; }

// Fill state with 9 canonical pattern keys and 4 tactical counts for every
// legal non-true-eye move.  Optional ladderInfo skips re-running the ladder
// analysis (useful if the caller already computed it for this board).
// Optional game3 is a Game3 kept in lockstep with `game`; passing it avoids a
// game3FromGame2 rebuild.  Optional weights is a WeightsStore; when supplied,
// each canonical key is interned and patIds[] holds dense indices (used by
// the scoring/softmax/REINFORCE hot paths).  When omitted, patIds[] holds
// raw canonKey ints.
function extractFeatures(game, state, ladderInfo, game3, weights) {
  const N      = game.N;
  const cells  = game.cells;
  const cur    = game.current;
  const emC    = game._emptyCells;
  const ec     = game.emptyCount;

  let count = 0;
  const moves    = state.moves;
  const tact     = state.tact;
  const patchNbr = state.patchNbr;
  const wc9      = _windowCells;
  const patch    = _patch5;
  const wMap     = weights ? weights.map : null;
  // Feature-extraction gates: when a weights store with cfg is supplied, skip
  // shape types that are disabled.  Without weights (tests extracting raw keys
  // only), extract everything for backward compatibility.
  const cfg         = weights ? weights.cfg : null;
  const doUseTact   = !cfg || cfg.useTactical !== false;
  // Skip ladder annotation entirely when tactical features are off — that's
  // the dominant per-position cost when no shape extracts depend on it.
  const li = doUseTact ? (ladderInfo || annotateLadders(game, state.ladder, game3)) : null;
  const tactCount = li ? li.tactCount : null;
  const doUse33c = !cfg || cfg.use33c;
  const doUseE   = !cfg || cfg.useE;
  const patIds33c = state.patIds33c;
  const patIdsE   = state.patIdsE;

  for (let ei = 0; ei < ec; ei++) {
    const idx = emC[ei];
    if (!game.isLegal(idx) || game.isTrueEye(idx)) continue;

    // Build the 5×5 patch of encoded cell values centred on the candidate.
    // patchNbr pre-stores the 25 toroidal flat indices for each idx; 25
    // indexed loads replace 25 _wrap calls.  Layout: patch[(pr+2)*5 + (pc+2)].
    const nbrBase = idx * 25;
    for (let p = 0; p < 25; p++) {
      const bi = patchNbr[nbrBase + p];
      const ci = cells[bi];
      let v;
      if (ci === 0)        v = CELL_EMPTY;
      else if (ci === cur) v = CELL_FRIEND;
      else                 v = CELL_FOE;
      patch[p] = v;
    }

    // Centered 3×3 window — candidate at relPos 4, reads the central 3×3 of
    // the patch.  Canonicalised via canonKey with its SHAPE33C raw-base
    // offset so keys sit in a disjoint range.
    if (doUse33c) {
      for (let i = 0; i < 9; i++) {
        const dr = (i / 3) | 0;
        const dc = i - dr * 3;
        wc9[i] = patch[(dr - 1 + 2) * 5 + (dc - 1 + 2)];
      }
      const raw33c = SHAPE33C_RAW_BASE + canonKey(4, wc9);
      patIds33c[count] = wMap ? _internWeight(weights, raw33c) : raw33c;
    }

    // Type E shape — 12-cell 3×4 block including the candidate cell.
    if (doUseE) {
      const rawE = TYPE_E_RAW_BASE + canonKeyE(patch);
      patIdsE[count] = wMap ? _internWeight(weights, rawE) : rawE;
    }

    // Copy tactical counts for this candidate (N_TACT_SLOTS bytes = 4 types
    // × TACT_STONE_LIMIT stone-indices).  Skipped when useTactical is off.
    if (doUseTact) {
      const tSrc = idx * N_TACT_SLOTS;
      const tDst = count * N_TACT_SLOTS;
      for (let k = 0; k < N_TACT_SLOTS; k++) tact[tDst + k] = tactCount[tSrc + k];
    }

    moves[count] = idx;
    count++;
  }

  state.count = count;
}

// ── Scoring helpers ───────────────────────────────────────────────────────────
//
// After extractFeatures(game, state, _, _, weights), patIds holds dense indices
// into weights.vals (the raw canonKey has been interned) and tact holds per-
// move tactical counts.  Scoring is a pure typed-array sum.
function _score(state, i, weights) {
  const vals = weights.vals;
  const tact = state.tact;
  const tIds = weights.tactIds;
  const cfg  = weights.cfg;
  let s = 0;
  if (cfg.useTactical !== false) {
    const tOff = i * N_TACT_SLOTS;
    for (let k = 0; k < N_TACT_SLOTS; k++) s += tact[tOff + k] * vals[tIds[k]];
  }
  if (cfg.use33c) s += vals[state.patIds33c[i]];
  if (cfg.useE)   s += vals[state.patIdsE[i]];
  return s;
}

// Evaluate all moves and return them sorted by score descending.
function evaluate(game, state, weights) {
  extractFeatures(game, state, undefined, undefined, weights);
  const out = [];
  for (let i = 0; i < state.count; i++) {
    out.push({ move: state.moves[i], score: _score(state, i, weights) });
  }
  return out.sort((a, b) => b.score - a.score);
}

function _computeSoftmax(state, weights) {
  const n   = state.count;
  if (n === 0) return 0;
  const tact  = state.tact;
  const lg    = state.logits;
  const pr    = state.probs;
  const vals  = weights.vals;
  const tIds  = weights.tactIds;
  const cfg   = weights.cfg;
  const useTact = cfg.useTactical !== false;
  const tW    = new Float64Array(N_TACT_SLOTS);
  if (useTact) {
    for (let k = 0; k < N_TACT_SLOTS; k++) tW[k] = vals[tIds[k]];
  }
  const pid33c = cfg.use33c ? state.patIds33c : null;
  const pidE   = cfg.useE   ? state.patIdsE   : null;

  let maxL = -Infinity;
  for (let i = 0; i < n; i++) {
    let s = 0;
    if (useTact) {
      const tOff = i * N_TACT_SLOTS;
      for (let k = 0; k < N_TACT_SLOTS; k++) s += tact[tOff + k] * tW[k];
    }
    if (pid33c) s += vals[pid33c[i]];
    if (pidE)   s += vals[pidE[i]];
    lg[i] = s;
    if (s > maxL) maxL = s;
  }
  let sum = 0;
  for (let i = 0; i < n; i++) { pr[i] = Math.exp(lg[i] - maxL); sum += pr[i]; }
  const inv = 1 / sum;
  for (let i = 0; i < n; i++) pr[i] *= inv;
  return n;
}

// Extract features, softmax over logits, sample an action.
// Returns { move, index, prob }.  Optional game3 kept in lockstep with game.
function policyMove(game, state, weights, rng, game3) {
  extractFeatures(game, state, undefined, game3, weights);
  const n = state.count;
  if (n === 0) return { move: PASS, index: -1, prob: 1 };

  _computeSoftmax(state, weights);
  const probs = state.probs;

  const R = (rng || Math).random();
  let r = R, chosen = n - 1;
  for (let i = 0; i < n; i++) {
    r -= probs[i];
    if (r <= 0) { chosen = i; break; }
  }
  return { move: state.moves[chosen], index: chosen, prob: probs[chosen] };
}

// Greedy argmax move (no sampling).  Optional game3 kept in lockstep.
function greedyMove(game, state, weights, game3) {
  extractFeatures(game, state, undefined, game3, weights);
  const n = state.count;
  if (n === 0) return PASS;
  let best = -Infinity, bestI = 0;
  for (let i = 0; i < n; i++) {
    const s = _score(state, i, weights);
    if (s > best) { best = s; bestI = i; }
  }
  return state.moves[bestI];
}

// ── REINFORCE step ────────────────────────────────────────────────────────────
//
// For each weight w, ∂ log π_m / ∂ w = feat_m(w) − Σ_i π_i · feat_i(w), where
// feat_i(w) is the count of w's feature in move i's feature bag (the 9 shape
// pids each count 1 per window; the 4 tactical counts come straight from the
// tact[] buffer).  Update: Δw = lr · adv · ∂ log π_m / ∂ w.
//
// state.probs must be populated (by policyMove / _computeSoftmax) before the
// call.

const _tactScratch = new Float64Array(N_TACT_SLOTS);

function reinforceUpdate(state, chosenIndex, advantage, weights, lr) {
  const n = state.count;
  if (n === 0 || chosenIndex < 0) return;

  const step = lr * advantage;
  if (step === 0) return;

  const probs    = state.probs;
  const tact     = state.tact;
  const vals     = weights.vals;
  const delta    = weights.delta;
  const tIds     = weights.tactIds;
  const touched  = state.touched;
  const cfg      = weights.cfg;
  const patIds33c = cfg.use33c ? state.patIds33c : null;
  const patIdsE   = cfg.useE   ? state.patIdsE   : null;
  let tc = 0;

  // ── Shape features (use delta buffer to dedupe repeats across moves) ──
  // Chosen move contributes +step to each of its active shape pids.
  if (patIds33c) {
    const idx = patIds33c[chosenIndex];
    touched[tc++] = idx;
    delta[idx] += step;
  }
  if (patIdsE) {
    const idx = patIdsE[chosenIndex];
    touched[tc++] = idx;
    delta[idx] += step;
  }
  // Every legal move i contributes -step * π_i to each of its active shape pids.
  for (let i = 0; i < n; i++) {
    const pi = probs[i];
    if (pi === 0) continue;
    const sub = step * pi;
    if (patIds33c) {
      const idx = patIds33c[i];
      touched[tc++] = idx;
      delta[idx] -= sub;
    }
    if (patIdsE) {
      const idx = patIdsE[i];
      touched[tc++] = idx;
      delta[idx] -= sub;
    }
  }
  // Apply accumulated deltas and clear them.  Duplicates are tolerated by the
  // zero-check: the first visit applies the net delta; later visits see 0.
  for (let i = 0; i < tc; i++) {
    const idx = touched[i];
    const d = delta[idx];
    if (d !== 0) {
      vals[idx] += d;
      delta[idx] = 0;
    }
  }

  // ── Tactical features (N_TACT_SLOTS, always touched; no delta buffer) ──
  if (cfg.useTactical !== false) {
    const cOff = chosenIndex * N_TACT_SLOTS;
    const neg  = _tactScratch;  // Float64 scratch buffer sized N_TACT_SLOTS
    for (let k = 0; k < N_TACT_SLOTS; k++) neg[k] = 0;
    for (let i = 0; i < n; i++) {
      const pi = probs[i];
      if (pi === 0) continue;
      const off = i * N_TACT_SLOTS;
      for (let k = 0; k < N_TACT_SLOTS; k++) neg[k] += pi * tact[off + k];
    }
    for (let k = 0; k < N_TACT_SLOTS; k++) {
      const d = step * (tact[cOff + k] - neg[k]);
      if (d !== 0) vals[tIds[k]] += d;
    }
  }
  // Return total shape-feature touches (including duplicates across moves) —
  // train-npat.js accumulates these to report mean updates-per-pattern.
  return tc + (cfg.useTactical !== false ? N_TACT_SLOTS : 0);
}

// ── Entropy-bonus step ────────────────────────────────────────────────────────
//
// Keep the softmax distribution from collapsing by pushing weights in the
// direction of higher per-state entropy H = −Σ π_i log π_i.  The gradient
//   ∂H/∂w = Σ_i π_i · (−log π_i − H) · (feat_i − <feat>)
// has the same sum-over-moves structure as REINFORCE, so per move i the
// contribution is c_i · feat_i with
//   c_i = β · π_i · (−log π_i − H)
// and a balancing −Σ_i π_i·c_i term that we fold in via the same <feat>
// baseline trick: writing c_i as is, Σ_i c_i · <feat> = 0 because Σ c_i = 0
// (by construction of H).  So we can just apply Δw[k] += c_i · feat_i_k for
// every move i and the baseline cancels itself across moves.
//
// state.probs must already be populated (by policyMove / _computeSoftmax).

let _entScratch = new Float64Array(0);

function entropyBonusUpdate(state, weights, beta) {
  const n = state.count;
  if (n === 0 || beta === 0) return;

  const probs    = state.probs;
  const tact     = state.tact;
  const vals     = weights.vals;
  const delta    = weights.delta;
  const tIds     = weights.tactIds;
  const touched  = state.touched;
  const cfg      = weights.cfg;
  const patIds33c = cfg.use33c ? state.patIds33c : null;
  const patIdsE   = cfg.useE   ? state.patIdsE   : null;
  if (_entScratch.length < n) _entScratch = new Float64Array(n);
  const logs    = _entScratch;  // scratch for −log(π_i); not on state (snapshots omit it)
  let tc = 0;

  // Compute H = −Σ π_i log π_i and cache −log(π_i).
  let H = 0;
  for (let i = 0; i < n; i++) {
    const pi = probs[i];
    const lp = pi > 0 ? -Math.log(pi) : 0;
    logs[i] = lp;
    H += pi * lp;
  }

  // Accumulate per-move coefficients onto shape pids, and tactical totals.
  const acc = _tactScratch;
  for (let k = 0; k < N_TACT_SLOTS; k++) acc[k] = 0;
  for (let i = 0; i < n; i++) {
    const pi = probs[i];
    if (pi === 0) continue;
    const ci = beta * pi * (logs[i] - H);
    if (ci === 0) continue;
    if (patIds33c) {
      const idx = patIds33c[i];
      touched[tc++] = idx;
      delta[idx] += ci;
    }
    if (patIdsE) {
      const idx = patIdsE[i];
      touched[tc++] = idx;
      delta[idx] += ci;
    }
    if (cfg.useTactical !== false) {
      const tOff = i * N_TACT_SLOTS;
      for (let k = 0; k < N_TACT_SLOTS; k++) acc[k] += ci * tact[tOff + k];
    }
  }
  for (let i = 0; i < tc; i++) {
    const idx = touched[i];
    const d = delta[idx];
    if (d !== 0) { vals[idx] += d; delta[idx] = 0; }
  }
  if (cfg.useTactical !== false) {
    for (let k = 0; k < N_TACT_SLOTS; k++) {
      if (acc[k] !== 0) vals[tIds[k]] += acc[k];
    }
  }
}

// ── Module exports ────────────────────────────────────────────────────────────

const NPatterns = {
  createState,
  createWeights,
  internWeight: _internWeight,
  extractFeatures,
  evaluate,
  policyMove,
  greedyMove,
  reinforceUpdate,
  entropyBonusUpdate,
  annotateLadders,
  canonKey,
  canonKeyE,
  // Constants.
  CELL_BASE,
  CELLS_BASE,
  CELL_EMPTY, CELL_FRIEND, CELL_FOE,
  N_TACT, TACT_STONE_LIMIT, N_TACT_SLOTS,
  TACT_URGENT_KILL, TACT_URGENT_SAVE,
  TACT_WASTED_EXTEND, TACT_WASTED_ATTACK,
  TACT_RAW_BASE,
  SHAPE33C_RAW_BASE,
  TYPE_E_CELLS, TYPE_E_BASE, TYPE_E_RAW_BASE,
  // Exposed for tests.
  _D4,
};

if (typeof module !== 'undefined') module.exports = NPatterns;
else window.NPatterns = NPatterns;

})();
