'use strict';

// npat-lib.js — pattern-based policy features with stone-indexed tactical bits.
//
// For each candidate move we extract zero or more shape windows (centered 3×3
// via cfg.useP8, 12-cell diamond P12 via cfg.useP12) plus four tactical
// feature types expanded into per-stone-index sub-features.  The move's logit
// is the sum of the active shape weights and the tactical contributions;
// moves are sampled by softmax over logits.
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
// a dense Float32Array indexed by the intern map's dense slot.
//
// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.

(function () {

const Util = (typeof require === 'function') ? require('./util.js') : window.Util;
const { PASS }                 = Util.load('./game2.js', 'Game2');
const { game3FromGame2 }       = Util.load('./game3.js', 'Game3');
const { getAllLadderStatuses } = Util.load('./ladder2.js', 'Ladder2');

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
// TACT_STONE_LIMIT controls per-stone-index expansion.
const TACT_STONE_LIMIT   = 8;
const N_TACT_SLOTS       = N_TACT * TACT_STONE_LIMIT;  // 32 at default
const TACT_URGENT_KILL   = 0;
const TACT_URGENT_SAVE   = 1;
const TACT_WASTED_EXTEND = 2;
const TACT_WASTED_ATTACK = 3;
// Feature families (raw-key base offsets are assigned below from one ordered
// layout table — see ── Raw-key layout ──):
//   P1  — tactical slots: 4 types (TACT_URGENT_* above) × TACT_STONE_LIMIT
//         per-stone-index sub-features.
//   P8  — centered 3×3 shape window (9 cells, candidate at relPos 4);
//         D4-canonicalised via canonKey(4, cells), range 9·CELLS_BASE.
//   P12 — the 12 closest cells (Manhattan distance 1–2) in a diamond; the
//         candidate is excluded (always empty).  D4-canonicalised, range 3^12.
//
//             (-2, 0)
//      (-1,-1)(-1, 0)(-1, 1)
//      ( 0,-2)( 0,-1)  *   ( 0, 1)( 0, 2)
//      ( 1,-1)( 1, 0)( 1, 1)
//             ( 2, 0)
//
//   P9  — P8 shape conjoined with a 4-bit tactical-type presence mask
//         (16 buckets): shape weights specialised by tactical context.
//   P13 — P12 shape conjoined with the same mask (the P12 analogue of P9).
const P12_CELLS     = 12;
const P12_BASE      = 531441; // 3^12
const P9_TACT_MASKS = 16;     // 4-bit tactical-type presence mask (P9/P13)

// ── Raw-key layout ───────────────────────────────────────────────────────────
//
// The raw-key line is partitioned into contiguous, non-overlapping blocks — one
// per family, plus reserved regions for legacy keys.  Each family's base offset
// is the running sum of all preceding block spans, so the offsets are DERIVED
// from this single ordered table: adding a family is one entry, with no
// hand-computed offset to keep non-overlapping.  Do NOT reorder or resize
// existing blocks — the offsets are baked into every trained model's keys, and
// the reserved blocks ('shape9', 'legacy') preserve the exact historical layout
// so old checkpoints stay valid.
const SHAPE9_SPAN = 9 * CELLS_BASE;                 // bare canonKey output range
const LEGACY_SPAN = 59049 + 177147 + 531441 + 81;   // removed A/B/D/T region
const _KEY_LAYOUT = [
  ['shape9', SHAPE9_SPAN],                  // reserved: legacy 9-window family
  ['tact',   N_TACT_SLOTS],                 // P1
  ['p8',     SHAPE9_SPAN],                  // P8  (centered 3×3)
  ['legacy', LEGACY_SPAN],                  // reserved: removed A/B/D/T
  ['p12',    P12_BASE],                     // P12 (diamond)
  ['p9',     P9_TACT_MASKS * SHAPE9_SPAN],  // P9  (16 × 3×3)
  ['p13',    P9_TACT_MASKS * P12_BASE],     // P13 (16 × diamond)
];
const _KEY_BASE = (() => {
  const base = {};
  let off = 0;
  for (const [name, span] of _KEY_LAYOUT) { base[name] = off; off += span; }
  return base;
})();
const TACT_RAW_BASE = _KEY_BASE.tact;
const P8_RAW_BASE   = _KEY_BASE.p8;
const P12_RAW_BASE  = _KEY_BASE.p12;
const P9_RAW_BASE   = _KEY_BASE.p9;
const P13_RAW_BASE  = _KEY_BASE.p13;

// ── Ladder-status annotation ─────────────────────────────────────────────────
//
// Build a per-cell tactical-feature count array for the current game position.
// tactCount is a Uint8Array of length N*N*N_TACT, laid out as
// tactCount[cellIdx * N_TACT + k] = how many qualifying chains make feature k
// apply at cell cellIdx.  Counts stay tiny (≤ number of distinct 1-2-lib
// chains sharing that liberty; in practice 1-2).

// Optional `infos` is a precomputed getAllLadderStatuses result for this
// exact position (same player to move); when given, the ladder pass is
// skipped and `infos` is consumed instead.
function annotateLadders(game, out, game3, infos) {
  const N   = game.N;
  const cap = N * N;

  if (!out) out = { tactCount: new Uint8Array(cap * N_TACT_SLOTS) };
  out.tactCount.fill(0);

  if (game.emptyCount === cap) return out;

  const g3 = game3 || game3FromGame2(game);
  if (!infos) infos = getAllLadderStatuses(g3);
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
// patIdsP8 [count]                       canonical 3×3c key per move
// patIdsP12 [count]                       canonical P12 key per move
// tact      [count * N_TACT_SLOTS]        tactical feature counts per move (uint8)
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
    patIdsP8:  new Int32Array(cap),
    patIdsP12: new Int32Array(cap),
    patIdsP9:  new Int32Array(cap),
    patIdsP13: new Int32Array(cap),
    tact:      new Uint8Array(cap * N_TACT_SLOTS),
    logits:    new Float64Array(cap),
    probs:     new Float64Array(cap),
    ladder:    { tactCount: new Uint8Array(cap * N_TACT_SLOTS) },
    patchNbr,
    // Reusable touched-index scratch for reinforceUpdate.  Up to four shape
    // types (P8 + P12 + P9 + P13), one dense idx per (chosen + each move).
    touched:   new Int32Array(4 * (cap + 1)),
    count:     0,
  };
}

// ── Weights store ─────────────────────────────────────────────────────────────
//
// Canonical pattern keys (the raw ints from canonKey) can range up to ~90M,
// so we can't index a flat array by them directly.  Instead we intern each
// raw key to a dense 0-based index the first time it is seen.  Weights and
// the reusable reinforce-delta buffer are plain Float32Arrays indexed by the
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
  //   { initialCapacity, useP1, useP8, useP12, useP9 }.
  // useP1 is the single-cell ladder/tactical feature; it defaults ON, the
  // shape windows (P8/P9/P12) default OFF.
  let initialCapacity = 1024;
  let useP1 = true;
  let useP8 = false, useP12 = false, useP9 = false, useP13 = false;
  if (typeof opts === 'number') {
    initialCapacity = opts;
  } else if (opts && typeof opts === 'object') {
    if (opts.initialCapacity) initialCapacity = opts.initialCapacity;
    if (opts.useP1 === false) useP1 = false;
    if (opts.useP8) useP8 = true;
    if (opts.useP12) useP12 = true;
    if (opts.useP9) useP9 = true;
    if (opts.useP13) useP13 = true;
  }
  const w = {
    map:   new Map(),                              // raw canonKey → dense idx
    vals:  new Float32Array(initialCapacity),      // weights[dense idx]
    delta: new Float32Array(initialCapacity),      // reusable reinforce buffer
    size:  0,                                      // next dense idx to assign
    tactIds: new Int32Array(N_TACT_SLOTS),
    cfg:   { useP1, useP8, useP12, useP9, useP13 },
  };
  if (useP1) {
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
    const nv = new Float32Array(cap); nv.set(w.vals); w.vals = nv;
    const nd = new Float32Array(cap); nd.set(w.delta); w.delta = nd;
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

// Type P12 shape: the 12 closest non-centre cells (Manhattan distance 1 and 2)
// in a diamond.  Canonical key = min over 8 D4 symmetries of
//   raw = Σ_i patch[shape[σ(i)]] · 3^i.
// Storage: _P12PatchIdx[s*12 + i] is the 5×5-patch flat index of the cell that
// occupies P12 position i under symmetry s.  Per-cell weight is 3^i.
const _P12Shape = [
  [-1,  0], [ 0,  1], [ 1,  0], [ 0, -1],   // 0..3: N, E, S, W (ring 1)
  [-2,  0], [-1,  1], [ 0,  2], [ 1,  1],   // 4..7: N2, NE, E2, SE
  [ 2,  0], [ 1, -1], [ 0, -2], [-1, -1],   // 8..11: S2, SW, W2, NW
];
const _P12PatchIdx    = new Int32Array(8 * P12_CELLS);
const _P12CellWeights = new Int32Array(P12_CELLS);
(function () {
  const d4 = [
    (r, c) => [ r,  c], (r, c) => [ c, -r], (r, c) => [-r, -c], (r, c) => [-c,  r],
    (r, c) => [ r, -c], (r, c) => [-r,  c], (r, c) => [ c,  r], (r, c) => [-c, -r],
  ];
  for (let s = 0; s < 8; s++) {
    for (let i = 0; i < P12_CELLS; i++) {
      const [r, c] = d4[s](_P12Shape[i][0], _P12Shape[i][1]);
      _P12PatchIdx[s * P12_CELLS + i] = (r + 2) * 5 + (c + 2);
    }
  }
  let w = 1;
  for (let i = 0; i < P12_CELLS; i++) { _P12CellWeights[i] = w; w *= CELL_BASE; }
})();

const _canonKeyP12Cache = new Int32Array(P12_BASE).fill(-1); // 2.1MB

function canonKeyP12(patch) {
  const idx = _P12PatchIdx;
  const cw  = _P12CellWeights;
  let cacheKey = 0;
  for (let i = 0; i < P12_CELLS; i++) cacheKey += patch[idx[i]] * cw[i];
  const cached = _canonKeyP12Cache[cacheKey];
  if (cached !== -1) return cached;
  let best = cacheKey;
  for (let s = 1; s < 8; s++) {
    const o = s * P12_CELLS;
    let raw = 0;
    for (let i = 0; i < P12_CELLS; i++) raw += patch[idx[o + i]] * cw[i];
    if (raw < best) best = raw;
  }
  _canonKeyP12Cache[cacheKey] = best;
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
// raw canonKey ints.  Optional ladderStatuses is a precomputed
// getAllLadderStatuses result for this exact position, shared with other
// extractors (see annotateLadders).
function extractFeatures(game, state, ladderInfo, game3, weights, ladderStatuses) {
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
  const doUseP1   = !cfg || cfg.useP1 !== false;
  const doUseP8 = !cfg || cfg.useP8;
  const doUseP12 = !cfg || cfg.useP12;
  const doUseP9 = !cfg || cfg.useP9;
  const doUseP13 = !cfg || cfg.useP13;
  // Skip ladder annotation entirely when nothing depends on it — that's the
  // dominant per-position cost.  P9/P13 conjoin a shape with the tactical mask,
  // so they need the ladder pass even when the plain P1 features are off.
  const li = (doUseP1 || doUseP9 || doUseP13)
    ? (ladderInfo || annotateLadders(game, state.ladder, game3, ladderStatuses)) : null;
  const tactCount = li ? li.tactCount : null;
  const patIdsP8 = state.patIdsP8;
  const patIdsP12 = state.patIdsP12;
  const patIdsP9 = state.patIdsP9;
  const patIdsP13 = state.patIdsP13;

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

    // 4-bit tactical-type presence mask at this candidate (slot 0 of each type
    // is set whenever that type applies — see annotateLadders).  Shared by the
    // P9 (3×3) and P13 (P12) shape×tactics conjunctions.
    let tactMask = 0;
    if (doUseP9 || doUseP13) {
      const tcOff = idx * N_TACT_SLOTS;
      if (tactCount[tcOff + TACT_URGENT_KILL   * TACT_STONE_LIMIT]) tactMask |= 1;
      if (tactCount[tcOff + TACT_URGENT_SAVE   * TACT_STONE_LIMIT]) tactMask |= 2;
      if (tactCount[tcOff + TACT_WASTED_EXTEND * TACT_STONE_LIMIT]) tactMask |= 4;
      if (tactCount[tcOff + TACT_WASTED_ATTACK * TACT_STONE_LIMIT]) tactMask |= 8;
    }

    // Centered 3×3 window — candidate at relPos 4, reads the central 3×3 of
    // the patch.  Canonicalised via canonKey; the same canonical key feeds both
    // the plain P8 shape and the tactical-conjoined P9 shape (disjoint bases).
    if (doUseP8 || doUseP9) {
      for (let i = 0; i < 9; i++) {
        const dr = (i / 3) | 0;
        const dc = i - dr * 3;
        wc9[i] = patch[(dr - 1 + 2) * 5 + (dc - 1 + 2)];
      }
      const canon3 = canonKey(4, wc9);
      if (doUseP8) {
        const rawP8 = P8_RAW_BASE + canon3;
        patIdsP8[count] = wMap ? _internWeight(weights, rawP8) : rawP8;
      }
      if (doUseP9) {
        const rawP9 = P9_RAW_BASE + tactMask * (9 * CELLS_BASE) + canon3;
        patIdsP9[count] = wMap ? _internWeight(weights, rawP9) : rawP9;
      }
    }

    // P12 diamond — 12 closest non-centre cells, D4-canonicalised; the same
    // canonical key feeds the plain P12 shape and the tactical-conjoined P13.
    if (doUseP12 || doUseP13) {
      const canonP12 = canonKeyP12(patch);
      if (doUseP12) {
        const rawP12 = P12_RAW_BASE + canonP12;
        patIdsP12[count] = wMap ? _internWeight(weights, rawP12) : rawP12;
      }
      if (doUseP13) {
        const rawP13 = P13_RAW_BASE + tactMask * P12_BASE + canonP12;
        patIdsP13[count] = wMap ? _internWeight(weights, rawP13) : rawP13;
      }
    }

    // Copy tactical counts for this candidate (N_TACT_SLOTS bytes = 4 types
    // × TACT_STONE_LIMIT stone-indices).  Skipped when P1 is off.
    if (doUseP1) {
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
  if (cfg.useP1 !== false) {
    const tOff = i * N_TACT_SLOTS;
    for (let k = 0; k < N_TACT_SLOTS; k++) s += tact[tOff + k] * vals[tIds[k]];
  }
  if (cfg.useP8) s += vals[state.patIdsP8[i]];
  if (cfg.useP12) s += vals[state.patIdsP12[i]];
  if (cfg.useP9) s += vals[state.patIdsP9[i]];
  if (cfg.useP13) s += vals[state.patIdsP13[i]];
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

// Populate state.logits and state.probs from the current weights.  At
// `temperature` = 1 this is the standard softmax; at 0 the distribution
// collapses to one-hot on the argmax (argmax fast-path); at intermediate T it
// is exp(logit / T) / Z.
function _computeSoftmax(state, weights, temperature = 1) {
  const n   = state.count;
  if (n === 0) return 0;
  const tact  = state.tact;
  const lg    = state.logits;
  const pr    = state.probs;
  const vals  = weights.vals;
  const tIds  = weights.tactIds;
  const cfg   = weights.cfg;
  const useTact = cfg.useP1 !== false;
  const tW    = new Float64Array(N_TACT_SLOTS);
  if (useTact) {
    for (let k = 0; k < N_TACT_SLOTS; k++) tW[k] = vals[tIds[k]];
  }
  const pidP8 = cfg.useP8 ? state.patIdsP8 : null;
  const pidP12 = cfg.useP12 ? state.patIdsP12 : null;
  const pidP9 = cfg.useP9 ? state.patIdsP9 : null;
  const pidP13 = cfg.useP13 ? state.patIdsP13 : null;

  let maxL = -Infinity, maxI = 0;
  for (let i = 0; i < n; i++) {
    let s = 0;
    if (useTact) {
      const tOff = i * N_TACT_SLOTS;
      for (let k = 0; k < N_TACT_SLOTS; k++) s += tact[tOff + k] * tW[k];
    }
    if (pidP8) s += vals[pidP8[i]];
    if (pidP12) s += vals[pidP12[i]];
    if (pidP9) s += vals[pidP9[i]];
    if (pidP13) s += vals[pidP13[i]];
    lg[i] = s;
    if (s > maxL) { maxL = s; maxI = i; }
  }
  if (temperature === 0) {
    for (let i = 0; i < n; i++) pr[i] = 0;
    pr[maxI] = 1;
    return n;
  }
  const invT = 1 / temperature;
  let sum = 0;
  for (let i = 0; i < n; i++) { pr[i] = Math.exp((lg[i] - maxL) * invT); sum += pr[i]; }
  const inv = 1 / sum;
  for (let i = 0; i < n; i++) pr[i] *= inv;
  return n;
}

// Extract features, softmax over logits (scaled by `temperature`), sample an
// action.  Returns { move, index, prob }.  `temperature` defaults to 1 (the
// standard softmax sample); 0 short-circuits to a deterministic argmax (no rng
// is consulted); values in between sharpen (< 1) or flatten (> 1) the
// distribution.  Optional game3 kept in lockstep with game.
function policyMove(game, state, weights, rng, game3, temperature = 1) {
  extractFeatures(game, state, undefined, game3, weights);
  const n = state.count;
  if (n === 0) return { move: PASS, index: -1, prob: 1 };

  _computeSoftmax(state, weights, temperature);
  const probs = state.probs;

  if (temperature === 0) {
    // Argmax fast-path — _computeSoftmax wrote a one-hot vector.
    let bestI = 0;
    for (let i = 1; i < n; i++) if (probs[i] > probs[bestI]) bestI = i;
    return { move: state.moves[bestI], index: bestI, prob: 1 };
  }

  const R = (rng || Math).random();
  let r = R, chosen = n - 1;
  for (let i = 0; i < n; i++) {
    r -= probs[i];
    if (r <= 0) { chosen = i; break; }
  }
  return { move: state.moves[chosen], index: chosen, prob: probs[chosen] };
}

// Near-greedy move.  Thin wrapper around policyMove with temperature = 0.1 —
// almost always picks the argmax but breaks near-ties stochastically.  For
// strict argmax use policyMove with T = 0; for full softmax sampling use the
// policyMove default T = 1.  Optional game3 kept in lockstep.
function greedyMove(game, state, weights, game3) {
  return policyMove(game, state, weights, undefined, game3, 0.1).move;
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
  const patIdsP8 = cfg.useP8 ? state.patIdsP8 : null;
  const patIdsP12 = cfg.useP12 ? state.patIdsP12 : null;
  const patIdsP9 = cfg.useP9 ? state.patIdsP9 : null;
  const patIdsP13 = cfg.useP13 ? state.patIdsP13 : null;
  let tc = 0;

  // ── Shape features (use delta buffer to dedupe repeats across moves) ──
  // Chosen move contributes +step to each of its active shape pids.
  if (patIdsP8) {
    const idx = patIdsP8[chosenIndex];
    touched[tc++] = idx;
    delta[idx] += step;
  }
  if (patIdsP12) {
    const idx = patIdsP12[chosenIndex];
    touched[tc++] = idx;
    delta[idx] += step;
  }
  if (patIdsP9) {
    const idx = patIdsP9[chosenIndex];
    touched[tc++] = idx;
    delta[idx] += step;
  }
  if (patIdsP13) {
    const idx = patIdsP13[chosenIndex];
    touched[tc++] = idx;
    delta[idx] += step;
  }
  // Every legal move i contributes -step * π_i to each of its active shape pids.
  for (let i = 0; i < n; i++) {
    const pi = probs[i];
    if (pi === 0) continue;
    const sub = step * pi;
    if (patIdsP8) {
      const idx = patIdsP8[i];
      touched[tc++] = idx;
      delta[idx] -= sub;
    }
    if (patIdsP12) {
      const idx = patIdsP12[i];
      touched[tc++] = idx;
      delta[idx] -= sub;
    }
    if (patIdsP9) {
      const idx = patIdsP9[i];
      touched[tc++] = idx;
      delta[idx] -= sub;
    }
    if (patIdsP13) {
      const idx = patIdsP13[i];
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

  // ── P1 tactical features (N_TACT_SLOTS, always touched; no delta buffer) ──
  if (cfg.useP1 !== false) {
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
  return tc + (cfg.useP1 !== false ? N_TACT_SLOTS : 0);
}

// ── Model loading ─────────────────────────────────────────────────────────────

// Normalise a raw npat data file's weight table to { count, forEach(cb) },
// where cb(key, val) is called once per entry.  Supports three forms:
//   - int16-quantised  ({ keys: Int32Array, qvals: Int16Array, scale, count })
//     — produced by train-npat.js; weight = qvals[i] / scale.
//   - float32 typed arrays ({ keys: Int32Array, vals: Float32Array, count })
//   - legacy literal Map   ({ weights: Map })
function modelWeights(raw) {
  if (raw.keys && raw.qvals) {
    const keys = raw.keys, qvals = raw.qvals;
    const count = raw.count != null ? raw.count : keys.length;
    const inv = 1 / raw.scale;
    return { count, forEach(cb) { for (let i = 0; i < count; i++) cb(keys[i], qvals[i] * inv); } };
  }
  if (raw.keys && raw.vals) {
    const keys = raw.keys, vals = raw.vals;
    const count = raw.count != null ? raw.count : keys.length;
    return { count, forEach(cb) { for (let i = 0; i < count; i++) cb(keys[i], vals[i]); } };
  }
  const m = raw.weights;
  return { count: m.size, forEach(cb) { for (const [k, v] of m) cb(k, v); } };
}

// Build runtime weights from a raw model object (a required npat data file or
// window.npatModel; new base64 or legacy Map form — see modelWeights).
// Validates the tactical stone limit and interns every weight.  The active
// feature families come from the file's explicit `cfg` when present; for legacy
// files that lack it, they're inferred from the raw key ranges (P1 always on).
// `name` prefixes error messages.  The active families are readable from the
// result via weights.cfg.
function prepareWeights(raw, name = 'npat') {
  if (raw.tactStoneLimit !== undefined && raw.tactStoneLimit !== TACT_STONE_LIMIT) {
    throw new Error(
      `${name}: TACT_STONE_LIMIT mismatch — model was trained at ${raw.tactStoneLimit}, ` +
      `runtime is ${TACT_STONE_LIMIT}. Set NPAT_STONE_LIMIT=${raw.tactStoneLimit} before launching.`
    );
  }
  const mw = modelWeights(raw);

  // Active families: prefer the file's explicit set; fall back to inferring them
  // from key magnitudes for legacy files written before `cfg` was stored.
  let cfg = raw.cfg;
  if (!cfg) {
    let hasP8 = false, hasP12 = false, hasP9 = false, hasP13 = false;
    mw.forEach(k => {
      if (typeof k === 'string') return; // ignore any orphan string keys
      if      (k >= P13_RAW_BASE)                    hasP13 = true;
      else if (k >= P9_RAW_BASE)                     hasP9 = true;
      else if (k >= P12_RAW_BASE)                    hasP12 = true;
      else if (k >= P8_RAW_BASE)                     hasP8 = true;
    });
    cfg = { useP8: hasP8, useP12: hasP12, useP9: hasP9, useP13: hasP13 }; // P1 defaults on
  }

  const weights = createWeights({ initialCapacity: Math.max(1024, mw.count | 0), ...cfg });
  mw.forEach((k, v) => {
    weights.vals[_internWeight(weights, k)] = v;
  });
  return weights;
}

// Resolve the npat model source and build runtime weights from it.
//   name — message prefix (typically the agent name)      (default 'npat')
//   path — model file path, Node only                     (default ../npat-data.js)
// In the browser the source is always window.npatModel.
// Returns { weights, modelName }.
function loadModel({ name = 'npat', path: pathOverride } = {}) {
  let raw, modelName;
  if (typeof window !== 'undefined') {
    if (!window.npatModel) {
      throw new Error(`${name}: window.npatModel is not set — load the npat weights script first`);
    }
    raw = window.npatModel;
    modelName = 'window.npatModel';
  } else {
    const path = require('path');
    const file = pathOverride || path.join(__dirname, 'npat-data.js');
    raw = require(path.resolve(file));
    modelName = path.basename(file);
  }
  return { weights: prepareWeights(raw, name), modelName };
}

// ── Module exports ────────────────────────────────────────────────────────────

const NPatterns = {
  createState,
  createWeights,
  internWeight: _internWeight,
  prepareWeights,
  modelWeights,
  loadModel,
  extractFeatures,
  evaluate,
  policyMove,
  greedyMove,
  computeSoftmax: _computeSoftmax,
  reinforceUpdate,
  annotateLadders,
  canonKey,
  canonKeyP12,
  // Constants.
  CELL_BASE,
  CELLS_BASE,
  CELL_EMPTY, CELL_FRIEND, CELL_FOE,
  N_TACT, TACT_STONE_LIMIT, N_TACT_SLOTS,
  TACT_URGENT_KILL, TACT_URGENT_SAVE,
  TACT_WASTED_EXTEND, TACT_WASTED_ATTACK,
  TACT_RAW_BASE,
  P8_RAW_BASE,
  P12_CELLS, P12_BASE, P12_RAW_BASE,
  P9_RAW_BASE, P9_TACT_MASKS,
  P13_RAW_BASE,
  // Exposed for tests.
  _D4,
};

if (typeof module !== 'undefined') module.exports = NPatterns;
else window.NPatterns = NPatterns;

})();
