'use strict';

// npat-lib.js — "nine-pattern" policy features with orthogonal tactical bits.
//
// For each candidate move, we extract the NINE 3×3 shape windows that overlap
// the candidate point, plus FOUR orthogonal tactical features.  The move's
// logit is the sum of the 9 shape-feature weights plus the tactical
// contributions; moves are sampled by softmax over logits.
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
const N_TACT             = 4;
const TACT_URGENT_KILL   = 0;
const TACT_URGENT_SAVE   = 1;
const TACT_WASTED_EXTEND = 2;
const TACT_WASTED_ATTACK = 3;
const TACT_RAW_BASE      = 9 * CELLS_BASE; // 177147; above the shape-key range

// 3×4 shape window (12 cells, single window per candidate with the candidate
// CENTERED at relPos=5 — middle row, col index 1 in the 3×4 layout).  Under
// D4 canonicalisation this collapses with its hflip (candidate at relPos=6),
// its 4×3 transpose, and other symmetries.  Theoretical canonical-key count
// ≈ 3^11 / avg-orbit = 22k–100k.  The window spans rows [-1,+1] and cols
// [-1,+2] around the candidate.
const WINDOWS_34         = 1;
const CELLS12_BASE       = 531441; // 3^12
const SHAPE34_RAW_BASE   = TACT_RAW_BASE + N_TACT; // 177151; above tactical ids

// Full-L shape window (14 cells — the 3×3 centered on the candidate plus 5
// extras extending into the SE corner: 2 cells East at row offsets {0, +1}
// col offset +2, 2 cells South at row offset +2 col offsets {0, +1}, AND the
// (+2, +2) corner completing the L).  Single window per candidate, handle
// always SE; D4 canonicalisation merges the 8 rotated / reflected versions of
// the pattern.  The shape is 180°- and diagonal-symmetric so only 2 of the
// 8 D4 orbits land on distinct footprints, but we still canonicalise over all
// 8 to get the orbit-minimum integer.  All 8 footprints lie within rows
// [-2,+2] × cols [-2,+2], so we read them from the existing 5×7 patch via
// pre-computed patch indices.  Canonical-key count ≈ 3^13 / 4 ≈ 400k.
const SHAPE_L_CELLS      = 14;
const SHAPE_L_BASE       = 4782969; // 3^14
const SHAPE_L_RAW_BASE   = SHAPE34_RAW_BASE + 12 * CELLS12_BASE; // 6554443

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

  if (!out) out = { tactCount: new Uint8Array(cap * N_TACT) };
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
    if (urgentLibs.length > 0) {
      const fi = defending ? TACT_URGENT_SAVE : TACT_URGENT_KILL;
      for (const lib of urgentLibs) tc[lib * N_TACT + fi]++;
    } else if (!moverSucceeds) {
      const fi = defending ? TACT_WASTED_EXTEND : TACT_WASTED_ATTACK;
      for (const lib of libs) tc[lib * N_TACT + fi]++;
    }
  }

  return out;
}

// ── State ─────────────────────────────────────────────────────────────────────
//
// moves   [count]              flat board index of move i
// patIds  [count * 9]          canonical shape pattern keys per window
// tact    [count * N_TACT]     tactical feature counts per move (uint8)
// logits  [count]              scratch buffer for softmax
// probs   [count]              scratch buffer for softmax
// ladder  { tactCount }        reusable ladder-annotation buffer
//
// Window order w = (wr+2)*3 + (wc+2) where (wr, wc) ∈ {-2,-1,0}² is the
// offset from the candidate to the window's top-left corner.

function createState(N) {
  const cap = N * N;
  // Precomputed toroidal 5×7 neighbour table (rows pr ∈ [-2, 2], cols
  // pc ∈ [-3, 3]).  For each board index idx and each offset, patchNbr[idx*35
  // + (pr+2)*7 + (pc+3)] gives the flat board index of (r+pr mod N, c+pc mod N).
  // This replaces 35 `_wrap` calls per candidate in extractFeatures with one
  // indexed load.  5 rows suffice for both 3×3 (rows [-2,0]) and 3×4 (rows
  // [-2,0]) windows; 7 cols accommodate 3×4 windows with wc ∈ [-3, 0].
  const patchNbr = new Int32Array(cap * 35);
  for (let idx = 0; idx < cap; idx++) {
    const r = (idx / N) | 0;
    const c = idx - r * N;
    const base = idx * 35;
    for (let pr = -2; pr <= 2; pr++) {
      const br = _wrap(r + pr, N);
      const rowBase = br * N;
      const prBase  = (pr + 2) * 7;
      for (let pc = -3; pc <= 3; pc++) {
        patchNbr[base + prBase + (pc + 3)] = rowBase + _wrap(c + pc, N);
      }
    }
  }
  return {
    N,
    moves:    new Int32Array(cap),
    patIds:   new Int32Array(cap * 9),
    patIds34: new Int32Array(cap * WINDOWS_34),
    patIdsL:  new Int32Array(cap),
    tact:     new Uint8Array(cap * N_TACT),
    logits:   new Float64Array(cap),
    probs:    new Float64Array(cap),
    ladder:   { tactCount: new Uint8Array(cap * N_TACT) },
    patchNbr,
    // Reusable touched-index scratch for reinforceUpdate.  Upper bound is
    // (9 + WINDOWS_34 + 1) * (n + 1) dense idxs, n ≤ cap.
    touched:  new Int32Array((9 + WINDOWS_34 + 1) * (cap + 1)),
    count:    0,
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

function createWeights(initialCapacity = 1024) {
  const w = {
    map:   new Map(),                              // raw canonKey → dense idx
    vals:  new Float64Array(initialCapacity),      // weights[dense idx]
    delta: new Float64Array(initialCapacity),      // reusable reinforce buffer
    size:  0,                                      // next dense idx to assign
    tactIds: new Int32Array(N_TACT),
  };
  for (let k = 0; k < N_TACT; k++) {
    w.tactIds[k] = _internWeight(w, TACT_RAW_BASE + k);
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

function canonKey(relPos, cells) {
  const c0 = cells[0], c1 = cells[1], c2 = cells[2];
  const c3 = cells[3], c4 = cells[4], c5 = cells[5];
  const c6 = cells[6], c7 = cells[7], c8 = cells[8];
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
  return best;
}

const _windowCells  = new Int32Array(9);
const _patch5       = new Int32Array(35); // 5×7 (rows [-2,+2] × cols [-3,+3])

// ── Runtime D4 canonicalisation for 3×4 windows ──────────────────────────────
//
// A 3×4 rectangle is NOT closed under D4 — 90°/270°/diagonal flips take it to
// a 4×3.  We still canonicalise over all 8 symmetries so a 3×4 window and the
// 4×3 window obtained by rotating the board share a canonical key.  For each
// σ we precompute P_σ : {0..11} → {0..11}, the permutation of row-major cell
// indices (and of relPos) when cells are re-indexed row-major in the
// post-σ bounding box (which is 3×4 for σ ∈ {id,hflip,vflip,180°} and 4×3 for
// σ ∈ {90°,270°,diag,antidiag}).  Canonical raw for a 3×4 window:
//   raw = P_σ(relPos) · 3^12 + Σ_i cells[i] · 3^{P_σ(i)}
// minimised over σ.  Precompute cell weights and relPos offsets analogous to
// the 3×3 tables above.

const _D4rect = [
  [0, 1, 2, 3,  4, 5, 6, 7,   8, 9,10,11], // id           (r,c) -> (r,c),         3×4
  [3, 2, 1, 0,  7, 6, 5, 4,  11,10, 9, 8], // hflip        (r,c) -> (r, 3-c),      3×4
  [8, 9,10,11,  4, 5, 6, 7,   0, 1, 2, 3], // vflip        (r,c) -> (2-r, c),      3×4
  [11,10, 9, 8, 7, 6, 5, 4,   3, 2, 1, 0], // 180°         (r,c) -> (2-r, 3-c),    3×4
  [2, 5, 8,11,  1, 4, 7,10,   0, 3, 6, 9], // 90° CW       (r,c) -> (c, 2-r),      4×3
  [9, 6, 3, 0, 10, 7, 4, 1,  11, 8, 5, 2], // 270° CW      (r,c) -> (3-c, r),      4×3
  [0, 3, 6, 9,  1, 4, 7,10,   2, 5, 8,11], // diag         (r,c) -> (c, r),        4×3
  [11, 8, 5, 2,10, 7, 4, 1,   9, 6, 3, 0], // antidiag     (r,c) -> (3-c, 2-r),    4×3
];

const _D4rectCw = new Int32Array(8 * 12);
const _D4rectRp = new Int32Array(8 * 12);
(function () {
  const pow = new Int32Array(13);
  pow[0] = 1;
  for (let k = 1; k < 13; k++) pow[k] = pow[k - 1] * CELL_BASE;
  for (let di = 0; di < 8; di++) {
    const perm = _D4rect[di];
    for (let i = 0; i < 12; i++) {
      _D4rectCw[di * 12 + i] = pow[perm[i]];
      _D4rectRp[di * 12 + i] = perm[i] * CELLS12_BASE;
    }
  }
})();

// canonKey34(relPos, cells12) returns the D4-canonical raw int for a 3×4
// window, WITHOUT the SHAPE34_RAW_BASE offset (the caller adds it at intern
// time so raw ids don't collide with 3×3 keys or tactical ids).
function canonKey34(relPos, cells) {
  const c0 = cells[0],  c1 = cells[1],  c2 = cells[2],  c3 = cells[3];
  const c4 = cells[4],  c5 = cells[5],  c6 = cells[6],  c7 = cells[7];
  const c8 = cells[8],  c9 = cells[9],  c10 = cells[10], c11 = cells[11];
  const cw = _D4rectCw, rp = _D4rectRp;
  let best = 0x7fffffff;
  for (let di = 0; di < 8; di++) {
    const o = di * 12;
    const raw = rp[o + relPos]
      + c0  * cw[o]      + c1  * cw[o + 1]  + c2  * cw[o + 2]  + c3  * cw[o + 3]
      + c4  * cw[o + 4]  + c5  * cw[o + 5]  + c6  * cw[o + 6]  + c7  * cw[o + 7]
      + c8  * cw[o + 8]  + c9  * cw[o + 9]  + c10 * cw[o + 10] + c11 * cw[o + 11];
    if (raw < best) best = raw;
  }
  return best;
}

const _windowCells12 = new Int32Array(12);

// ── Runtime D4 canonicalisation for the almost-L 13-cell shape ────────────────
//
// Shape offsets (row, col) in the original orientation (handle-SE), row-major:
//    ( -1,-1) ( -1, 0) ( -1,+1)
//    (  0,-1) (  0, 0) (  0,+1) (  0,+2)     ← candidate at index 4 = (0, 0)
//    ( +1,-1) ( +1, 0) ( +1,+1) ( +1,+2)
//                      ( +2, 0) ( +2,+1) ( +2,+2)
// 14 cells total (4×4 NW sub-block with the NE and SW corners removed).
// Under each D4 perm σ, the 14 offsets map to a D4-transformed footprint;
// the shape has 4-fold internal symmetry (id, 180°, diag, antidiag) so only
// 2 of the 8 orbits land on distinct footprints.  Canonicalisation reads all
// 8 variants and takes the min integer.  All footprints lie within rows
// [-2,+2] × cols [-2,+2], reusable from the 5×7 patch.

const _LShape = [
  [-1, -1], [-1,  0], [-1, +1],
  [ 0, -1], [ 0,  0], [ 0, +1], [ 0, +2],
  [+1, -1], [+1,  0], [+1, +1], [+1, +2],
            [+2,  0], [+2, +1], [+2, +2],
];

const _LPatchIdx    = new Int32Array(8 * SHAPE_L_CELLS);
const _LCellWeights = new Int32Array(SHAPE_L_CELLS);
(function () {
  const d4 = [
    (r, c) => [ r,  c],   // 0: id
    (r, c) => [ c, -r],   // 1: 90° CW
    (r, c) => [-r, -c],   // 2: 180°
    (r, c) => [-c,  r],   // 3: 270° CW
    (r, c) => [ r, -c],   // 4: hflip
    (r, c) => [-r,  c],   // 5: vflip
    (r, c) => [ c,  r],   // 6: diag
    (r, c) => [-c, -r],   // 7: antidiag
  ];
  for (let s = 0; s < 8; s++) {
    for (let i = 0; i < SHAPE_L_CELLS; i++) {
      const [r, c] = d4[s](_LShape[i][0], _LShape[i][1]);
      // patch layout: (pr+2)*7 + (pc+3), rows [-2,+2] × cols [-3,+3].
      _LPatchIdx[s * SHAPE_L_CELLS + i] = (r + 2) * 7 + (c + 3);
    }
  }
  let w = 1;
  for (let i = 0; i < SHAPE_L_CELLS; i++) { _LCellWeights[i] = w; w *= CELL_BASE; }
})();

// canonKeyL(patch) returns the D4-canonical raw int for the almost-L shape
// around the candidate at patch centre.  The patch is the 5×7 byte array
// filled by extractFeatures.  Result is WITHOUT the SHAPE_L_RAW_BASE offset.
function canonKeyL(patch) {
  const idx = _LPatchIdx;
  const cw  = _LCellWeights;
  let best = 0x7fffffff;
  for (let s = 0; s < 8; s++) {
    const o = s * SHAPE_L_CELLS;
    let raw = 0;
    for (let i = 0; i < SHAPE_L_CELLS; i++) {
      raw += patch[idx[o + i]] * cw[i];
    }
    if (raw < best) best = raw;
  }
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

  const li = ladderInfo || annotateLadders(game, state.ladder, game3);
  const tactCount = li.tactCount;

  let count = 0;
  const moves    = state.moves;
  const patIds   = state.patIds;
  const patIds34 = state.patIds34;
  const patIdsL  = state.patIdsL;
  const tact     = state.tact;
  const patchNbr = state.patchNbr;
  const wc9      = _windowCells;
  const wc12     = _windowCells12;
  const patch    = _patch5;
  const wMap     = weights ? weights.map : null;

  for (let ei = 0; ei < ec; ei++) {
    const idx = emC[ei];
    if (!game.isLegal(idx) || game.isTrueEye(idx)) continue;

    // Build the 5×7 patch of encoded cell values centred on the candidate.
    // patchNbr pre-stores the 35 toroidal flat indices for each idx; 35
    // indexed loads replace 35 _wrap calls.  Layout: patch[(pr+2)*7 + (pc+3)].
    const nbrBase = idx * 35;
    for (let p = 0; p < 35; p++) {
      const bi = patchNbr[nbrBase + p];
      const ci = cells[bi];
      let v;
      if (ci === 0)        v = CELL_EMPTY;
      else if (ci === cur) v = CELL_FRIEND;
      else                 v = CELL_FOE;
      patch[p] = v;
    }

    // 3×3 shape windows (9 per candidate).
    const off = count * 9;
    for (let wr = -2; wr <= 0; wr++) {
      for (let wc = -2; wc <= 0; wc++) {
        const relRow = -wr;
        const relCol = -wc;
        const relPos = relRow * 3 + relCol;

        // Fill wc9 from the patch — no board indexing at all here.
        for (let i = 0; i < 9; i++) {
          const dr = (i / 3) | 0;
          const dc = i - dr * 3;
          wc9[i] = patch[(wr + dr + 2) * 7 + (wc + dc + 3)];
        }

        const raw = canonKey(relPos, wc9);
        patIds[off + (wr + 2) * 3 + (wc + 2)] = wMap ? _internWeight(weights, raw) : raw;
      }
    }

    // 3×4 shape window — single window with the candidate centered at
    // relPos=5 (middle row, col index 1 in the 3×4 layout).  Window spans 1
    // row above/below and 1 col left / 2 cols right of the candidate.  D4
    // canonicalisation absorbs the left-right asymmetry (hflip swaps 5↔6).
    {
      const wr = -1, wc = -1;
      const relPos = 5;
      for (let i = 0; i < 12; i++) {
        const dr = (i / 4) | 0;
        const dc = i - dr * 4;
        wc12[i] = patch[(wr + dr + 2) * 7 + (wc + dc + 3)];
      }
      const raw34 = SHAPE34_RAW_BASE + canonKey34(relPos, wc12);
      patIds34[count] = wMap ? _internWeight(weights, raw34) : raw34;
    }

    // Almost-L 13-cell shape (3×3 + 4 SE extras) — one canonical key per
    // candidate.  canonKeyL reads cells directly from the 5×7 patch at 8
    // precomputed D4 orientations and returns the minimum raw int.
    {
      const rawL = SHAPE_L_RAW_BASE + canonKeyL(patch);
      patIdsL[count] = wMap ? _internWeight(weights, rawL) : rawL;
    }

    // Copy tactical counts for this candidate.
    const tSrc = idx * N_TACT;
    const tDst = count * N_TACT;
    tact[tDst]     = tactCount[tSrc];
    tact[tDst + 1] = tactCount[tSrc + 1];
    tact[tDst + 2] = tactCount[tSrc + 2];
    tact[tDst + 3] = tactCount[tSrc + 3];

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
  const vals     = weights.vals;
  const patIds   = state.patIds;
  const patIds34 = state.patIds34;
  const patIdsL  = state.patIdsL;
  const tact     = state.tact;
  const tIds     = weights.tactIds;
  const off9     = i * 9;
  const off12    = i * WINDOWS_34;
  const off4     = i * N_TACT;
  let s = vals[patIds[off9]]     + vals[patIds[off9 + 1]] + vals[patIds[off9 + 2]]
        + vals[patIds[off9 + 3]] + vals[patIds[off9 + 4]] + vals[patIds[off9 + 5]]
        + vals[patIds[off9 + 6]] + vals[patIds[off9 + 7]] + vals[patIds[off9 + 8]]
        + tact[off4]     * vals[tIds[0]]
        + tact[off4 + 1] * vals[tIds[1]]
        + tact[off4 + 2] * vals[tIds[2]]
        + tact[off4 + 3] * vals[tIds[3]];
  for (let k = 0; k < WINDOWS_34; k++) s += vals[patIds34[off12 + k]];
  s += vals[patIdsL[i]];
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
  const pid   = state.patIds;
  const pid34 = state.patIds34;
  const pidL  = state.patIdsL;
  const tact  = state.tact;
  const lg    = state.logits;
  const pr    = state.probs;
  const vals  = weights.vals;
  const tIds  = weights.tactIds;
  const w0 = vals[tIds[0]], w1 = vals[tIds[1]], w2 = vals[tIds[2]], w3 = vals[tIds[3]];

  let maxL = -Infinity;
  for (let i = 0; i < n; i++) {
    const off9  = i * 9;
    const off12 = i * WINDOWS_34;
    const off4  = i * N_TACT;
    let s = vals[pid[off9]]     + vals[pid[off9 + 1]] + vals[pid[off9 + 2]]
          + vals[pid[off9 + 3]] + vals[pid[off9 + 4]] + vals[pid[off9 + 5]]
          + vals[pid[off9 + 6]] + vals[pid[off9 + 7]] + vals[pid[off9 + 8]]
          + tact[off4] * w0 + tact[off4 + 1] * w1
          + tact[off4 + 2] * w2 + tact[off4 + 3] * w3;
    for (let k = 0; k < WINDOWS_34; k++) s += vals[pid34[off12 + k]];
    s += vals[pidL[i]];
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

function reinforceUpdate(state, chosenIndex, advantage, weights, lr) {
  const n = state.count;
  if (n === 0 || chosenIndex < 0) return;

  const step = lr * advantage;
  if (step === 0) return;

  const patIds   = state.patIds;
  const patIds34 = state.patIds34;
  const patIdsL  = state.patIdsL;
  const probs    = state.probs;
  const tact     = state.tact;
  const vals     = weights.vals;
  const delta    = weights.delta;
  const tIds     = weights.tactIds;
  const touched  = state.touched;
  let tc = 0;

  // ── Shape features (use delta buffer to dedupe repeats across moves) ──
  // Chosen move contributes +step to each of its 9 + WINDOWS_34 + 1 dense pids.
  {
    const off   = chosenIndex * 9;
    const off34 = chosenIndex * WINDOWS_34;
    for (let k = 0; k < 9; k++) {
      const idx = patIds[off + k];
      touched[tc++] = idx;
      delta[idx] += step;
    }
    for (let k = 0; k < WINDOWS_34; k++) {
      const idx = patIds34[off34 + k];
      touched[tc++] = idx;
      delta[idx] += step;
    }
    {
      const idx = patIdsL[chosenIndex];
      touched[tc++] = idx;
      delta[idx] += step;
    }
  }
  // Every legal move i contributes -step * π_i to each of its shape pids.
  for (let i = 0; i < n; i++) {
    const pi = probs[i];
    if (pi === 0) continue;
    const sub = step * pi;
    const off   = i * 9;
    const off34 = i * WINDOWS_34;
    for (let k = 0; k < 9; k++) {
      const idx = patIds[off + k];
      touched[tc++] = idx;
      delta[idx] -= sub;
    }
    for (let k = 0; k < WINDOWS_34; k++) {
      const idx = patIds34[off34 + k];
      touched[tc++] = idx;
      delta[idx] -= sub;
    }
    {
      const idx = patIdsL[i];
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

  // ── Tactical features (just 4, always touched; no delta buffer needed) ──
  const cOff = chosenIndex * N_TACT;
  let neg0 = 0, neg1 = 0, neg2 = 0, neg3 = 0;
  for (let i = 0; i < n; i++) {
    const pi = probs[i];
    if (pi === 0) continue;
    const off = i * N_TACT;
    neg0 += pi * tact[off];
    neg1 += pi * tact[off + 1];
    neg2 += pi * tact[off + 2];
    neg3 += pi * tact[off + 3];
  }
  const d0 = step * (tact[cOff]     - neg0);
  const d1 = step * (tact[cOff + 1] - neg1);
  const d2 = step * (tact[cOff + 2] - neg2);
  const d3 = step * (tact[cOff + 3] - neg3);
  if (d0 !== 0) vals[tIds[0]] += d0;
  if (d1 !== 0) vals[tIds[1]] += d1;
  if (d2 !== 0) vals[tIds[2]] += d2;
  if (d3 !== 0) vals[tIds[3]] += d3;
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

  const patIds   = state.patIds;
  const patIds34 = state.patIds34;
  const patIdsL  = state.patIdsL;
  const probs    = state.probs;
  const tact     = state.tact;
  const vals     = weights.vals;
  const delta    = weights.delta;
  const tIds     = weights.tactIds;
  const touched  = state.touched;
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
  let acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
  for (let i = 0; i < n; i++) {
    const pi = probs[i];
    if (pi === 0) continue;
    const ci = beta * pi * (logs[i] - H);
    if (ci === 0) continue;
    const off9  = i * 9;
    const off12 = i * WINDOWS_34;
    for (let k = 0; k < 9; k++) {
      const idx = patIds[off9 + k];
      touched[tc++] = idx;
      delta[idx] += ci;
    }
    for (let k = 0; k < WINDOWS_34; k++) {
      const idx = patIds34[off12 + k];
      touched[tc++] = idx;
      delta[idx] += ci;
    }
    {
      const idx = patIdsL[i];
      touched[tc++] = idx;
      delta[idx] += ci;
    }
    const off4 = i * N_TACT;
    acc0 += ci * tact[off4];
    acc1 += ci * tact[off4 + 1];
    acc2 += ci * tact[off4 + 2];
    acc3 += ci * tact[off4 + 3];
  }
  for (let i = 0; i < tc; i++) {
    const idx = touched[i];
    const d = delta[idx];
    if (d !== 0) { vals[idx] += d; delta[idx] = 0; }
  }
  if (acc0 !== 0) vals[tIds[0]] += acc0;
  if (acc1 !== 0) vals[tIds[1]] += acc1;
  if (acc2 !== 0) vals[tIds[2]] += acc2;
  if (acc3 !== 0) vals[tIds[3]] += acc3;
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
  canonKey34,
  canonKeyL,
  // Constants.
  CELL_BASE,
  CELLS_BASE,
  CELL_EMPTY, CELL_FRIEND, CELL_FOE,
  N_TACT,
  TACT_URGENT_KILL, TACT_URGENT_SAVE,
  TACT_WASTED_EXTEND, TACT_WASTED_ATTACK,
  TACT_RAW_BASE,
  WINDOWS_34, CELLS12_BASE, SHAPE34_RAW_BASE,
  SHAPE_L_CELLS, SHAPE_L_BASE, SHAPE_L_RAW_BASE,
  // Exposed for tests.
  _D4,
  _D4rect,
  _LShape,
};

if (typeof module !== 'undefined') module.exports = NPatterns;
else window.NPatterns = NPatterns;

})();
