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

// Type A shape window (10 cells, hflip-symmetric about the vertical axis
// through the candidate).  Extends 1 row up and 2 rows down from the
// candidate.  Cells (row, col) offsets from the candidate, row-major:
//
//     .    x    .    (-1, 0)
//     x    *    x    (0, -1) (0, 0) (0, +1)
//     x    x    x    (+1, -1) (+1, 0) (+1, +1)
//     x    x    x    (+2, -1) (+2, 0) (+2, +1)
//
// Under D4 the 8 orbits produce distinct footprints (4 unique up to the
// shape's hflip symmetry).  All lie within rows [-2, +2] × cols [-2, +2],
// so we read them from the existing 5×7 patch via pre-computed patch
// indices.  Canonical-key count ≈ 3^9 / 4 ≈ 5k.
const TYPE_A_CELLS       = 10;
const TYPE_A_BASE        = 59049; // 3^10
const TYPE_A_RAW_BASE    = SHAPE33C_RAW_BASE + 9 * CELLS_BASE; // above 33c range

// Type B shape window (11 cells, diagonal-symmetric about the NW-SE axis
// through the candidate).  Extends 1 row up, 2 cells right, 2 rows down and
// 2 cols right from the candidate:
//
//     .    x    .    .      (-1, 0)
//     x    *    x    x      (0, -1) (0, 0) (0, +1) (0, +2)
//     .    x    x    x      (+1, 0) (+1, +1) (+1, +2)
//     .    x    x    x      (+2, 0) (+2, +1) (+2, +2)
//
// Under D4 the 4 orbits produce 4 distinct footprints (diag and id stabilise
// the shape).  All lie within rows [-2, +2] × cols [-2, +2], reusable from
// the 5×7 patch.  Canonical-key count ≈ 3^10 / 4 ≈ 15k.
const TYPE_B_CELLS       = 11;
const TYPE_B_BASE        = 177147; // 3^11
const TYPE_B_RAW_BASE    = TYPE_A_RAW_BASE + TYPE_A_BASE; // above A range

// Grown shape window — one pattern per candidate whose cell set is chosen
// dynamically.  The 9 cells of the 3×3 core are always included; we then add
// the next-closest board cell (by game2's blended-distance function) one at a
// time until the pattern contains at least PAT_STONES stones or reaches
// MAX_PAT_SIZE cells.  The raw key is the byte sequence of cell values in
// grow order, prefixed by the size, canonicalised over D4 by taking the
// lex-min string across 8 σ-grows (each σ sees a rotated virtual-coord
// tiebreaker, so rotated boards produce the same canonical key).  Raw keys
// are STRINGS (not integers) so intern uses a Map<string, denseIdx>; they
// live outside any flat base-offset.
const MAX_PAT_SIZE       = 40;
const PAT_STONES         = (typeof process !== 'undefined' && process.env && process.env.NPAT_PAT_STONES)
  ? parseInt(process.env.NPAT_PAT_STONES, 10) : 1;

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
// patIdsA   [count]                        canonical Type A key per move
// patIdsB   [count]                        canonical Type B key per move
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

  // Precomputed distance-sorted growth order per (candidate, σ).  For each
  // candidate c and each D4 perm σ (indexed 0..7), growOrder[c * 8 *
  // MAX_PAT_SIZE + σ * MAX_PAT_SIZE + k] = the k-th board cell to add under
  // σ-growth.  Sort key is (distance(c, j), σ(j − c) lex) so distance drives
  // the growth and the virtual σ-coord breaks ties.  Rotating the board by σ'
  // swaps which σ produces which grown pattern, so canonicalising by lex-min
  // over the 8 resulting encodings yields a D4-invariant key.
  const growOrder = new Int32Array(cap * 8 * MAX_PAT_SIZE);
  const d4fns = [
    (dr, dc) => [ dr,  dc],
    (dr, dc) => [ dc, -dr],
    (dr, dc) => [-dr, -dc],
    (dr, dc) => [-dc,  dr],
    (dr, dc) => [ dr, -dc],
    (dr, dc) => [-dr,  dc],
    (dr, dc) => [ dc,  dr],
    (dr, dc) => [-dc, -dr],
  ];
  const halfN = N / 2;
  function _torDiff(a, b) {
    let d = a - b;
    if (d >  halfN) d -= N;
    if (d < -halfN) d += N;
    return d;
  }
  function _blendedDist(dr, dc) {
    const adr = Math.abs(dr), adc = Math.abs(dc);
    return 0.4 * (adr + adc) + 0.6 * Math.sqrt(adr * adr + adc * adc);
  }
  const entries = new Array(cap - 1);
  for (let c = 0; c < cap; c++) {
    const cr = (c / N) | 0, cc = c - cr * N;
    let ei = 0;
    for (let j = 0; j < cap; j++) {
      if (j === c) continue;
      const jr = (j / N) | 0, jc = j - jr * N;
      const dr = _torDiff(jr, cr);
      const dc = _torDiff(jc, cc);
      entries[ei++] = { j, dr, dc, dist: _blendedDist(dr, dc) };
    }
    for (let s = 0; s < 8; s++) {
      const perm = d4fns[s];
      for (let k = 0; k < entries.length; k++) {
        const [vr, vc] = perm(entries[k].dr, entries[k].dc);
        entries[k].vr = vr;
        entries[k].vc = vc;
      }
      entries.sort((a, b) => {
        if (a.dist !== b.dist) return a.dist - b.dist;
        if (a.vr   !== b.vr)   return a.vr   - b.vr;
        return a.vc - b.vc;
      });
      const baseK = c * 8 * MAX_PAT_SIZE + s * MAX_PAT_SIZE;
      const limit = Math.min(MAX_PAT_SIZE, entries.length);
      for (let k = 0; k < limit; k++) growOrder[baseK + k] = entries[k].j;
    }
  }

  // Precomputed octant-growth order per (candidate, σ).  Slot k at
  // octantOrder[c * 8 * MAX_PAT_SIZE + σ * MAX_PAT_SIZE + k] gives the k-th
  // cell in the σ-octant-grow sequence: slots 0..7 are the 8 core cells (same
  // as growOrder), slots 8.. are the cells in σ's octant (outside the core)
  // in distance order with σ-virtual-coord tiebreak.
  //
  // The canonical octant (for σ=0) is SSE: cells (dr, dc) with dr > 0 and
  // 0 ≤ dc ≤ dr — the closed 45° wedge between "directly south" (the dc=0
  // vertical) and the "SE" diagonal (dc=dr), inclusive of both boundary
  // rays.  The other 7 σs rotate / reflect this wedge onto the other 7
  // octants of the plane.
  const octantOrder = new Int32Array(cap * 8 * MAX_PAT_SIZE).fill(-1);
  const sigmaInv = [
    (dr, dc) => [ dr,  dc],
    (dr, dc) => [-dc,  dr],
    (dr, dc) => [-dr, -dc],
    (dr, dc) => [ dc, -dr],
    (dr, dc) => [ dr, -dc],
    (dr, dc) => [-dr,  dc],
    (dr, dc) => [ dc,  dr],
    (dr, dc) => [-dc, -dr],
  ];
  for (let c = 0; c < cap; c++) {
    for (let s = 0; s < 8; s++) {
      const baseK = c * 8 * MAX_PAT_SIZE + s * MAX_PAT_SIZE;
      // Slots 0..7: copy core from growOrder.
      for (let k = 0; k < 8; k++) octantOrder[baseK + k] = growOrder[baseK + k];
      // Slots 8..: cells in σ-octant outside the core.
      const cr = (c / N) | 0, cc = c - cr * N;
      const invPerm = sigmaInv[s];
      const perm    = d4fns[s];
      const out = [];
      for (let j = 0; j < cap; j++) {
        if (j === c) continue;
        const jr = (j / N) | 0, jc = j - jr * N;
        const dr = _torDiff(jr, cr);
        const dc = _torDiff(jc, cc);
        // Skip core cells.
        if (Math.abs(dr) <= 1 && Math.abs(dc) <= 1) continue;
        // σ-octant membership: apply σ⁻¹ and test canonical-SSE condition.
        const [pr, pc] = invPerm(dr, dc);
        if (!(pr > 0 && pc >= 0 && pc <= pr)) continue;
        const [vr, vc] = perm(dr, dc);
        out.push({ j, dist: _blendedDist(dr, dc), vr, vc });
      }
      out.sort((a, b) => {
        if (a.dist !== b.dist) return a.dist - b.dist;
        if (a.vr   !== b.vr)   return a.vr   - b.vr;
        return a.vc - b.vc;
      });
      const lim = Math.min(out.length, MAX_PAT_SIZE - 8);
      for (let k = 0; k < lim; k++) octantOrder[baseK + 8 + k] = out[k].j;
    }
  }

  // Precomputed per-candidate read sets: the union of all cells touched by
  // ANY σ-grow at MAX_PAT_SIZE, sorted by RELATIVE OFFSET (Δr, Δc) so the
  // hash of the mover-relative values is position-invariant (same local
  // pattern anywhere on the board → same hash → same cache entry).
  function _buildReadSet(orderArr) {
    const out = new Array(cap);
    const seen = new Uint8Array(cap);
    for (let c = 0; c < cap; c++) {
      seen.fill(0);
      const cr = (c / N) | 0, cc = c - cr * N;
      const list = [];
      for (let s = 0; s < 8; s++) {
        const baseK = c * 8 * MAX_PAT_SIZE + s * MAX_PAT_SIZE;
        for (let k = 0; k < MAX_PAT_SIZE; k++) {
          const bi = orderArr[baseK + k];
          if (bi < 0) break;
          if (!seen[bi]) {
            seen[bi] = 1;
            const br = (bi / N) | 0, bc = bi - br * N;
            const dr = _torDiff(br, cr);
            const dc = _torDiff(bc, cc);
            list.push({ bi, dr, dc });
          }
        }
      }
      list.sort((a, b) => a.dr !== b.dr ? a.dr - b.dr : a.dc - b.dc);
      const arr = new Int32Array(list.length);
      for (let i = 0; i < list.length; i++) arr[i] = list[i].bi;
      out[c] = arr;
    }
    return out;
  }
  const readSetG = _buildReadSet(growOrder);
  const readSetO = _buildReadSet(octantOrder);

  return {
    N,
    moves:     new Int32Array(cap),
    patIds33c: new Int32Array(cap),
    patIdsA:   new Int32Array(cap),
    patIdsB:   new Int32Array(cap),
    patIdsG:   new Int32Array(cap),
    patIdsO:   new Int32Array(cap),
    tact:      new Uint8Array(cap * N_TACT_SLOTS),
    logits:    new Float64Array(cap),
    probs:     new Float64Array(cap),
    ladder:    { tactCount: new Uint8Array(cap * N_TACT_SLOTS) },
    patchNbr,
    growOrder,
    octantOrder,
    readSetG,
    readSetO,
    // Reusable touched-index scratch for reinforceUpdate.  Upper bound is
    // 5 * (n + 1) dense idxs (one per shape type, chosen + all moves).
    touched:   new Int32Array(5 * (cap + 1)),
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
  let use33c = false, useA = false, useB = false, useG = false, useO = false;
  if (typeof opts === 'number') {
    initialCapacity = opts;
  } else if (opts && typeof opts === 'object') {
    if (opts.initialCapacity) initialCapacity = opts.initialCapacity;
    if (opts.use33c) use33c = true;
    if (opts.useA)   useA   = true;
    if (opts.useB)   useB   = true;
    if (opts.useG)   useG   = true;
    if (opts.useO)   useO   = true;
  }
  const w = {
    map:   new Map(),                              // raw canonKey → dense idx
    vals:  new Float64Array(initialCapacity),      // weights[dense idx]
    delta: new Float64Array(initialCapacity),      // reusable reinforce buffer
    size:  0,                                      // next dense idx to assign
    tactIds: new Int32Array(N_TACT_SLOTS),
    cfg:   { use33c, useA, useB, useG, useO },     // feature gating
  };
  for (let k = 0; k < N_TACT_SLOTS; k++) {
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

// Type A shape (10 cells, hflip-symmetric).  Row-major layout with candidate
// at index 2.  All 8 D4 orbits are exercised even though the shape has a
// 2-fold internal symmetry — the orbit-minimum collapses the redundancy
// automatically.

const _AShape = [
                    [-1,  0],
  [ 0, -1], [ 0,  0], [ 0, +1],
  [+1, -1], [+1,  0], [+1, +1],
  [+2, -1], [+2,  0], [+2, +1],
];

const _APatchIdx    = new Int32Array(8 * TYPE_A_CELLS);
const _ACellWeights = new Int32Array(TYPE_A_CELLS);
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
    for (let i = 0; i < TYPE_A_CELLS; i++) {
      const [r, c] = d4[s](_AShape[i][0], _AShape[i][1]);
      _APatchIdx[s * TYPE_A_CELLS + i] = (r + 2) * 5 + (c + 2);
    }
  }
  let w = 1;
  for (let i = 0; i < TYPE_A_CELLS; i++) { _ACellWeights[i] = w; w *= CELL_BASE; }
})();

const _canonKeyACache = new Int32Array(TYPE_A_BASE).fill(-1); // 236KB

function canonKeyA(patch) {
  const idx = _APatchIdx;
  const cw  = _ACellWeights;
  // s=0 identity raw = cache key.
  let cacheKey = 0;
  for (let i = 0; i < TYPE_A_CELLS; i++) cacheKey += patch[idx[i]] * cw[i];
  const cached = _canonKeyACache[cacheKey];
  if (cached !== -1) return cached;
  let best = cacheKey;
  for (let s = 1; s < 8; s++) {
    const o = s * TYPE_A_CELLS;
    let raw = 0;
    for (let i = 0; i < TYPE_A_CELLS; i++) raw += patch[idx[o + i]] * cw[i];
    if (raw < best) best = raw;
  }
  _canonKeyACache[cacheKey] = best;
  return best;
}

// Type B shape (11 cells, diag-symmetric).  Row-major layout with candidate
// at index 2.  Canonicalisation over all 8 D4 perms (stabiliser {id, diag}
// makes 4 of the 8 redundant; the orbit-minimum collapses them).

const _BShape = [
                    [-1,  0],
  [ 0, -1], [ 0,  0], [ 0, +1], [ 0, +2],
            [+1,  0], [+1, +1], [+1, +2],
            [+2,  0], [+2, +1], [+2, +2],
];

const _BPatchIdx    = new Int32Array(8 * TYPE_B_CELLS);
const _BCellWeights = new Int32Array(TYPE_B_CELLS);
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
    for (let i = 0; i < TYPE_B_CELLS; i++) {
      const [r, c] = d4[s](_BShape[i][0], _BShape[i][1]);
      _BPatchIdx[s * TYPE_B_CELLS + i] = (r + 2) * 5 + (c + 2);
    }
  }
  let w = 1;
  for (let i = 0; i < TYPE_B_CELLS; i++) { _BCellWeights[i] = w; w *= CELL_BASE; }
})();

const _canonKeyBCache = new Int32Array(TYPE_B_BASE).fill(-1); // 708KB

function canonKeyB(patch) {
  const idx = _BPatchIdx;
  const cw  = _BCellWeights;
  let cacheKey = 0;
  for (let i = 0; i < TYPE_B_CELLS; i++) cacheKey += patch[idx[i]] * cw[i];
  const cached = _canonKeyBCache[cacheKey];
  if (cached !== -1) return cached;
  let best = cacheKey;
  for (let s = 1; s < 8; s++) {
    const o = s * TYPE_B_CELLS;
    let raw = 0;
    for (let i = 0; i < TYPE_B_CELLS; i++) raw += patch[idx[o + i]] * cw[i];
    if (raw < best) best = raw;
  }
  _canonKeyBCache[cacheKey] = best;
  return best;
}

// Scratch byte buffer for grown-pattern serialization (size prefix + up to
// MAX_PAT_SIZE cells).
const _growBytes = new Uint8Array(MAX_PAT_SIZE + 1);

// ── Canonical-key caches keyed by 32-bit FNV-1a hash ─────────────────────────
//
// Each candidate's readSet (cells any σ-grow might touch, sorted by relative
// offset) is hashed to a 32-bit int via FNV-1a over mover-relative values.
// The cache maps that int to the canonical string — the cache NEVER clears,
// so memory grows with the working set of distinct local patterns.
// 32-bit hashes mean rare collisions (~1 per 1M entries by birthday); those
// return a wrong canonical for those few patterns, which we treat as noise
// in the gradient signal.
//
// V8 caps a single Map at ~16.7M entries, so we shard into 16 Maps keyed by
// the low 4 bits of the hash — total capacity ~268M entries.
const CACHE_SHARD_BITS = 4;
const CACHE_SHARD_COUNT = 1 << CACHE_SHARD_BITS;
const CACHE_SHARD_MASK  = CACHE_SHARD_COUNT - 1;
const _canonKeyGCaches = new Array(CACHE_SHARD_COUNT);
const _canonKeyOCaches = new Array(CACHE_SHARD_COUNT);
for (let i = 0; i < CACHE_SHARD_COUNT; i++) {
  _canonKeyGCaches[i] = new Map();
  _canonKeyOCaches[i] = new Map();
}

function _hashReadSet(readSet, cells, cur) {
  let h = 2166136261;  // FNV-1a offset basis
  const n = readSet.length;
  for (let i = 0; i < n; i++) {
    const ci = cells[readSet[i]];
    let v;
    if (ci === 0)        v = CELL_EMPTY;
    else if (ci === cur) v = CELL_FRIEND;
    else                 v = CELL_FOE;
    h = Math.imul(h ^ v, 16777619);  // FNV-1a prime; Math.imul keeps int32
  }
  return h | 0;
}
// Helper: grow the pattern under σ starting from a given precomputed order
// array, writing the bytes into out[1..size], with out[0] = size.  Returns
// the built string encoding.
function _growAndEncode(order, baseK, cells, cur, out) {
  let size = 0, stones = 0;
  // Core (8 cells).
  for (let k = 0; k < 8 && k < MAX_PAT_SIZE; k++) {
    const bi = order[baseK + k];
    if (bi < 0) break;
    const ci = cells[bi];
    let v;
    if (ci === 0)        v = CELL_EMPTY;
    else if (ci === cur) v = CELL_FRIEND;
    else                 v = CELL_FOE;
    out[size + 1] = v;
    if (v !== 0) stones++;
    size++;
  }
  // Grow.
  while (size < MAX_PAT_SIZE && stones < PAT_STONES) {
    const bi = order[baseK + size];
    if (bi < 0) break;
    const ci = cells[bi];
    let v;
    if (ci === 0)        v = CELL_EMPTY;
    else if (ci === cur) v = CELL_FRIEND;
    else                 v = CELL_FOE;
    out[size + 1] = v;
    if (v !== 0) stones++;
    size++;
  }
  out[0] = size;
  let str = '';
  for (let k = 0; k <= size; k++) str += String.fromCharCode(out[k]);
  return str;
}

// canonKeyG(game, candIdx, state, cur) — grow the pattern 8 times (one per
// D4 σ) and return the lex-min string encoding.  The 8 cells of the 3×3
// core (the candidate's neighbours; the candidate itself is always empty
// and carries no info) are ALWAYS included unconditionally.  Beyond the
// core, we add the next-closest board cell by blended distance (with
// σ-virtual (vr, vc) tiebreak) one at a time as long as the stone count is
// strictly less than PAT_STONES, up to MAX_PAT_SIZE total cells.
function canonKeyG(game, candIdx, state, cur) {
  const cells = game.cells;
  const h = _hashReadSet(state.readSetG[candIdx], cells, cur);
  const shard = _canonKeyGCaches[h & CACHE_SHARD_MASK];
  const cached = shard.get(h);
  if (cached !== undefined) return cached;
  const growOrd = state.growOrder;
  const bytes   = _growBytes;
  const strideC = 8 * MAX_PAT_SIZE;
  let best = null;
  for (let s = 0; s < 8; s++) {
    const str = _growAndEncode(growOrd, candIdx * strideC + s * MAX_PAT_SIZE,
      cells, cur, bytes);
    if (best === null || str < best) best = str;
  }
  shard.set(h, best);
  return best;
}

// canonKeyO(game, candIdx, state, cur) — 3×3 core + one auto-expanding octant.
// Like canonKeyG but growth is restricted to cells in the σ-octant (see
// createState for the octant definition).  The 8 D4 variants rotate the
// octant through all 8 directions; taking lex-min across the encodings
// gives a D4-invariant canonical key.
function canonKeyO(game, candIdx, state, cur) {
  const cells = game.cells;
  const h = _hashReadSet(state.readSetO[candIdx], cells, cur);
  const shard = _canonKeyOCaches[h & CACHE_SHARD_MASK];
  const cached = shard.get(h);
  if (cached !== undefined) return cached;
  const octOrd  = state.octantOrder;
  const bytes   = _growBytes;
  const strideC = 8 * MAX_PAT_SIZE;
  let best = null;
  for (let s = 0; s < 8; s++) {
    const str = _growAndEncode(octOrd, candIdx * strideC + s * MAX_PAT_SIZE,
      cells, cur, bytes);
    if (best === null || str < best) best = str;
  }
  shard.set(h, best);
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
  const tact     = state.tact;
  const patchNbr = state.patchNbr;
  const wc9      = _windowCells;
  const patch    = _patch5;
  const wMap     = weights ? weights.map : null;
  // Feature-extraction gates: when a weights store with cfg is supplied, skip
  // shape types that are disabled.  Without weights (tests extracting raw keys
  // only), extract everything for backward compatibility.
  const cfg      = weights ? weights.cfg : null;
  const doUse33c = !cfg || cfg.use33c;
  const doUseA   = !cfg || cfg.useA;
  const doUseB   = !cfg || cfg.useB;
  const doUseG   = !cfg || cfg.useG;
  const doUseO   = !cfg || cfg.useO;
  const patIds33c = state.patIds33c;
  const patIdsA   = state.patIdsA;
  const patIdsB   = state.patIdsB;
  const patIdsG   = state.patIdsG;
  const patIdsO   = state.patIdsO;

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

    // Type A shape (10 cells, hflip-symmetric).  canonKeyA reads cells
    // directly from the 5×5 patch at 8 precomputed D4 orientations.
    if (doUseA) {
      const rawA = TYPE_A_RAW_BASE + canonKeyA(patch);
      patIdsA[count] = wMap ? _internWeight(weights, rawA) : rawA;
    }

    // Type B shape (11 cells, diag-symmetric).  canonKeyB likewise.
    if (doUseB) {
      const rawB = TYPE_B_RAW_BASE + canonKeyB(patch);
      patIdsB[count] = wMap ? _internWeight(weights, rawB) : rawB;
    }

    // Grown shape — single canonical string key per candidate.  Raw key is
    // a string so it bypasses the integer raw-id ranges used by the other
    // shapes; interning still goes through the same weights.map.
    if (doUseG) {
      const rawG = canonKeyG(game, idx, state, cur);
      patIdsG[count] = wMap ? _internWeight(weights, rawG) : -1;
    }

    // Octant-grown shape — 3×3 core + auto-expanding into one of 8 D4-
    // canonicalised octants until patStones are accumulated (or the octant
    // cells are exhausted).
    if (doUseO) {
      const rawO = canonKeyO(game, idx, state, cur);
      patIdsO[count] = wMap ? _internWeight(weights, rawO) : -1;
    }

    // Copy tactical counts for this candidate (N_TACT_SLOTS bytes = 4 types
    // × TACT_STONE_LIMIT stone-indices).
    const tSrc = idx * N_TACT_SLOTS;
    const tDst = count * N_TACT_SLOTS;
    for (let k = 0; k < N_TACT_SLOTS; k++) tact[tDst + k] = tactCount[tSrc + k];

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
  const tOff = i * N_TACT_SLOTS;
  let s = 0;
  for (let k = 0; k < N_TACT_SLOTS; k++) s += tact[tOff + k] * vals[tIds[k]];
  if (cfg.use33c) s += vals[state.patIds33c[i]];
  if (cfg.useA)   s += vals[state.patIdsA[i]];
  if (cfg.useB)   s += vals[state.patIdsB[i]];
  if (cfg.useG)   s += vals[state.patIdsG[i]];
  if (cfg.useO)   s += vals[state.patIdsO[i]];
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
  const tW    = new Float64Array(N_TACT_SLOTS);
  for (let k = 0; k < N_TACT_SLOTS; k++) tW[k] = vals[tIds[k]];
  const pid33c = cfg.use33c ? state.patIds33c : null;
  const pidA   = cfg.useA   ? state.patIdsA   : null;
  const pidB   = cfg.useB   ? state.patIdsB   : null;
  const pidG   = cfg.useG   ? state.patIdsG   : null;
  const pidO   = cfg.useO   ? state.patIdsO   : null;

  let maxL = -Infinity;
  for (let i = 0; i < n; i++) {
    const tOff = i * N_TACT_SLOTS;
    let s = 0;
    for (let k = 0; k < N_TACT_SLOTS; k++) s += tact[tOff + k] * tW[k];
    if (pid33c) s += vals[pid33c[i]];
    if (pidA)   s += vals[pidA[i]];
    if (pidB)   s += vals[pidB[i]];
    if (pidG)   s += vals[pidG[i]];
    if (pidO)   s += vals[pidO[i]];
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
  const patIdsA   = cfg.useA   ? state.patIdsA   : null;
  const patIdsB   = cfg.useB   ? state.patIdsB   : null;
  const patIdsG   = cfg.useG   ? state.patIdsG   : null;
  const patIdsO   = cfg.useO   ? state.patIdsO   : null;
  let tc = 0;

  // ── Shape features (use delta buffer to dedupe repeats across moves) ──
  // Chosen move contributes +step to each of its active shape pids.
  if (patIds33c) {
    const idx = patIds33c[chosenIndex];
    touched[tc++] = idx;
    delta[idx] += step;
  }
  if (patIdsA) {
    const idx = patIdsA[chosenIndex];
    touched[tc++] = idx;
    delta[idx] += step;
  }
  if (patIdsB) {
    const idx = patIdsB[chosenIndex];
    touched[tc++] = idx;
    delta[idx] += step;
  }
  if (patIdsG) {
    const idx = patIdsG[chosenIndex];
    touched[tc++] = idx;
    delta[idx] += step;
  }
  if (patIdsO) {
    const idx = patIdsO[chosenIndex];
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
    if (patIdsA) {
      const idx = patIdsA[i];
      touched[tc++] = idx;
      delta[idx] -= sub;
    }
    if (patIdsB) {
      const idx = patIdsB[i];
      touched[tc++] = idx;
      delta[idx] -= sub;
    }
    if (patIdsG) {
      const idx = patIdsG[i];
      touched[tc++] = idx;
      delta[idx] -= sub;
    }
    if (patIdsO) {
      const idx = patIdsO[i];
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
  // Return total shape-feature touches (including duplicates across moves) —
  // train-npat.js accumulates these to report mean updates-per-pattern.
  return tc + N_TACT_SLOTS;
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
  const patIdsA   = cfg.useA   ? state.patIdsA   : null;
  const patIdsB   = cfg.useB   ? state.patIdsB   : null;
  const patIdsG   = cfg.useG   ? state.patIdsG   : null;
  const patIdsO   = cfg.useO   ? state.patIdsO   : null;
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
    if (patIdsA) {
      const idx = patIdsA[i];
      touched[tc++] = idx;
      delta[idx] += ci;
    }
    if (patIdsB) {
      const idx = patIdsB[i];
      touched[tc++] = idx;
      delta[idx] += ci;
    }
    if (patIdsG) {
      const idx = patIdsG[i];
      touched[tc++] = idx;
      delta[idx] += ci;
    }
    if (patIdsO) {
      const idx = patIdsO[i];
      touched[tc++] = idx;
      delta[idx] += ci;
    }
    const tOff = i * N_TACT_SLOTS;
    for (let k = 0; k < N_TACT_SLOTS; k++) acc[k] += ci * tact[tOff + k];
  }
  for (let i = 0; i < tc; i++) {
    const idx = touched[i];
    const d = delta[idx];
    if (d !== 0) { vals[idx] += d; delta[idx] = 0; }
  }
  for (let k = 0; k < N_TACT_SLOTS; k++) {
    if (acc[k] !== 0) vals[tIds[k]] += acc[k];
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
  canonKeyA,
  canonKeyB,
  canonKeyG,
  canonKeyO,
  // Constants.
  CELL_BASE,
  CELLS_BASE,
  CELL_EMPTY, CELL_FRIEND, CELL_FOE,
  N_TACT, TACT_STONE_LIMIT, N_TACT_SLOTS,
  TACT_URGENT_KILL, TACT_URGENT_SAVE,
  TACT_WASTED_EXTEND, TACT_WASTED_ATTACK,
  TACT_RAW_BASE,
  SHAPE33C_RAW_BASE,
  TYPE_A_CELLS, TYPE_A_BASE, TYPE_A_RAW_BASE,
  TYPE_B_CELLS, TYPE_B_BASE, TYPE_B_RAW_BASE,
  MAX_PAT_SIZE, PAT_STONES,
  // Exposed for tests.
  _D4,
  _AShape,
  _BShape,
};

if (typeof module !== 'undefined') module.exports = NPatterns;
else window.NPatterns = NPatterns;

})();
