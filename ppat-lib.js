'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
const { PASS, isEyelike } = _isNode ? require('./game2.js') : window.Game2;
const Util = _isNode ? require('./util.js') : window.Util;

// ── D4 position permutations ──────────────────────────────────────────────────
// Positions 0–7: N=0, E=1, S=2, W=3, NE=4, SE=5, SW=6, NW=7
// (dr,dc): N=(−1,0), E=(0,+1), S=(+1,0), W=(0,−1),
//           NE=(−1,+1), SE=(+1,+1), SW=(+1,−1), NW=(−1,−1)
// perm[src]=dst: value at src in original goes to dst in transformed.
const _D4 = [
  [0,1,2,3,4,5,6,7],  // Identity
  [1,2,3,0,5,6,7,4],  // Rot90CW      (dr,dc)→(dc,−dr)
  [2,3,0,1,6,7,4,5],  // Rot180       (dr,dc)→(−dr,−dc)
  [3,0,1,2,7,4,5,6],  // Rot270CW     (dr,dc)→(−dc,dr)
  [0,3,2,1,7,6,5,4],  // FlipH        (dr,dc)→(dr,−dc)
  [2,1,0,3,5,4,7,6],  // FlipV        (dr,dc)→(−dr,dc)
  [3,2,1,0,6,5,4,7],  // TransposeMD  (dr,dc)→(dc,dr)
  [1,0,3,2,4,7,6,5],  // TransposeAD  (dr,dc)→(−dc,−dr)
];

// ── Load-time canonicalisation ────────────────────────────────────────────────
//
// Raw pattern index (positions in encoding order: N, E, S, W, NE, SE, SW, NW),
// with R = 2*adjLib + 1 the orthogonal radix and 3 the diagonal radix:
//   rawIdx = vN + R*(vE + R*(vS + R*(vW + R*(vNE + 3*(vSE + 3*(vSW + 3*vNW))))))
// Range: [0, R^4 × 3^4).  adjLib 2 (R = 5, 50625 configurations) is the
// historical encoding and the SB paper's: an orthogonal is atari or not.
// Higher caps resolve more liberty levels — the cap is a property of a trained
// model and travels in its weights file as `adjLib`; files without the field
// are cap 2.  MUST stay bit-identical to c/ppat.c (encode8 / adj_val).
//
// The encoding is already relative to the current mover (FRIEND/FOE), so color
// swap is NOT a symmetry.  Only the 8 D4 spatial transforms are applied.
// canonId maps raw → dense canonical ID (0-based); Int32Array because cap 4
// canonicalises to ~71k patterns, past Int16's range.

const MIN_ADJ_LIB = 1, MAX_ADJ_LIB = 4;   // cap 1 = presence-only (pure shape)
const _tablesByCap = new Map();   // adjLib → { canonId, numPatterns, rawSize }

function _buildTables(adjLib) {
  if (!(adjLib >= MIN_ADJ_LIB && adjLib <= MAX_ADJ_LIB))
    throw new Error(`ppat: adjLib ${adjLib} out of range [${MIN_ADJ_LIB},${MAX_ADJ_LIB}]`);
  const cached = _tablesByCap.get(adjLib);
  if (cached) return cached;

  const R = 2 * adjLib + 1;
  const rawSize = R * R * R * R * 81;
  const canonId = new Int32Array(rawSize);
  const v  = new Int32Array(8);
  const tv = new Int32Array(8);
  const idMap = new Map(); // minVariant rawIdx → assigned canonId
  let nextId = 0;

  for (let raw = 0; raw < rawSize; raw++) {
    // Decode: positions 0-3 have base R, positions 4-7 have base 3.
    let r = raw;
    v[0] = r % R; r = (r / R) | 0;
    v[1] = r % R; r = (r / R) | 0;
    v[2] = r % R; r = (r / R) | 0;
    v[3] = r % R; r = (r / R) | 0;
    v[4] = r % 3; r = (r / 3) | 0;
    v[5] = r % 3; r = (r / 3) | 0;
    v[6] = r % 3;
    v[7] = (r / 3) | 0;

    let minV = raw;

    for (let di = 0; di < 8; di++) {
      const p = _D4[di];
      for (let i = 0; i < 8; i++) tv[p[i]] = v[i];
      const enc = tv[0] + R*(tv[1] + R*(tv[2] + R*(tv[3] + R*(tv[4] + 3*(tv[5] + 3*(tv[6] + 3*tv[7]))))));
      if (enc < minV) minV = enc;
    }

    if (!idMap.has(minV)) idMap.set(minV, nextId++);
    canonId[raw] = idMap.get(minV);
  }
  const t = { canonId, numPatterns: nextId, rawSize, R, adjLib };
  _tablesByCap.set(adjLib, t);
  return t;
}

// Default tables (cap 2) — the historical encoding, built eagerly so callers that
// never load a model (tests, counters) work unchanged.
const _T2 = _buildTables(2);
const NUM_PATTERNS = _T2.numPatterns;

// ── Twelvecell extension ──────────────────────────────────────────────────────
// When a candidate's NINECELL is entirely empty the pattern feature cannot tell
// one open-area move from another: every such candidate shares a single weight,
// so the policy picks among them uniformly.  The extension emits a SECOND key
// describing the four DISTANCE-2 orthogonals — the twelvecell's arms — coded
// empty / mine / theirs.
//
// It STACKS on the ninecell rather than replacing it: the pattern key is still
// emitted, so an existing model fine-tunes into the extension with its weights
// untouched and the new keys starting at zero.
//
// TWO mutually exclusive triggers, both sharing this one 21-weight block:
//   mode 1 (--twelvecell)   the whole NINECELL is empty — orthogonals and
//                           diagonals alike.
//   mode 2 (--twelvecell2)  the four ADJACENT points are empty, whatever the
//                           diagonals hold.  Fires strictly more often.
//
// Each mode canonicalises EXACTLY, over whatever cells its trigger leaves
// informative — cells sharing one spatial window are canonicalised together,
// never independently, or the key loses their relative arrangement:
//   mode 1: the inner ninecell is all-empty, hence fixed by every element of
//           D4, so the four ARMS alone are exact.  81 raw -> 21 orbits.
//   mode 2: the four orthogonals are empty and carry nothing, but the DIAGONALS
//           may be occupied, so diagonals and arms are canonicalised JOINTLY —
//           one enemy diagonal beside an enemy arm is a connected shape, the
//           same two stones opposite each other are unrelated, and an
//           independently-canonicalised arm key would collide them.
//           3^8 = 6561 raw -> 954 orbits (the same structure as the health
//           ninecell: eight cells in two D4-orbits of four).
const T12_RAW = 81;
const T12B_RAW = 6561;
const _T12 = (() => {
  // Arm order N, E, S, W — the ninecell's D4 restricted to the orthogonals.
  const ROT = [1, 2, 3, 0];   // 90 degrees: N->E, E->S, S->W, W->N
  const REF = [0, 3, 2, 1];   // mirror: E<->W
  const ap = (p, q) => q.map(i => p[i]);
  const perms = [];
  let cur = [0, 1, 2, 3];
  for (let r = 0; r < 4; r++) { perms.push(cur.slice()); perms.push(ap(cur, REF)); cur = ap(cur, ROT); }
  const canonId = new Int32Array(T12_RAW);
  const v = new Int32Array(4), tv = new Int32Array(4);
  const idMap = new Map();
  let nextId = 0;
  for (let raw = 0; raw < T12_RAW; raw++) {
    let r = raw;
    for (let i = 0; i < 4; i++) { v[i] = r % 3; r = (r / 3) | 0; }
    let minV = raw;
    for (const p of perms) {
      for (let i = 0; i < 4; i++) tv[p[i]] = v[i];
      const enc = tv[0] + 3 * (tv[1] + 3 * (tv[2] + 3 * tv[3]));
      if (enc < minV) minV = enc;
    }
    if (!idMap.has(minV)) idMap.set(minV, nextId++);
    canonId[raw] = idMap.get(minV);
  }
  return { canonId, numPatterns: nextId };
})();
const NUM_T12 = _T12.numPatterns;

// Mode 2's joint table: positions 0-3 are the DIAGONALS (NE, SE, SW, NW) and
// 4-7 the ARMS (N, E, S, W), so one D4 element permutes both sets at once.
const _T12B = (() => {
  // Diagonal quarter-turn NE->SE->SW->NW; arm quarter-turn N->E->S->W.
  const ROT = [1, 2, 3, 0, 5, 6, 7, 4];
  // Mirror about the N-S axis: NE<->NW, SE<->SW, E<->W, N and S fixed.
  const REF = [3, 2, 1, 0, 4, 7, 6, 5];
  const ap = (p, q) => q.map(i => p[i]);
  const perms = [];
  let cur = [0, 1, 2, 3, 4, 5, 6, 7];
  for (let r = 0; r < 4; r++) { perms.push(cur.slice()); perms.push(ap(cur, REF)); cur = ap(cur, ROT); }
  const canonId = new Int32Array(T12B_RAW);
  const v = new Int32Array(8), tv = new Int32Array(8);
  const idMap = new Map();
  let nextId = 0;
  for (let raw = 0; raw < T12B_RAW; raw++) {
    let r = raw;
    for (let i = 0; i < 8; i++) { v[i] = r % 3; r = (r / 3) | 0; }
    let minV = raw;
    for (const p of perms) {
      for (let i = 0; i < 8; i++) tv[p[i]] = v[i];
      let enc = 0;
      for (let i = 7; i >= 0; i--) enc = enc * 3 + tv[i];
      if (enc < minV) minV = enc;
    }
    if (!idMap.has(minV)) idMap.set(minV, nextId++);
    canonId[raw] = idMap.get(minV);
  }
  return { canonId, numPatterns: nextId };
})();
const NUM_T12B = _T12B.numPatterns;
// Weights the twelvecell block needs for a given mode.
function t12Block(mode) { return mode === 1 ? NUM_T12 : mode === 2 ? NUM_T12B : 0; }

// ── Private helpers ───────────────────────────────────────────────────────────

// Iterate the liberty bitset of gid; return the first liberty index, or -1.
// Caller must ensure lsArr[gid] >= 1.
function _firstLib(gid, lw, W, cap) {
  const lb = gid * W;
  for (let wi = 0; wi < W; wi++) {
    const w = lw[lb + wi];
    if (w) { const i = wi * 32 + (31 - Math.clz32(w & -w)); if (i < cap) return i; }
  }
  return -1;
}

// Returns true if playing at idx would capture an enemy group that is adjacent to
// any of the friendly groups in new1LibGids, saving them from atari.
function _canSaveByCapture(idx, new1LibGids, nbr, cells, gidArr, lsArr, lw, sw, W, cap, foe) {
  for (const sgid of new1LibGids) {
    // Walk all stones of the atari'd string.
    const sb = sgid * W;
    for (let wi = 0; wi < W; wi++) {
      let w = sw[sb + wi];
      while (w) {
        const lsb = w & -w;
        const si = wi * 32 + (31 - Math.clz32(lsb));
        if (si < cap) {
          const b4 = si * 4;
          for (let di = 0; di < 4; di++) {
            const ni = nbr[b4 + di];
            if (cells[ni] !== foe) continue;
            const egid = gidArr[ni];
            // Enemy group with 1 liberty == idx? Capturing it saves our string.
            if (lsArr[egid] === 1 && (lw[egid * W + (idx >> 5)] & (1 << (idx & 31)))) return true;
          }
        }
        w ^= lsb;
      }
    }
  }
  return false;
}




// Quick self-atari pre-check.  Returns true if we can guarantee the move at idx
// is NOT self-atari, without simulating the move.
// Two types of guaranteed liberties:
//   • empty orthogonal neighbor (stays empty after the move)
//   • enemy neighbor with exactly 1 liberty == idx (will be captured, cell freed)
// Also: if any adjacent friendly group has ≥3 liberties, connecting to it still
// leaves ≥2 after idx is consumed from its liberty set.
// If this returns true, skip the expensive clone; otherwise fall through to clone.

// ── Static buffers (avoid per-call allocation / GC pressure) ─────────────────
const _prevN8       = new Int32Array(8);
const _atariGids    = new Int32Array(8);
const _atariLibsArr = new Int32Array(8);
let _sbcCells       = new Int32Array(64);   // save-by-capture cell indices (grown to cap)
const _koSolveLibs  = new Int32Array(4);
const _seenBuf      = new Int32Array(16);   // dedup scratch

// ── Public API ────────────────────────────────────────────────────────────────

// Allocate reusable output buffers for a board of size N.
function createState(N) {
  const cap = N * N;
  return {
    moves:          new Int32Array(cap),
    feat:           new Int32Array(cap * 10),  // flat feature keys (1 pat + 7 prev + 1 twelvecell + 1 self-atari)
    featStart:      new Int32Array(cap + 1),  // featStart[i]..featStart[i+1] = keys for candidate i
    count:          0,
  };
}

// The twelvecell block is APPENDED after the pattern and local blocks, so every
// pre-extension weight index keeps its meaning and an old file loads unchanged.
// The twelvecell block's size depends on the MODE (21 arms-only vs 954 joint),
// so callers pass the mode rather than a boolean.
// Graded self-atari (the C twin's PPAT_SA_N): when a legal candidate would
// leave its own group in atari, one gated key fires, one-hot on
// min(merged size, SA_N) — size-1 self-atari (throw-ins, snapbacks) is often
// correct while large is almost always a blunder, so the grades let training
// find the sign flip.  Appended after the twelvecell block.
const SA_N = 4;
let _saLib = null;

// Cheap bound: true = provably NOT self-atari (>= 2 liberties after placing).
function _notSelfAtariCheap(game, idx, cur) {
  const cells = game.cells, nbr = game._nbr, gid = game._gid, ls = game._ls,
        lw = game._lw, W = game._W;
  let free = 0;
  const wi = idx >> 5, m = 1 << (idx & 31), b4 = idx * 4;
  for (let d = 0; d < 4; d++) {
    const ni = nbr[b4 + d], c = cells[ni];
    if (c === 0) { if (++free >= 2) return true; }
    else if (c === cur) { if (ls[gid[ni]] >= 3) return true; }
    else {
      const eg = gid[ni];
      if (ls[eg] === 1 && (lw[eg * W + wi] & m) !== 0) { if (++free >= 2) return true; }
    }
  }
  return false;
}

// Exact self-atari size for the LEGAL candidate idx: 0 when the placed group
// would keep >= 2 liberties, else its merged stone count.  Mirrors the C
// twin: OR the joined chains' liberty bitsets and credit capture-freed
// points, so snapbacks label correctly as size-1 self-atari.
function _selfAtariSize(game, idx, cur) {
  if (_notSelfAtariCheap(game, idx, cur)) return 0;
  const cells = game.cells, nbr = game._nbr, gid = game._gid, ls = game._ls,
        lw = game._lw, sw = game._sw, ss = game._ss, W = game._W;
  if (!_saLib || _saLib.length < W) _saLib = new Int32Array(W);
  else _saLib.fill(0, 0, W);
  const lib = _saLib, b4 = idx * 4;
  const fr = [], capg = [];
  for (let d = 0; d < 4; d++) {
    const ni = nbr[b4 + d], c = cells[ni];
    if (c === 0) { lib[ni >> 5] |= 1 << (ni & 31); continue; }
    const g = gid[ni];
    if (c === cur) { if (!fr.includes(g)) fr.push(g); }
    else if (ls[g] === 1) {
      // adjacent enemy in atari: its lone liberty is idx, so it dies
      if (!capg.includes(g)) capg.push(g);
    }
  }
  let size = 1;
  for (let k = 0; k < fr.length; k++) {
    const base = fr[k] * W;
    for (let w = 0; w < W; w++) lib[w] |= lw[base + w];
    size += ss[fr[k]];
  }
  lib[idx >> 5] &= ~(1 << (idx & 31));
  // Capture-freed points: a captured stone is a liberty of the merged group
  // iff adjacent to it (the placed stone or a joined chain).
  for (let k = 0; k < capg.length; k++) {
    const base = capg[k] * W;
    for (let w = 0; w < W; w++) {
      let bits = sw[base + w];
      while (bits !== 0) {
        const p = (w << 5) + (31 - Math.clz32(bits & -bits));
        bits &= bits - 1;
        for (let d = 0; d < 4; d++) {
          const np = nbr[p * 4 + d];
          if (np === idx || (cells[np] === cur && fr.includes(gid[np]))) {
            lib[p >> 5] |= 1 << (p & 31);
            break;
          }
        }
      }
    }
  }
  let libs = 0;
  for (let w = 0; w < W && libs < 2; w++) {
    let bits = lib[w];
    while (bits !== 0 && libs < 2) { bits &= bits - 1; libs++; }
  }
  return libs >= 2 ? 0 : size;
}

function totalWeights(phaseCount, adjLib = 2, t12mode = 0, selfAtari = false, atariN = 0) {
  return phaseCount * (_buildTables(adjLib).numPatterns + 7) +
         phaseCount * t12Block(t12mode | 0) +
         (selfAtari ? phaseCount * SA_N : 0) +
         phaseCount * (atariN | 0);
}

// Extract features for all legal non-true-eye moves from game into state.
//
// state.moves[i]:     flat board index of move i
// state.patIds[i]:    canonical 3×3 pattern ID in [0, NUM_PATTERNS)
// state.prevMasks[i]: bitmask of active previous-move features:
//   bit 0 — Feature 1: in 8-neighborhood of previous move
//   bit 1 — Feature 2: save string in new atari by capture (not self-atari)
//   bit 2 — Feature 3: save string in new atari by capture (is self-atari)
//   bit 3 — Feature 4: save string in new atari by extension (not self-atari)
//   bit 4 — Feature 5: save string in new atari by extension (is self-atari)
//   bit 5 — Feature 6: solve a new ko by capturing
//   bit 6 — Feature 7: 2-point semeai (give atari to adjacent enemy)
//
// skipLocal skips features 1-7 entirely (pre-scans, per-candidate mask, emit);
// only the pattern feature is extracted.  Exactly equivalent for a model whose
// local weights are all zero: scores are plain sums, so a zero weight
// contributes nothing.
function extractFeatures(game, state, phaseCount = 1, adjLib = 2, skipLocal = false, t12mode = 0, selfAtari = false, atariN = 0) {
  const N      = game.N;
  const cap    = N * N;
  // Per-cap canonical tables.  _T2 is the common case (historical encoding).
  const _tab   = adjLib === 2 ? _T2 : _buildTables(adjLib);
  const _CANON = _tab.canonId, _NPAT = _tab.numPatterns, _R = _tab.R;
  const _LC    = adjLib;
  if (_sbcCells.length < cap) {
    _sbcCells = new Int32Array(cap);
  }
  const cells  = game.cells;
  const gidArr = game._gid;
  const lsArr  = game._ls;
  const lwArr  = game._lw;
  const swArr  = game._sw;
  const ssArr  = game._ss;
  const W      = game._W;

  const phase = phaseCount * (cap - game.emptyCount) / cap | 0;
  const patOffset = phase * _NPAT;
  const prevOffset = phaseCount * _NPAT + phase * 7;
  const _T12C = t12mode === 2 ? _T12B.canonId : _T12.canonId;
  const t12Offset = phaseCount * (_NPAT + 7) + phase * (t12mode === 2 ? NUM_T12B : NUM_T12);
  const saOffset  = phaseCount * (_NPAT + 7 + t12Block(t12mode)) + phase * SA_N;
  const atOffset  = phaseCount * (_NPAT + 7 + t12Block(t12mode) + (selfAtari ? SA_N : 0)) + phase * atariN;
  const nbr    = game._nbr;
  const dnbr   = game._dnbr;
  const cur    = game.current;
  const foe    = -cur;
  const emC    = game._emptyCells;
  const ec     = game.emptyCount;
  const prev   = game.lastMove;
  const hasPrev = !skipLocal && prev !== PASS;
  const myKoStone = game.koStone[cur + 1];

  // ── Pre-scan (mirrors c/ppat.c): friendly strings put in atari by prev,
  // their save-by-capture cells, prev's 8-neighborhood, and ko-solve libs. ──
  const atariGids = _atariGids;     // reuse static arrays (no GC)
  const atariLibsArr = _atariLibsArr;
  let nAtari = 0;
  const prevN8 = _prevN8;
  let nPrevN8 = 0;

  if (hasPrev) {
    const pb4 = prev * 4;
    for (let di = 0; di < 4; di++) {
      prevN8[nPrevN8++] = nbr[pb4 + di];
      prevN8[nPrevN8++] = dnbr[pb4 + di];
      const ni = nbr[pb4 + di];
      if (cells[ni] !== cur) continue;
      const gid = gidArr[ni];
      if (lsArr[gid] === 1) {
        let dup = false;
        for (let j = 0; j < nAtari; j++) if (atariGids[j] === gid) { dup = true; break; }
        if (!dup) atariGids[nAtari++] = gid;
      }
    }
    for (let i = 0; i < nAtari; i++)
      atariLibsArr[i] = _firstLib(atariGids[i], lwArr, W, cap);
  }

  // Precompute save-by-capture cells.
  let nSbc = 0;
  if (nAtari > 0) {
    const seen = _seenBuf;
    let nSeen = 0;
    for (let ai = 0; ai < nAtari; ai++) {
      const sgid = atariGids[ai];
      const sb = sgid * W;
      for (let wi = 0; wi < W; wi++) {
        let w = swArr[sb + wi];
        while (w) {
          const lsb = w & -w;
          const si = wi * 32 + (31 - Math.clz32(lsb));
          if (si < cap) {
            const b4s = si * 4;
            for (let d = 0; d < 4; d++) {
              const ni = nbr[b4s + d];
              if (cells[ni] !== foe) continue;
              const egid = gidArr[ni];
              if (lsArr[egid] !== 1) continue;
              let dup = false;
              for (let j = 0; j < nSeen; j++) if (seen[j] === egid) { dup = true; break; }
              if (dup) continue;
              if (nSeen < 16) seen[nSeen++] = egid;
              const lib = _firstLib(egid, lwArr, W, cap);
              if (lib >= 0) _sbcCells[nSbc++] = lib;
            }
          }
          w ^= lsb;
        }
      }
    }
  }

  // Feature 6 pre-scan
  let nKoSolve = 0;
  if (!skipLocal && myKoStone !== PASS) {
    const ks4 = myKoStone * 4;
    for (let d = 0; d < 4; d++) {
      const ni = nbr[ks4 + d];
      if (cells[ni] !== foe) continue;
      const egid = gidArr[ni];
      if (lsArr[egid] === 1) {
        const lib = _firstLib(egid, lwArr, W, cap);
        if (lib >= 0) _koSolveLibs[nKoSolve++] = lib;
      }
    }
  }

  const ko = game.ko;
  let count = 0;
  let nf = 0;

  for (let ei = 0; ei < ec; ei++) {
    const idx = emC[ei];
    const b4 = idx * 4;

    // Inlined legality check
    const niN = nbr[b4], niS = nbr[b4 + 1], niW = nbr[b4 + 2], niE = nbr[b4 + 3];
    const cN = cells[niN], cS = cells[niS], cW = cells[niW], cE = cells[niE];
    const anyEmpty = (cN === 0) | (cS === 0) | (cW === 0) | (cE === 0);
    if (anyEmpty) {
      if (idx === ko && game._isKo(idx, cur)) continue;
    } else {
      if (game._isSingleSuicide(idx, cur)) continue;
      if (game._isMultiSuicide(idx, cur)) continue;
      if (idx === ko && game._isKo(idx, cur)) continue;
    }

    // Inlined true-eye check + adj_val computation
    let friendCount = 0, emptyNbr = 0, firstGid = -2, sameGroup = 0;
    let vN, vS2, vW2, vE2;
    if (cN === 0) { emptyNbr++; vN = 0; }
    else { const g_ = gidArr[niN]; let l_ = lsArr[g_]; if (l_ > _LC) l_ = _LC; const s_ = _LC + 1 - l_;
      if (cN === cur) { friendCount++; if (firstGid === -2) { firstGid = g_; sameGroup = 1; } else if (g_ === firstGid) sameGroup++; vN = s_; }
      else vN = _LC + s_; }
    if (cS === 0) { emptyNbr++; vS2 = 0; }
    else { const g_ = gidArr[niS]; let l_ = lsArr[g_]; if (l_ > _LC) l_ = _LC; const s_ = _LC + 1 - l_;
      if (cS === cur) { friendCount++; if (firstGid === -2) { firstGid = g_; sameGroup = 1; } else if (g_ === firstGid) sameGroup++; vS2 = s_; }
      else vS2 = _LC + s_; }
    if (cW === 0) { emptyNbr++; vW2 = 0; }
    else { const g_ = gidArr[niW]; let l_ = lsArr[g_]; if (l_ > _LC) l_ = _LC; const s_ = _LC + 1 - l_;
      if (cW === cur) { friendCount++; if (firstGid === -2) { firstGid = g_; sameGroup = 1; } else if (g_ === firstGid) sameGroup++; vW2 = s_; }
      else vW2 = _LC + s_; }
    if (cE === 0) { emptyNbr++; vE2 = 0; }
    else { const g_ = gidArr[niE]; let l_ = lsArr[g_]; if (l_ > _LC) l_ = _LC; const s_ = _LC + 1 - l_;
      if (cE === cur) { friendCount++; if (firstGid === -2) { firstGid = g_; sameGroup = 1; } else if (g_ === firstGid) sameGroup++; vE2 = s_; }
      else vE2 = _LC + s_; }

    // Diag values.  Read BEFORE the eye check, which needs the hostile-diagonal
    // count: these are the same four reads the pattern index needs below, so
    // sharing them costs nothing but the handful of eye points that used to
    // skip out first.
    const cNE2 = cells[dnbr[b4 + 1]], vNE = cNE2 === 0 ? 0 : (cNE2 === cur ? 1 : 2);
    const cSE2 = cells[dnbr[b4 + 3]], vSE = cSE2 === 0 ? 0 : (cSE2 === cur ? 1 : 2);
    const cSW2 = cells[dnbr[b4 + 2]], vSW = cSW2 === 0 ? 0 : (cSW2 === cur ? 1 : 2);
    const cNW2 = cells[dnbr[b4]],     vNW = cNW2 === 0 ? 0 : (cNW2 === cur ? 1 : 2);

    // THE playout eye rule lives in game2.isEyelike; the counts above are
    // handed to it so it need not rescan.  Playouts prune MORE than a root
    // generator may: isEyelike adds the multi-chain wall with one hostile
    // diagonal, which isTrueEye leaves legal because it cannot prove it is never a
    // move.  This used to be an inlined copy that had drifted, which made the
    // playout fill multi-chain eyes and kill live groups.
    if (isEyelike(friendCount, emptyNbr, sameGroup,
                  (vNE === 2) + (vSE === 2) + (vSW === 2) + (vNW === 2))) continue;

    const rawIdx = vN + _R*(vE2 + _R*(vS2 + _R*(vW2 + _R*(vNE + 3*(vSE + 3*(vSW + 3*vNW))))));

    state.moves[count] = idx;
    state.featStart[count] = nf;

    // Pattern feature
    state.feat[nf++] = patOffset + _CANON[rawIdx];

    // Twelvecell extension.  Mode 1 needs the whole ninecell empty (rawIdx 0 —
    // every cell codes 0 when empty); mode 2 only the four adjacent points,
    // which emptyNbr already counts.
    if (t12mode === 1 ? rawIdx === 0 : t12mode === 2 && emptyNbr === 4) {
      const a0 = cells[nbr[niN * 4 + 0]], a1 = cells[nbr[niE * 4 + 3]];
      const a2 = cells[nbr[niS * 4 + 1]], a3 = cells[nbr[niW * 4 + 2]];
      const w0 = a0 === 0 ? 0 : a0 === cur ? 1 : 2, w1 = a1 === 0 ? 0 : a1 === cur ? 1 : 2;
      const w2 = a2 === 0 ? 0 : a2 === cur ? 1 : 2, w3 = a3 === 0 ? 0 : a3 === cur ? 1 : 2;
      let t12;
      if (t12mode === 1) {
        t12 = w0 + 3 * (w1 + 3 * (w2 + 3 * w3));
      } else {
        // Diagonals first (NE, SE, SW, NW — vNE.. already coded 0/1/2), then
        // the arms, jointly canonicalised so their relative placement survives.
        t12 = vNE + 3 * (vSE + 3 * (vSW + 3 * (vNW +
              3 * (w0 + 3 * (w1 + 3 * (w2 + 3 * w3))))));
      }
      state.feat[nf++] = t12Offset + _T12C[t12];
    }

    // Computed once; feeds the graded feature AND the save-slot split.
    const sa = (selfAtari || hasPrev) ? _selfAtariSize(game, idx, cur) : 0;
    if (selfAtari && sa > 0) state.feat[nf++] = saOffset + (sa < SA_N ? sa : SA_N) - 1;

    // Gives-atari: the largest adjacent enemy chain this move reduces to one
    // liberty (pre-move ls === 2 is exact: an adjacent empty point is always
    // one of its liberties).
    if (atariN > 0) {
      let biggest = 0;
      for (let d = 0; d < 4; d++) {
        const ni = nbr[b4 + d], c = cells[ni];
        if (c !== 0 && c !== cur) {
          const eg = gidArr[ni];
          if (lsArr[eg] === 2 && ssArr[eg] > biggest) biggest = ssArr[eg];
        }
      }
      if (biggest > 0) state.feat[nf++] = atOffset + (biggest < atariN ? biggest : atariN) - 1;
    }

    // ── Previous-move features (mirrors c/ppat.c: slots 1-5 + contiguity) ────
    const nfLocals = nf;

    // Save-atari: capture (slots 1/2) takes priority over extension (3/4),
    // split by whether the rescue itself is a self-atari.
    if (nAtari > 0) {
      let feat2 = false;
      for (let si = 0; si < nSbc; si++)
        if (_sbcCells[si] === idx) { feat2 = true; break; }
      if (feat2) state.feat[nf++] = prevOffset + (sa > 0 ? 2 : 1);
      else {
        for (let i = 0; i < nAtari; i++)
          if (atariLibsArr[i] === idx) { state.feat[nf++] = prevOffset + (sa > 0 ? 4 : 3); break; }
      }
    }

    // Ko-solve (slot 5)
    for (let ki = 0; ki < nKoSolve; ki++)
      if (idx === _koSolveLibs[ki]) { state.feat[nf++] = prevOffset + 5; break; }

    // Contiguity (slot 0): within prev's 8-neighborhood, or any tactical
    // local (slots 1-5) fired.
    if (hasPrev) {
      let local = nf > nfLocals;
      for (let k = 0; !local && k < nPrevN8; k++) local = (prevN8[k] === idx);
      if (local) state.feat[nf++] = prevOffset + 0;
    }

    count++;
  }

  state.featStart[count] = nf;
  state.count = count;

}

// Score all moves with a model { phaseCount, weights } and return them sorted by
// score descending.
function evaluate(game, state, model) {
  extractFeatures(game, state, model.phaseCount, model.adjLib, model.skipLocal, model.t12mode, model.selfAtari, model.atariN);
  const weights = model.weights;
  const out = [];
  for (let i = 0; i < state.count; i++) {
    let score = 0;
    for (let fi = state.featStart[i]; fi < state.featStart[i + 1]; fi++)
      score += weights[state.feat[fi]];
    out.push({ move: state.moves[i], score });
  }
  return out.sort((a, b) => b.score - a.score);
}

// ── Policy move selection ─────────────────────────────────────────────────────
//
// Extract features, compute softmax over logits, sample an action.
// model: { phaseCount, weights } as returned by loadWeights.
// Returns the flat board index of the chosen move, or PASS if no legal non-eye moves.
// After return, state is populated with the extracted features.

let _logits = new Float32Array(512);

// Fast approximate exp using the Schraudolph IEEE-754 trick.
const _expBuf = new Float64Array(1);
const _expInt = new Int32Array(_expBuf.buffer);
function _fastExp(x) {
  if (x < -20) return 0;
  if (x > 20) x = 20;
  // Schraudolph: write to high 32 bits of float64
  _expInt[1] = (1512775 * x + 1072632447) | 0;
  _expInt[0] = 0;
  return _expBuf[0];
}

function ppatMove(game, state, model, rng = Math) {
  // Uniform fast-path: in the early game the trained policy is ≈ uniform while
  // feature extraction is the dominant per-step cost, so below a board-fullness
  // threshold skip extraction and pick a uniform random legal move.
  // model.uniformBelowPhase is a runtime/deployment knob (set by the agent), a
  // fraction in [0,1] of board fullness (cap-empty)/cap; 0/undefined = off.
  const ubp = model.uniformBelowPhase;
  if (ubp > 0) {
    const cap = game.N * game.N;
    const fullness = (cap - game.emptyCount) / cap;
    if (fullness < ubp) return game.randomLegalMove(rng);
  }

  extractFeatures(game, state, model.phaseCount, model.adjLib, model.skipLocal, model.t12mode, model.selfAtari, model.atariN);
  const weights = model.weights;
  const n = state.count;
  if (n === 0) return PASS;

  if (_logits.length < n) _logits = new Float32Array(n * 2);

  const feat = state.feat;
  const fs = state.featStart;

  // Compute logits and find max in one pass
  let max = -1e30;
  for (let i = 0; i < n; i++) {
    let v = 0;
    for (let fi = fs[i]; fi < fs[i + 1]; fi++) v += weights[feat[fi]];
    _logits[i] = v;
    if (v > max) max = v;
  }

  // PASS as a candidate, available ONLY to a model whose weights file declares
  // earlyPass — see loadWeights.  Its logit is the model's LEARNED passWeight:
  // SB cannot fit it (the update along that direction is Cov(z, passed), which
  // vanishes when an early mutual stop is outcome-neutral), so train_ppat drives
  // it with a control loop on whether the final board still had an atari.
  // Because softmax is shift-invariant it is a threshold against the LOG-SUM-EXP
  // of the board moves, not the best one, so passing also gets likelier as
  // candidates run out rather than only as they get worse.
  // model.passLogit overrides the file's weight, for tuning experiments only.
  const passOn = model.earlyPass === true;
  const passLogit = !passOn ? 0
                  : model.passLogit !== undefined ? model.passLogit
                  : (model.passWeight || 0);
  if (passOn && passLogit > max) max = passLogit;

  // Compute unnormalized weights and sample in two passes
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const e = _fastExp(_logits[i] - max);
    _logits[i] = e;
    sum += e;
  }
  const ePass = passOn ? _fastExp(passLogit - max) : 0;
  sum += ePass;

  let r = rng.random() * sum;
  if (passOn) { r -= ePass; if (r <= 0) return PASS; }
  let chosen = n - 1;
  for (let i = 0; i < n; i++) { r -= _logits[i]; if (r <= 0) { chosen = i; break; } }
  return state.moves[chosen];
}

// Load a weights file (JS module with { weights, phases, numPatterns }).
// Returns the model { phaseCount, weights } (weights = flat Float32Array), or null.
function loadWeights(pathOrObj) {
  let raw = pathOrObj;
  if (typeof raw === 'string') {
    const _path = _isNode ? require('path') : null;
    try { raw = require(_path.resolve(raw)); } catch (e) { return null; }
  }
  if (!raw || !raw.weights) return null;
  // adjLib travels with the model; files predating the field are the historical
  // cap 2.  Build that cap's tables first so numPatterns is checked like-for-like.
  // adjLib travels with the model; older files spell the field `libCap`, and
  // files predating it entirely are cap 2.
  const adjLib = raw.adjLib != null ? raw.adjLib : (raw.libCap != null ? raw.libCap : 2);
  if (!(adjLib >= MIN_ADJ_LIB && adjLib <= MAX_ADJ_LIB)) {
    console.error(`ppat loadWeights: adjLib=${adjLib} out of range [${MIN_ADJ_LIB},${MAX_ADJ_LIB}]`);
    return null;
  }
  const nPat = _buildTables(adjLib).numPatterns;
  if (raw.numPatterns != null && raw.numPatterns !== nPat) {
    console.error(`ppat loadWeights: numPatterns=${raw.numPatterns} in file but ${nPat} expected (adjLib ${adjLib})`);
    return null;
  }
  if (raw.ladder === true) {
    console.error('ppat loadWeights: ladder models are no longer supported (ladder features removed)');
    return null;
  }
  // All-zero local weights (e.g. strip-ppat-local.js output) contribute
  // nothing to any score, so inference skips extracting features 1-7 entirely.
  // This detection must stay on the load path: a freshly initialised model's
  // local weights are also all zero, and a trainer that skipped them would pin
  // their gradients at zero forever.
  const phases = raw.phases || 1;
  const twelvecell = raw.twelvecell === true, twelvecell2 = raw.twelvecell2 === true;
  const selfAtari = raw.selfAtari === true;
  const atariN = raw.atari | 0;
  if (twelvecell && twelvecell2) {
    console.error('ppat loadWeights: twelvecell and twelvecell2 are mutually exclusive');
    return null;
  }
  const t12mode = twelvecell ? 1 : twelvecell2 ? 2 : 0;
  // Scan the LOCAL block only — the twelvecell block is appended after it, and
  // a nonzero twelvecell weight says nothing about features 1-7.
  let skipLocal = true;
  const localEnd = phases * (nPat + 7);   // the twelvecell block starts here
  for (let i = phases * nPat; i < localEnd && i < raw.weights.length; i++)
    if (raw.weights[i] !== 0) { skipLocal = false; break; }
  if (skipLocal) console.log('ppat loadWeights: local weights all zero, skipping local feature extraction');
  // Early pass travels with the model and is OFF unless the file says the
  // model was trained for it.  The pass logit is pinned at 0, so the board
  // weights' ABSOLUTE level is the threshold — and in a model trained without
  // the anchor that level is arbitrary, because adding a constant to every
  // pattern weight shifts all logits equally and leaves the softmax unchanged.
  // Enabling it on such a model therefore produces a pass frequency that is an
  // accident of training, not a decision.
  return { phaseCount: phases, weights: raw.weights, adjLib, skipLocal,
           twelvecell, twelvecell2, t12mode, selfAtari, atariN,
           earlyPass: raw.earlyPass === true,
           passWeight: typeof raw.passWeight === 'number' ? raw.passWeight : 0 };
}

const PPatterns = {
  createState, extractFeatures, evaluate, ppatMove,
  totalWeights, loadWeights,
  NUM_PATTERNS, NUM_T12, NUM_T12B, SA_N,
};
if (typeof module !== 'undefined') module.exports = PPatterns;
else window.PPatterns = PPatterns;

})();
