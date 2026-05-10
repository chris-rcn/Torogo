'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
const { PASS } = _isNode ? require('./game2.js') : window.Game2;

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
// Raw pattern index (positions in encoding order: N, E, S, W, NE, SE, SW, NW):
//   rawIdx = vN + 5*(vE + 5*(vS + 5*(vW + 5*(vNE + 3*(vSE + 3*(vSW + 3*vNW))))))
// Range: [0, 5^4 × 3^4) = [0, 50625).
//
// The encoding is already relative to the current mover (FRIEND/FOE), so color
// swap is NOT a symmetry.  Only the 8 D4 spatial transforms are applied.
// CANON_ID maps raw → dense canonical ID (0-based).

const _CANON_ID = new Int16Array(50625);
let PHASE_COUNT = 1;  // set via setPhaseCount() before use

const NUM_PATTERNS = (function _buildTables() {
  const v  = new Int32Array(8);
  const tv = new Int32Array(8);
  const idMap = new Map(); // minVariant rawIdx → assigned canonId
  let nextId = 0;

  for (let raw = 0; raw < 50625; raw++) {
    // Decode: positions 0-3 have base 5, positions 4-7 have base 3.
    let r = raw;
    v[0] = r % 5; r = (r / 5) | 0;
    v[1] = r % 5; r = (r / 5) | 0;
    v[2] = r % 5; r = (r / 5) | 0;
    v[3] = r % 5; r = (r / 5) | 0;
    v[4] = r % 3; r = (r / 3) | 0;
    v[5] = r % 3; r = (r / 3) | 0;
    v[6] = r % 3;
    v[7] = (r / 3) | 0;

    let minV = raw;

    for (let di = 0; di < 8; di++) {
      const p = _D4[di];
      for (let i = 0; i < 8; i++) tv[p[i]] = v[i];
      const enc = tv[0] + 5*(tv[1] + 5*(tv[2] + 5*(tv[3] + 5*(tv[4] + 3*(tv[5] + 3*(tv[6] + 3*tv[7]))))));
      if (enc < minV) minV = enc;
    }

    if (!idMap.has(minV)) idMap.set(minV, nextId++);
    _CANON_ID[raw] = idMap.get(minV);
  }
  return nextId;
}());

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

// Find both liberties of a group with exactly 2 libs.
function _twoLibs(gid, lw, W, cap) {
  const lb = gid * W;
  const libs = [-1, -1];
  let found = 0;
  for (let wi = 0; wi < W && found < 2; wi++) {
    let w = lw[lb + wi];
    while (w && found < 2) {
      const i = wi * 32 + (31 - Math.clz32(w & -w));
      if (i < cap) libs[found++] = i;
      w &= w - 1;
    }
  }
  return libs;
}

// Check if the other liberty of egid (not idx) connects to a same-color group
// (excluding egid) with ≥2 liberties. If so, the opponent can save by joining.
function _opponentCanSave(idx, egid, nbr, cells, gidArr, lsArr, lw, W, cap, foe) {
  const [l0, l1] = _twoLibs(egid, lw, W, cap);
  const other = (l0 === idx) ? l1 : l0;
  const ob4 = other * 4;
  for (let d = 0; d < 4; d++) {
    const ni = nbr[ob4 + d];
    if (cells[ni] !== foe) continue;
    const ngid = gidArr[ni];
    if (ngid === egid) continue;
    if (lsArr[ngid] >= 2) return true;
  }
  return false;
}

// Returns true if playing at idx gives atari to an enemy group adjacent to
// any of the friendly 2-liberty groups in new2LibGids, AND the opponent
// cannot save by joining at the other liberty.
function _gives2LibAtari(idx, new2LibGids, nbr, cells, gidArr, lsArr, lw, sw, W, cap, foe) {
  for (const sgid of new2LibGids) {
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
            if (lsArr[egid] === 2 && (lw[egid * W + (idx >> 5)] & (1 << (idx & 31)))
                && !_opponentCanSave(idx, egid, nbr, cells, gidArr, lsArr, lw, W, cap, foe))
              return true;
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
function _notSelfAtariCheap(idx, b4, nbr, cells, gidArr, lsArr, lw, W, cur, foe) {
  let free = 0;
  for (let di = 0; di < 4; di++) {
    const ni = nbr[b4 + di];
    const c  = cells[ni];
    if (c === 0) {
      if (++free >= 2) return true;
    } else if (c === cur) {
      if (lsArr[gidArr[ni]] >= 3) return true;
    } else {
      const egid = gidArr[ni];
      if (lsArr[egid] === 1 && (lw[egid * W + (idx >> 5)] & (1 << (idx & 31))))
        if (++free >= 2) return true;
    }
  }
  return false;
}

// ── Static buffers (avoid per-call allocation / GC pressure) ─────────────────
const _atariGids    = new Int32Array(8);
const _atariLibsArr = new Int32Array(8);
const _twoLibGids   = new Int32Array(8);
const _sbcCells     = new Int32Array(64);   // save-by-capture cell indices
const _semCells     = new Int32Array(64);   // semeai candidate cell indices
const _semEgids     = new Int32Array(64);   // semeai candidate enemy gids
const _koSolveLibs  = new Int32Array(4);
const _seenBuf      = new Int32Array(16);   // dedup scratch

// ── Public API ────────────────────────────────────────────────────────────────

// Allocate reusable output buffers for a board of size N.
function createState(N) {
  const cap = N * N;
  return {
    moves:          new Int32Array(cap),
    feat:           new Int32Array(cap * 8),  // flat feature keys (max 8 per candidate)
    featStart:      new Int32Array(cap + 1),  // featStart[i]..featStart[i+1] = keys for candidate i
    prevNeighborSet: new Uint8Array(cap),
    count:          0,
  };
}

function setPhaseCount(n) { PHASE_COUNT = n; }
function totalWeights() { return PHASE_COUNT * (NUM_PATTERNS + 7); }

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
function extractFeatures(game, state) {
  const N      = game.N;
  const cap    = N * N;
  const cells  = game.cells;
  const gidArr = game._gid;
  const lsArr  = game._ls;
  const lwArr  = game._lw;
  const swArr  = game._sw;
  const W      = game._W;

  const phase = PHASE_COUNT * (cap - game.emptyCount) / cap | 0;
  const patOffset = phase * NUM_PATTERNS;
  const prevOffset = PHASE_COUNT * NUM_PATTERNS + phase * 7;
  const nbr    = game._nbr;
  const dnbr   = game._dnbr;
  const cur    = game.current;
  const foe    = -cur;
  const emC    = game._emptyCells;
  const ec     = game.emptyCount;
  const prev   = game.lastMove;
  const hasPrev = prev !== PASS;
  const myKoStone = game.koStone[cur + 1];

  // ── Pre-scan: build prevNeighborSet + find friendly strings in atari or with 2 libs ──
  const prevNeighborSet = state.prevNeighborSet;
  const atariGids = _atariGids;     // reuse static arrays (no GC)
  const atariLibsArr = _atariLibsArr;
  let nAtari = 0;
  const twoLibGids = _twoLibGids;
  let nTwo = 0;

  if (hasPrev) {
    const pb4 = prev * 4;
    for (let di = 0; di < 4; di++) {
      prevNeighborSet[nbr[pb4 + di]]  = 1;
      prevNeighborSet[dnbr[pb4 + di]] = 1;
      const ni = nbr[pb4 + di];
      if (cells[ni] !== cur) continue;
      const gid = gidArr[ni];
      const ls  = lsArr[gid];
      if (ls === 1) {
        let dup = false;
        for (let j = 0; j < nAtari; j++) if (atariGids[j] === gid) { dup = true; break; }
        if (!dup) atariGids[nAtari++] = gid;
      } else if (ls === 2) {
        let dup = false;
        for (let j = 0; j < nTwo; j++) if (twoLibGids[j] === gid) { dup = true; break; }
        if (!dup) twoLibGids[nTwo++] = gid;
      }
    }
    for (let i = 0; i < nAtari; i++)
      atariLibsArr[i] = _firstLib(atariGids[i], lwArr, W, cap);
  }

  // Precompute save-by-capture cells (avoid per-candidate _canSaveByCapture).
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

  // Precompute semeai candidates (avoid per-candidate _gives2LibAtari).
  let nSem = 0;
  if (nTwo > 0) {
    const seen = _seenBuf;
    let nSeen = 0;
    for (let ti = 0; ti < nTwo; ti++) {
      const sgid = twoLibGids[ti];
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
              if (lsArr[egid] !== 2) continue;
              let dup = false;
              for (let j = 0; j < nSeen; j++) if (seen[j] === egid) { dup = true; break; }
              if (dup) continue;
              if (nSeen < 16) seen[nSeen++] = egid;
              const [l0, l1] = _twoLibs(egid, lwArr, W, cap);
              if (l0 >= 0) { _semCells[nSem] = l0; _semEgids[nSem] = egid; nSem++; }
              if (l1 >= 0) { _semCells[nSem] = l1; _semEgids[nSem] = egid; nSem++; }
            }
          }
          w ^= lsb;
        }
      }
    }
  }

  // Feature 6 pre-scan
  let nKoSolve = 0;
  if (myKoStone !== PASS) {
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
    else { const g_ = gidArr[niN]; const a_ = lsArr[g_] === 1;
      if (cN === cur) { friendCount++; if (firstGid === -2) { firstGid = g_; sameGroup = 1; } else if (g_ === firstGid) sameGroup++; vN = a_ ? 2 : 1; }
      else vN = a_ ? 4 : 3; }
    if (cS === 0) { emptyNbr++; vS2 = 0; }
    else { const g_ = gidArr[niS]; const a_ = lsArr[g_] === 1;
      if (cS === cur) { friendCount++; if (firstGid === -2) { firstGid = g_; sameGroup = 1; } else if (g_ === firstGid) sameGroup++; vS2 = a_ ? 2 : 1; }
      else vS2 = a_ ? 4 : 3; }
    if (cW === 0) { emptyNbr++; vW2 = 0; }
    else { const g_ = gidArr[niW]; const a_ = lsArr[g_] === 1;
      if (cW === cur) { friendCount++; if (firstGid === -2) { firstGid = g_; sameGroup = 1; } else if (g_ === firstGid) sameGroup++; vW2 = a_ ? 2 : 1; }
      else vW2 = a_ ? 4 : 3; }
    if (cE === 0) { emptyNbr++; vE2 = 0; }
    else { const g_ = gidArr[niE]; const a_ = lsArr[g_] === 1;
      if (cE === cur) { friendCount++; if (firstGid === -2) { firstGid = g_; sameGroup = 1; } else if (g_ === firstGid) sameGroup++; vE2 = a_ ? 2 : 1; }
      else vE2 = a_ ? 4 : 3; }

    if (friendCount === 3 && emptyNbr === 1 && sameGroup === 3) continue;
    if (friendCount === 4) {
      if (sameGroup === 4) continue;
      let dc = 0;
      if (cells[dnbr[b4]]     === cur) dc++;
      if (cells[dnbr[b4 + 1]] === cur) dc++;
      if (cells[dnbr[b4 + 2]] === cur) dc++;
      if (cells[dnbr[b4 + 3]] === cur) dc++;
      if (dc >= 3) continue;
    }

    // Diag values
    const cNE2 = cells[dnbr[b4 + 1]], vNE = cNE2 === 0 ? 0 : (cNE2 === cur ? 1 : 2);
    const cSE2 = cells[dnbr[b4 + 3]], vSE = cSE2 === 0 ? 0 : (cSE2 === cur ? 1 : 2);
    const cSW2 = cells[dnbr[b4 + 2]], vSW = cSW2 === 0 ? 0 : (cSW2 === cur ? 1 : 2);
    const cNW2 = cells[dnbr[b4]],     vNW = cNW2 === 0 ? 0 : (cNW2 === cur ? 1 : 2);

    const rawIdx = vN + 5*(vE2 + 5*(vS2 + 5*(vW2 + 5*(vNE + 3*(vSE + 3*(vSW + 3*vNW))))));

    state.moves[count] = idx;
    state.featStart[count] = nf;

    // Pattern feature
    state.feat[nf++] = patOffset + _CANON_ID[rawIdx];

    // ── Previous-move features ────────────────────────────────────────────────
    let mask = 0;
    if (hasPrev && prevNeighborSet[idx]) mask = 1;

    // Features 2–5: precomputed save-by-capture + extension check
    if (nAtari > 0) {
      let feat2 = false;
      for (let si = 0; si < nSbc; si++) {
        if (_sbcCells[si] === idx) { feat2 = true; break; }
      }
      let feat4 = false;
      if (!feat2) {
        for (let i = 0; i < nAtari; i++) {
          if (atariLibsArr[i] === idx) { feat4 = true; break; }
        }
      }
      if (feat2 || feat4) {
        let sa = false;
        if (!_notSelfAtariCheap(idx, b4, nbr, cells, gidArr, lsArr, lwArr, W, cur, foe)) {
          const cg = game.clone();
          cg.play(idx);
          const cid = cg._gid[idx];
          sa = cid !== -1 && cg._ls[cid] === 1;
        }
        if (feat2) mask |= sa ? 4 : 2;
        if (feat4) mask |= sa ? 16 : 8;
      }
    }

    // Feature 7: precomputed semeai candidates
    if (nSem > 0) {
      for (let si = 0; si < nSem; si++) {
        if (_semCells[si] === idx &&
            !_opponentCanSave(idx, _semEgids[si], nbr, cells, gidArr, lsArr, lwArr, W, cap, foe)) {
          mask |= 64; break;
        }
      }
    }

    // Feature 6: ko-solve capture
    for (let ki = 0; ki < nKoSolve; ki++) {
      if (idx === _koSolveLibs[ki]) { mask |= 32; break; }
    }

    if (mask & 0x7E) mask |= 1;

    // Emit prev feature keys
    for (let b = 0; b < 7; b++)
      if (mask & (1 << b)) state.feat[nf++] = prevOffset + b;

    count++;
  }

  state.featStart[count] = nf;
  state.count = count;

  if (hasPrev) {
    const pb4 = prev * 4;
    for (let di = 0; di < 4; di++) {
      prevNeighborSet[nbr[pb4 + di]]  = 0;
      prevNeighborSet[dnbr[pb4 + di]] = 0;
    }
  }
}

// Score all moves with a flat weight array and return sorted by score descending.
function evaluate(game, state, weights) {
  extractFeatures(game, state);
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
// weights: { pat: Float32Array(NUM_PATTERNS), prev: Float32Array(7) }
// Returns the flat board index of the chosen move, or PASS if no legal non-eye moves.
// After return, state is populated with the extracted features.

let _logits = new Float32Array(512);

// Fast approximate exp using the Schraudolph IEEE-754 trick.
const _expBuf = new Float64Array(1);
const _expInt = new Int32Array(_expBuf.buffer);
function _fastExp(x) {
  if (x < -20) return 0;
  // Schraudolph: write to high 32 bits of float64
  _expInt[1] = (1512775 * x + 1072632447) | 0;
  _expInt[0] = 0;
  return _expBuf[0];
}

function ppatMove(game, state, weights) {
  extractFeatures(game, state);
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

  // Compute unnormalized weights and sample in two passes
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const e = _fastExp(_logits[i] - max);
    _logits[i] = e;
    sum += e;
  }

  let r = Math.random() * sum, chosen = n - 1;
  for (let i = 0; i < n; i++) { r -= _logits[i]; if (r <= 0) { chosen = i; break; } }
  return state.moves[chosen];
}

// Load a weights file (JS module with { weights, phases, numPatterns }).
// Sets PHASE_COUNT and returns the flat Float32Array, or null.
function loadWeights(pathOrObj) {
  let raw = pathOrObj;
  if (typeof raw === 'string') {
    const _path = _isNode ? require('path') : null;
    try { raw = require(_path.resolve(raw)); } catch (e) { return null; }
  }
  if (!raw || !raw.weights) return null;
  if (raw.phases) PHASE_COUNT = raw.phases;
  return raw.weights;
}

const PPatterns = {
  createState, extractFeatures, evaluate, ppatMove,
  setPhaseCount, totalWeights, loadWeights,
  NUM_PATTERNS, get PHASE_COUNT() { return PHASE_COUNT; },
};
if (typeof module !== 'undefined') module.exports = PPatterns;
else window.PPatterns = PPatterns;

})();
