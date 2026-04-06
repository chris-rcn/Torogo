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

const _CANON_ID = new Int32Array(50625);

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
// any of the friendly groups in newAtariGids, saving them from atari.
function _canSaveByCapture(idx, newAtariGids, nbr, cells, gidArr, lsArr, lw, sw, W, cap, foe) {
  for (const sgid of newAtariGids) {
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

// Returns true if playing at idx gives atari (reduces to 1 liberty) to an enemy group
// that is adjacent to any of the friendly 2-liberty groups in twoLibGids.
function _gives2LibAtari(idx, twoLibGids, nbr, cells, gidArr, lsArr, lw, sw, W, cap, foe) {
  for (const sgid of twoLibGids) {
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
            // Enemy group with 2 liberties where idx is one of them → playing idx gives atari.
            if (lsArr[egid] === 2 && (lw[egid * W + (idx >> 5)] & (1 << (idx & 31)))) return true;
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

// ── Public API ────────────────────────────────────────────────────────────────

// Allocate reusable output buffers for a board of size N.
function createState(N) {
  const cap = N * N;
  return {
    moves:          new Int32Array(cap),
    patIds:         new Int32Array(cap),
    prevMasks:      new Uint8Array(cap),
    prevNeighborSet: new Uint8Array(cap),
    count:          0,
  };
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
function extractFeatures(game, state) {
  const N      = game.N;
  const cap    = N * N;
  const cells  = game.cells;
  const gidArr = game._gid;
  const lsArr  = game._ls;
  const lwArr  = game._lw;
  const swArr  = game._sw;
  const W      = game._W;
  const nbr    = game._nbr;
  const dnbr   = game._dnbr;
  const cur    = game.current;
  const foe    = -cur;
  const emC    = game._emptyCells;
  const ec     = game.emptyCount;
  const prev   = game.lastMove;
  const hasPrev = prev !== PASS;
  const koPoint = game.ko;

  // ── Pre-scan: build prevNeighborSet + find friendly strings in new atari or with 2 libs ──
  const prevNeighborSet = state.prevNeighborSet;
  let newAtariGids = null;  // Set<gid>: friendly groups with 1 liberty adjacent to prev
  let twoLibGids   = null;  // Set<gid>: friendly groups with 2 liberties adjacent to prev
  if (hasPrev) {
    const pb4 = prev * 4;
    for (let di = 0; di < 4; di++) {
      const n1 = nbr[pb4 + di], n2 = dnbr[pb4 + di];
      prevNeighborSet[n1] = 1;
      prevNeighborSet[n2] = 1;
      for (let k = 0; k < 2; k++) {
        const ni = k === 0 ? n1 : n2;
        if (cells[ni] !== cur) continue;
        const gid = gidArr[ni];
        const ls  = lsArr[gid];
        if (ls === 1) { if (!newAtariGids) newAtariGids = new Set(); newAtariGids.add(gid); }
        else if (ls === 2) { if (!twoLibGids) twoLibGids = new Set(); twoLibGids.add(gid); }
      }
    }
  }

  // Cache single liberty for each new-atari group (for Feature 4/5 check).
  let atariLibs = null;
  if (newAtariGids) {
    atariLibs = new Map();
    for (const gid of newAtariGids) atariLibs.set(gid, _firstLib(gid, lwArr, W, cap));
  }

  let count = 0;

  for (let ei = 0; ei < ec; ei++) {
    const idx = emC[ei];
    if (!game.isLegal(idx) || game.isTrueEye(idx)) continue;

    // ── 3×3 pattern ──────────────────────────────────────────────────────────
    // game2 neighbor order: _nbr[i*4+0]=N(row−1), +1=S(row+1), +2=W(col−1), +3=E(col+1)
    //                       _dnbr[i*4+0]=NW, +1=NE, +2=SW, +3=SE
    const b4 = idx * 4;
    // Inline _adj: 0=EMPTY, 1=FRIEND, 2=FRIEND_ATARI, 3=FOE, 4=FOE_ATARI
    const niN = nbr[b4],     cN = cells[niN];
    const niE = nbr[b4 + 3], cE = cells[niE];
    const niS = nbr[b4 + 1], cS = cells[niS];
    const niW = nbr[b4 + 2], cW = cells[niW];
    const vN = cN === 0 ? 0 : (lsArr[gidArr[niN]] === 1 ? (cN === cur ? 2 : 4) : (cN === cur ? 1 : 3));
    const vE = cE === 0 ? 0 : (lsArr[gidArr[niE]] === 1 ? (cE === cur ? 2 : 4) : (cE === cur ? 1 : 3));
    const vS = cS === 0 ? 0 : (lsArr[gidArr[niS]] === 1 ? (cS === cur ? 2 : 4) : (cS === cur ? 1 : 3));
    const vW = cW === 0 ? 0 : (lsArr[gidArr[niW]] === 1 ? (cW === cur ? 2 : 4) : (cW === cur ? 1 : 3));
    // Inline _diag: 0=EMPTY, 1=FRIEND, 2=FOE
    const cNE = cells[dnbr[b4 + 1]], vNE = cNE === 0 ? 0 : (cNE === cur ? 1 : 2);
    const cSE = cells[dnbr[b4 + 3]], vSE = cSE === 0 ? 0 : (cSE === cur ? 1 : 2);
    const cSW = cells[dnbr[b4 + 2]], vSW = cSW === 0 ? 0 : (cSW === cur ? 1 : 2);
    const cNW = cells[dnbr[b4]],     vNW = cNW === 0 ? 0 : (cNW === cur ? 1 : 2);

    const rawIdx = vN + 5*(vE + 5*(vS + 5*(vW + 5*(vNE + 3*(vSE + 3*(vSW + 3*vNW))))));

    state.moves[count]  = idx;
    state.patIds[count] = _CANON_ID[rawIdx];

    // ── Previous-move features ────────────────────────────────────────────────
    let mask = 0;

    if (hasPrev && prevNeighborSet[idx]) {
      mask = 1; // bit 0 = Feature 1

      // Features 2–5: save a string in new atari by capture or extension.
      if (newAtariGids) {
        // Feature 4/5: idx is the single liberty of an atari'd string (extension).
        let feat4 = false;
        for (const gid of newAtariGids) {
          if (atariLibs.get(gid) === idx) { feat4 = true; break; }
        }

        // Feature 2/3: idx captures an adjacent enemy group, freeing an atari'd string.
        let feat2 = !feat4 && _canSaveByCapture(idx, newAtariGids, nbr, cells, gidArr, lsArr, lwArr, swArr, W, cap, foe);

        if (feat2 || feat4) {
          // Cheap check: if 2+ guaranteed liberties, definitely not self-atari.
          let sa = false;
          if (!_notSelfAtariCheap(idx, b4, nbr, cells, gidArr, lsArr, lwArr, W, cur, foe)) {
            const cg = game.clone();
            cg.play(idx);
            const cid = cg._gid[idx];
            sa = cid !== -1 && cg._ls[cid] === 1;
          }
          if (feat2) mask |= sa ? 4 : 2;   // bit 2 or bit 1
          if (feat4) mask |= sa ? 16 : 8;  // bit 4 or bit 3
        }
      }

      // Feature 6: there is a new ko and this move is a capture (solves the ko).
      if (koPoint !== PASS && game.isCapture(idx)) mask |= 32; // bit 5

      // Feature 7: 2-point semeai — give atari to an enemy group adjacent to our 2-lib string.
      if (twoLibGids && _gives2LibAtari(idx, twoLibGids, nbr, cells, gidArr, lsArr, lwArr, swArr, W, cap, foe))
        mask |= 64; // bit 6
    }

    state.prevMasks[count] = mask;
    count++;
  }

  state.count = count;

  // Clear prevNeighborSet for reuse.
  if (hasPrev) {
    const pb4 = prev * 4;
    for (let di = 0; di < 4; di++) {
      prevNeighborSet[nbr[pb4 + di]]  = 0;
      prevNeighborSet[dnbr[pb4 + di]] = 0;
    }
  }
}

// Score all moves with a linear model and return them sorted by score descending.
// score(move) = patWeights[patId] + sum_{active bits b} prevWeights[b]
// patWeights : Float32Array(NUM_PATTERNS) or Map<int32, number>
// prevWeights: Float32Array(7) indexed by bit position 0–6  (may be null/undefined)
function evaluate(game, state, patWeights, prevWeights) {
  extractFeatures(game, state);
  const isMap = patWeights instanceof Map;
  const out = [];
  for (let i = 0; i < state.count; i++) {
    let score = isMap ? (patWeights.get(state.patIds[i]) || 0) : (patWeights[state.patIds[i]] || 0);
    if (prevWeights) {
      let m = state.prevMasks[i];
      for (let b = 0; m; b++, m >>= 1) if (m & 1) score += prevWeights[b];
    }
    out.push({ move: state.moves[i], score });
  }
  return out.sort((a, b) => b.score - a.score);
}

const PPatterns = { createState, extractFeatures, evaluate, NUM_PATTERNS };
if (typeof module !== 'undefined') module.exports = PPatterns;
else window.PPatterns = PPatterns;

})();
