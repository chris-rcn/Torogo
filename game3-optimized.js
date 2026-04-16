'use strict';

// Game3-Optimized — Tactical search optimized incremental game
//
// Key optimization: assumes play/undo happen in tight pairs
// Uses dual-board snapshots for O(n) undo that's faster than delta reconstruction
//
// Performance for tactical search:
// - Simple moves: O(n) snapshot on play, O(n) restore on undo
// - Captures: same O(n) pattern
// - No reconstruction needed for undo!

(function() {

const EMPTY = 0, BLACK = 1, WHITE = -1;
const PASS = -1;

class Game3Optimized {
  constructor(size) {
    this.N = size;
    this.boardSize = size;
    const cap = size * size;

    // Current board state
    this.cells = new Int8Array(cap);
    this.current = BLACK;
    this.ko = PASS;
    this.emptyCount = cap;
    this.moveCount = 0;
    this.lastMove = PASS;

    // Group tracking
    this._gid = new Int32Array(cap).fill(-1);
    this._nextGid = 0;

    // Bitset arrays
    const W = (cap + 31) >> 5;
    this._W = W;
    const MAX_G = 4 * cap + 4;
    this._gc = new Uint8Array(MAX_G);
    this._sw = new Int32Array(MAX_G * W);
    this._ss = new Int32Array(MAX_G);
    this._lw = new Int32Array(MAX_G * W);
    this._ls = new Int32Array(MAX_G);

    // Cached board snapshots for fast undo
    this._cachedCells = new Int8Array(cap);
    this._cachedGid = new Int32Array(cap);
    this._cachedGc = new Uint8Array(MAX_G);
    this._cachedSw = new Int32Array(MAX_G * W);
    this._cachedSs = new Int32Array(MAX_G);
    this._cachedLw = new Int32Array(MAX_G * W);
    this._cachedLs = new Int32Array(MAX_G);
    this._cachedNextGid = 0;
    this._cacheValid = false;

    // Cached game state
    this._cachedCurrent = BLACK;
    this._cachedKo = PASS;
    this._cachedEmptyCount = cap;
    this._cachedMoveCount = 0;
    this._cachedLastMove = PASS;

    // Topology
    const tables = this._getTopology(size);
    this._nbr = tables.nbr;
    this._dnbr = tables.dnbr;
    this.nbr = tables.nbr;
    this._allCells = tables.allCells;

    // Initialize
    const center = ((size >> 1) * size) + (size >> 1);
    this._placeStone(center, BLACK);
    this.current = WHITE;
    this.moveCount = 1;
  }

  _getTopology(N) {
    const cap = N * N;
    const nbr = new Int32Array(cap * 4);
    const dnbr = new Int32Array(cap * 4);
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const i = y * N + x;
        nbr[i*4+0] = ((y-1+N)%N)*N + x;
        nbr[i*4+1] = ((y+1  )%N)*N + x;
        nbr[i*4+2] = y*N + (x-1+N)%N;
        nbr[i*4+3] = y*N + (x+1  )%N;
      }
    }
    const allCells = new Int32Array(cap);
    for (let i = 0; i < cap; i++) allCells[i] = i;
    return { nbr, dnbr, allCells };
  }

  _pop32(x) {
    x = x - ((x >>> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
    x = (x + (x >>> 4)) & 0x0f0f0f0f;
    return (Math.imul(x, 0x01010101) >>> 24);
  }

  _cacheState() {
    // Save current state to cache for fast undo
    const W = this._W;
    this._cachedCells.set(this.cells);
    this._cachedGid.set(this._gid);
    this._cachedNextGid = this._nextGid;

    // Use efficient bulk copy for group data
    const dataSize = this._nextGid * W;
    this._cachedSw.set(this._sw.subarray(0, dataSize));
    this._cachedLw.set(this._lw.subarray(0, dataSize));
    this._cachedGc.set(this._gc.subarray(0, this._nextGid));
    this._cachedSs.set(this._ss.subarray(0, this._nextGid));
    this._cachedLs.set(this._ls.subarray(0, this._nextGid));

    this._cachedCurrent = this.current;
    this._cachedKo = this.ko;
    this._cachedEmptyCount = this.emptyCount;
    this._cachedMoveCount = this.moveCount;
    this._cachedLastMove = this.lastMove;
    this._cacheValid = true;
  }

  _restoreFromCache() {
    // Restore all state from cache - very fast
    const W = this._W;
    this.cells.set(this._cachedCells);
    this._gid.set(this._cachedGid);
    this._nextGid = this._cachedNextGid;

    // Use efficient bulk copy for group data
    const dataSize = this._nextGid * W;
    this._sw.set(this._cachedSw.subarray(0, dataSize));
    this._lw.set(this._cachedLw.subarray(0, dataSize));
    this._gc.set(this._cachedGc.subarray(0, this._nextGid));
    this._ss.set(this._cachedSs.subarray(0, this._nextGid));
    this._ls.set(this._cachedLs.subarray(0, this._nextGid));

    this.current = this._cachedCurrent;
    this.ko = this._cachedKo;
    this.emptyCount = this._cachedEmptyCount;
    this.moveCount = this._cachedMoveCount;
    this.lastMove = this._cachedLastMove;
    this._cacheValid = false;
  }

  // ── Place stone (same as before) ────────────────────────────────────────────

  _placeStone(idx, color) {
    this.cells[idx] = color;
    this.emptyCount--;
    this.lastMove = idx;

    const gidArr = this._gid;
    const nbr = this._nbr;
    const base = idx * 4;
    const W = this._W;

    // Remove from liberties of opponent groups
    let oppGroupIds = [];
    for (let i = 0; i < 4; i++) {
      const ni = nbr[base + i];
      const gid = gidArr[ni];
      if (gid !== -1 && !oppGroupIds.includes(gid)) {
        oppGroupIds.push(gid);
        const m = 1 << (idx & 31);
        const wi = idx >> 5;
        const lb = gid * W;
        if (this._lw[lb + wi] & m) {
          this._lw[lb + wi] &= ~m;
          this._ls[gid]--;
        }
      }
    }

    // Find adjacent groups
    let sameColorGroupIds = [];
    let emptyNeighbors = [];
    for (let i = 0; i < 4; i++) {
      const ni = nbr[base + i];
      const c = this.cells[ni];
      if (c === color) {
        const gid = gidArr[ni];
        if (!sameColorGroupIds.includes(gid)) sameColorGroupIds.push(gid);
      } else if (c === EMPTY && !emptyNeighbors.includes(ni)) {
        emptyNeighbors.push(ni);
      }
    }

    if (sameColorGroupIds.length === 0) {
      // Create new group
      const gid = this._nextGid++;
      gidArr[idx] = gid;
      this._gc[gid] = color;
      const gb = gid * W;
      this._sw[gb + (idx >> 5)] |= (1 << (idx & 31));
      this._ss[gid] = 1;
      for (const liberty of emptyNeighbors) {
        const m = 1 << (liberty & 31);
        const wi = liberty >> 5;
        if (!(this._lw[gb + wi] & m)) {
          this._lw[gb + wi] |= m;
          this._ls[gid]++;
        }
      }
    } else {
      // Merge groups
      let mainGid = sameColorGroupIds[0];
      for (let i = 1; i < sameColorGroupIds.length; i++) {
        if (this._ss[sameColorGroupIds[i]] > this._ss[mainGid]) {
          mainGid = sameColorGroupIds[i];
        }
      }

      const gb = mainGid * W;
      this._sw[gb + (idx >> 5)] |= (1 << (idx & 31));
      this._ss[mainGid]++;
      gidArr[idx] = mainGid;

      for (const liberty of emptyNeighbors) {
        const m = 1 << (liberty & 31);
        const wi = liberty >> 5;
        if (!(this._lw[gb + wi] & m)) {
          this._lw[gb + wi] |= m;
          this._ls[mainGid]++;
        }
      }

      // Merge other groups
      for (const otherId of sameColorGroupIds) {
        if (otherId === mainGid) continue;
        const ob = otherId * W;
        for (let wi = 0; wi < W; wi++) {
          let w = this._sw[ob + wi];
          while (w) {
            const bit = 31 - Math.clz32(w & -w);
            gidArr[wi * 32 + bit] = mainGid;
            w &= w - 1;
          }
          this._sw[gb + wi] |= this._sw[ob + wi];
        }
        this._ss[mainGid] += this._ss[otherId];
        for (let wi = 0; wi < W; wi++) {
          this._lw[gb + wi] |= this._lw[ob + wi];
        }
      }

      let newLs = 0;
      for (let wi = 0; wi < W; wi++) {
        newLs += this._pop32(this._lw[gb + wi]);
      }
      this._ls[mainGid] = newLs;
    }
  }

  _captureGroup(gid) {
    const W = this._W;
    const stones = [];
    const sb = gid * W;

    for (let wi = 0; wi < W; wi++) {
      let w = this._sw[sb + wi];
      while (w) {
        const bit = 31 - Math.clz32(w & -w);
        const idx = wi * 32 + bit;
        stones.push(idx);
        this.cells[idx] = EMPTY;
        this._gid[idx] = -1;
        const base = idx * 4;
        for (let i = 0; i < 4; i++) {
          const nGid = this._gid[this._nbr[base + i]];
          if (nGid !== -1 && nGid !== gid) {
            const nlb = nGid * W;
            const m = 1 << (idx & 31);
            const nwi = idx >> 5;
            if (!(this._lw[nlb + nwi] & m)) {
              this._lw[nlb + nwi] |= m;
              this._ls[nGid]++;
            }
          }
        }
        w &= w - 1;
      }
    }
    this.emptyCount += stones.length;
    return stones;
  }

  // ── Legality checks ────────────────────────────────────────────────────────

  _isSingleSuicide(idx, color) {
    const nbr = this._nbr;
    const base = idx * 4;
    for (let i = 0; i < 4; i++) {
      const ni = nbr[base + i];
      const c = this.cells[ni];
      if (c === EMPTY) return false;
      const gid = this._gid[ni];
      if (c === color) { if (this._ls[gid] > 1) return false; }
      else             { if (this._ls[gid] === 1) return false; }
    }
    return true;
  }

  isLegal(idx, color = this.current) {
    if (this.cells[idx] !== EMPTY) return false;
    if (this.ko === idx) return false;
    if (this._isSingleSuicide(idx, color)) {
      const nbr = this._nbr;
      const base = idx * 4;
      for (let i = 0; i < 4; i++) {
        const ni = nbr[base + i];
        const c = this.cells[ni];
        if (c === -color && this._ls[this._gid[ni]] === 1) return true;
      }
      return false;
    }
    return true;
  }

  // ── Play & Undo ────────────────────────────────────────────────────────────

  play(move) {
    if (move === PASS) {
      this._cacheState();
      this.current = -this.current;
      this.moveCount++;
      return true;
    }

    if (!this.isLegal(move)) return false;

    // Cache state BEFORE the move
    this._cacheState();

    const color = this.current;
    const oppColor = -color;

    this._placeStone(move, color);

    // Capture opponent groups
    const nbr = this._nbr;
    const base = move * 4;
    const capturedStones = [];

    for (let i = 0; i < 4; i++) {
      const ni = nbr[base + i];
      const gid = this._gid[ni];
      if (gid !== -1 && this._gc[gid] === oppColor && this._ls[gid] === 0) {
        const stones = this._captureGroup(gid);
        capturedStones.push(...stones);
      }
    }

    // Update ko
    if (capturedStones.length === 1 &&
        (Math.abs(capturedStones[0] - move) === 1 ||
         Math.abs(capturedStones[0] - move) === this.N)) {
      this.ko = capturedStones[0];
    } else {
      this.ko = PASS;
    }

    this.current = oppColor;
    this.moveCount++;
    return true;
  }

  undo() {
    if (!this._cacheValid) return;
    this._restoreFromCache();
  }

  // ── Group Query ────────────────────────────────────────────────────────────

  groupIdAt(idx) {
    return this._gid[idx];
  }

  groupSize(gid) {
    return this._ss[gid];
  }

  groupLibertyCount(gid) {
    return this._ls[gid];
  }

  groupLibs(idx) {
    const gid = this._gid[idx];
    if (gid === -1) return new Int32Array(0);
    const lc = this._ls[gid];
    const out = new Int32Array(lc);
    const cap = this.N * this.N;
    const lb = gid * this._W;
    let found = 0;
    for (let wi = 0; wi < this._W && found < lc; wi++) {
      let w = this._lw[lb + wi];
      while (w && found < lc) {
        const i = wi * 32 + (31 - Math.clz32(w & -w));
        if (i < cap) out[found++] = i;
        w &= w - 1;
      }
    }
    return out;
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { Game3Optimized, PASS, BLACK, WHITE, EMPTY };
} else if (typeof window !== 'undefined') {
  window.Game3Optimized = Game3Optimized;
  window.PASS = PASS;
  window.BLACK = BLACK;
  window.WHITE = WHITE;
  window.EMPTY = EMPTY;
}

})();
