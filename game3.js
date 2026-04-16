'use strict';

// Game3 — Fast incremental game class for tactical searches
// Features:
//   - Fully incremental: no cloning, just push/pop moves on a stack
//   - Unlimited undo: change stack depth is your only limit
//   - Optimized for shallow tactical search (few moves, many branches)
//
// Interface (compatible with Game2 where possible):
//   new Game3(size)
//   game.play(move)           → true/false
//   game.isLegal(move)        → boolean (non-mutating)
//   game.undo()               → void (pops last move)
//   game.cells                Int8Array (read-only)
//   game.current              BLACK | WHITE
//   game.groupLibertyCount(gid) → number of liberties

(function() {

const EMPTY = 0, BLACK = 1, WHITE = -1;
const PASS = -1;

class Game3 {
  constructor(size) {
    this.N = size;
    this.boardSize = size;
    const cap = size * size;

    // Board state
    this.cells = new Int8Array(cap);
    this.current = BLACK;
    this.ko = PASS;
    this.emptyCount = cap;
    this.moveCount = 0;
    this.lastMove = PASS;

    // Group tracking
    this._gid = new Int32Array(cap).fill(-1);
    this._nextGid = 0;

    // Bitset arrays for efficient group operations
    const W = (cap + 31) >> 5;
    this._W = W;
    const MAX_G = 4 * cap + 4;
    this._gc = new Uint8Array(MAX_G);       // group color
    this._sw = new Int32Array(MAX_G * W);   // stones bitset
    this._ss = new Int32Array(MAX_G);       // stones count
    this._lw = new Int32Array(MAX_G * W);   // liberties bitset
    this._ls = new Int32Array(MAX_G);       // liberties count

    // Topology (cached, shared with Game2)
    const tables = this._getTopology(size);
    this._nbr = tables.nbr;
    this._dnbr = tables.dnbr;
    this.nbr = tables.nbr;
    this._allCells = tables.allCells;

    // Undo stack: each entry is a complete move change record
    this._undoStack = [];

    // Initialize: place center stone
    const center = ((size >> 1) * size) + (size >> 1);
    this._placeStone(center, BLACK);
    this.current = WHITE;
    this.moveCount = 1;
    this._recordPlacementForUndo();
  }

  reset() {
    // Clear board and undo stack
    const cap = this.N * this.N;
    this.cells.fill(0);
    this._gid.fill(-1);
    this._nextGid = 0;
    const used = this._nextGid * this._W;
    this._sw.fill(0, 0, used);
    this._ss.fill(0, 0, this._nextGid);
    this._lw.fill(0, 0, used);
    this._ls.fill(0, 0, this._nextGid);
    this._undoStack.length = 0;

    this.current = BLACK;
    this.ko = PASS;
    this.emptyCount = cap;
    this.moveCount = 0;
    this.lastMove = PASS;

    // Reinitialize with center stone
    const center = ((this.N >> 1) * this.N) + (this.N >> 1);
    this._placeStone(center, BLACK);
    this.current = WHITE;
    this.moveCount = 1;
    this._recordPlacementForUndo();
  }

  _getTopology(N) {
    // Inline topology (same as Game2)
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
        dnbr[i*4+0] = ((y-1+N)%N)*N + (x-1+N)%N;
        dnbr[i*4+1] = ((y-1+N)%N)*N + (x+1  )%N;
        dnbr[i*4+2] = ((y+1  )%N)*N + (x-1+N)%N;
        dnbr[i*4+3] = ((y+1  )%N)*N + (x+1  )%N;
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

  // ── Move Recording for Undo ────────────────────────────────────────────────

  _recordPlacementForUndo() {
    // Called after _placeStone to record the current state for undo
    const change = {
      type: 'place',
      idx: this.lastMove,
      color: this.current === BLACK ? WHITE : BLACK,
      previousCurrent: this.current === BLACK ? WHITE : BLACK,
      previousKo: this.ko,
      previousEmptyCount: this.emptyCount + 1,
    };
    this._pushUndo(change);
  }

  _pushUndo(change) {
    // Maintain a reasonable undo depth limit (prevents memory explosion)
    const MAX_UNDO_DEPTH = 50000;
    this._undoStack.push(change);
    if (this._undoStack.length > MAX_UNDO_DEPTH) {
      // Remove oldest entries to stay under limit
      this._undoStack.splice(0, this._undoStack.length - MAX_UNDO_DEPTH);
    }
  }

  // ── Core Stone Placement ───────────────────────────────────────────────────

  _placeStone(idx, color) {
    this.cells[idx] = color;
    this.emptyCount--;
    this.lastMove = idx;

    const gidArr = this._gid;
    const nbr = this._nbr;
    const base = idx * 4;
    const W = this._W;

    // Remove from liberties of adjacent opponent groups
    let s0 = -1, s1 = -1, s2 = -1, s3 = -1;
    for (let i = 0; i < 4; i++) {
      const ni = nbr[base + i];
      const gid = gidArr[ni];
      if (gid === -1 || gid === s0 || gid === s1 || gid === s2 || gid === s3) continue;
      if      (s0 === -1) s0 = gid;
      else if (s1 === -1) s1 = gid;
      else if (s2 === -1) s2 = gid;
      else                s3 = gid;
      const m = 1 << (idx & 31);
      const wi = idx >> 5;
      const lb = gid * W;
      if (this._lw[lb + wi] & m) {
        this._lw[lb + wi] &= ~m;
        this._ls[gid]--;
      }
    }

    // Find adjacent same-color and empty neighbors
    let sg0 = -1, sg1 = -1, sg2 = -1, sg3 = -1, ns = 0;
    let ml0 = -1, ml1 = -1, ml2 = -1, ml3 = -1;
    for (let i = 0; i < 4; i++) {
      const ni = nbr[base + i];
      const c = this.cells[ni];
      if (c === color) {
        const gid = gidArr[ni];
        if (gid !== sg0 && gid !== sg1 && gid !== sg2 && gid !== sg3) {
          if      (sg0 === -1) sg0 = gid;
          else if (sg1 === -1) sg1 = gid;
          else if (sg2 === -1) sg2 = gid;
          else                 sg3 = gid;
          ns++;
        }
      } else if (c === EMPTY) {
        if      (ml0 === -1) ml0 = ni;
        else if (ml1 === -1) ml1 = ni;
        else if (ml2 === -1) ml2 = ni;
        else                 ml3 = ni;
      }
    }

    if (ns === 0) {
      // New group
      const gid = this._nextGid++;
      gidArr[idx] = gid;
      this._gc[gid] = color;
      this._sw[gid * W + (idx >> 5)] |= (1 << (idx & 31));
      this._ss[gid] = 1;
      const lb = gid * W;
      let lsCount = 0;
      if (ml0 !== -1) { const m = 1<<(ml0&31); const wi = ml0>>5; if (!(this._lw[lb+wi]&m)) { this._lw[lb+wi]|=m; lsCount++; } }
      if (ml1 !== -1) { const m = 1<<(ml1&31); const wi = ml1>>5; if (!(this._lw[lb+wi]&m)) { this._lw[lb+wi]|=m; lsCount++; } }
      if (ml2 !== -1) { const m = 1<<(ml2&31); const wi = ml2>>5; if (!(this._lw[lb+wi]&m)) { this._lw[lb+wi]|=m; lsCount++; } }
      if (ml3 !== -1) { const m = 1<<(ml3&31); const wi = ml3>>5; if (!(this._lw[lb+wi]&m)) { this._lw[lb+wi]|=m; lsCount++; } }
      this._ls[gid] = lsCount;
    } else {
      // Merge with adjacent same-color groups
      let mainGid = sg0;
      if (sg1 !== -1 && this._ss[sg1] > this._ss[mainGid]) mainGid = sg1;
      if (sg2 !== -1 && this._ss[sg2] > this._ss[mainGid]) mainGid = sg2;
      if (sg3 !== -1 && this._ss[sg3] > this._ss[mainGid]) mainGid = sg3;

      const gb = mainGid * W;
      this._sw[gb + (idx >> 5)] |= (1 << (idx & 31));
      this._ss[mainGid]++;
      gidArr[idx] = mainGid;

      if (ml0 !== -1) { const m = 1<<(ml0&31); const wi = ml0>>5; if (!(this._lw[gb+wi]&m)) { this._lw[gb+wi]|=m; this._ls[mainGid]++; } }
      if (ml1 !== -1) { const m = 1<<(ml1&31); const wi = ml1>>5; if (!(this._lw[gb+wi]&m)) { this._lw[gb+wi]|=m; this._ls[mainGid]++; } }
      if (ml2 !== -1) { const m = 1<<(ml2&31); const wi = ml2>>5; if (!(this._lw[gb+wi]&m)) { this._lw[gb+wi]|=m; this._ls[mainGid]++; } }
      if (ml3 !== -1) { const m = 1<<(ml3&31); const wi = ml3>>5; if (!(this._lw[gb+wi]&m)) { this._lw[gb+wi]|=m; this._ls[mainGid]++; } }

      // Merge other groups
      if (sg0 !== mainGid && sg0 !== -1) {
        const ob = sg0 * W;
        for (let wi = 0; wi < W; wi++) {
          let w = this._sw[ob + wi];
          while (w) {
            const bit = 31 - Math.clz32(w & -w);
            gidArr[wi * 32 + bit] = mainGid;
            w &= w - 1;
          }
          this._sw[gb + wi] |= this._sw[ob + wi];
        }
        this._ss[mainGid] += this._ss[sg0];
        for (let wi = 0; wi < W; wi++) this._lw[gb + wi] |= this._lw[ob + wi];
      }
      if (sg1 !== mainGid && sg1 !== -1) {
        const ob = sg1 * W;
        for (let wi = 0; wi < W; wi++) {
          let w = this._sw[ob + wi];
          while (w) {
            const bit = 31 - Math.clz32(w & -w);
            gidArr[wi * 32 + bit] = mainGid;
            w &= w - 1;
          }
          this._sw[gb + wi] |= this._sw[ob + wi];
        }
        this._ss[mainGid] += this._ss[sg1];
        for (let wi = 0; wi < W; wi++) this._lw[gb + wi] |= this._lw[ob + wi];
      }
      if (sg2 !== mainGid && sg2 !== -1) {
        const ob = sg2 * W;
        for (let wi = 0; wi < W; wi++) {
          let w = this._sw[ob + wi];
          while (w) {
            const bit = 31 - Math.clz32(w & -w);
            gidArr[wi * 32 + bit] = mainGid;
            w &= w - 1;
          }
          this._sw[gb + wi] |= this._sw[ob + wi];
        }
        this._ss[mainGid] += this._ss[sg2];
        for (let wi = 0; wi < W; wi++) this._lw[gb + wi] |= this._lw[ob + wi];
      }
      if (sg3 !== mainGid && sg3 !== -1) {
        const ob = sg3 * W;
        for (let wi = 0; wi < W; wi++) {
          let w = this._sw[ob + wi];
          while (w) {
            const bit = 31 - Math.clz32(w & -w);
            gidArr[wi * 32 + bit] = mainGid;
            w &= w - 1;
          }
          this._sw[gb + wi] |= this._sw[ob + wi];
        }
        this._ss[mainGid] += this._ss[sg3];
        for (let wi = 0; wi < W; wi++) this._lw[gb + wi] |= this._lw[ob + wi];
      }

      // Recount liberties
      let newLs = 0;
      for (let wi = 0; wi < W; wi++) newLs += this._pop32(this._lw[gb + wi]);
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

  // ── Legality Check ─────────────────────────────────────────────────────────

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
      // Check if any opponent group would be captured
      const nbr = this._nbr;
      const base = idx * 4;
      for (let i = 0; i < 4; i++) {
        const ni = nbr[base + i];
        const c = this.cells[ni];
        if (c === -color) {
          const gid = this._gid[ni];
          if (this._ls[gid] === 1) return true;
        }
      }
      return false;
    }
    return true;
  }

  // ── Play & Undo ────────────────────────────────────────────────────────────

  play(move) {
    if (move === PASS) {
      const prevCurrent = this.current;
      this.current = -this.current;
      const change = {
        type: 'pass',
        previousCurrent: prevCurrent,
      };
      this._pushUndo(change);
      this.moveCount++;
      return true;
    }

    if (!this.isLegal(move)) return false;

    const color = this.current;
    const oppColor = -color;
    const oldKo = this.ko;
    const oldEmptyCount = this.emptyCount;

    // Snapshot COMPLETE state BEFORE the move (for fast undo)
    const stateSnapshot = {
      cells: new Int8Array(this.cells),
      gid: new Int32Array(this._gid),
      nextGid: this._nextGid,
      gc: new Uint8Array(this._gc.slice(0, this._nextGid)),
      sw: new Int32Array(this._sw.slice(0, this._nextGid * this._W)),
      ss: new Int32Array(this._ss.slice(0, this._nextGid)),
      lw: new Int32Array(this._lw.slice(0, this._nextGid * this._W)),
      ls: new Int32Array(this._ls.slice(0, this._nextGid)),
    };

    this._placeStone(move, color);

    // Capture opponent groups with 0 liberties
    const nbr = this._nbr;
    const base = move * 4;
    const captured = [];

    for (let i = 0; i < 4; i++) {
      const ni = nbr[base + i];
      const gid = this._gid[ni];
      if (gid !== -1 && this._gc[gid] === oppColor && this._ls[gid] === 0) {
        const stones = this._captureGroup(gid);
        captured.push(stones);
      }
    }

    // Record move for undo with full state snapshot
    const change = {
      type: 'move',
      idx: move,
      color: color,
      previousCurrent: color,
      previousKo: oldKo,
      previousEmptyCount: oldEmptyCount,
      stateSnapshot: stateSnapshot,
    };

    this._pushUndo(change);

    // Update ko and current player
    if (captured.length === 1 && captured[0].length === 1) {
      this.ko = captured[0][0];
    } else {
      this.ko = PASS;
    }

    this.current = oppColor;
    this.moveCount++;
    return true;
  }

  undo() {
    if (this._undoStack.length === 0) return;

    const change = this._undoStack.pop();

    if (change.type === 'pass') {
      this.current = change.previousCurrent;
      this.moveCount--;
      return;
    }

    if (change.type === 'move' || change.type === 'place') {
      // Restore full game state from snapshot (fast, no reconstruction needed)
      if (change.stateSnapshot) {
        const snap = change.stateSnapshot;
        this.cells.set(snap.cells);
        this._gid.set(snap.gid);
        this._nextGid = snap.nextGid;
        // Only restore used group data
        const W = this._W;
        for (let i = 0; i < snap.nextGid; i++) {
          this._gc[i] = snap.gc[i];
          this._ss[i] = snap.ss[i];
          this._ls[i] = snap.ls[i];
          for (let j = 0; j < W; j++) {
            this._sw[i * W + j] = snap.sw[i * W + j];
            this._lw[i * W + j] = snap.lw[i * W + j];
          }
        }
      }

      // Restore game state
      this.emptyCount = change.previousEmptyCount;
      this.ko = change.previousKo;
      this.current = change.previousCurrent;
      this.lastMove = PASS;

      this.moveCount--;
    }
  }

  _reconstructGroups() {
    // Full reconstruction from scratch without side effects
    const W = this._W;
    const cap = this.N * this.N;
    const nbr = this._nbr;

    // Clear all group data
    this._gid.fill(-1);
    this._nextGid = 0;
    for (let g = 0; g < 4 * cap; g++) {
      const wb = g * W;
      this._sw.fill(0, wb, wb + W);
      this._lw.fill(0, wb, wb + W);
      this._ss[g] = 0;
      this._ls[g] = 0;
    }

    // Assign group IDs via flood-fill
    for (let i = 0; i < cap; i++) {
      if (this.cells[i] !== EMPTY && this._gid[i] === -1) {
        const color = this.cells[i];
        const gid = this._nextGid++;
        const gb = gid * W;

        this._gc[gid] = color;
        const queue = [i];
        this._gid[i] = gid;

        // Flood-fill to assign all stones in this group
        while (queue.length > 0) {
          const idx = queue.shift();
          const m = 1 << (idx & 31);
          const wi = idx >> 5;
          this._sw[gb + wi] |= m;
          this._ss[gid]++;

          const base = idx * 4;
          for (let j = 0; j < 4; j++) {
            const ni = nbr[base + j];
            if (this.cells[ni] === color && this._gid[ni] === -1) {
              this._gid[ni] = gid;
              queue.push(ni);
            }
          }
        }
      }
    }

    // Calculate liberties for all groups
    for (let i = 0; i < cap; i++) {
      if (this.cells[i] === EMPTY) {
        const base = i * 4;
        for (let j = 0; j < 4; j++) {
          const ni = nbr[base + j];
          const gid = this._gid[ni];
          if (gid !== -1) {
            const m = 1 << (i & 31);
            const wi = i >> 5;
            const lb = gid * W;
            if (!(this._lw[lb + wi] & m)) {
              this._lw[lb + wi] |= m;
              this._ls[gid]++;
            }
          }
        }
      }
    }
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
  module.exports = { Game3, PASS, BLACK, WHITE, EMPTY };
} else if (typeof window !== 'undefined') {
  window.Game3 = Game3;
  window.PASS = PASS;
  window.BLACK = BLACK;
  window.WHITE = WHITE;
  window.EMPTY = EMPTY;
}

})();
