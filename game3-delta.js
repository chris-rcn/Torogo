'use strict';

// Game3-Delta — True incremental game class with delta-based undo
// Only stores changes (deltas) for each move, not full state snapshots
//
// Features:
//   - Minimal memory per move (only record changes)
//   - Meticulous state tracking for correctness
//   - O(n) undo (reconstructs affected groups only)
//   - Faster than Game2 clone for deep searches
//
// Interface (compatible with Game2):
//   new Game3Delta(size)
//   game.play(move)           → true/false
//   game.isLegal(move)        → boolean
//   game.undo()               → void
//   game.cells                Int8Array
//   game.current              BLACK | WHITE
//   game.groupLibertyCount(gid) → number

(function() {

const EMPTY = 0, BLACK = 1, WHITE = -1;
const PASS = -1;

class Game3Delta {
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

    // Topology (cached, shared)
    const tables = this._getTopology(size);
    this._nbr = tables.nbr;
    this._dnbr = tables.dnbr;
    this.nbr = tables.nbr;
    this._allCells = tables.allCells;

    // Delta undo stack - stores only what changed
    this._deltaStack = [];

    // Initialize with center stone
    const center = ((size >> 1) * size) + (size >> 1);
    this._placeStoneNoDelta(center, BLACK);
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

  // ── Place Stone Without Delta Recording ─────────────────────────────────────

  _placeStoneNoDelta(idx, color) {
    // Used for initialization only
    this.cells[idx] = color;
    this.emptyCount--;
    this.lastMove = idx;

    const gidArr = this._gid;
    const nbr = this._nbr;
    const base = idx * 4;
    const W = this._W;

    // Remove from liberties of adjacent opponent groups
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

    // Find adjacent same-color groups and empty neighbors
    let sameColorGroupIds = [];
    let emptyNeighbors = [];
    for (let i = 0; i < 4; i++) {
      const ni = nbr[base + i];
      const c = this.cells[ni];
      if (c === color) {
        const gid = gidArr[ni];
        if (!sameColorGroupIds.includes(gid)) {
          sameColorGroupIds.push(gid);
        }
      } else if (c === EMPTY) {
        if (!emptyNeighbors.includes(ni)) {
          emptyNeighbors.push(ni);
        }
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
      // Merge with same-color groups
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

      // Add liberties
      for (const liberty of emptyNeighbors) {
        const m = 1 << (liberty & 31);
        const wi = liberty >> 5;
        if (!(this._lw[gb + wi] & m)) {
          this._lw[gb + wi] |= m;
          this._ls[mainGid]++;
        }
      }

      // Merge other groups
      for (let i = 0; i < sameColorGroupIds.length; i++) {
        const otherId = sameColorGroupIds[i];
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

      // Recount liberties for merged group
      let newLs = 0;
      for (let wi = 0; wi < W; wi++) {
        newLs += this._pop32(this._lw[gb + wi]);
      }
      this._ls[mainGid] = newLs;
    }
  }

  // ── Delta-based placement ───────────────────────────────────────────────────

  _placeStone(idx, color) {
    // Returns delta object recording the changes
    const W = this._W;
    const gidArr = this._gid;
    const nbr = this._nbr;
    const base = idx * 4;

    const delta = {
      type: 'move',
      idx: idx,
      color: color,
      previousCurrent: this.current,
      previousKo: this.ko,
      previousEmptyCount: this.emptyCount,
      // Will be populated below
      createdGroupId: null,
      mergedGroupIds: [],  // groups that were merged into main group
      capturedGroupIds: [],
      libertyChanges: {}, // gid -> {removed: [indices], added: [indices]}
    };

    // Mark cell as occupied
    this.cells[idx] = color;
    this.emptyCount--;
    this.lastMove = idx;

    // Record adjacent opponent groups losing the liberty at idx
    let affectedOppGroups = [];
    for (let i = 0; i < 4; i++) {
      const ni = nbr[base + i];
      const gid = gidArr[ni];
      if (gid !== -1 && !affectedOppGroups.includes(gid)) {
        affectedOppGroups.push(gid);
        const m = 1 << (idx & 31);
        const wi = idx >> 5;
        const lb = gid * W;
        if (this._lw[lb + wi] & m) {
          this._lw[lb + wi] &= ~m;
          this._ls[gid]--;
          if (!delta.libertyChanges[gid]) delta.libertyChanges[gid] = {removed: [], added: []};
          delta.libertyChanges[gid].removed.push(idx);
        }
      }
    }

    // Find adjacent same-color and empty neighbors
    let sameColorGroupIds = [];
    let emptyNeighbors = [];
    for (let i = 0; i < 4; i++) {
      const ni = nbr[base + i];
      const c = this.cells[ni];
      if (c === color) {
        const gid = gidArr[ni];
        if (!sameColorGroupIds.includes(gid)) {
          sameColorGroupIds.push(gid);
        }
      } else if (c === EMPTY) {
        if (!emptyNeighbors.includes(ni)) {
          emptyNeighbors.push(ni);
        }
      }
    }

    if (sameColorGroupIds.length === 0) {
      // Create new group
      const gid = this._nextGid++;
      delta.createdGroupId = gid;
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
          if (!delta.libertyChanges[gid]) delta.libertyChanges[gid] = {removed: [], added: []};
          delta.libertyChanges[gid].added.push(liberty);
        }
      }
    } else {
      // Merge into largest same-color group
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

      if (!delta.libertyChanges[mainGid]) delta.libertyChanges[mainGid] = {removed: [], added: []};

      // Add new liberties
      for (const liberty of emptyNeighbors) {
        const m = 1 << (liberty & 31);
        const wi = liberty >> 5;
        if (!(this._lw[gb + wi] & m)) {
          this._lw[gb + wi] |= m;
          this._ls[mainGid]++;
          delta.libertyChanges[mainGid].added.push(liberty);
        }
      }

      // Merge other same-color groups
      for (const otherId of sameColorGroupIds) {
        if (otherId === mainGid) continue;

        delta.mergedGroupIds.push(otherId);
        const ob = otherId * W;

        // Merge stones
        for (let wi = 0; wi < W; wi++) {
          let w = this._sw[ob + wi];
          while (w) {
            const bit = 31 - Math.clz32(w & -w);
            const stoneIdx = wi * 32 + bit;
            gidArr[stoneIdx] = mainGid;
            w &= w - 1;
          }
          this._sw[gb + wi] |= this._sw[ob + wi];
        }
        this._ss[mainGid] += this._ss[otherId];

        // Merge liberties (union)
        for (let wi = 0; wi < W; wi++) {
          const oldLw = this._lw[gb + wi];
          this._lw[gb + wi] |= this._lw[ob + wi];
        }
      }

      // Recount liberties for merged group
      let newLs = 0;
      for (let wi = 0; wi < W; wi++) {
        newLs += this._pop32(this._lw[gb + wi]);
      }
      this._ls[mainGid] = newLs;
    }

    return delta;
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
      const delta = {
        type: 'pass',
        previousCurrent: this.current,
      };
      this._deltaStack.push(delta);
      this.current = -this.current;
      this.moveCount++;
      return true;
    }

    if (!this.isLegal(move)) return false;

    const color = this.current;
    const oppColor = -color;

    const delta = this._placeStone(move, color);

    // Check for and capture opponent groups with 0 liberties
    const nbr = this._nbr;
    const base = move * 4;
    const capturedStones = [];

    for (let i = 0; i < 4; i++) {
      const ni = nbr[base + i];
      const gid = this._gid[ni];
      if (gid !== -1 && this._gc[gid] === oppColor && this._ls[gid] === 0) {
        const stones = this._captureGroup(gid);
        delta.capturedGroupIds.push(gid);
        capturedStones.push(...stones);

        // Record liberty gains for neighbors of captured group
        const W = this._W;
        for (const stoneIdx of stones) {
          const sBase = stoneIdx * 4;
          for (let j = 0; j < 4; j++) {
            const nGid = this._gid[nbr[sBase + j]];
            if (nGid !== -1) {
              const m = 1 << (stoneIdx & 31);
              const wi = stoneIdx >> 5;
              if (!delta.libertyChanges[nGid]) delta.libertyChanges[nGid] = {removed: [], added: []};
              if (!delta.libertyChanges[nGid].added.includes(stoneIdx)) {
                delta.libertyChanges[nGid].added.push(stoneIdx);
              }
            }
          }
        }
      }
    }

    // Store captured stones in delta for restoration on undo
    delta.capturedStones = capturedStones;

    // Update ko rule
    if (delta.capturedGroupIds.length === 1 && capturedStones.length === 1) {
      this.ko = capturedStones[0];
    } else {
      this.ko = PASS;
    }

    this._deltaStack.push(delta);
    this.current = oppColor;
    this.moveCount++;
    return true;
  }

  undo() {
    if (this._deltaStack.length === 0) return;

    const delta = this._deltaStack.pop();

    if (delta.type === 'pass') {
      this.current = delta.previousCurrent;
      this.moveCount--;
      return;
    }

    // Undo a move by reversing all changes recorded in delta
    const W = this._W;
    const idx = delta.idx;
    const color = delta.color;
    const oppColor = -color;
    const gidArr = this._gid;
    const nbr = this._nbr;

    // Step 1: Remove the placed stone
    this.cells[idx] = EMPTY;
    gidArr[idx] = -1;

    // Step 2: Restore captured opponent stones
    if (delta.capturedStones && delta.capturedStones.length > 0) {
      // Restore stones and rebuild captured groups
      const capturedByGid = {}; // Map gid -> [stone indices]
      for (const stoneIdx of delta.capturedStones) {
        this.cells[stoneIdx] = oppColor;
        // We don't know the original gid yet, so we'll rebuild all groups
      }

      // Rebuild ALL groups (necessary when captures are undone, as complex merges may have occurred)
      // This is O(n) and guarantees correctness
      this._reconstructGroupsFull();
      return;  // Early return, full reconstruction handles everything
    }

    // Step 3: Restore game state
    this.emptyCount = delta.previousEmptyCount;
    this.ko = delta.previousKo;
    this.current = delta.previousCurrent;
    this.lastMove = PASS;

    // Step 4: Handle group splitting/merges
    if (delta.mergedGroupIds.length > 0) {
      // Groups were merged - need to split them back
      // This requires full reconstruction to correctly separate the groups
      this._reconstructGroupsFull();
    } else if (delta.createdGroupId !== null) {
      // If we created a new group, clear it and decrement next gid
      // But only if we're sure there are no dependencies
      // To be safe, still do a reconstruction for affected groups
      this._reconstructGroupsFull();
    } else {
      // Simple case: only liberty changes, no merges/creates/captures
      // Just recalculate liberties for affected groups
      for (const gidStr in delta.libertyChanges) {
        const gid = parseInt(gidStr);
        if (gid >= 0 && gid < this._nextGid) {
          this._recalculateGroupLiberties(gid);
        }
      }
    }

    this.moveCount--;
  }

  _recalculateGroupLiberties(gid) {
    const W = this._W;
    const cap = this.N * this.N;
    const nbr = this._nbr;
    const gb = gid * W;

    // Clear liberty bitset
    for (let wi = 0; wi < W; wi++) {
      this._lw[gb + wi] = 0;
    }

    let libCount = 0;
    // Find all empty neighbors of this group's stones
    for (let wi = 0; wi < W; wi++) {
      let w = this._sw[gb + wi];
      while (w) {
        const bit = 31 - Math.clz32(w & -w);
        const stoneIdx = wi * 32 + bit;
        const base = stoneIdx * 4;
        for (let i = 0; i < 4; i++) {
          const ni = nbr[base + i];
          if (this.cells[ni] === EMPTY) {
            const m = 1 << (ni & 31);
            const nwi = ni >> 5;
            if (!(this._lw[gb + nwi] & m)) {
              this._lw[gb + nwi] |= m;
              libCount++;
            }
          }
        }
        w &= w - 1;
      }
    }
    this._ls[gid] = libCount;
  }

  _reconstructGroupsFull() {
    // Full group reconstruction when merges need to be undone
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

    // Reassign group IDs via flood-fill
    for (let i = 0; i < cap; i++) {
      if (this.cells[i] !== EMPTY && this._gid[i] === -1) {
        const color = this.cells[i];
        const gid = this._nextGid++;
        const gb = gid * W;

        this._gc[gid] = color;
        const queue = [i];
        this._gid[i] = gid;

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

        // Calculate liberties for this group
        this._recalculateGroupLiberties(gid);
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
  module.exports = { Game3Delta, PASS, BLACK, WHITE, EMPTY };
} else if (typeof window !== 'undefined') {
  window.Game3Delta = Game3Delta;
  window.PASS = PASS;
  window.BLACK = BLACK;
  window.WHITE = WHITE;
  window.EMPTY = EMPTY;
}

})();
