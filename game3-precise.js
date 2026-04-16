'use strict';

// Game3-Precise — Fully reversible incremental game with NO reconstruction
//
// Core principle: Every operation has an exact, precise undo.
// No state reconstruction ever needed.
//
// Operations are:
//   - AddStone(idx, gid): Add stone at index to group
//   - RemoveStone(idx, gid): Remove stone from group
//   - MergeGroup(mainGid, otherId, stones, liberties): Merge group into main
//   - SplitGroup(mainGid, otherId, stones, liberties): Split group back
//   - AddLiberty(gid, idx): Add liberty to group
//   - RemoveLiberty(gid, idx): Remove liberty from group
//
// Each operation can be undone by reversing its exact changes.

(function() {

const EMPTY = 0, BLACK = 1, WHITE = -1;
const PASS = -1;

class Game3Precise {
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
    this.gameOver = false;
    this.consecutivePasses = 0;

    // Group tracking
    this._gid = new Int32Array(cap).fill(-1);
    this._nextGid = 0;

    // Bitset arrays
    const W = (cap + 31) >> 5;
    this._W = W;
    const MAX_G = 4 * cap + 4;
    this._gc = new Uint8Array(MAX_G);       // group color
    this._sw = new Int32Array(MAX_G * W);   // stones bitset
    this._ss = new Int32Array(MAX_G);       // stones count
    this._lw = new Int32Array(MAX_G * W);   // liberties bitset
    this._ls = new Int32Array(MAX_G);       // liberties count

    // Topology
    const tables = this._getTopology(size);
    this._nbr = tables.nbr;
    this._dnbr = tables.dnbr;
    this.nbr = tables.nbr;
    this._allCells = tables.allCells;

    // Operation stack - stores reversible operations
    this._opStack = [];

    // Initialize center stone
    const center = ((size >> 1) * size) + (size >> 1);
    const centerGid = this._nextGid++;
    this._gc[centerGid] = BLACK;
    this._gid[center] = centerGid;
    this.cells[center] = BLACK;
    this._addStone(center, centerGid, BLACK);
    this.emptyCount--;
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

  // ── Precise Operations ─────────────────────────────────────────────────────
  // Raw versions that modify state without recording operations (for undo)

  _addStone_raw(idx, gid) {
    const W = this._W;
    const m = 1 << (idx & 31);
    const wi = idx >> 5;
    const gb = gid * W;
    this._sw[gb + wi] |= m;
    this._ss[gid]++;
  }

  _removeStone_raw(idx, gid) {
    const W = this._W;
    const m = 1 << (idx & 31);
    const wi = idx >> 5;
    const gb = gid * W;
    this._sw[gb + wi] &= ~m;
    this._ss[gid]--;
  }

  _addLiberty_raw(gid, idx) {
    const W = this._W;
    const m = 1 << (idx & 31);
    const wi = idx >> 5;
    const lb = gid * W;
    this._lw[lb + wi] |= m;
    this._ls[gid]++;
  }

  _removeLiberty_raw(gid, idx) {
    const W = this._W;
    const m = 1 << (idx & 31);
    const wi = idx >> 5;
    const lb = gid * W;
    this._lw[lb + wi] &= ~m;
    this._ls[gid]--;
  }

  // Recording versions that also push operations
  _addStone(idx, gid, color) {
    this._addStone_raw(idx, gid);
    this._opStack.push({
      type: 'addStone',
      idx: idx,
      gid: gid,
    });
  }

  _removeStone(idx, gid) {
    this._removeStone_raw(idx, gid);
    this._opStack.push({
      type: 'removeStone',
      idx: idx,
      gid: gid,
    });
  }

  _addLiberty(gid, idx) {
    const W = this._W;
    const m = 1 << (idx & 31);
    const wi = idx >> 5;
    const lb = gid * W;
    const wasSet = !!(this._lw[lb + wi] & m);
    if (!wasSet) {
      this._addLiberty_raw(gid, idx);
      this._opStack.push({
        type: 'addLiberty',
        gid: gid,
        idx: idx,
      });
    }
  }

  _removeLiberty(gid, idx) {
    const W = this._W;
    const m = 1 << (idx & 31);
    const wi = idx >> 5;
    const lb = gid * W;
    const wasSet = !!(this._lw[lb + wi] & m);
    if (wasSet) {
      this._removeLiberty_raw(gid, idx);
      this._opStack.push({
        type: 'removeLiberty',
        gid: gid,
        idx: idx,
      });
    }
  }

  _mergeGroups(mainGid, otherId) {
    // Merge group into main with exact reversal
    // Records which stones and liberties came from the other group
    const W = this._W;
    const gb = mainGid * W;
    const ob = otherId * W;

    // Snapshot the other group's data for reversal
    const otherStones = new Int32Array(W);
    const otherLibs = new Int32Array(W);
    for (let wi = 0; wi < W; wi++) {
      otherStones[wi] = this._sw[ob + wi];
      otherLibs[wi] = this._lw[ob + wi];
    }
    const otherSize = this._ss[otherId];
    const otherLibCount = this._ls[otherId];

    // Merge stones
    for (let wi = 0; wi < W; wi++) {
      let w = this._sw[ob + wi];
      while (w) {
        const bit = 31 - Math.clz32(w & -w);
        this._gid[wi * 32 + bit] = mainGid;
        w &= w - 1;
      }
      this._sw[gb + wi] |= this._sw[ob + wi];
    }
    this._ss[mainGid] += this._ss[otherId];

    // Merge liberties
    for (let wi = 0; wi < W; wi++) {
      this._lw[gb + wi] |= this._lw[ob + wi];
    }
    this._ls[mainGid] = this._pop32Count(mainGid, W);

    // Record merge operation with snapshots for reversal
    this._opStack.push({
      type: 'mergeGroups',
      mainGid: mainGid,
      otherId: otherId,
      otherStones: otherStones,
      otherLibs: otherLibs,
      otherSize: otherSize,
      otherLibCount: otherLibCount,
    });
  }

  _pop32Count(gid, W) {
    let count = 0;
    const gb = gid * W;
    for (let wi = 0; wi < W; wi++) {
      count += this._pop32(this._lw[gb + wi]);
    }
    return count;
  }

  // ── Play & Undo ────────────────────────────────────────────────────────────

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

  isValidMove(idx, color = this.current) {
    return this.isLegal(idx, color) && !this.isTrueEye(idx);
  }

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

  play(move) {
    if (move === PASS) {
      const previousConsecutivePasses = this.consecutivePasses;
      this._opStack.push({
        type: 'pass',
        previousCurrent: this.current,
        previousConsecutivePasses: previousConsecutivePasses,
      });
      this.consecutivePasses++;
      if (this.consecutivePasses >= 2) {
        this.gameOver = true;
      }
      this.current = -this.current;
      this.moveCount++;
      return true;
    }

    if (!this.isLegal(move)) return false;

    const color = this.current;
    const oppColor = -color;
    const nbr = this._nbr;
    const base = move * 4;
    const W = this._W;

    // Record move start for grouping operations
    const opCountBefore = this._opStack.length;

    // Save state before any modifications
    const previousKo = this.ko;
    const previousEmptyCount = this.emptyCount;
    const previousConsecutivePasses = this.consecutivePasses;
    const previousLastMove = this.lastMove;

    // Step 1: Mark cell as occupied
    this.cells[move] = color;
    this.emptyCount--;
    this.lastMove = move;

    // Step 2: Remove liberty from adjacent opponent groups
    const oppGroupIds = [];
    for (let i = 0; i < 4; i++) {
      const ni = nbr[base + i];
      const gid = this._gid[ni];
      if (gid !== -1 && !oppGroupIds.includes(gid)) {
        oppGroupIds.push(gid);
        this._removeLiberty(gid, move);
      }
    }

    // Step 3: Find adjacent same-color groups
    const sameColorGroupIds = [];
    const emptyNeighbors = [];
    for (let i = 0; i < 4; i++) {
      const ni = nbr[base + i];
      const c = this.cells[ni];
      if (c === color) {
        const gid = this._gid[ni];
        if (!sameColorGroupIds.includes(gid)) sameColorGroupIds.push(gid);
      } else if (c === EMPTY && !emptyNeighbors.includes(ni)) {
        emptyNeighbors.push(ni);
      }
    }

    // Step 4: Create or merge group
    let mainGid;
    if (sameColorGroupIds.length === 0) {
      // Create new group
      mainGid = this._nextGid++;
      this._gc[mainGid] = color;
      this._gid[move] = mainGid;
      this._addStone(move, mainGid, color);
    } else {
      // Find largest group to merge into
      mainGid = sameColorGroupIds[0];
      for (let i = 1; i < sameColorGroupIds.length; i++) {
        if (this._ss[sameColorGroupIds[i]] > this._ss[mainGid]) {
          mainGid = sameColorGroupIds[i];
        }
      }
      // Add stone to main group
      this._gid[move] = mainGid;
      this._addStone(move, mainGid, color);

      // Merge other groups
      for (const otherId of sameColorGroupIds) {
        if (otherId !== mainGid) {
          this._mergeGroups(mainGid, otherId);
        }
      }
    }

    // Step 5: Add liberties
    for (const lib of emptyNeighbors) {
      this._addLiberty(mainGid, lib);
    }

    // Step 6: Capture opponent groups with 0 liberties
    const captured = [];
    for (const oppGid of oppGroupIds) {
      if (this._ls[oppGid] === 0) {
        // Remove all stones from this group
        const gb = oppGid * W;
        for (let wi = 0; wi < W; wi++) {
          let w = this._sw[gb + wi];
          while (w) {
            const bit = 31 - Math.clz32(w & -w);
            const stoneIdx = wi * 32 + bit;
            this.cells[stoneIdx] = EMPTY;
            this._gid[stoneIdx] = -1;
            this._removeStone(stoneIdx, oppGid);
            captured.push(stoneIdx);
            this.emptyCount++;

            // Add liberties to adjacent groups
            const sBase = stoneIdx * 4;
            for (let j = 0; j < 4; j++) {
              const nGid = this._gid[nbr[sBase + j]];
              if (nGid !== -1 && nGid !== oppGid) {
                this._addLiberty(nGid, stoneIdx);
              }
            }
            w &= w - 1;
          }
        }
      }
    }

    // Step 7: Update ko rule
    if (captured.length === 1) {
      this.ko = captured[0];
    } else {
      this.ko = PASS;
    }

    // Wrap operations in a move record
    this._opStack.push({
      type: 'move',
      move: move,
      color: color,
      previousCurrent: color,
      previousKo: previousKo,
      previousEmptyCount: previousEmptyCount,
      previousConsecutivePasses: previousConsecutivePasses,
      previousLastMove: previousLastMove,
      opsStart: opCountBefore,
      captured: captured,
    });

    this.consecutivePasses = 0;
    this.current = oppColor;
    this.moveCount++;
    return true;
  }

  undo() {
    if (this._opStack.length === 0) return;

    // Pop operations in reverse order until we hit a move marker
    while (this._opStack.length > 0) {
      const op = this._opStack.pop();

      if (op.type === 'pass') {
        this.current = op.previousCurrent;
        this.consecutivePasses = op.previousConsecutivePasses;
        if (this.consecutivePasses < 2) {
          this.gameOver = false;
        }
        this.moveCount--;
        return;
      }

      if (op.type === 'move') {
        // Undo all operations from this move
        while (this._opStack.length > op.opsStart) {
          this._undoOperation(this._opStack.pop());
        }

        // Restore board cell and group ID
        this.cells[op.move] = EMPTY;
        this._gid[op.move] = -1;

        // Restore captured stones
        if (op.captured) {
          for (const stoneIdx of op.captured) {
            this.cells[stoneIdx] = -op.color; // Opponent color
            // _gid will be set by mergeGroups undo
          }
        }

        this.current = op.previousCurrent;
        this.ko = op.previousKo;
        this.emptyCount = op.previousEmptyCount;
        this.consecutivePasses = op.previousConsecutivePasses;
        this.lastMove = op.previousLastMove;
        this.moveCount--;
        return;
      }

      // Undo individual operation
      this._undoOperation(op);
    }
  }

  _undoOperation(op) {
    const W = this._W;

    if (op.type === 'addStone') {
      // Undo: remove stone (use raw to avoid pushing new operation)
      this._removeStone_raw(op.idx, op.gid);
    } else if (op.type === 'removeStone') {
      // Undo: add stone back (use raw to avoid pushing new operation)
      this._addStone_raw(op.idx, op.gid);
    } else if (op.type === 'addLiberty') {
      // Undo: remove liberty
      this._removeLiberty_raw(op.gid, op.idx);
    } else if (op.type === 'removeLiberty') {
      // Undo: add liberty back
      this._addLiberty_raw(op.gid, op.idx);
    } else if (op.type === 'mergeGroups') {
      // Undo: restore other group
      const gb = op.mainGid * W;
      const ob = op.otherId * W;

      // Restore stones from other group
      for (let wi = 0; wi < W; wi++) {
        let w = op.otherStones[wi];
        while (w) {
          const bit = 31 - Math.clz32(w & -w);
          this._gid[wi * 32 + bit] = op.otherId;
          w &= w - 1;
        }
        // Remove merged stones from main group
        this._sw[gb + wi] &= ~op.otherStones[wi];
        // Restore other group's stones
        this._sw[ob + wi] = op.otherStones[wi];
        // Restore other group's liberties
        this._lw[ob + wi] = op.otherLibs[wi];
      }

      // Restore sizes and liberty counts
      this._ss[op.mainGid] -= op.otherSize;
      this._ss[op.otherId] = op.otherSize;
      this._ls[op.mainGid] = this._pop32Count(op.mainGid, W);
      this._ls[op.otherId] = op.otherLibCount;
    }
  }

  // ── Display ────────────────────────────────────────────────────────────────

  // Render the board as a ● ○ · string. Optional 'markIdx' marks that
  // cell with bracket separators: · ·(●)· · — row width is unchanged.
  toString(markIdx = this.lastMove, { centerAt = null } = {}) {
    const N = this.N;
    const cells = this.cells;
    const markX = (markIdx !== PASS) ? markIdx % N : -1;
    const markY = (markIdx !== PASS) ? (markIdx / N | 0) : -1;

    const half = (N / 2) | 0;
    const x0 = (centerAt !== null && centerAt !== PASS)
      ? ((centerAt % N) - half + N) % N : 0;
    const y0 = (centerAt !== null && centerAt !== PASS)
      ? (((centerAt / N) | 0) - half + N) % N : 0;

    // Convert board-space mark to display-space coordinates.
    const dmx = markX >= 0 ? (markX - x0 + N) % N : -1;
    const dmy = markY >= 0 ? (markY - y0 + N) % N : -1;

    const rows = [];
    for (let dy = 0; dy < N; dy++) {
      const by = (y0 + dy) % N;
      const mx = (dy === dmy) ? dmx : -1;
      let row = (mx === 0) ? '(' : ' ';
      for (let dx = 0; dx < N; dx++) {
        const bx = (x0 + dx) % N;
        const c = cells[by * N + bx];
        const ch = c === BLACK ? '●' : c === WHITE ? '○' : '·';
        if (dx > 0) row += (dx === mx) ? '(' : (dx - 1 === mx) ? ')' : ' ';
        row += ch;
      }
      row += (mx === N - 1) ? ')' : ' ';
      rows.push(row);
    }
    return rows.join('\n');
  }

  // ── Scoring ────────────────────────────────────────────────────────────────

  // Fast 1-step area estimate plus komi.
  // Returns { black, white } where white already includes komi.
  estimateScore() {
    const N = this.N;
    const cap = N * N;
    const cells = this.cells;
    const nbr = this._nbr;
    let black = 0;
    let white = 0;

    for (let i = 0; i < cap; i++) {
      const c = cells[i];
      if (c === BLACK) {
        black++;
        continue;
      }
      if (c === WHITE) {
        white++;
        continue;
      }

      // Empty cell - check adjacent stones
      const base = i * 4;
      let bAdj = false;
      let wAdj = false;
      for (let k = 0; k < 4; k++) {
        const nc = cells[nbr[base + k]];
        if (nc === BLACK) bAdj = true;
        else if (nc === WHITE) wAdj = true;
      }

      // Territory: assign to color if only one color is adjacent
      if (bAdj && !wAdj) black++;
      else if (wAdj && !bAdj) white++;
      // else: neutral territory (adjacent to both or neither)
    }

    // Add komi (board-size dependent)
    // Standard komi values: 3.5 (default), 6.5 (9x9), 7.5 (13x13), 35.5 (19x19)
    const komiOverrides = new Map([
      [5, 3.5],
      [7, 3.5],
      [9, 6.5],
      [11, 3.5],
      [13, 7.5],
      [19, 35.5],
    ]);
    const komi = komiOverrides.get(N) ?? 3.5;
    white += komi;

    return { black, white };
  }

  // Fast 1-step area estimate. Returns BLACK or WHITE.
  estimateWinner() {
    const score = this.estimateScore();
    return score.black > score.white ? BLACK : WHITE;
  }

  // ── Eye Detection ─────────────────────────────────────────────────────────

  isTrueEye(idx) {
    const color = this.current;
    const cells = this.cells;
    const gidArr = this._gid;
    const nbr = this._nbr;
    const dnbr = this._dnbr;
    const base = idx * 4;

    // Check all 4 neighbors
    let firstGid = -2, friendCount = 0, emptyCount = 0, sameGroup = 0;
    for (let i = 0; i < 4; i++) {
      const ni = nbr[base + i];
      const c = cells[ni];
      if (c === color) {
        friendCount++;
        const gid = gidArr[ni];
        if (firstGid === -2) { firstGid = gid; sameGroup = 1; }
        else if (gid === firstGid) sameGroup++;
      } else if (c === EMPTY) {
        emptyCount++;
      }
    }

    // 3 same-group friends + 1 empty: proto-eye, treat as true eye
    if (friendCount === 3 && emptyCount === 1 && sameGroup === 3) return true;
    if (friendCount < 4) return false;
    if (sameGroup === 4) return true;

    // Check diagonals for friendly color
    let dc = 0;
    for (let i = 0; i < 4; i++) if (cells[dnbr[base + i]] === color) dc++;
    return dc >= 3;
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

  groupLibs2(idx) {
    const gid = this._gid[idx];
    if (gid === -1) return { count: 0, lib0: -1, lib1: -1 };
    const lc = this._ls[gid];
    if (lc === 0) return { count: 0, lib0: -1, lib1: -1 };
    const W = this._W;
    const lb = gid * W;
    const cap = this.N * this.N;
    let lib0 = -1, lib1 = -1, found = 0;
    for (let wi = 0; wi < W && found < 2; wi++) {
      let w = this._lw[lb + wi];
      while (w && found < 2) {
        const i = wi * 32 + (31 - Math.clz32(w & -w));
        if (i < cap) {
          if (found === 0) lib0 = i;
          else if (found === 1) lib1 = i;
          found++;
        }
        w &= w - 1;
      }
    }
    return { count: lc, lib0, lib1 };
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { Game3Precise, PASS, BLACK, WHITE, EMPTY };
} else if (typeof window !== 'undefined') {
  window.Game3Precise = Game3Precise;
  window.PASS = PASS;
  window.BLACK = BLACK;
  window.WHITE = WHITE;
  window.EMPTY = EMPTY;
}

})();
