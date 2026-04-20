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

// Import coordStr from game2 for coordinate display
const { coordStr } = require('./game2.js');

// Operation types (integers, not strings)
const OP_ADD_STONE = 0;
const OP_REMOVE_STONE = 1;
const OP_ADD_LIBERTY = 2;
const OP_REMOVE_LIBERTY = 3;
const OP_MERGE_GROUPS = 4;
const OP_MOVE = 5;
const OP_PASS = 6;

const _topologyCache = new Map();

class Game3 {
  // Initialize game with center black stone, ready for white's first move
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

    this.current = BLACK;
    this.moveCount = 0;
  }

  // Get or compute neighbor tables and cell list for board size (with memoization)
  _getTopology(N) {
    if (_topologyCache.has(N)) {
      return _topologyCache.get(N);
    }

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
    const result = { nbr, dnbr, allCells };
    _topologyCache.set(N, result);
    return result;
  }

  // Population count: count set bits in 32-bit integer
  _pop32(x) {
    x = x - ((x >>> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
    x = (x + (x >>> 4)) & 0x0f0f0f0f;
    return (Math.imul(x, 0x01010101) >>> 24);
  }

  // ── Precise Operations ─────────────────────────────────────────────────────
  // Raw versions that modify state without recording operations (for undo)

  // Add stone to group's bitset without recording operation
  _addStone_raw(idx, gid) {
    const W = this._W;
    const m = 1 << (idx & 31);
    const wi = idx >> 5;
    const gb = gid * W;
    this._sw[gb + wi] |= m;
    this._ss[gid]++;
  }

  // Remove stone from group's bitset without recording operation
  _removeStone_raw(idx, gid) {
    const W = this._W;
    const m = 1 << (idx & 31);
    const wi = idx >> 5;
    const gb = gid * W;
    this._sw[gb + wi] &= ~m;
    this._ss[gid]--;
  }

  // Add liberty to group's bitset without recording operation
  _addLiberty_raw(gid, idx) {
    const W = this._W;
    const m = 1 << (idx & 31);
    const wi = idx >> 5;
    const lb = gid * W;
    this._lw[lb + wi] |= m;
    this._ls[gid]++;
  }

  // Remove liberty from group's bitset without recording operation
  _removeLiberty_raw(gid, idx) {
    const W = this._W;
    const m = 1 << (idx & 31);
    const wi = idx >> 5;
    const lb = gid * W;
    this._lw[lb + wi] &= ~m;
    this._ls[gid]--;
  }

  // Recording versions that also push operations
  // Add stone to group and record operation for undo
  _addStone(idx, gid, color) {
    this._addStone_raw(idx, gid);
    this._opStack.push({
      type: OP_ADD_STONE,
      idx: idx,
      gid: gid,
    });
  }

  // Remove stone from group and record operation for undo
  _removeStone(idx, gid) {
    this._removeStone_raw(idx, gid);
    this._opStack.push({
      type: OP_REMOVE_STONE,
      idx: idx,
      gid: gid,
    });
  }

  // Add liberty to group and record operation for undo
  _addLiberty(gid, idx) {
    const W = this._W;
    const m = 1 << (idx & 31);
    const wi = idx >> 5;
    const lb = gid * W;
    const wasSet = !!(this._lw[lb + wi] & m);
    if (!wasSet) {
      this._addLiberty_raw(gid, idx);
      this._opStack.push({
        type: OP_ADD_LIBERTY,
        gid: gid,
        idx: idx,
      });
    }
  }

  // Remove liberty from group and record operation for undo
  _removeLiberty(gid, idx) {
    const W = this._W;
    const m = 1 << (idx & 31);
    const wi = idx >> 5;
    const lb = gid * W;
    const wasSet = !!(this._lw[lb + wi] & m);
    if (wasSet) {
      this._removeLiberty_raw(gid, idx);
      this._opStack.push({
        type: OP_REMOVE_LIBERTY,
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

    // Snapshot the main group's data before merge (for exact reversal)
    const mainStones = new Int32Array(W);
    const mainLibs = new Int32Array(W);
    for (let wi = 0; wi < W; wi++) {
      mainStones[wi] = this._sw[gb + wi];
      mainLibs[wi] = this._lw[gb + wi];
    }
    const mainSize = this._ss[mainGid];
    const mainLibCount = this._ls[mainGid];

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
      type: OP_MERGE_GROUPS,
      mainGid: mainGid,
      otherId: otherId,
      mainStones: mainStones,
      mainLibs: mainLibs,
      mainSize: mainSize,
      mainLibCount: mainLibCount,
      otherStones: otherStones,
      otherLibs: otherLibs,
      otherSize: otherSize,
      otherLibCount: otherLibCount,
    });
  }

  // Count liberties in group using bitset population count
  _pop32Count(gid, W) {
    let count = 0;
    const gb = gid * W;
    for (let wi = 0; wi < W; wi++) {
      count += this._pop32(this._lw[gb + wi]);
    }
    return count;
  }

  // ── Play & Undo ────────────────────────────────────────────────────────────

  // Check if move is legal (empty, not ko, not suicide)
  isLegal(idx, color = this.current) {
    if (idx === PASS) return true;
    if (this.cells[idx] !== EMPTY) return false;
    if (this.ko === idx) return false;
    if (this._isSingleSuicide(idx, color)) return false;
    if (this._isMultiSuicide(idx, color)) return false;
    return true;
  }

  // Check if move is legal and not a true eye
  isValidMove(idx, color = this.current) {
    return this.isLegal(idx, color) && !this.isTrueEye(idx);
  }

  // Detect single suicide (all neighbors filled, none can help)
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

  _isMultiSuicide(idx, color) {
    // Checks if playing here would capture enemy stones but leave our own with 0 liberties
    const cells = this.cells;
    const gidArr = this._gid;
    const nbr = this._nbr;
    const base = idx * 4;
    const ls = this._ls;
    const lw = this._lw;
    const W = this._W;
    let hasFriendly = false;
    let s0 = -1, s1 = -1, s2 = -1, s3 = -1;

    for (let i = 0; i < 4; i++) {
      const ni = nbr[base + i];
      const c = cells[ni];
      if (c === EMPTY) return false;
      const gid = gidArr[ni];
      if (gid === s0 || gid === s1 || gid === s2 || gid === s3) continue;
      if      (s0 === -1) s0 = gid;
      else if (s1 === -1) s1 = gid;
      else if (s2 === -1) s2 = gid;
      else                s3 = gid;
      if (c === color) {
        hasFriendly = true;
        if (ls[gid] > 1) return false;
        if (ls[gid] === 1 && !((lw[gid * W + (idx >> 5)] >>> (idx & 31)) & 1)) return false;
      } else {
        if (ls[gid] === 1 && ((lw[gid * W + (idx >> 5)] >>> (idx & 31)) & 1)) return false;
      }
    }
    return hasFriendly;
  }

  // Play a move (or pass), record operation, update game state
  play(move) {
    if (move === PASS) {
      const previousConsecutivePasses = this.consecutivePasses;
      this._opStack.push({
        type: OP_PASS,
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
      if (this.cells[ni] === oppColor) {
        const gid = this._gid[ni];
        if (gid !== -1 && !oppGroupIds.includes(gid)) {
          oppGroupIds.push(gid);
          this._removeLiberty(gid, move);
        }
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

      // Remove the newly placed stone from liberties (it was empty before placement)
      this._removeLiberty(mainGid, move);
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
            // Store both index and gid for proper restoration on undo
            captured.push({ idx: stoneIdx, gid: oppGid });
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
      this.ko = captured[0].idx;
    } else {
      this.ko = PASS;
    }

    // Wrap operations in a move record
    this._opStack.push({
      type: OP_MOVE,
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

  // Undo last move by reversing all recorded operations
  undo() {
    if (this._opStack.length === 0) return;

    // Pop operations in reverse order until we hit a move marker
    while (this._opStack.length > 0) {
      const op = this._opStack.pop();

      if (op.type === OP_PASS) {
        this.current = op.previousCurrent;
        this.consecutivePasses = op.previousConsecutivePasses;
        if (this.consecutivePasses < 2) {
          this.gameOver = false;
        }
        this.moveCount--;
        return;
      }

      if (op.type === OP_MOVE) {
        // Undo all operations from this move
        while (this._opStack.length > op.opsStart) {
          this._undoOperation(this._opStack.pop());
        }

        // Restore board cell and group ID
        this.cells[op.move] = EMPTY;
        this._gid[op.move] = -1;

        // Restore captured stones with their group IDs
        if (op.captured) {
          for (const stone of op.captured) {
            const stoneIdx = stone.idx || stone;  // Handle both old and new format
            const gid = stone.gid;
            // op.color is always stored as 1 or -1, so negation is safe
            // (op.color is the color that made the move, so opponent is -op.color)
            this.cells[stoneIdx] = -op.color; // Opponent color
            if (gid !== undefined) {
              this._gid[stoneIdx] = gid;  // Restore group ID
            }
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

  // Reverse a single recorded operation
  _undoOperation(op) {
    const W = this._W;

    if (op.type === OP_ADD_STONE) {
      // Undo: remove stone (use raw to avoid pushing new operation)
      this._removeStone_raw(op.idx, op.gid);
    } else if (op.type === OP_REMOVE_STONE) {
      // Undo: add stone back (use raw to avoid pushing new operation)
      this._addStone_raw(op.idx, op.gid);
    } else if (op.type === OP_ADD_LIBERTY) {
      // Undo: remove liberty
      this._removeLiberty_raw(op.gid, op.idx);
    } else if (op.type === OP_REMOVE_LIBERTY) {
      // Undo: add liberty back
      this._addLiberty_raw(op.gid, op.idx);
    } else if (op.type === OP_MERGE_GROUPS) {
      // Undo: restore both main and other groups to pre-merge state
      const gb = op.mainGid * W;
      const ob = op.otherId * W;

      // Restore main group's stones and liberties
      for (let wi = 0; wi < W; wi++) {
        // Restore main group stones (remove merged-in stones)
        this._sw[gb + wi] = op.mainStones[wi];
        // Restore main group liberties
        this._lw[gb + wi] = op.mainLibs[wi];
      }

      // Restore stones from other group
      for (let wi = 0; wi < W; wi++) {
        let w = op.otherStones[wi];
        while (w) {
          const bit = 31 - Math.clz32(w & -w);
          this._gid[wi * 32 + bit] = op.otherId;
          w &= w - 1;
        }
        // Restore other group's stones
        this._sw[ob + wi] = op.otherStones[wi];
        // Restore other group's liberties
        this._lw[ob + wi] = op.otherLibs[wi];
      }

      // Restore sizes and liberty counts exactly as they were before merge
      this._ss[op.mainGid] = op.mainSize;
      this._ss[op.otherId] = op.otherSize;
      this._ls[op.mainGid] = op.mainLibCount;
      this._ls[op.otherId] = op.otherLibCount;
    }
  }

  // ── Display ────────────────────────────────────────────────────────────────

  // Render the board as a ● ○ · string. Optional 'markIdx' marks that
  // cell with bracket separators: · ·(●)· · — row width is unchanged.
  // Render board as ASCII string with optional marked position
  toString(markIdx = this.lastMove, { centerAt = null, showAxes = true } = {}) {
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

    // Add column labels if requested
    if (showAxes) {
      let header = '   ';
      for (let dx = 0; dx < N; dx++) {
        const bx = (x0 + dx) % N;
        header += String.fromCharCode(97 + bx) + ' ';
      }
      rows.push(header);
    }

    for (let dy = N - 1; dy >= 0; dy--) {
      const by = (y0 + dy) % N;
      const mx = (dy === dmy) ? dmx : -1;
      let row = '';

      // Add row label if requested
      if (showAxes) {
        const rowNum = by + 1;  // 1-based row numbers matching coordStr
        row = rowNum.toString().padStart(2) + ' ';
      }

      row += (mx === 0) ? '(' : ' ';
      for (let dx = 0; dx < N; dx++) {
        const bx = (x0 + dx) % N;
        const c = cells[by * N + bx];
        const ch = c === BLACK ? '●' : c === WHITE ? '○' : '·';
        if (dx > 0) row += (dx === mx) ? '(' : (dx - 1 === mx) ? ')' : ' ';
        row += ch;
      }
      row += (mx === N - 1) ? ')' : ' ';

      if (showAxes) {
        row += ' ' + (by + 1);
      }

      rows.push(row);
    }

    // Add bottom axis labels if requested
    if (showAxes) {
      let footer = '   ';
      for (let dx = 0; dx < N; dx++) {
        const bx = (x0 + dx) % N;
        footer += String.fromCharCode(97 + bx) + ' ';
      }
      rows.push(footer);
    }

    return rows.join('\n');
  }

  // ── Scoring ────────────────────────────────────────────────────────────────

  // Fast 1-step area estimate plus komi.
  // Returns { black, white } where white already includes komi.
  // ── Eye Detection ─────────────────────────────────────────────────────

  // Check if position is a true eye (solid border, not multi-color)
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

  // Get group ID at position
  groupIdAt(idx) {
    return this._gid[idx];
  }

  // Get number of stones in group
  groupSize(gid) {
    return this._ss[gid];
  }

  // Get number of liberties for group
  groupLibertyCount(gid) {
    return this._ls[gid];
  }

  // Get array of liberty indices for group at position
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

  // Get up to 2 liberties and count for group (fast path)
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

// Convert a Game2 instance to a Game3 instance
function game3FromGame2(game2) {
  const N = game2.N;
  const game3 = new Game3(N);
  const cap = N * N;

  // Place stones from Game2 board state using play()
  // Set current player to the color we want to place before each play()
  for (let i = 0; i < cap; i++) {
    if (game2.cells[i] !== EMPTY) {
      game3.current = game2.cells[i];
      if (!game3.isLegal(i)) {
        console.log('\n=== ILLEGAL MOVE ===');
        console.log('Cell', coordStr(i, N), 'is not legal in Game3');
        console.log('Game2 cell:', game2.cells[i] === BLACK ? 'BLACK' : 'WHITE');
        console.log('\nGame2 board state:');
        console.log(game2.toString(PASS));
        console.log('\nGame3 board state:');
        console.log(game3.toString(PASS));
        process.exit(1);
      }

      // Count stones before placement
      let stonesBefore = 0;
      for (let j = 0; j < cap; j++) {
        if (game3.cells[j] !== EMPTY) stonesBefore++;
      }

      if (!game3.play(i)) {
        console.log('illegal move in board position??', i);
      }

      // Count stones after placement
      let stonesAfter = 0;
      for (let j = 0; j < cap; j++) {
        if (game3.cells[j] !== EMPTY) stonesAfter++;
      }

      // Check if a capture occurred (more stones removed than the one we placed)
      // After placing one stone, we should have exactly 1 more stone
      // If we have fewer, a capture happened
      if (stonesAfter < stonesBefore + 1) {
        console.log('\n=== CAPTURE DETECTED ===');
        console.log('Placed stone at', coordStr(i, N));
        console.log('Stones before:', stonesBefore, 'after:', stonesAfter);
        console.log('\nGame2 board state:');
        console.log(game2.toString(PASS));
        console.log('\nGame3 board state at capture:');
        console.log(game3.toString(PASS));
        process.exit(1);
      }
    }
  }

  // Copy game state from Game2
  game3.current = game2.current;
  game3.ko = game2.ko;
  game3.consecutivePasses = game2.consecutivePasses;
  game3.gameOver = game2.gameOver;
  game3.moveCount = game2.moveCount;
  game3.lastMove = game2.lastMove;

  return game3;
}

const _exports = { Game3, game3FromGame2, PASS, BLACK, WHITE, EMPTY };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = _exports;
} else if (typeof window !== 'undefined') {
  window.Game3 = _exports;
}

})();
