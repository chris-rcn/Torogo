// BROWSER-COMPATIBLE: no Node.js-only APIs (require, process, etc.).
// Loaded as a plain <script> tag; do not use require/module/process at top level.

// ─── Zobrist hash table ───────────────────────────────────────────────────────
// One deterministic 64-bit random value per (row, col, color) triple,
// sized for the largest supported board (19×19).  Used for O(1) Ko detection.
const ZOBRIST = (() => {
  let s = 0xdeadbeef;
  function rand32() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s >>> 0;
  }
  function rand64() {
    return (BigInt(rand32()) << 32n) | BigInt(rand32());
  }
  const t = [];
  for (let y = 0; y < 19; y++) {
    t.push([]);
    for (let x = 0; x < 19; x++) {
      t[y].push({ black: rand64(), white: rand64() });
    }
  }
  return t;
})();

// ─── Board ────────────────────────────────────────────────────────────────────

class Board {
  constructor(size) {
    this.size = size;
    this.grid = Array.from({ length: size }, () => Array(size).fill(null));
    // Incremental group tracking
    this._gid = new Int16Array(size * size).fill(-1); // groupId per cell
    this._groups = new Map(); // gid → { color, stones: Set<idx>, liberties: Set<idx> }
    this._nextGid = 0;
    // Precomputed orthogonal neighbor indices (toroidal wrapping)
    if (!Board._nbrCache) Board._nbrCache = new Map();
    if (Board._nbrCache.has(size)) {
      this._nbr = Board._nbrCache.get(size);
    } else {
      const n = size * size;
      const nbr = new Int16Array(n * 4);
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const base = (y * size + x) * 4;
          nbr[base]     = ((x + 1) % size) + y * size;
          nbr[base + 1] = ((x - 1 + size) % size) + y * size;
          nbr[base + 2] = x + ((y + 1) % size) * size;
          nbr[base + 3] = x + ((y - 1 + size) % size) * size;
        }
      }
      Board._nbrCache.set(size, nbr);
      this._nbr = nbr;
    }
  }

  clone() {
    const b = Object.create(Board.prototype);
    b.size = this.size;
    b.grid = this.grid.map(row => row.slice());
    b._gid = new Int16Array(this._gid);
    b._nextGid = this._nextGid;
    b._nbr = this._nbr; // immutable, share reference
    b._groups = new Map();
    for (const [gid, g] of this._groups) {
      b._groups.set(gid, {
        color: g.color,
        stones: new Set(g.stones),
        liberties: new Set(g.liberties),
      });
    }
    return b;
  }

  get(x, y) {
    return this.grid[y][x];
  }

  set(x, y, color) {
    this.grid[y][x] = color;
  }

  // ─── Coordinate helpers ───────────────────────────────────────────────────

  _idx(x, y) { return y * this.size + x; }
  _xy(idx) { return [idx % this.size, (idx / this.size) | 0]; }

  // Toroidal wrapping: 4 neighbors with modular arithmetic
  getNeighbors(x, y) {
    const N = this.size;
    return [
      [(x + 1) % N, y],
      [(x - 1 + N) % N, y],
      [x, (y + 1) % N],
      [x, (y - 1 + N) % N],
    ];
  }

  _neighborIndices(x, y) {
    const N = this.size;
    return [
      ((x + 1) % N) + y * N,
      ((x - 1 + N) % N) + y * N,
      x + ((y + 1) % N) * N,
      x + ((y - 1 + N) % N) * N,
    ];
  }

  // ─── Incremental group tracking ──────────────────────────────────────────

  // Update tracking after a stone has been placed in the grid at (x, y).
  // Returns the groupId of the placed stone's group.
  _trackPlace(x, y) {
    const color = this.grid[y][x];
    const N = this.size;
    const idx = y * N + x;
    const nbr = this._nbr;
    const base = idx * 4;
    const grid = this.grid;
    const gidArr = this._gid;

    // Remove idx from liberties of all adjacent groups (max 4 distinct gids)
    let s0 = -1, s1 = -1, s2 = -1, s3 = -1;
    for (let i = 0; i < 4; i++) {
      const gid = gidArr[nbr[base + i]];
      if (gid === -1 || gid === s0 || gid === s1 || gid === s2 || gid === s3) continue;
      if (s0 === -1) s0 = gid; else if (s1 === -1) s1 = gid; else if (s2 === -1) s2 = gid; else s3 = gid;
      this._groups.get(gid).liberties.delete(idx);
    }

    // Find same-color neighbor groups and compute liberties for new stone
    const sameGids = [];
    const myLibs = [];
    for (let i = 0; i < 4; i++) {
      const ni = nbr[base + i];
      const c = grid[(ni / N) | 0][ni % N];
      if (c === color) {
        const gid = gidArr[ni];
        let dup = false;
        for (let j = 0; j < sameGids.length; j++) {
          if (sameGids[j] === gid) { dup = true; break; }
        }
        if (!dup) sameGids.push(gid);
      } else if (c === null) {
        myLibs.push(ni);
      }
    }

    if (sameGids.length === 0) {
      // New solo group
      const gid = this._nextGid++;
      gidArr[idx] = gid;
      this._groups.set(gid, { color, stones: new Set([idx]), liberties: new Set(myLibs) });
      return gid;
    }

    // Merge into the largest same-color neighbor group
    let mainGid = sameGids[0];
    let mainGroup = this._groups.get(mainGid);
    for (let i = 1; i < sameGids.length; i++) {
      const g = this._groups.get(sameGids[i]);
      if (g.stones.size > mainGroup.stones.size) {
        mainGid = sameGids[i];
        mainGroup = g;
      }
    }

    // Add new stone to main group
    mainGroup.stones.add(idx);
    gidArr[idx] = mainGid;
    for (let i = 0; i < myLibs.length; i++) mainGroup.liberties.add(myLibs[i]);

    // Merge other groups into main
    for (let i = 0; i < sameGids.length; i++) {
      const gid = sameGids[i];
      if (gid === mainGid) continue;
      const other = this._groups.get(gid);
      for (const si of other.stones) {
        mainGroup.stones.add(si);
        gidArr[si] = mainGid;
      }
      for (const li of other.liberties) {
        mainGroup.liberties.add(li);
      }
      this._groups.delete(gid);
    }

    return mainGid;
  }

  // Remove a tracked group from the board, updating neighbors' liberties.
  // Returns array of flat cell indices removed.
  _trackRemove(gid) {
    const group = this._groups.get(gid);
    const removed = [];
    const N = this.size;
    const nbr = this._nbr;
    const gidArr = this._gid;
    const grid = this.grid;

    // Single pass: clear stone, then update neighbor liberties.
    // Check nGid !== gid (not -1) since earlier stones in this group
    // already had their gidArr cleared.
    for (const idx of group.stones) {
      grid[(idx / N) | 0][idx % N] = null;
      gidArr[idx] = -1;
      removed.push(idx);
      const base = idx * 4;
      for (let i = 0; i < 4; i++) {
        const nGid = gidArr[nbr[base + i]];
        if (nGid !== -1 && nGid !== gid) {
          this._groups.get(nGid).liberties.add(idx);
        }
      }
    }

    this._groups.delete(gid);
    return removed;
  }

  // Rebuild all tracking data from the grid (used after rollback).
  _rebuildGroups() {
    const N = this.size;
    const n = N * N;
    this._gid.fill(-1);
    this._groups.clear();
    this._nextGid = 0;

    const nbr = this._nbr;
    const grid = this.grid;
    const gidArr = this._gid;
    const visited = new Uint8Array(n);
    for (let idx = 0; idx < n; idx++) {
      const y = (idx / N) | 0, x = idx % N;
      if (visited[idx] || grid[y][x] === null) continue;

      const color = grid[y][x];
      const gid = this._nextGid++;
      const stones = new Set();
      const liberties = new Set();
      const queue = [idx];
      visited[idx] = 1;

      while (queue.length) {
        const ci = queue.pop();
        stones.add(ci);
        gidArr[ci] = gid;
        const base = ci * 4;
        for (let i = 0; i < 4; i++) {
          const ni = nbr[base + i];
          const ny = (ni / N) | 0, nx = ni % N;
          if (grid[ny][nx] === null) {
            liberties.add(ni);
          } else if (!visited[ni] && grid[ny][nx] === color) {
            visited[ni] = 1;
            queue.push(ni);
          }
        }
      }

      this._groups.set(gid, { color, stones, liberties });
    }
  }

  // ─── Stochastic verification ──────────────────────────────────────────────

  _verifyGroups() {
    const N = this.size;
    const verified = new Set();
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const idx = y * N + x;
        if (this.grid[y][x] === null) {
          if (this._gid[idx] !== -1)
            throw new Error(`Group verify: empty cell (${x},${y}) has gid=${this._gid[idx]}`);
          continue;
        }

        const gid = this._gid[idx];
        if (gid === -1) throw new Error(`Group verify: stone at (${x},${y}) has no group`);
        if (verified.has(gid)) continue;
        verified.add(gid);

        const group = this._groups.get(gid);
        if (!group) throw new Error(`Group verify: gid ${gid} not in _groups map`);
        if (group.color !== this.grid[y][x])
          throw new Error(`Group verify: color mismatch at (${x},${y})`);

        // BFS to find expected group and liberties
        const bfsGroup = this._getGroupBFS(x, y);
        const bfsStones = new Set(bfsGroup.map(([gx, gy]) => gy * N + gx));
        if (bfsStones.size !== group.stones.size)
          throw new Error(`Group verify: size mismatch at (${x},${y}): ` +
            `BFS=${bfsStones.size}, tracked=${group.stones.size}`);
        for (const si of group.stones) {
          if (!bfsStones.has(si))
            throw new Error(`Group verify: tracked stone ${si} not in BFS group`);
        }

        const bfsLibs = this._getLibertiesBFS(bfsGroup);
        const bfsLibSet = new Set();
        for (const key of bfsLibs) {
          const [lx, ly] = key.split(',').map(Number);
          bfsLibSet.add(ly * N + lx);
        }
        if (bfsLibSet.size !== group.liberties.size)
          throw new Error(`Group verify: liberty count mismatch at (${x},${y}): ` +
            `BFS=${bfsLibSet.size}, tracked=${group.liberties.size}`);
        for (const li of group.liberties) {
          if (!bfsLibSet.has(li))
            throw new Error(`Group verify: tracked liberty ${li} not in BFS liberties`);
        }
      }
    }
  }

  // ─── BFS reference implementations (for verification) ────────────────────

  _getGroupBFS(x, y) {
    const color = this.get(x, y);
    if (!color) return [];
    const visited = new Set();
    const queue = [[x, y]];
    const group = [];
    visited.add(`${x},${y}`);
    while (queue.length) {
      const [cx, cy] = queue.shift();
      group.push([cx, cy]);
      for (const [nx, ny] of this.getNeighbors(cx, cy)) {
        const key = `${nx},${ny}`;
        if (!visited.has(key) && this.get(nx, ny) === color) {
          visited.add(key);
          queue.push([nx, ny]);
        }
      }
    }
    return group;
  }

  _getLibertiesBFS(group) {
    const liberties = new Set();
    for (const [x, y] of group) {
      for (const [nx, ny] of this.getNeighbors(x, y)) {
        if (this.get(nx, ny) === null) {
          liberties.add(`${nx},${ny}`);
        }
      }
    }
    return liberties;
  }

  // ─── Public group API (backed by tracker) ─────────────────────────────────

  // Returns array of [x,y] for connected same-color stones
  getGroup(x, y) {
    const color = this.get(x, y);
    if (!color) return [];
    const gid = this._gid[this._idx(x, y)];
    const group = this._groups.get(gid);
    const N = this.size;
    return [...group.stones].map(idx => [idx % N, (idx / N) | 0]);
  }

  // Count unique empty intersections adjacent to the group.
  // Returns Set<string> of "x,y" keys (matching original API).
  getLiberties(group) {
    if (group.length === 0) return new Set();
    const [x, y] = group[0];
    const gid = this._gid[this._idx(x, y)];
    if (gid === -1) return new Set();
    const tracked = this._groups.get(gid);
    const N = this.size;
    const result = new Set();
    for (const idx of tracked.liberties) {
      result.add(`${idx % N},${(idx / N) | 0}`);
    }
    return result;
  }

  // Check if a group has zero liberties
  hasNoLiberties(group) {
    if (group.length === 0) return true;
    const [x, y] = group[0];
    const gid = this._gid[this._idx(x, y)];
    if (gid === -1) return true;
    return this._groups.get(gid).liberties.size === 0;
  }

  // Returns true if placing a lone stone at (x, y) would be suicide:
  // all 4 neighbours are occupied by opponents, no opponent is in atari at (x,y).
  // Does not modify board state.
  isSingleSuicide(x, y, color) {
    const N = this.size;
    const placedIdx = y * N + x;
    const nbr = this._nbr;
    const base = placedIdx * 4;
    const grid = this.grid;
    for (let i = 0; i < 4; i++) {
      const ni = nbr[base + i];
      const cell = grid[(ni / N) | 0][ni % N];
      if (cell === null) return false;   // empty → immediate liberty
      if (cell === color) return false;  // friendly neighbour → not a lone stone
      // Opponent: would it be captured? (its only liberty is (x,y))
      const group = this._groups.get(this._gid[ni]);
      if (group.liberties.size === 1 && group.liberties.has(placedIdx)) return false;
    }
    return true;
  }

  // Returns true if placing `color` at (x, y) would fill the last shared liberty
  // of the adjacent friendly groups: every friendly neighbour group's only liberty
  // is (x,y), and no opponent capture would free a new liberty.
  // Does not modify board state.
  isMultiSuicide(x, y, color) {
    const N = this.size;
    const placedIdx = y * N + x;
    const nbr = this._nbr;
    const base = placedIdx * 4;
    const grid = this.grid;
    const gidArr = this._gid;
    let hasFriendly = false;
    let s0 = -1, s1 = -1, s2 = -1, s3 = -1;

    for (let i = 0; i < 4; i++) {
      const ni = nbr[base + i];
      const cell = grid[(ni / N) | 0][ni % N];
      if (cell === null) return false; // empty → immediate liberty

      const gid = gidArr[ni];
      if (gid === s0 || gid === s1 || gid === s2 || gid === s3) continue;
      if (s0 === -1) s0 = gid; else if (s1 === -1) s1 = gid; else if (s2 === -1) s2 = gid; else s3 = gid;

      const group = this._groups.get(gid);
      if (cell === color) {
        hasFriendly = true;
        // Friendly group contributes liberties besides (x,y)?
        if (group.liberties.size > 1) return false;
        if (group.liberties.size === 1 && !group.liberties.has(placedIdx)) return false;
        // else: sole liberty is (x,y) — no gain after placement
      } else {
        // Opponent captured → frees at least one cell adjacent to us
        if (group.liberties.size === 1 && group.liberties.has(placedIdx)) return false;
      }
    }

    return hasFriendly; // must have at least one friendly group to qualify
  }

  // Returns true if placing `color` at (x, y) would be suicide (either kind).
  // Does not modify board state.
  isSuicide(x, y, color) {
    return this.isSingleSuicide(x, y, color) || this.isMultiSuicide(x, y, color);
  }

  // Returns true if placing `color` at (x, y) is Ko-illegal given koFlag.
  // koFlag is {x, y} of the single stone captured on the previous move, or null.
  // A move is Ko-illegal when it is on the koFlag point and would itself capture
  // exactly one stone (which would recreate the previous board position).
  // Does not modify board state.
  isKo(x, y, color, koFlag) {
    if (!koFlag || x !== koFlag.x || y !== koFlag.y) return false;
    const N = this.size;
    const placedIdx = y * N + x;
    const nbr = this._nbr;
    const base = placedIdx * 4;
    const grid = this.grid;
    const gidArr = this._gid;
    const opponentColor = color === 'black' ? 'white' : 'black';
    let s0 = -1, s1 = -1, s2 = -1, s3 = -1;
    let capturedCount = 0;
    for (let i = 0; i < 4; i++) {
      const ni = nbr[base + i];
      if (grid[(ni / N) | 0][ni % N] !== opponentColor) continue;
      const gid = gidArr[ni];
      if (gid === s0 || gid === s1 || gid === s2 || gid === s3) continue;
      if (s0 === -1) s0 = gid; else if (s1 === -1) s1 = gid; else if (s2 === -1) s2 = gid; else s3 = gid;
      const group = this._groups.get(gid);
      if (group.liberties.size === 1 && group.liberties.has(placedIdx)) {
        capturedCount += group.stones.size;
      }
    }
    return capturedCount === 1;
  }

  // After placing at (lastX, lastY), capture groups with 0 liberties.
  // Returns flat indices of captured stones per color: { black: [idx,...], white: [...] }
  captureGroups(lastX, lastY) {
    const placedColor = this.get(lastX, lastY);
    const opponentColor = placedColor === 'black' ? 'white' : 'black';
    const captured = { black: [], white: [] };

    // Update tracking for placed stone
    const placedGid = this._trackPlace(lastX, lastY);

    // Check opponent groups adjacent to the placed stone
    const N = this.size;
    const idx = lastY * N + lastX;
    const nbr = this._nbr;
    const base = idx * 4;
    const gidArr = this._gid;
    let c0 = -1, c1 = -1, c2 = -1, c3 = -1;
    for (let i = 0; i < 4; i++) {
      const ni = nbr[base + i];
      if (this.grid[(ni / N) | 0][ni % N] !== opponentColor) continue;
      const gid = gidArr[ni];
      if (gid === -1 || gid === c0 || gid === c1 || gid === c2 || gid === c3) continue;
      if (c0 === -1) c0 = gid; else if (c1 === -1) c1 = gid; else if (c2 === -1) c2 = gid; else c3 = gid;

      const group = this._groups.get(gid);
      if (group.liberties.size === 0) {
        const removed = this._trackRemove(gid);
        for (let j = 0; j < removed.length; j++) captured[opponentColor].push(removed[j]);
      }
    }

    // Check own group for suicide
    const ownGroup = this._groups.get(placedGid);
    if (ownGroup && ownGroup.liberties.size === 0) {
      const removed = this._trackRemove(placedGid);
      for (let j = 0; j < removed.length; j++) captured[placedColor].push(removed[j]);
    }

    // Stochastic verification
    if (Board.verifyGroupRatio > 0 && Math.random() < Board.verifyGroupRatio) {
      this._verifyGroups();
    }

    return captured;
  }

  // Single-point true eye detection (toroidal board — every cell has exactly
  // 4 orthogonal and 4 diagonal neighbours).
  // An empty cell (x, y) is a true eye for `color` when:
  //   1. All 4 orthogonal neighbours are occupied by `color`.
  //   2. All 4 ortho neighbours belong to the same group (unconditional true
  //      eye), OR at least 3 of the 4 diagonal neighbours are `color`.
  isTrueEye(x, y, color) {
    const N = this.size;
    const idx = y * N + x;
    const base = idx * 4;
    const nbr = this._nbr;
    const grid = this.grid;
    const gidArr = this._gid;

    // Count friendly neighbors and track group IDs.
    let friendCount = 0;
    let emptyCount = 0;
    let sameGroupCount = 0;
    let firstGid = -2; // sentinel distinct from untracked (-1)
    for (let i = 0; i < 4; i++) {
      const ni = nbr[base + i];
      const c = grid[(ni / N) | 0][ni % N];
      if (c === color) {
        friendCount++;
        const gid = gidArr[ni];
        if (gid !== -1) { // only count tracked stones
          if (firstGid === -2) firstGid = gid;
          if (gid === firstGid) sameGroupCount++;
        }
      } else if (c === null) {
        emptyCount++;
      }
    }

    // All 4 neighbors are friendly: classic true-eye check.
    if (friendCount === 4) {
      if (sameGroupCount === 4) return true; // same group → unconditionally true
      // Different friendly groups: fall back to the diagonal heuristic.
      let dc = 0;
      if (grid[(y + 1) % N][(x + 1) % N] === color) dc++;
      if (grid[(y + 1) % N][(x - 1 + N) % N] === color) dc++;
      if (grid[(y - 1 + N) % N][(x + 1) % N] === color) dc++;
      if (grid[(y - 1 + N) % N][(x - 1 + N) % N] === color) dc++;
      return dc >= 3;
    }

    // 3 neighbors are the same friendly group + 1 empty → proto-eye, skip.
    if (friendCount === 3 && emptyCount === 1 && sameGroupCount === 3) return true;

    return false;
  }

  // Combined eye + empty-neighbor check for playout loops.
  // Returns { isTrueEye: bool, hasEmptyNeighbor: bool } from a single scan.
  classifyEmpty(x, y, color) {
    const N = this.size;
    const idx = y * N + x;
    const base = idx * 4;
    const nbr = this._nbr;
    const grid = this.grid;
    const gidArr = this._gid;

    let friendCount = 0;
    let emptyCount = 0;
    let sameGroupCount = 0;
    let firstGid = -2;
    let hasEmptyNeighbor = false;

    for (let i = 0; i < 4; i++) {
      const ni = nbr[base + i];
      const c = grid[(ni / N) | 0][ni % N];
      if (c === color) {
        friendCount++;
        const gid = gidArr[ni];
        if (gid !== -1) {
          if (firstGid === -2) firstGid = gid;
          if (gid === firstGid) sameGroupCount++;
        }
      } else if (c === null) {
        emptyCount++;
        hasEmptyNeighbor = true;
      }
    }

    // True eye checks (same logic as isTrueEye).
    if (friendCount === 4) {
      if (sameGroupCount === 4) return { isTrueEye: true, hasEmptyNeighbor };
      let dc = 0;
      if (grid[(y + 1) % N][(x + 1) % N] === color) dc++;
      if (grid[(y + 1) % N][(x - 1 + N) % N] === color) dc++;
      if (grid[(y - 1 + N) % N][(x + 1) % N] === color) dc++;
      if (grid[(y - 1 + N) % N][(x - 1 + N) % N] === color) dc++;
      if (dc >= 3) return { isTrueEye: true, hasEmptyNeighbor };
    }
    if (friendCount === 3 && emptyCount === 1 && sameGroupCount === 3) {
      return { isTrueEye: true, hasEmptyNeighbor };
    }

    return { isTrueEye: false, hasEmptyNeighbor };
  }

  // ─── ASCII serialization ──────────────────────────────────────────────────

  // Render the board as a ● ○ · string (rows separated by '\n').
  // Optional mark {x, y} highlights that cell using the adjacent separators as
  // brackets: · ·(●)· · — so the row width is unchanged for interior cells.
  toAscii(mark) {
    const rows = [];
    for (let y = 0; y < this.size; y++) {
      const rowMarked = mark && mark.y === y;
      let row = rowMarked && mark.x === 0 ? '(' : ' ';
      for (let x = 0; x < this.size; x++) {
        const v = this.grid[y][x];
        const ch = v === 'black' ? '●' : v === 'white' ? '○' : '·';
        const isMarked   = rowMarked && x === mark.x;
        const prevMarked = rowMarked && x - 1 === mark.x;
        if (x > 0) row += isMarked ? '(' : prevMarked ? ')' : ' ';
        row += ch;
      }
      row += rowMarked && mark.x === this.size - 1 ? ')' : ' ';
      rows.push(row);
    }
    return rows.join('\n');
  }

  // Parse a ● ○ · board string produced by toAscii().
  // Mark decoration ( and ) are stripped before splitting.
  // Returns { size, stones } where stones is [[x, y, color], ...].
  static parse(str) {
    const rows = str.trim().split('\n').map(r => r.trim().replace(/[()]/g, ' ').split(/\s+/));
    const size = rows.length;
    const stones = [];
    for (let y = 0; y < size; y++)
      for (let x = 0; x < size; x++) {
        const c = rows[y][x].replace(/[()]/g, '');
        if (c === '●') stones.push([x, y, 'black']);
        else if (c === '○') stones.push([x, y, 'white']);
      }
    return { size, stones };
  }
}

// Fraction of captureGroups calls that run BFS verification (0 = off)
Board.verifyGroupRatio = 0;

// ─── Game ─────────────────────────────────────────────────────────────────────

class Game {
  constructor(boardSize = 9, komi = 4.5) {
    this.boardSize = boardSize;
    this.board = new Board(boardSize);
    this.current = 'black';
    this.hash     = 0n;       // Zobrist hash of current board position
    this.koFlag   = null;     // {x, y} of the single stone captured last move, or null
    this.consecutivePasses = 0;
    this.gameOver = false;
    this.lastMove = null;
    this.komi = komi;         // compensation for white going second
    this.scores = null;       // set on game end
    this.illegalFlash = null; // {x, y} of last rejected move, for visual feedback
    this.moveCount = 0;

    // First move is always at the center of the board.
    const c = Math.floor(boardSize / 2);
    this.placeStone(c, c);
  }

  clone() {
    const g = Object.create(Game.prototype);
    g.boardSize         = this.boardSize;
    g.board             = this.board.clone();
    g.current           = this.current;
    g.hash              = this.hash;
    g.koFlag            = this.koFlag;
    g.consecutivePasses = this.consecutivePasses;
    g.gameOver          = this.gameOver;
    g.lastMove          = null;
    g.komi              = this.komi;
    g.scores            = null;
    g.illegalFlash      = null;
    g.moveCount         = this.moveCount;
    return g;
  }

  // Returns true if move was legal and placed
  placeStone(x, y) {
    if (this.gameOver) return false;
    if (this.board.get(x, y) !== null) {
      this.illegalFlash = { x, y };
      return false;
    }

    // All legality checks use the tracker — no board mutation needed for rejection
    if (this.board.isSuicide(x, y, this.current)) {
      this.illegalFlash = { x, y };
      return false;
    }
    if (this.board.isKo(x, y, this.current, this.koFlag)) {
      this.illegalFlash = { x, y };
      return false;
    }

    // Move is legal — commit (no rollback path remains)
    const N = this.boardSize;
    this.board.set(x, y, this.current);
    this.hash ^= ZOBRIST[y][x][this.current];
    const caps = this.board.captureGroups(x, y);
    for (let i = 0; i < caps.black.length; i++) {
      const ci = caps.black[i];
      this.hash ^= ZOBRIST[(ci / N) | 0][ci % N].black;
    }
    for (let i = 0; i < caps.white.length; i++) {
      const ci = caps.white[i];
      this.hash ^= ZOBRIST[(ci / N) | 0][ci % N].white;
    }

    this.illegalFlash = null;

    // Update koFlag: set to the captured position only when exactly one stone was taken
    const totalCaptured = caps.black.length + caps.white.length;
    if (totalCaptured === 1) {
      const ci = caps.black.length === 1 ? caps.black[0] : caps.white[0];
      this.koFlag = { x: ci % N, y: (ci / N) | 0 };
    } else {
      this.koFlag = null;
    }

    this.lastMove = { x, y };
    this.consecutivePasses = 0;
    this.current = this.current === 'black' ? 'white' : 'black';
    this._incrementMoveCount();
    return totalCaptured || true;
  }

  // Like placeStone() but skips all legality checks.  Only call this when the
  // move is already known to be legal (not occupied, not suicide, not ko).
  applyLegal(x, y) {
    const N = this.boardSize;
    this.board.set(x, y, this.current);
    this.hash ^= ZOBRIST[y][x][this.current];
    const caps = this.board.captureGroups(x, y);
    for (let i = 0; i < caps.black.length; i++) {
      const ci = caps.black[i];
      this.hash ^= ZOBRIST[(ci / N) | 0][ci % N].black;
    }
    for (let i = 0; i < caps.white.length; i++) {
      const ci = caps.white[i];
      this.hash ^= ZOBRIST[(ci / N) | 0][ci % N].white;
    }
    const totalCaptured = caps.black.length + caps.white.length;
    this.koFlag = totalCaptured === 1
      ? { x: (caps.black.length === 1 ? caps.black[0] : caps.white[0]) % N,
          y: ((caps.black.length === 1 ? caps.black[0] : caps.white[0]) / N) | 0 }
      : null;
    this.lastMove = { x, y };
    this.consecutivePasses = 0;
    this.current = this.current === 'black' ? 'white' : 'black';
    this._incrementMoveCount();
    return totalCaptured;
  }

  _incrementMoveCount() {
    this.moveCount++;
    const threshold = 4 * this.boardSize * this.boardSize;
    if (this.moveCount > threshold) {
      this.endGame();
    }
  }

  pass() {
    if (this.gameOver) return;
    this.consecutivePasses++;
    this.lastMove = null;
    this.illegalFlash = null;
    this.koFlag = null;
    const passer = this.current;
    this.current = this.current === 'black' ? 'white' : 'black';
    this._incrementMoveCount();
    if (this.consecutivePasses >= 2) {
      this.endGame();
    }
    return passer;
  }

  endGame() {
    this.gameOver = true;
    const territory = this.calcTerritory();
    this.scores = {
      black: { territory: territory.black, total: territory.black },
      white: { territory: territory.white, total: territory.white + this.komi },
    };
  }

  // For each empty point: check orthogonal neighbors.
  // If they are stones of a single color, assign to that color.
  // If all orthogonal neighbors are empty, check diagonals;
  // if diagonals are empty too, assign no point.
  calcTerritory() {
    const grid = this.board.grid;
    const N = this.board.size;
    const territory = { black: 0, white: 0, neutral: 0 };
    const ortho = [[-1,0],[1,0],[0,-1],[0,1]];
    const diag  = [[-1,-1],[-1,1],[1,-1],[1,1]];

    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const cell = grid[y][x];
        if (cell !== null) {
          // Stones count as territory for their color (Chinese scoring)
          if (cell === 'black') territory.black++;
          else territory.white++;
          continue;
        }

        let hasBlack = false, hasWhite = false, allOrthoEmpty = true;
        for (const [dy, dx] of ortho) {
          const c = grid[(y + dy + N) % N][(x + dx + N) % N];
          if (c !== null) {
            allOrthoEmpty = false;
            if (c === 'black') hasBlack = true; else hasWhite = true;
          }
        }

        if (!allOrthoEmpty) {
          if (hasBlack && !hasWhite) territory.black++;
          else if (hasWhite && !hasBlack) territory.white++;
          // else mixed → neutral, no count
        } else {
          // All orthogonal neighbors empty — check diagonals
          let diagAllEmpty = true;
          for (const [dy, dx] of diag) {
            if (grid[(y + dy + N) % N][(x + dx + N) % N] !== null) {
              diagAllEmpty = false;
              break;
            }
          }
          if (!diagAllEmpty) territory.neutral++;
          // else completely isolated → no point assigned
        }
      }
    }

    return territory;
  }

  statusText() {
    if (this.gameOver) return 'Game over';
    const name = this.current === 'black' ? 'Black' : 'White';
    return `${name} to play`;
  }
}

const DEFAULT_KOMI = new Game().komi;

// boardTurnToString(board, toPlay?) — serialize board to ASCII; if toPlay ('●'/'○')
// is given, prepend it as the first line so parseBoard can recover it.
function boardTurnToString(board, toPlay) {
  const body = board.toAscii();
  return toPlay ? toPlay + '\n' + body : body;
}

// parseBoard(str) — returns { size, stones, toPlay? }.
// toPlay is '●' or '○' if the string was produced with boardTurnToString(board, toPlay).
function parseBoard(boardStr) {
  const lines = boardStr.trim().split('\n').map(r => r.trim());
  let toPlay;
  if (lines[0] === '●' || lines[0] === '○') {
    toPlay = lines.shift();
  }
  const result = Board.parse(lines.join('\n'));
  if (toPlay !== undefined) result.toPlay = toPlay;
  return result;
}

if (typeof module !== 'undefined') module.exports = { Board, Game, DEFAULT_KOMI, ZOBRIST, parseBoard, boardTurnToString };
