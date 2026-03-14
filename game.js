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
  }

  clone() {
    const b = new Board(this.size);
    b.grid = this.grid.map(row => row.slice());
    b._gid = new Int16Array(this._gid);
    b._nextGid = this._nextGid;
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
    const idx = this._idx(x, y);
    const N = this.size;
    const nbors = this.getNeighbors(x, y);
    const nborIdxs = this._neighborIndices(x, y);

    // Remove idx from liberties of all adjacent groups
    const seenGids = new Set();
    for (const ni of nborIdxs) {
      const gid = this._gid[ni];
      if (gid !== -1 && !seenGids.has(gid)) {
        seenGids.add(gid);
        this._groups.get(gid).liberties.delete(idx);
      }
    }

    // Find same-color neighbor groups (deduplicated)
    const sameGids = new Set();
    for (let i = 0; i < 4; i++) {
      const [nx, ny] = nbors[i];
      if (this.grid[ny][nx] === color) {
        sameGids.add(this._gid[nborIdxs[i]]);
      }
    }

    // Compute liberties for the new stone position
    const myLibs = new Set();
    for (let i = 0; i < 4; i++) {
      const [nx, ny] = nbors[i];
      if (this.grid[ny][nx] === null) {
        myLibs.add(nborIdxs[i]);
      }
    }

    if (sameGids.size === 0) {
      // New solo group
      const gid = this._nextGid++;
      this._gid[idx] = gid;
      this._groups.set(gid, { color, stones: new Set([idx]), liberties: myLibs });
      return gid;
    }

    // Merge into the largest same-color neighbor group
    const gidArr = [...sameGids];
    let mainGid = gidArr[0];
    let mainGroup = this._groups.get(mainGid);
    for (let i = 1; i < gidArr.length; i++) {
      const g = this._groups.get(gidArr[i]);
      if (g.stones.size > mainGroup.stones.size) {
        mainGid = gidArr[i];
        mainGroup = g;
      }
    }

    // Add new stone to main group
    mainGroup.stones.add(idx);
    this._gid[idx] = mainGid;
    for (const lib of myLibs) mainGroup.liberties.add(lib);

    // Merge other groups into main
    for (const gid of gidArr) {
      if (gid === mainGid) continue;
      const other = this._groups.get(gid);
      for (const si of other.stones) {
        mainGroup.stones.add(si);
        this._gid[si] = mainGid;
      }
      for (const li of other.liberties) {
        mainGroup.liberties.add(li);
      }
      this._groups.delete(gid);
    }

    return mainGid;
  }

  // Remove a tracked group from the board, updating neighbors' liberties.
  // Returns array of [x,y] positions removed.
  _trackRemove(gid) {
    const group = this._groups.get(gid);
    const removed = [];

    // First pass: clear all stones in grid and groupId map
    for (const idx of group.stones) {
      const [x, y] = this._xy(idx);
      this.grid[y][x] = null;
      this._gid[idx] = -1;
      removed.push([x, y]);
    }

    // Second pass: add removed positions as liberties to adjacent groups
    for (const idx of group.stones) {
      for (const ni of this._neighborIndices(...this._xy(idx))) {
        const nGid = this._gid[ni];
        if (nGid !== -1) {
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

    const visited = new Uint8Array(n);
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const idx = y * N + x;
        if (visited[idx] || this.grid[y][x] === null) continue;

        const color = this.grid[y][x];
        const gid = this._nextGid++;
        const stones = new Set();
        const liberties = new Set();
        const queue = [idx];
        visited[idx] = 1;

        while (queue.length) {
          const ci = queue.pop();
          stones.add(ci);
          this._gid[ci] = gid;
          const cx = ci % N, cy = (ci / N) | 0;
          for (const [nx, ny] of this.getNeighbors(cx, cy)) {
            const ni = ny * N + nx;
            if (this.grid[ny][nx] === null) {
              liberties.add(ni);
            } else if (!visited[ni] && this.grid[ny][nx] === color) {
              visited[ni] = 1;
              queue.push(ni);
            }
          }
        }

        this._groups.set(gid, { color, stones, liberties });
      }
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

  // After placing at (lastX, lastY), capture groups with 0 liberties.
  // Returns positions of captured stones per color: { black: [[x,y],...], white: [...] }
  captureGroups(lastX, lastY) {
    const placedColor = this.get(lastX, lastY);
    const opponentColor = placedColor === 'black' ? 'white' : 'black';
    const captured = { black: [], white: [] };

    // Update tracking for placed stone
    const placedGid = this._trackPlace(lastX, lastY);

    // Check opponent groups adjacent to the placed stone
    const checked = new Set();
    for (const [nx, ny] of this.getNeighbors(lastX, lastY)) {
      if (this.grid[ny][nx] !== opponentColor) continue;
      const gid = this._gid[this._idx(nx, ny)];
      if (gid === -1 || checked.has(gid)) continue;
      checked.add(gid);

      const group = this._groups.get(gid);
      if (group.liberties.size === 0) {
        const removed = this._trackRemove(gid);
        for (const pos of removed) captured[opponentColor].push(pos);
      }
    }

    // Check own group for suicide
    const ownGroup = this._groups.get(placedGid);
    if (ownGroup && ownGroup.liberties.size === 0) {
      const removed = this._trackRemove(placedGid);
      for (const pos of removed) captured[placedColor].push(pos);
    }

    // Stochastic verification
    if (Board.verifyGroupRatio > 0 && Math.random() < Board.verifyGroupRatio) {
      this._verifyGroups();
    }

    return captured;
  }
}

// Fraction of captureGroups calls that run BFS verification (0 = off)
Board.verifyGroupRatio = 0;

// ─── Game ─────────────────────────────────────────────────────────────────────

class Game {
  constructor(boardSize = 9, komi = 3.5) {
    this.boardSize = boardSize;
    this.board = new Board(boardSize);
    this.current = 'black';
    this.captured = { black: 0, white: 0 }; // stones captured of each color
    this.hash     = 0n;       // Zobrist hash of current board position
    this.prevHash = 0n;       // hash before the most recent move (for Ko)
    this.consecutivePasses = 0;
    this.gameOver = false;
    this.lastMove = null;
    this.komi = komi;         // compensation for white going second
    this.scores = null;       // set on game end
    this.illegalFlash = null; // {x, y} of last rejected move, for visual feedback
    this.moveCount = 0;
  }

  clone() {
    const g = new Game(this.boardSize);
    g.board             = this.board.clone();
    g.current           = this.current;
    g.captured          = { ...this.captured };
    g.hash              = this.hash;
    g.prevHash          = this.prevHash;
    g.consecutivePasses = this.consecutivePasses;
    g.gameOver          = this.gameOver;
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

    const hashBefore = this.hash;

    // Clone board for potential rollback
    const savedGrid = this.board.grid.map(row => row.slice());

    this.board.set(x, y, this.current);
    this.hash ^= ZOBRIST[y][x][this.current];
    const caps = this.board.captureGroups(x, y);
    for (const [cx, cy] of caps.black) this.hash ^= ZOBRIST[cy][cx].black;
    for (const [cx, cy] of caps.white) this.hash ^= ZOBRIST[cy][cx].white;

    // Suicide: if our stone was removed, it's illegal
    if (this.board.get(x, y) === null) {
      this.board.grid = savedGrid;
      this.board._rebuildGroups();
      this.hash = hashBefore;
      this.illegalFlash = { x, y };
      return false;
    }

    // Ko rule: reject if this recreates the board state before previous move
    if (this.hash === this.prevHash) {
      this.board.grid = savedGrid;
      this.board._rebuildGroups();
      this.hash = hashBefore;
      this.illegalFlash = { x, y };
      return false;
    }

    // Move is legal — commit
    this.illegalFlash = null;
    this.captured.black += caps.black.length;
    this.captured.white += caps.white.length;
    this.prevHash = hashBefore;
    this.lastMove = { x, y };
    this.consecutivePasses = 0;
    this.current = this.current === 'black' ? 'white' : 'black';
    this._incrementMoveCount();
    return true;
  }

  _incrementMoveCount() {
    this.moveCount++;
    const threshold = 5 * this.boardSize * this.boardSize;
    if (this.moveCount === threshold + 1) {
      console.warn(`Game moveCount (${this.moveCount}) exceeded 5× board area (${threshold})`);
    }
  }

  pass() {
    if (this.gameOver) return;
    this.consecutivePasses++;
    this.lastMove = null;
    this.illegalFlash = null;
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
      black: { territory: territory.black, captures: this.captured.white },
      white: { territory: territory.white, captures: this.captured.black },
    };
    this.scores.black.total = this.scores.black.territory + this.scores.black.captures;
    this.scores.white.total = this.scores.white.territory + this.scores.white.captures + this.komi;
  }

  // BFS over empty cells; attribute to a color if all reachable boundary
  // stones are of one color (toroidal wrapping respected).
  calcTerritory() {
    const N = this.board.size;
    const visited = new Set();
    const territory = { black: 0, white: 0, neutral: 0 };

    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const key = `${x},${y}`;
        if (this.board.get(x, y) !== null || visited.has(key)) continue;

        // BFS to find the empty region and its border colors
        const region = [];
        const borderColors = new Set();
        const queue = [[x, y]];
        visited.add(key);

        while (queue.length) {
          const [cx, cy] = queue.shift();
          region.push([cx, cy]);
          for (const [nx, ny] of this.board.getNeighbors(cx, cy)) {
            const nkey = `${nx},${ny}`;
            const cell = this.board.get(nx, ny);
            if (cell !== null) {
              borderColors.add(cell);
            } else if (!visited.has(nkey)) {
              visited.add(nkey);
              queue.push([nx, ny]);
            }
          }
        }

        if (borderColors.size === 1) {
          const [owner] = borderColors;
          territory[owner] += region.length;
        } else {
          territory.neutral += region.length;
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
if (typeof module !== 'undefined') module.exports = { Board, Game, DEFAULT_KOMI, ZOBRIST };
