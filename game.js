// ─── Board ────────────────────────────────────────────────────────────────────

class Board {
  constructor(size) {
    this.size = size;
    this.grid = Array.from({ length: size }, () => Array(size).fill(null));
  }

  clone() {
    const b = new Board(this.size);
    b.grid = this.grid.map(row => row.slice());
    return b;
  }

  get(x, y) {
    return this.grid[y][x];
  }

  set(x, y, color) {
    this.grid[y][x] = color;
  }

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

  // BFS flood-fill: returns array of [x,y] for connected same-color stones
  getGroup(x, y) {
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

  // Count unique empty intersections adjacent to the group
  getLiberties(group) {
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

  // Check if a group has zero liberties
  hasNoLiberties(group) {
    return this.getLiberties(group).size === 0;
  }

  // After placing at (lastX, lastY), capture groups with 0 liberties.
  // Returns count of captured stones per color: { black: N, white: N }
  captureGroups(lastX, lastY) {
    const placedColor = this.get(lastX, lastY);
    const opponentColor = placedColor === 'black' ? 'white' : 'black';
    const captured = { black: 0, white: 0 };

    // Check opponent groups adjacent to the placed stone
    const checked = new Set();
    for (const [nx, ny] of this.getNeighbors(lastX, lastY)) {
      const key = `${nx},${ny}`;
      if (!checked.has(key) && this.get(nx, ny) === opponentColor) {
        const group = this.getGroup(nx, ny);
        group.forEach(([gx, gy]) => checked.add(`${gx},${gy}`));
        if (this.hasNoLiberties(group)) {
          for (const [gx, gy] of group) {
            this.set(gx, gy, null);
            captured[opponentColor]++;
          }
        }
      }
    }

    // Check own group for suicide
    const ownGroup = this.getGroup(lastX, lastY);
    if (this.hasNoLiberties(ownGroup)) {
      for (const [gx, gy] of ownGroup) {
        this.set(gx, gy, null);
        captured[placedColor]++;
      }
    }

    return captured;
  }
}

// ─── Game ─────────────────────────────────────────────────────────────────────

class Game {
  constructor(boardSize = 9) {
    this.boardSize = boardSize;
    this.board = new Board(boardSize);
    this.current = 'black';
    this.captured = { black: 0, white: 0 }; // stones captured of each color
    this.prevHash = null;     // board hash before the most recent move (for Ko)
    this.consecutivePasses = 0;
    this.gameOver = false;
    this.lastMove = null;
    this.scores = null;       // set on game end
    this.illegalFlash = null; // {x, y} of last rejected move, for visual feedback
  }

  boardHash() {
    return this.board.grid.map(row => row.map(c => c ? c[0] : '.').join('')).join('|');
  }

  // Returns true if move was legal and placed
  placeStone(x, y) {
    if (this.gameOver) return false;
    if (this.board.get(x, y) !== null) {
      this.illegalFlash = { x, y };
      return false;
    }

    const hashBefore = this.boardHash();

    // Clone board for potential rollback
    const savedGrid = this.board.grid.map(row => row.slice());

    this.board.set(x, y, this.current);
    const newCaptured = this.board.captureGroups(x, y);

    // Suicide: if our stone was removed, it's illegal
    if (this.board.get(x, y) === null) {
      this.board.grid = savedGrid;
      this.illegalFlash = { x, y };
      return false;
    }

    const hashAfter = this.boardHash();

    // Ko rule: reject if this recreates the board state before previous move
    if (hashAfter === this.prevHash) {
      this.board.grid = savedGrid;
      this.illegalFlash = { x, y };
      return false;
    }

    // Move is legal — commit
    this.illegalFlash = null;
    this.captured.black += newCaptured.black;
    this.captured.white += newCaptured.white;
    this.prevHash = hashBefore;
    this.lastMove = { x, y };
    this.consecutivePasses = 0;
    this.current = this.current === 'black' ? 'white' : 'black';
    return true;
  }

  pass() {
    if (this.gameOver) return;
    this.consecutivePasses++;
    this.lastMove = null;
    this.illegalFlash = null;
    const passer = this.current;
    this.current = this.current === 'black' ? 'white' : 'black';
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
    this.scores.white.total = this.scores.white.territory + this.scores.white.captures;
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

if (typeof module !== 'undefined') module.exports = { Board, Game };
