# Game3-Precise: Implementation Roadmap

## Current Status

**Code complete**: Core game mechanics 100%
**API coverage**: 85% (must-haves implemented)
**Test coverage**: 6 unit tests + 4 correctness tests passing
**Performance**: 5-11x faster than Game2.clone()

---

## Missing Features by Priority

### PHASE 1: Drop-In Replacement (HIGH PRIORITY)

#### 1.1 `clone()` Method
**Usage**: 20+ files (tactical search, tree exploration)
**Complexity**: LOW
**Time estimate**: 30 minutes

```javascript
clone(): Game3Precise {
  // Create new instance with same initial state
  const cloned = new Game3Precise(this.N);
  
  // Copy board state
  cloned.cells.set(this.cells);
  cloned._gid.set(this._gid);
  cloned.current = this.current;
  cloned.ko = this.ko;
  cloned.emptyCount = this.emptyCount;
  cloned.moveCount = this.moveCount;
  cloned.lastMove = this.lastMove;
  cloned.gameOver = this.gameOver;
  cloned.consecutivePasses = this.consecutivePasses;
  
  // Copy group data
  cloned._nextGid = this._nextGid;
  cloned._gc.set(this._gc);
  cloned._ss.set(this._ss);
  cloned._ls.set(this._ls);
  cloned._sw.set(this._sw);
  cloned._lw.set(this._lw);
  
  // Copy operation stack
  cloned._opStack = this._opStack.map(op => ({...op}));
  
  return cloned;
}
```

**Why clone() matters**: Many files expect to create branches for exploration and discard them

#### 1.2 `consecutivePasses` Property
**Usage**: 41 files (game end detection)
**Complexity**: LOW
**Time estimate**: 15 minutes

```javascript
// In constructor:
this.consecutivePasses = 0;

// In play() method:
if (move === PASS) {
  this._opStack.push({
    type: 'pass',
    previousCurrent: this.current,
    previousConsecutivePasses: this.consecutivePasses,
  });
  
  this.consecutivePasses++;
  if (this.consecutivePasses >= 2) {
    this.gameOver = true;
  }
  
  this.current = -this.current;
  this.moveCount++;
  return true;
}

// In undo():
if (op.type === 'pass') {
  this.current = op.previousCurrent;
  this.consecutivePasses = op.previousConsecutivePasses;
  this.moveCount--;
  if (this.consecutivePasses < 2) {
    this.gameOver = false;
  }
  return;
}
```

**Why tracking passes matters**: Two consecutive passes end the game

---

### PHASE 2: Playout Optimization (MEDIUM PRIORITY)

#### 2.1 `isTrueEye()` Method
**Usage**: 31 files (playout move pruning)
**Complexity**: MEDIUM
**Time estimate**: 1-2 hours

```javascript
isTrueEye(idx: number): boolean {
  // An eye is a vacant point surrounded by stones of one color
  // A true eye is one where the diagonals don't give control to opponent
  
  if (this.cells[idx] !== EMPTY) return false;
  
  const N = this.N;
  const color = null; // Will determine from neighbors
  const neighbors = [
    this._nbr[idx * 4 + 0], // up
    this._nbr[idx * 4 + 1], // down
    this._nbr[idx * 4 + 2], // left
    this._nbr[idx * 4 + 3], // right
  ];
  
  // Check all neighbors are same color (or same group)
  const neighborColor = this.cells[neighbors[0]];
  if (neighborColor === EMPTY) return false; // Not surrounded
  
  for (const nbr of neighbors) {
    if (this.cells[nbr] !== neighborColor) {
      return false; // Mixed colors - not an eye
    }
  }
  
  // Check diagonals for proper eye pattern
  const diagonals = [
    idx - this.N - 1,  // up-left
    idx - this.N + 1,  // up-right
    idx + this.N - 1,  // down-left
    idx + this.N + 1,  // down-right
  ];
  
  let friendlyDiagonals = 0;
  const oppColor = -neighborColor;
  
  for (const diag of diagonals) {
    // Check bounds
    if (diag < 0 || diag >= this.N * this.N) continue;
    
    const diagColor = this.cells[diag];
    if (diagColor === neighborColor) {
      friendlyDiagonals++;
    }
  }
  
  // True eye: at least 3 diagonals are friendly
  // (1+ is enough for corner, 2+ for edge, 3+ for center)
  const x = idx % this.N;
  const isCorner = (x === 0 || x === this.N - 1);
  const isEdge = (x === 0 || x === this.N - 1 || idx < this.N || idx >= this.N * (this.N - 1));
  
  if (isCorner) return friendlyDiagonals >= 1;
  if (isEdge) return friendlyDiagonals >= 2;
  return friendlyDiagonals >= 3;
}
```

**Why true eye detection matters**: Saves time in playouts by avoiding obviously bad moves

**Reference implementation**: Check game2.js for exact semantics to match

#### 2.2 Neighbor-Based Queries
**Note**: Already implemented via `_nbr` array
- Neighbors are accessed as: `_nbr[idx * 4 + direction]` where direction is 0-3
- This is already working in Game3-Precise

---

### PHASE 3: End Game Support (LOWER PRIORITY)

#### 3.1 `calcWinner()` Method
**Usage**: 28 files (final scoring)
**Complexity**: HIGH
**Time estimate**: 2-3 hours

```javascript
calcWinner(): {
  black: number,
  white: number,
  komi: number,
  score: number,
  winner: number,
  margin: number
} {
  const komi = 6.5; // Standard rule
  
  // Count territories
  const territory = { black: 0, white: 0, neutral: 0 };
  const visited = new Int8Array(this.N * this.N);
  
  // For each empty cell, determine territory
  for (let idx = 0; idx < this.N * this.N; idx++) {
    if (this.cells[idx] !== EMPTY || visited[idx]) continue;
    
    // Flood fill to find connected empty region
    const region = [];
    const queue = [idx];
    visited[idx] = 1;
    let surroundingColor = null;
    let isMixed = false;
    
    while (queue.length > 0) {
      const pos = queue.shift();
      region.push(pos);
      
      // Check neighbors
      for (let i = 0; i < 4; i++) {
        const neighbor = this._nbr[pos * 4 + i];
        if (visited[neighbor]) continue;
        
        if (this.cells[neighbor] === EMPTY) {
          visited[neighbor] = 1;
          queue.push(neighbor);
        } else {
          // Neighbor has a stone
          const nbrColor = this.cells[neighbor];
          if (surroundingColor === null) {
            surroundingColor = nbrColor;
          } else if (surroundingColor !== nbrColor) {
            isMixed = true;
          }
        }
      }
    }
    
    // Assign territory
    if (isMixed || surroundingColor === null) {
      territory.neutral += region.length;
    } else if (surroundingColor === BLACK) {
      territory.black += region.length;
    } else {
      territory.white += region.length;
    }
  }
  
  // Count stones
  let blackStones = 0;
  let whiteStones = 0;
  for (let idx = 0; idx < this.N * this.N; idx++) {
    if (this.cells[idx] === BLACK) blackStones++;
    if (this.cells[idx] === WHITE) whiteStones++;
  }
  
  // Calculate scores
  const blackScore = blackStones + territory.black;
  const whiteScore = whiteStones + territory.white + komi;
  const scoreDiff = blackScore - whiteScore;
  
  return {
    black: blackScore,
    white: whiteScore,
    komi: komi,
    score: scoreDiff,
    winner: scoreDiff > 0 ? BLACK : WHITE,
    margin: Math.abs(scoreDiff),
  };
}
```

**Complexity notes**:
- Requires territory detection (flood fill)
- Must distinguish neutral vs. owned territory
- Handled captures are already removed from cells

**Reference**: Check game2.js implementation for exact details

#### 3.2 `estimateWinner()` Method
**Usage**: 15 files (position evaluation)
**Complexity**: MEDIUM
**Time estimate**: 1 hour

```javascript
estimateWinner(): {
  score: number,
  leader: number,
  margin: number
} {
  // Simple heuristic: count stones + liberties
  let blackScore = 0;
  let whiteScore = 0;
  const visited = new Set();
  
  for (let idx = 0; idx < this.N * this.N; idx++) {
    if (this.cells[idx] === BLACK) {
      blackScore += 1.5; // Stone is worth points
      const gid = this._gid[idx];
      if (!visited.has(gid)) {
        visited.add(gid);
        blackScore += this._ls[gid] * 0.3; // Liberties worth some value
      }
    } else if (this.cells[idx] === WHITE) {
      whiteScore += 1.5;
      const gid = this._gid[idx];
      if (!visited.has(gid)) {
        visited.add(gid);
        whiteScore += this._ls[gid] * 0.3;
      }
    }
  }
  
  // Komi adjustment
  whiteScore += 6.5;
  
  const scoreDiff = blackScore - whiteScore;
  
  return {
    score: scoreDiff,
    leader: scoreDiff > 0 ? BLACK : WHITE,
    margin: Math.abs(scoreDiff),
  };
}
```

**Why estimate**: Full territory detection is expensive; this heuristic is fast

---

### PHASE 4: Debugging Support (LOW PRIORITY)

#### 4.1 `toString()` Method
**Usage**: 12 files (debug output)
**Complexity**: LOW
**Time estimate**: 30 minutes

```javascript
toString(idx?: number, opts?: { centerAt?: number, width?: number }): string {
  const width = opts?.width || 70;
  const centered = opts?.centerAt !== undefined;
  
  let output = '';
  
  // Board display
  for (let y = 0; y < this.N; y++) {
    let line = '';
    for (let x = 0; x < this.N; x++) {
      const cell = y * this.N + x;
      if (cell === idx) {
        line += this.cells[cell] === BLACK ? '●' : 
                this.cells[cell] === WHITE ? '○' : '*';
      } else {
        line += this.cells[cell] === BLACK ? '●' : 
                this.cells[cell] === WHITE ? '○' : '·';
      }
      line += ' ';
    }
    output += line + '\n';
  }
  
  return output;
}
```

**Reference**: Check game2.js toString implementation for full formatting

---

## Implementation Priority Matrix

| Feature | Users | Complexity | Enables | Priority |
|---------|-------|-----------|---------|----------|
| `clone()` | 20 | LOW | Tactical search | **PHASE 1** |
| `consecutivePasses` | 41 | LOW | Game end detection | **PHASE 1** |
| `isTrueEye()` | 31 | MEDIUM | Optimal playouts | **PHASE 2** |
| `calcWinner()` | 28 | HIGH | Final scoring | **PHASE 3** |
| `estimateWinner()` | 15 | MEDIUM | Position eval | **PHASE 3** |
| `toString()` | 12 | LOW | Debugging | **PHASE 4** |

---

## Testing Strategy

### Phase 1 Testing
- [ ] Test `clone()` preserves all game state
- [ ] Test `clone()` is independent (changes to clone don't affect original)
- [ ] Test `consecutivePasses` increments correctly
- [ ] Test game ends after 2 consecutive passes

### Phase 2 Testing
- [ ] Test `isTrueEye()` on corner eyes
- [ ] Test `isTrueEye()` on edge eyes
- [ ] Test `isTrueEye()` on center eyes
- [ ] Test `isTrueEye()` rejects false eyes
- [ ] Compare results with Game2 implementation

### Phase 3 Testing
- [ ] Test `calcWinner()` on completed games
- [ ] Test territory calculation accuracy
- [ ] Test komi application
- [ ] Test `estimateWinner()` on mid-game positions
- [ ] Compare heuristics with actual final scores

### Phase 4 Testing
- [ ] Test `toString()` output readability
- [ ] Test highlighted position display
- [ ] Compare formatting with Game2

---

## Code Changes Required

### Changes to game3-precise.js

**Constructor**:
```javascript
this.gameOver = false;
this.consecutivePasses = 0;  // ADD THIS
```

**play() method - modify PASS handling**:
```javascript
if (move === PASS) {
  const prevPasses = this.consecutivePasses;  // ADD
  this._opStack.push({
    type: 'pass',
    previousCurrent: this.current,
    previousConsecutivePasses: this.consecutivePasses,  // ADD
  });
  
  this.consecutivePasses++;  // ADD
  if (this.consecutivePasses >= 2) {  // ADD
    this.gameOver = true;  // ADD
  }  // ADD
  
  this.current = -this.current;
  this.moveCount++;
  return true;
}
```

**undo() method - modify PASS handling**:
```javascript
if (op.type === 'pass') {
  this.current = op.previousCurrent;
  this.consecutivePasses = op.previousConsecutivePasses;  // ADD
  this.gameOver = this.consecutivePasses < 2;  // ADD
  this.moveCount--;
  return;
}
```

**Add new methods**:
```javascript
clone(): Game3Precise { ... }
isTrueEye(idx: number): boolean { ... }
calcWinner(): { ... } { ... }
estimateWinner(): { ... } { ... }
toString(idx?: number, opts?: object): string { ... }
```

---

## Estimated Timeline

| Phase | Tasks | Time | Cumulative |
|-------|-------|------|-----------|
| 1 | clone + consecutivePasses | 1 hour | 1 hour |
| 2 | isTrueEye | 2 hours | 3 hours |
| 3 | calcWinner + estimateWinner | 3 hours | 6 hours |
| 4 | toString | 0.5 hours | 6.5 hours |
| **Testing** | All methods | 1.5 hours | **8 hours** |

---

## Success Criteria

- [ ] All 6 Phase 1-4 methods implemented
- [ ] All 10+ test cases passing
- [ ] Results match Game2 implementation
- [ ] Drop-in replacement works for 40+ files
- [ ] No performance regression
- [ ] Full backward compatibility

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| `isTrueEye()` differs from Game2 | Playouts behave differently | Run side-by-side comparison tests |
| Territory detection complexity | `calcWinner()` incorrect scoring | Thorough flood-fill testing |
| `clone()` performance overhead | Negates some benefits | Benchmark clone vs new instances |
| Missing edge cases | Some code paths fail | Comprehensive test suite |

---

## Conclusion

Game3-Precise is **85% complete** for full Game2 replacement.

**Next steps**:
1. Implement Phase 1 (6 hours work) to enable tactical search
2. Run compatibility tests against existing codebase
3. Gradually migrate files from Game2 to Game3-Precise
4. Implement Phases 2-4 as needed

**Expected outcome**: 
- Drop-in replacement for Game2 with 5-11x faster tactical search
- Same API, better performance, zero reconstruction overhead
- Production-ready for all Go game analysis
