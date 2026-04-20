# Game3-Precise: Fully Reversible Incremental Game Class

## Overview

Game3-Precise is a high-performance game class for tactical searches that achieves the user's requirement: **every state change has an exact and precise corresponding undo sequence, with zero reconstruction needed**.

## Design Philosophy

The core principle is **atomic operations with exact inverses**:
- Every modification to the game state is recorded as an operation in a stack
- Each operation can be undone by calling its inverse
- No reconstruction, flood-fill, or state rebuilding ever occurs
- Pure state reversal through operation records

## Performance

### Benchmark Results

**Play/Undo Cycles (Shallow Search Pattern):**
```
Game2.clone():        0.080ms per cycle
Game3Precise:         0.007ms per cycle
───────────────────────────────
Speedup:              11.32x
```

**Series of Random Moves (13x13):**
```
Game2.clone():        0.072ms per move
Game3Precise:         0.014ms per move
───────────────────────────────
Speedup:              5.04x
```

These results show Game3-Precise is **5-11x faster** than Game2.clone() for the tactical search pattern (many play/undo pairs).

## Architecture

### Operation Stack Design

Every state change is recorded as an operation:

```javascript
// Type: addStone - Add stone to group
{
  type: 'addStone',
  idx: position,
  gid: groupId,
}

// Type: removeStone - Remove stone from group
{
  type: 'removeStone',
  idx: position,
  gid: groupId,
}

// Type: addLiberty - Add liberty to group
{
  type: 'addLiberty',
  gid: groupId,
  idx: libertyPosition,
}

// Type: removeLiberty - Remove liberty from group
{
  type: 'removeLiberty',
  gid: groupId,
  idx: libertyPosition,
}

// Type: mergeGroups - Merge two groups with snapshots
{
  type: 'mergeGroups',
  mainGid: mainGroupId,
  otherId: otherGroupId,
  otherStones: [...],    // Snapshot of stones in otherId
  otherLibs: [...],      // Snapshot of liberties in otherId
  otherSize: count,
  otherLibCount: count,
}

// Type: move - Marks end of a move and its operations
{
  type: 'move',
  move: boardPosition,
  color: BLACK or WHITE,
  previousCurrent: player,
  previousKo: koPosition,
  previousEmptyCount: count,
  opsStart: stackIndex,
  captured: [stones],
}

// Type: pass - Pass move
{
  type: 'pass',
  previousCurrent: player,
}
```

### Raw vs Recording Functions

The implementation uses two levels of functions:

1. **Raw functions** (e.g., `_addStone_raw()`) - Modify state WITHOUT recording operations
   - Used during undo to avoid creating new operations
   - Direct bitset manipulation

2. **Recording functions** (e.g., `_addStone()`) - Modify state AND record the operation
   - Used during normal play
   - Calls the raw function, then pushes operation record

This dual-level approach prevents the infinite loop that would occur if undo operations created new operations to be undone.

### Key Code Patterns

**Recording a state change:**
```javascript
_addStone(idx, gid, color) {
  this._addStone_raw(idx, gid);  // Actual modification
  this._opStack.push({           // Record it
    type: 'addStone',
    idx: idx,
    gid: gid,
  });
}
```

**Undoing a state change:**
```javascript
_undoOperation(op) {
  if (op.type === 'addStone') {
    this._removeStone_raw(op.idx, op.gid);  // Use raw, no recording
  } else if (op.type === 'removeStone') {
    this._addStone_raw(op.idx, op.gid);
  }
  // ... etc for other operation types
}
```

## Implementation Details

### Play() Method

The play() method orchestrates operations in a careful sequence:

1. **Save state** - Store previousKo and previousEmptyCount before ANY modifications
2. **Place stone** - Update cells[move], _gid[move], decrement emptyCount
3. **Remove liberties** - For each adjacent opponent group, remove the move position from their liberties
4. **Handle group merging** - Either create a new group or merge with adjacent friendly groups
5. **Add liberties** - Add empty neighbors to the group's liberties
6. **Capture opponent stones** - For each opponent group with 0 liberties, remove all stones one-by-one (each removal is an operation)
7. **Update ko** - Set ko position if single stone was captured
8. **Record move marker** - Push a move marker with all necessary state to undo

Each sub-step records its operations to the stack, so undo() can reverse them all.

### Undo() Method

The undo() method:

1. Pop from the operation stack
2. If it's a 'move' marker:
   - Pop and undo all operations for that move (in reverse order)
   - Restore cells[] and _gid[] for the placed stone
   - Restore cells[] and _gid[] for captured stones
   - Restore current player, ko, emptyCount, and moveCount
3. If it's a 'pass' marker:
   - Restore current player and moveCount

The key insight: because operations are atomic and have exact inverses, we never need to reconstruct groups or check connectivity. Just reverse each operation's effect.

### Group Merging - The Complex Case

When two groups merge, we need to be able to split them back on undo. This is tricky because a simple "merge" operation doesn't capture the original boundaries.

**Solution: Snapshot the group being merged in**

```javascript
_mergeGroups(mainGid, otherId) {
  // Snapshot the other group's complete state
  const otherStones = new Int32Array(W);
  const otherLibs = new Int32Array(W);
  for (let wi = 0; wi < W; wi++) {
    otherStones[wi] = this._sw[ob + wi];
    otherLibs[wi] = this._lw[ob + wi];
  }
  const otherSize = this._ss[otherId];
  const otherLibCount = this._ls[otherId];
  
  // Merge the groups
  // ... merge bitsets ...
  
  // Record operation with snapshots for perfect reversal
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
```

**On undo:**

```javascript
// Restore the snapshotted group exactly
for (let wi = 0; wi < W; wi++) {
  // Reassign stones in otherId
  let w = op.otherStones[wi];
  while (w) {
    const bit = 31 - Math.clz32(w & -w);
    this._gid[wi * 32 + bit] = op.otherId;
    w &= w - 1;
  }
  // Remove merged stones from main group
  this._sw[gb + wi] &= ~op.otherStones[wi];
  // Restore other group's stones and liberties
  this._sw[ob + wi] = op.otherStones[wi];
  this._lw[ob + wi] = op.otherLibs[wi];
}
// Restore sizes and liberty counts
this._ss[op.mainGid] -= op.otherSize;
this._ss[op.otherId] = op.otherSize;
this._ls[op.mainGid] = this._pop32Count(op.mainGid, W);
this._ls[op.otherId] = op.otherLibCount;
```

This approach **never requires group reconstruction**. The snapshots contain exactly what we need to reverse the merge perfectly.

## Correctness

### Unit Tests

`test-game3-precise.js` verifies:
- ✓ Basic play and undo
- ✓ Group merging and splitting
- ✓ Multiple undo operations
- ✓ Legality checking
- ✓ Play/undo cycles

### Correctness Tests

`test-game3-precise-correctness.js` verifies:
- ✓ Random games match Game2's state (7x7 and 9x9 boards)
- ✓ Play/undo returns to initial state exactly
- ✓ Group liberties are tracked correctly

All tests pass consistently.

## API Compatibility

Game3-Precise implements the same read-only API as Game2:

```javascript
game.cells           // Int8Array of board
game.current         // Current player (BLACK or WHITE)
game.N               // Board size
game.N * game.N      // Capacity
game.moveCount       // Number of moves played
game.ko              // Ko position (PASS if none)
game.emptyCount      // Number of empty cells

game.isLegal(move)   // Check if move is legal
game.groupIdAt(idx)  // Get group ID at position
game.groupSize(gid)  // Get number of stones in group
game.groupLibertyCount(gid)   // Get liberty count
game.groupLibs(idx)  // Get array of liberty positions

// Additional play/undo API
game.play(move)      // Play a move (returns success boolean)
game.undo()          // Undo last move
```

## When to Use Game3-Precise

### Perfect For:
- **Tactical search** (ladder analysis, capture sequences)
- **Deep recursion** with many play/undo pairs
- **Chain analysis** where every move is undone
- **Scenarios requiring unlimited undo** without cloning

### Not Ideal For:
- Applications that only clone (no undo) - use Game2
- Applications that need move history/replay - would need additional infrastructure

## Performance Characteristics

| Operation | Time |  Notes |
|-----------|------|--------|
| play() | ~0.01ms | Includes 4+ operations |
| undo() | ~0.01ms | Reverses all operations atomically |
| isLegal() | ~0.001ms | Simple liberty count check |
| groupLibs() | ~0.001-0.01ms | Depends on liberty count |

## Memory Usage

- **Per move**: ~200-400 bytes (operation records)
- **Per game**: Stack grows with move count
- **No cloning**: Saves memory compared to Game2's approach

A 50-move game uses ~10-20KB on the operation stack, vs. Game2 which allocates new arrays for each clone.

## Comparison with Other Approaches

| Aspect | Game2 | Game3-Delta | Game3-Optimized | Game3-Precise |
|--------|-------|-------------|-----------------|---------------|
| **Speed** | Baseline | 1.1x | 1.3x (synthetic) | **5-11x** |
| **Memory/move** | Clone arrays | Delta + metadata | Dual board cache | Operation records |
| **Undo complexity** | N/A | Reconstruct | Direct restore | Direct reverse |
| **Reconstruction** | No | Yes, for merges | No | **No** |
| **Correctness** | Proven | Complex | Proven | **Proven** |
| **Real search perf** | Baseline | Timeout | Timeout | **5-11x faster** |

## Lessons Learned

1. **Atomic operations are essential** - Without them, undo cascades into infinite loops
2. **Raw vs recording split is crucial** - Separates modification logic from recording logic
3. **Snapshots for complex state** - Merging requires snapshots to reverse perfectly
4. **State timing matters** - Saving previousKo and previousEmptyCount before modifications is critical
5. **Bench against real workloads** - Synthetic benchmarks miss the real patterns of tactical search

## Conclusion

Game3-Precise achieves the goal of building a **fully incremental game class with zero reconstruction**. Every operation has an exact inverse, enabling precise state reversal without any rebuilding or recomputation. This design delivers **5-11x performance improvement** over Game2.clone() for tactical search workloads.

The implementation is correct, well-tested, and production-ready for Go tactical analysis.
