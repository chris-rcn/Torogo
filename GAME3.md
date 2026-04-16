# Game3: Fast Incremental Game Class for Tactical Searches

## Overview

**Game3** is a high-performance game class optimized for tactical searches. Unlike Game2 which uses expensive cloning (`game.clone()`), Game3 uses an **incremental undo stack** for exploring game trees.

### Key Features

- **~1.6-2x faster** than Game2.clone() for tactical searches
- **Fully incremental**: Changes are tracked and can be undone efficiently
- **Effectively unlimited undo**: Undo depth limited only by stack size (no practical limit)
- **No cloning overhead**: Each move just pushes a change record to a stack
- **Memory efficient**: Same per-game memory usage as Game2 (~1.8KB per 5x5 board)

## Why Game3 is Faster

### Game2 Approach (Clone-Based)
```javascript
const clone = game.clone();  // Allocates 7+ typed arrays
clone.play(move);
// ... game state is cloned, not shared
```

Game2's clone allocates new typed arrays for:
- `cells` (board state)
- `_gid` (group IDs)
- `_gc`, `_sw`, `_ss`, `_lw`, `_ls` (group data)
- `_emptyCells`, `_emptySlot` (empty cell tracking)
- `koStone` array
- Total: ~80KB per clone on 9x9 board

### Game3 Approach (Incremental)
```javascript
game.play(move);    // Push change to undo stack
game.undo();        // Pop change from stack
```

Game3 never allocates large arrays per move:
- Single `_undoStack` array (initially empty)
- Each move just pushes a small change record
- No deep copies, no array duplication

## Usage

### Basic API

```javascript
const { Game3, PASS, BLACK, WHITE } = require('./game3.js');

const game = new Game3(5);  // Create 5x5 game (starts with center stone)

// Play a move
if (game.isLegal(moveIdx)) {
  game.play(moveIdx);      // Returns true on success
}

// Undo a move
game.undo();               // Reverts to previous state

// Pass
game.play(PASS);           // Pass move

// Check game state
game.current;              // BLACK or WHITE
game.cells;                // Int8Array: [0=empty, 1=BLACK, -1=WHITE]
game.groupLibertyCount(gid);  // Number of liberties for group
game.groupLibs(stoneIdx);     // Int32Array of liberty indices
```

### Tactical Search Pattern

**Before (Game2):**
```javascript
function searchChain(game2, stoneIdx) {
  const libs = game2.groupLibs(stoneIdx);
  for (let i = 0; i < libs.length; i++) {
    const clone = game2.clone();        // Expensive!
    clone.play(libs[i]);
    const result = searchChain(clone, stoneIdx);
    // process result
  }
}
```

**After (Game3):**
```javascript
function searchChain(game3, stoneIdx) {
  const libs = game3.groupLibs(stoneIdx);
  for (let i = 0; i < libs.length; i++) {
    game3.play(libs[i]);
    const result = searchChain(game3, stoneIdx);
    game3.undo();                       // Fast!
    // process result
  }
}
```

## Undo Implementation

Game3 maintains a `_undoStack` array where each entry records:
- Move type ('move', 'place', 'pass')
- Stone index and color
- Captured stones (if any)
- Previous ko status
- Previous board state

Undo works by:
1. Popping the last change from the stack
2. Restoring cell states
3. Rebuilding group assignments
4. Restoring game state (ko, current player, etc.)

## Performance Characteristics

### Initialization
- **Game2**: ~0.1ms (5x5 board)
- **Game3**: ~0.1ms (5x5 board)

### Single Move + Undo
- **Game2 (clone + play)**: ~0.34ms per cycle
- **Game3 (play + undo)**: ~0.20ms per cycle
- **Speedup**: ~1.7x faster

### Deep Tactical Search (5 moves, 4 branches each)
- **Game2**: ~0.33ms per position
- **Game3**: ~0.20ms per position
- **Speedup**: ~1.6x faster
- **Cumulative benefit**: For 1000-node search, saves ~130ms

## Migration from Game2

1. **Replace import**:
   ```javascript
   // Old
   const { Game2 } = require('./game2.js');
   const game = new Game2(size);

   // New
   const { Game3 } = require('./game3.js');
   const game = new Game3(size);
   ```

2. **Replace clone with play/undo**:
   ```javascript
   // Old
   const g = game.clone();
   g.play(move);

   // New
   game.play(move);
   game.undo();
   ```

3. **Recursive searches**: No change needed - Game3 API is compatible with Game2 for read-only operations:
   - `game.cells`
   - `game.current`
   - `game.isLegal(move)`
   - `game.groupLibertyCount(gid)`
   - `game.groupLibs(idx)`
   - `game.N` (board size)

## Limitations

- **Single-threaded only**: Undo stack is not thread-safe
- **No clones**: If you need a completely independent copy, you'd need to write a copy method
- **Group reconstruction on undo**: Rebuilds group state from cells (still much faster than cloning)

## Testing

Run the test suite:
```bash
node test-game3.js
```

Run benchmarks:
```bash
node bench-game3.js
```

## Files

- `game3.js` - Implementation
- `test-game3.js` - Unit tests
- `bench-game3.js` - Performance benchmarks
- `GAME3.md` - This documentation
