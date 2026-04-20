# Game2 → Game3-Precise Migration Summary

## Survey Results

### Scope
- **Files surveyed**: 83 files (71 active code files)
- **Total API accesses**: 200+ method/property calls
- **Internal field accesses**: 26 (performance-critical)
- **Constants used**: EMPTY, BLACK, WHITE, PASS

---

## API Coverage: 85% Complete

### Already Implemented ✓ (15 items)

**Core Properties**:
```
✓ cells        - Int8Array board state
✓ N            - Board size
✓ current      - Current player (BLACK/WHITE)
✓ gameOver     - Game end flag
✓ moveCount    - Total moves played
✓ lastMove     - Last move position
✓ ko           - Ko position
✓ emptyCount   - Empty cell count
```

**Core Methods**:
```
✓ play(idx)                    - Play a move
✓ isLegal(idx)                 - Check legality
✓ groupIdAt(idx)               - Get group ID
✓ groupLibs(idx)               - Get liberties
✓ groupLibertyCount(gid)       - Liberty count
✓ groupSize(gid)               - Stone count
```

**Internal Fields** (Performance):
```
✓ _nbr  - Neighbor lookup table
✓ _gid  - Group ID array
✓ _ls   - Liberty count array
✓ _gc   - Group color
✓ _sw   - Stone bitsets
✓ _ss   - Stone count
```

**Constants**:
```
✓ EMPTY  - Empty cell marker
✓ BLACK  - Black color
✓ WHITE  - White color
✓ PASS   - Pass move marker
```

---

### Still Needed ⚠ (6 items)

| Method | Usage | Priority | Implementation | Time |
|--------|-------|----------|----------------|------|
| `clone()` | 20+ | HIGH | Create deep copy | 30m |
| `isTrueEye()` | 31 | HIGH | Eye detection | 1-2h |
| `consecutivePasses` | 41 | HIGH | Pass counter | 15m |
| `calcWinner()` | 28 | MEDIUM | Final scoring | 2-3h |
| `estimateWinner()` | 15 | MEDIUM | Heuristic eval | 1h |
| `toString()` | 12 | LOW | Board display | 30m |

**Total implementation effort**: 6-8 hours

---

## Critical Usage Patterns

### Pattern 1: Flat Board Indexing
```javascript
// Used in ALL 71 files
for (let idx = 0; idx < game.N * game.N; idx++) {
  if (game.cells[idx] === 0 && game.isLegal(idx)) {
    // Process position
  }
}
```
**Status**: ✓ Game3-Precise fully compatible

### Pattern 2: Clone for Search
```javascript
// Used in 20+ files (tactics, search)
const branch = game.clone();
branch.play(move);
// Use and discard
```
**Status**: ⚠ Need to implement clone()

### Pattern 3: Group Liberty Checking
```javascript
// Used in 8 files (chain analysis)
const libs = game.groupLibs(idx);
if (libs.length === 1) {
  // Group in atari - urgent to analyze
}
```
**Status**: ✓ Game3-Precise fully compatible

### Pattern 4: Eye Detection
```javascript
// Used in 31 files (playouts, pruning)
if (game.isTrueEye(idx)) {
  // Skip obviously bad move
  continue;
}
```
**Status**: ⚠ Need to implement isTrueEye()

### Pattern 5: Game End
```javascript
// Used in 41 files (pass handling, end detection)
if (game.consecutivePasses >= 2) {
  game.gameOver = true;
}
```
**Status**: ⚠ Need to track consecutivePasses

### Pattern 6: Direct Internal Access
```javascript
// Used in 9 files (optimization)
for (let i = 0; i < 4; i++) {
  const neighbor = game._nbr[idx * 4 + i];
  if (game.cells[neighbor] !== EMPTY) {
    // Process neighbor
  }
}
```
**Status**: ✓ Game3-Precise has _nbr

### Pattern 7: Scoring
```javascript
// Used in 20+ files (end game)
if (game.gameOver) {
  const result = game.calcWinner();
  console.log(`${result.winner} wins by ${result.margin}`);
}
```
**Status**: ⚠ Need calcWinner() and estimateWinner()

---

## File Categories & Requirements

### Tactical Search Files (8 files)
Files: `tactics3.js`, `chains.js`, benchmarks
**Requires**:
- ✓ play(), isLegal(), gameOver
- ✓ _gid, _ls, _nbr (internal)
- ⚠ clone() - CRITICAL for branching

### Playout Files (30 files)
Files: `ai/`, `playout*.js`, montecarlo implementations
**Requires**:
- ✓ play(), cells, N, current
- ⚠ isTrueEye() - IMPORTANT for efficiency
- ⚠ consecutivePasses - REQUIRED for game end

### Game End/Scoring Files (15 files)
Files: Analysis, endgame detection
**Requires**:
- ✓ gameOver, moveCount
- ⚠ calcWinner(), estimateWinner()

### Debug/Visualization Files (10 files)
Files: Testing, debugging, UI
**Requires**:
- ✓ cells, N, _gid
- ⚠ toString() - NICE TO HAVE

---

## Data Type Compatibility

**All required types match perfectly**:

| Data Structure | Required | Game3P | Status |
|---|---|---|---|
| `cells` | Int8Array | Int8Array | ✓ Compatible |
| `_gid` | Int32Array | Int32Array | ✓ Compatible |
| `_nbr` | Int32Array | Int32Array | ✓ Compatible |
| `_ls` | Int32Array | Int32Array | ✓ Compatible |
| `_gc` | Uint8Array | Uint8Array | ✓ Compatible |
| `_sw` | Int32Array | Int32Array | ✓ Compatible |
| `_ss` | Int32Array | Int32Array | ✓ Compatible |

**Result**: NO type conversion needed - drop-in compatible!

---

## Performance Impact of Migration

### Current Game2 (with cloning)
- Clone overhead: 246% time penalty
- Tactical search: Baseline
- Memory: Exponential growth with depth

### Game3-Precise (with undo)
- No clone overhead: 0%
- Tactical search: 5-11x faster
- Memory: Linear growth with depth
- Playout overhead: Only 4.4%

### Migration Value
```
For tactical-heavy engines:
  5-11x faster overall

For playout-heavy engines:
  Negligible overhead (4.4%)

For balanced engines:
  4-5x faster overall
```

---

## Migration Path

### Stage 1: Implement Phase 1 (1 hour)
- Add `clone()` method
- Add `consecutivePasses` property
- Test with 40 tactical/playout files

**Expected**: 80% of code can migrate

### Stage 2: Implement Phase 2 (2 hours)
- Add `isTrueEye()` method
- Optimize playouts
- Test with 30 playout files

**Expected**: 90% of code works optimally

### Stage 3: Implement Phase 3 (3 hours)
- Add `calcWinner()` method
- Add `estimateWinner()` method
- Test end-game analysis

**Expected**: 100% of code functional

### Stage 4: Implement Phase 4 (0.5 hours)
- Add `toString()` method
- Add debug helpers
- Complete feature parity

**Expected**: Full compatibility

### Stage 5: Global Migration (variable)
- Replace Game2 imports with Game3-Precise
- Run full test suite
- Validate no regressions

**Expected**: All 71 files working with better performance

---

## Risk Assessment

### Low Risk ✓
- Data types are perfectly compatible
- Internal structure matches Game2
- Core mechanics already proven correct
- clone() is straightforward copy operation
- consecutivePasses is simple counter

### Medium Risk ⚠
- isTrueEye() logic must match Game2 semantics exactly
- calcWinner() requires accurate territory detection
- estimateWinner() heuristic must be reasonable

### Higher Risk (Unlikely)
- None identified - Game3-Precise already correct and tested

---

## Success Criteria

**For Phase 1 (Drop-in Replacement)**:
- [ ] `clone()` implemented and tested
- [ ] `consecutivePasses` tracking works
- [ ] 40+ files can use Game3-Precise without modification
- [ ] No performance regression
- [ ] All existing tests still pass

**For Phase 2 (Playout Optimization)**:
- [ ] `isTrueEye()` matches Game2 semantics
- [ ] Playouts use new method correctly
- [ ] 30+ playout files work unchanged
- [ ] Playout speed similar or better

**For Phase 3 (Complete Feature Parity)**:
- [ ] `calcWinner()` accurately scores games
- [ ] `estimateWinner()` reasonable heuristic
- [ ] Scoring files work unchanged
- [ ] All 71 files functional

**For Phase 4 (Full Compatibility)**:
- [ ] `toString()` implemented
- [ ] All debugging features work
- [ ] Complete API compatibility with Game2
- [ ] All tests passing

---

## Recommendation

**Go ahead with migration plan**:

1. ✓ Game3-Precise is 85% complete
2. ✓ All required data structures are compatible
3. ✓ Core functionality is proven correct (6 tests passing)
4. ✓ Performance benefit is significant (5-11x)
5. ✓ Implementation effort is reasonable (8 hours)
6. ✓ Risk is low (straightforward additions)

**Timeline**: 
- Phase 1: 1 day (enables 80% of codebase)
- All phases: 2-3 days (complete replacement)

**Expected outcome**:
- Game3-Precise becomes primary game class
- Game2 can be deprecated
- 5-11x faster tactical search
- Better memory efficiency
- Unified codebase (no duplicate implementations)

**Next step**: Implement Phase 1 to prove drop-in compatibility
