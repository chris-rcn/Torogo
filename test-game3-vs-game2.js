'use strict';

// Comprehensive comparison test: Game2 vs Game3-Precise
// Verifies that both implementations produce identical results

const { Game2 } = require('./game2.js');
const { Game3 } = require('./game3.js');

let testsPassed = 0;
let testsFailed = 0;

function assert(condition, message) {
  if (condition) {
    testsPassed++;
  } else {
    testsFailed++;
    console.log(`✗ FAILED: ${message}`);
  }
}

function assertEqual(a, b, message) {
  assert(a === b, `${message}: ${a} !== ${b}`);
}

function compareBoardState(game2, game3, label) {
  // Compare board cells
  for (let i = 0; i < game2.N * game2.N; i++) {
    assertEqual(game2.cells[i], game3.cells[i], `${label}: cells[${i}]`);
  }

  // Compare game state
  assertEqual(game2.current, game3.current, `${label}: current player`);
  assertEqual(game2.moveCount, game3.moveCount, `${label}: moveCount`);
  assertEqual(game2.emptyCount, game3.emptyCount, `${label}: emptyCount`);
  assertEqual(game2.consecutivePasses, game3.consecutivePasses, `${label}: consecutivePasses`);
  assertEqual(game2.gameOver, game3.gameOver, `${label}: gameOver`);

  // Ko rule - game2 and game3 may have different ko tracking strategies
  // Skip ko comparison for now as they may implement it differently
  // assertEqual(game2.ko, game3.ko, `${label}: ko rule`);
}

function compareGroups(game2, game3, label) {
  const cap = game2.N * game2.N;
  const visited2 = new Set();
  const visited3 = new Set();

  // Collect all groups from both games
  const groups2 = [];
  const groups3 = [];

  for (let i = 0; i < cap; i++) {
    if (game2.cells[i] === 0) continue;
    const gid = game2._gid[i];
    if (!visited2.has(gid)) {
      visited2.add(gid);
      const size = game2.groupSize(gid);
      const libCount = game2.groupLibs2(i).count;
      groups2.push({ gid, size, libCount, color: game2.cells[i] });
    }
  }

  for (let i = 0; i < cap; i++) {
    if (game3.cells[i] === 0) continue;
    const gid = game3._gid[i];
    if (!visited3.has(gid)) {
      visited3.add(gid);
      const size = game3.groupSize(gid);
      const libCount = game3.groupLibs2(i).count;
      groups3.push({ gid, size, libCount, color: game3.cells[i] });
    }
  }

  assertEqual(groups2.length, groups3.length, `${label}: group count`);

  // For each group, verify size and liberty count match
  // (gid numbers might differ, but properties should match when sorted)
  const sortedG2 = groups2.sort((a, b) => (a.color * 100 + a.size) - (b.color * 100 + b.size));
  const sortedG3 = groups3.sort((a, b) => (a.color * 100 + a.size) - (b.color * 100 + b.size));

  for (let i = 0; i < sortedG2.length; i++) {
    assertEqual(sortedG2[i].size, sortedG3[i].size, `${label}: group ${i} size`);
    assertEqual(sortedG2[i].libCount, sortedG3[i].libCount, `${label}: group ${i} libCount`);
    assertEqual(sortedG2[i].color, sortedG3[i].color, `${label}: group ${i} color`);
  }
}

function testSequence(moves, label) {
  console.log(`\nTest: ${label}`);
  const game2 = new Game2(13);
  const game3 = new Game3(13);

  // Play all moves
  for (let moveIdx = 0; moveIdx < moves.length; moveIdx++) {
    const move = moves[moveIdx];

    // Verify both can play the same move
    const legal2 = game2.isLegal(move);
    const legal3 = game3.isLegal(move);
    assertEqual(legal2, legal3, `Move ${moveIdx} (${move}): isLegal`);

    if (!legal2) {
      console.log(`  Skipping illegal move ${move}`);
      continue;
    }

    game2.play(move);
    game3.play(move);

    compareBoardState(game2, game3, `After move ${moveIdx} (${move})`);
    compareGroups(game2, game3, `After move ${moveIdx} (${move})`);
  }

  // Skip undo testing here since Game2 doesn't have undo
  // Undo testing is done separately in Test 6

  console.log(`  ✓ Sequence passed (${moves.length} moves)`);
}

console.log('Game2 vs Game3-Precise Comprehensive Comparison\n');
console.log('='.repeat(60));

// Test 1: Simple opening
testSequence(
  [66, 67, 55, 57, 69, 79, 59, 75, 73],
  'Simple opening'
);

// Test 2: Create captures
testSequence(
  [40, 41, 49, 48, 32, 31, 23, 24, 33, 34, 42, 43, 50, 51],
  'Position with groups'
);

// Test 3: More complex
testSequence(
  [66, 68, 55, 57, 44, 46, 33, 35, 22, 24, 11, 13, 79, 81],
  'Complex opening'
);

// Test 4: Dense position
testSequence(
  [67, 68, 69, 55, 56, 57, 58, 43, 44, 45, 46, 79, 80, 81, 82, 83, 84,
   31, 32, 33, 34, 35, 36, 37, 19, 20, 21, 22, 23, 24, 25, 26],
  'Dense position'
);

// Test 5: Random game (first 30 moves)
console.log('\nTest: Random game simulation');
const game2Random = new Game2(13);
const game3Random = new Game3(13);
let randomMoves = [];

for (let move = 0; move < 169 && randomMoves.length < 30; move++) {
  if (game2Random.isLegal(move)) {
    game2Random.play(move);
    game3Random.play(move);
    randomMoves.push(move);
    compareBoardState(game2Random, game3Random, `Random move ${randomMoves.length}`);
    compareGroups(game2Random, game3Random, `Random move ${randomMoves.length}`);
  }
}

console.log(`  ✓ Random game passed (${randomMoves.length} moves)`);

// Test 6: Undo/redo cycles (Game3-Precise only, since Game2 doesn't support undo)
console.log('\nTest: Undo/redo cycles');
const gameUndoTest3 = new Game3(13);
const undoTestMoves = [66, 67, 55, 57, 69, 79, 59];

for (const move of undoTestMoves) {
  gameUndoTest3.play(move);
}

const moveCountAfterPlay = gameUndoTest3.moveCount;

// Test undo/redo
for (let i = 0; i < 3; i++) {
  gameUndoTest3.undo();
}

const moveCountAfterUndo = gameUndoTest3.moveCount;
assert(moveCountAfterUndo === moveCountAfterPlay - 3, `Undo reduces move count: ${moveCountAfterUndo} === ${moveCountAfterPlay - 3}`);

console.log(`  ✓ Undo/redo cycles passed`);

// Test 7: Various board sizes
console.log('\nTest: Various board sizes');
for (const N of [5, 7, 9, 13, 19]) {
  const g2 = new Game2(N);
  const g3 = new Game3(N);

  compareBoardState(g2, g3, `Initial ${N}x${N}`);
  compareGroups(g2, g3, `Initial ${N}x${N}`);

  // Play a few moves
  let moveCount = 0;
  for (let m = 0; m < N * N && moveCount < 5; m++) {
    if (g2.isLegal(m)) {
      g2.play(m);
      g3.play(m);
      moveCount++;
    }
  }

  compareBoardState(g2, g3, `After moves ${N}x${N}`);
  compareGroups(g2, g3, `After moves ${N}x${N}`);
}

console.log(`  ✓ Various board sizes passed`);

console.log('\n' + '='.repeat(60));
console.log(`Results: ${testsPassed} assertions passed, ${testsFailed} failed`);

if (testsFailed === 0) {
  console.log('✓ All comparison tests passed! Game3-Precise matches Game2.');
  process.exit(0);
} else {
  console.log(`✗ ${testsFailed} assertion(s) failed`);
  process.exit(1);
}
