'use strict';

const { Game3Precise, PASS, BLACK, WHITE, EMPTY } = require('./game3-precise.js');

function testBasicPlay() {
  console.log('Testing basic play...');
  const game = new Game3Precise(5);

  console.assert(game.cells[12] === BLACK, 'Center should be BLACK');
  console.assert(game.current === WHITE, 'Current should be WHITE');
  console.assert(game.moveCount === 1, 'Move count should be 1');

  const move1 = 6;
  console.assert(game.isLegal(move1), `Move ${move1} should be legal`);

  const played = game.play(move1);
  console.assert(played, 'Move should succeed');
  console.assert(game.cells[move1] === WHITE, 'Cell should have WHITE');
  console.assert(game.current === BLACK, 'Turn should flip');

  console.log('✓ Basic play works');
}

function testUndo() {
  console.log('Testing undo...');
  const game = new Game3Precise(5);

  const initialCurrent = game.current;
  const initialMoveCount = game.moveCount;

  const move = 6;
  game.play(move);
  console.assert(game.cells[move] === WHITE, 'Stone placed');
  console.assert(game.moveCount === 2, 'Move count incremented');

  game.undo();
  console.assert(game.cells[move] === EMPTY, 'Stone removed after undo');
  console.assert(game.current === initialCurrent, 'Turn reverted');
  console.assert(game.moveCount === initialMoveCount, 'Move count reverted');

  console.log('✓ Undo works');
}

function testGroupMerge() {
  console.log('Testing group merge...');
  const game = new Game3Precise(5);

  // Build a position that causes merges
  game.play(6);   // W at 6
  game.play(11);  // B at 11
  game.play(7);   // W at 7
  game.play(2);   // B at 2

  const gid6 = game.groupIdAt(6);
  const gid7 = game.groupIdAt(7);
  console.assert(gid6 >= 0 && gid7 >= 0, 'Both stones in groups');

  // Play a stone that merges the groups
  game.play(1);   // W at 1 (should merge with 6 and 7)
  const gid1 = game.groupIdAt(1);
  console.assert(game.groupIdAt(6) === gid1, 'Groups merged');
  console.assert(game.groupSize(gid1) >= 3, 'Merged group has multiple stones');

  // Undo should split the group back
  game.undo();
  const gid6After = game.groupIdAt(6);
  const gid1After = game.groupIdAt(1);
  console.assert(gid1After === -1, 'Placed stone removed');
  console.assert(gid6After === gid6, 'Original groups restored');

  console.log('✓ Group merge/split works');
}

function testMultipleUndos() {
  console.log('Testing multiple undos...');
  const game = new Game3Precise(5);

  const move1 = 6;
  const move2 = 11;
  const move3 = 7;

  game.play(move1);
  game.play(move2);
  game.play(move3);

  console.assert(game.moveCount === 4, 'All moves played');
  console.assert(game.cells[move3] === WHITE, 'Last move placed');

  game.undo();
  console.assert(game.cells[move3] === EMPTY, 'Last move undone');
  console.assert(game.moveCount === 3, 'Move count decremented');

  game.undo();
  console.assert(game.cells[move2] === EMPTY, 'Second move undone');
  console.assert(game.moveCount === 2, 'Move count decremented again');

  game.undo();
  console.assert(game.cells[move1] === EMPTY, 'First move undone');
  console.assert(game.moveCount === 1, 'Back to initial state');

  console.log('✓ Multiple undos work');
}

function testLegalityCheck() {
  console.log('Testing legality checks...');
  const game = new Game3Precise(5);

  const center = 12;
  console.assert(!game.isLegal(center), 'Occupied center should be illegal');
  console.assert(game.isLegal(6), 'Empty cell should be legal initially');

  game.play(6);
  console.assert(!game.isLegal(6), 'Now occupied cell should be illegal');

  console.log('✓ Legality checks work');
}

function testPlayUndoCycles() {
  console.log('Testing play/undo cycles (10 times)...');
  const game = new Game3Precise(7);

  for (let cycle = 0; cycle < 10; cycle++) {
    const move = 10 + cycle;
    if (move >= game.N * game.N) break;

    const before = game.moveCount;
    game.play(move);
    const after = game.moveCount;
    console.assert(after === before + 1, `Move ${cycle}: move count increased`);
    console.assert(game.cells[move] !== EMPTY, `Move ${cycle}: stone placed`);

    game.undo();
    console.assert(game.moveCount === before, `Move ${cycle}: move count restored`);
    console.assert(game.cells[move] === EMPTY, `Move ${cycle}: stone removed`);
  }

  console.log('✓ Play/undo cycles work');
}

function testToString() {
  console.log('Testing toString() display...');
  const game = new Game3Precise(7);

  // Play a few moves
  game.play(10);
  game.play(11);
  game.play(18);

  // Test basic toString
  const str = game.toString();
  console.assert(typeof str === 'string', 'toString should return a string');
  console.assert(str.length > 0, 'String should not be empty');
  console.assert(str.includes('●'), 'Should contain black stones');
  console.assert(str.includes('○'), 'Should contain white stones');
  console.assert(str.includes('·'), 'Should contain empty cells');
  console.assert(str.includes('\n'), 'Should contain newlines');

  // Test with markIdx
  const marked = game.toString(11);
  console.assert(marked.includes('('), 'Marked stone should have parentheses');
  console.assert(marked.includes(')'), 'Marked stone should have parentheses');

  // Test with centerAt
  const centered = game.toString(10, { centerAt: 10 });
  console.assert(typeof centered === 'string', 'Centered toString should return a string');
  console.assert(centered.length > 0, 'Centered string should not be empty');

  console.log('✓ toString() works correctly');
}

function testClone() {
  console.log('Testing clone()...');
  const game = new Game3Precise(9);

  // Play several moves
  game.play(28);   // W moves
  game.play(37);   // B moves
  game.play(47);   // W moves
  game.play(19);   // B moves

  // Clone the game
  const cloned = game.clone();

  // Verify all properties are copied
  console.assert(cloned.N === game.N, 'Board size should match');
  console.assert(cloned.current === game.current, 'Current player should match');
  console.assert(cloned.moveCount === game.moveCount, 'Move count should match');
  console.assert(cloned.lastMove === game.lastMove, 'Last move should match');
  console.assert(cloned.ko === game.ko, 'Ko position should match');
  console.assert(cloned.emptyCount === game.emptyCount, 'Empty count should match');
  console.assert(cloned.gameOver === game.gameOver, 'Game over flag should match');
  console.assert(cloned.consecutivePasses === game.consecutivePasses, 'Consecutive passes should match');

  // Verify board state is identical
  for (let i = 0; i < game.N * game.N; i++) {
    console.assert(cloned.cells[i] === game.cells[i], `Cell ${i} should match`);
  }

  // Clone should be independent: make a move in clone and verify original is unchanged
  const clonedLastMove = cloned.lastMove;
  cloned.play(20);
  console.assert(game.lastMove === clonedLastMove, 'Original game should not change after clone move');
  console.assert(game.moveCount === 5, 'Original game move count should be unchanged');
  console.assert(cloned.moveCount === 6, 'Cloned game move count should be incremented');

  // Verify undo works independently
  cloned.undo();
  console.assert(cloned.moveCount === 5, 'Cloned game move count should revert');
  console.assert(cloned.lastMove === clonedLastMove, 'Cloned game last move should revert');

  console.log('✓ clone() works correctly');
}

function testConsecutivePasses() {
  console.log('Testing consecutivePasses...');
  const game = new Game3Precise(5);

  console.assert(game.consecutivePasses === 0, 'Initially should be 0');
  console.assert(!game.gameOver, 'Game should not be over');

  // Play one move
  game.play(6);
  console.assert(game.consecutivePasses === 0, 'After regular move should reset to 0');

  // Pass
  game.play(PASS);
  console.assert(game.consecutivePasses === 1, 'After first pass should be 1');
  console.assert(!game.gameOver, 'Game should not be over after 1 pass');

  // Pass again
  game.play(PASS);
  console.assert(game.consecutivePasses === 2, 'After second pass should be 2');
  console.assert(game.gameOver, 'Game should be over after 2 consecutive passes');

  // Undo pass
  game.undo();
  console.assert(game.consecutivePasses === 1, 'After undo should be 1');
  console.assert(!game.gameOver, 'Game should not be over after undo');

  // Play a move instead of pass
  game.play(7);
  console.assert(game.consecutivePasses === 0, 'After regular move should reset to 0');
  console.assert(!game.gameOver, 'Game should not be over');

  // Verify undo restores consecutivePasses for moves
  game.undo();
  console.assert(game.consecutivePasses === 1, 'After undo should restore to 1');

  console.log('✓ consecutivePasses works correctly');
}

function testCalcWinner() {
  console.log('Testing calcWinner()...');

  // Test 1: 5x5 board with only center BLACK stone
  // All empty territory is adjacent to BLACK, so BLACK gets all 24 empty cells
  const game = new Game3Precise(5);
  let result = game.calcWinner();
  console.assert(result.black === 25, `5x5 with center BLACK should have black=25 (all territory), got ${result.black}`);
  console.assert(result.white === 3.5, `5x5 with center BLACK should have white=3.5 (komi only), got ${result.white}`);
  console.assert(result.winner === BLACK, 'BLACK should win board with center stone');
  console.assert(result.komi === 3.5, 'Komi should be 3.5 for size 5');
  console.assert(result.margin > 0, 'BLACK should have positive margin');

  // Test 2: 9x9 board with different komi
  const game2 = new Game3Precise(9);
  result = game2.calcWinner();
  console.assert(result.komi === 6.5, 'Komi should be 6.5 for size 9');

  // Test 3: With some stones creating territory
  const game3 = new Game3Precise(7);
  // Play W at position that creates simple territory
  game3.play(10);  // W at 10
  game3.play(PASS); // B
  game3.play(11);  // W at 11
  game3.play(PASS); // B

  result = game3.calcWinner();
  console.assert(result.black > 0 || result.white > 0, 'Score calculation should work with stones');
  console.assert(result.winner === BLACK || result.winner === WHITE, 'Winner should be BLACK or WHITE');
  console.assert(typeof result.margin === 'number', 'Margin should be a number');

  // Test 4: Return object has all required fields
  const game4 = new Game3Precise(5);
  result = game4.calcWinner();
  console.assert(typeof result.black === 'number', 'Result should have black score');
  console.assert(typeof result.white === 'number', 'Result should have white score');
  console.assert(typeof result.komi === 'number', 'Result should have komi');
  console.assert(typeof result.score === 'number', 'Result should have score difference');
  console.assert(result.winner === BLACK || result.winner === WHITE, 'Result should have winner');
  console.assert(typeof result.margin === 'number', 'Result should have margin');

  console.log('✓ calcWinner() works correctly');
}

function testIsTrueEye() {
  console.log('Testing isTrueEye()...');

  // Test 1: Simple surrounded position (all 4 neighbors friendly, 3+ diagonals friendly)
  const game = new Game3Precise(9);

  // Build: W at (3,3), (3,5), (5,3), (5,5) and at diagonals (2,4), (4,2), (4,6), (6,4)
  // This creates a cross pattern with position (4,4)=position 40 as center eye
  // Position 40 neighbors: up=31, down=49, left=39, right=41
  // Diagonals: ul=30, ur=32, dl=48, dr=50

  game.play(30);  // W at 30 (up-left diagonal)
  game.play(PASS); // B
  game.play(32);  // W at 32 (up-right diagonal)
  game.play(PASS); // B
  game.play(48);  // W at 48 (down-left diagonal)
  game.play(PASS); // B
  game.play(50);  // W at 50 (down-right diagonal)
  game.play(PASS); // B
  game.play(31);  // W at 31 (up)
  game.play(PASS); // B
  game.play(49);  // W at 49 (down)
  game.play(PASS); // B
  game.play(39);  // W at 39 (left)
  game.play(PASS); // B
  game.play(41);  // W at 41 (right)
  game.play(PASS); // B

  // Now position 40 is surrounded by WHITE on all 4 sides and has 4 diagonal neighbors
  // Should be a true eye for WHITE
  const isEye = game.isTrueEye(40);
  console.assert(isEye, 'Position surrounded by 4 friendly stones with 3+ friendly diagonals should be true eye');

  // Test 2: Occupied position is not an eye
  console.assert(!game.isTrueEye(30), 'Occupied position should not be eye');
  console.assert(!game.isTrueEye(31), 'Occupied position should not be eye');

  // Test 3: Position with opponent neighbors is not an eye
  const game2 = new Game3Precise(7);
  game2.play(17); // W at 17
  game2.play(16); // B at 16 (adjacent)
  game2.play(PASS); // W passes
  game2.play(PASS); // B passes (current = WHITE)

  // Position 18 has B to left, empty to right/above/below
  // Should not be true eye for WHITE (not enough friendly neighbors)
  const notEye = !game2.isTrueEye(18);
  console.assert(notEye, 'Position with opponent neighbor and few friendly neighbors should not be eye');

  console.log('✓ isTrueEye() works correctly');
}

// Run all tests
console.log('Game3-Precise Unit Tests');
console.log('='.repeat(60));

try {
  testBasicPlay();
  testUndo();
  testGroupMerge();
  testMultipleUndos();
  testLegalityCheck();
  testPlayUndoCycles();
  testToString();
  testClone();
  testConsecutivePasses();
  testIsTrueEye();
  testCalcWinner();

  console.log('='.repeat(60));
  console.log('All tests passed! ✓');
} catch (e) {
  console.error('Test failed with error:');
  console.error(e);
  process.exit(1);
}
