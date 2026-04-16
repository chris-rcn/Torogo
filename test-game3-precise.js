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

  console.log('='.repeat(60));
  console.log('All tests passed! ✓');
} catch (e) {
  console.error('Test failed with error:');
  console.error(e);
  process.exit(1);
}
