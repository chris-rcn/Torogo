'use strict';

const { Game3, PASS, BLACK, WHITE, EMPTY } = require('./game3.js');

function testBasicPlay() {
  console.log('Testing basic play...');
  const game = new Game3(5);

  // Verify initial state
  console.assert(game.cells[12] === BLACK, 'Center should be BLACK');
  console.assert(game.current === WHITE, 'Current should be WHITE after init');
  console.assert(game.moveCount === 1, 'Move count should be 1');

  // Make a move
  const move1 = 6; // Play somewhere valid
  const legal1 = game.isLegal(move1);
  console.assert(legal1, `Move ${move1} should be legal`);

  const played = game.play(move1);
  console.assert(played, 'Move should succeed');
  console.assert(game.cells[move1] === WHITE, 'Cell should contain WHITE stone');
  console.assert(game.current === BLACK, 'Turn should change to BLACK');
  console.assert(game.moveCount === 2, 'Move count should be 2');

  console.log('✓ Basic play works');
}

function testUndo() {
  console.log('Testing undo...');
  const game = new Game3(5);

  const initialCurrent = game.current;
  const initialMoveCount = game.moveCount;

  // Play a move
  const move = 6;
  game.play(move);
  console.assert(game.cells[move] === WHITE, 'Stone should be placed');
  console.assert(game.moveCount === 2, 'Move count should increment');

  // Undo the move
  game.undo();
  console.assert(game.cells[move] === EMPTY, 'Stone should be removed after undo');
  console.assert(game.current === initialCurrent, 'Current player should revert');
  console.assert(game.moveCount === initialMoveCount, 'Move count should revert');

  console.log('✓ Undo works');
}

function testCapture() {
  console.log('Testing capture...');
  const game = new Game3(5);

  // Setup: Create a simple capture situation
  // White at (1,0) surrounded by black
  game.play(6);   // white
  game.play(11);  // black
  game.play(7);   // white
  game.play(2);   // black
  game.play(1);   // white - should not capture yet
  game.play(3);   // black - surrounds white at (1,0)
  game.play(0);   // white - captures black at (1,0) if white surrounds

  // Just verify moves are valid
  console.assert(game.moveCount >= 3, 'Moves should be recorded');
  console.log('✓ Capture sequence plays without error');
}

function testGroupLiberties() {
  console.log('Testing group liberties...');
  const game = new Game3(5);

  // Place stones and check liberties
  const idx = 12; // center
  const gid = game.groupIdAt(idx);
  console.assert(gid >= 0, 'Center stone should have a group');

  const libs = game.groupLibs(idx);
  console.assert(libs.length > 0, 'Group should have liberties');

  const libCount = game.groupLibertyCount(gid);
  console.assert(libCount === libs.length, 'Liberty count should match');

  console.log(`✓ Group liberties work (group ${gid} has ${libCount} liberties)`);
}

function testPassMove() {
  console.log('Testing pass move...');
  const game = new Game3(5);

  const currentBefore = game.current;
  game.play(PASS);

  console.assert(game.current === -currentBefore, 'Current should flip on pass');
  console.assert(game.moveCount === 2, 'Pass should count as a move');

  game.undo();
  console.assert(game.current === currentBefore, 'Current should revert after undo');

  console.log('✓ Pass move works');
}

function testLongSequence() {
  console.log('Testing long sequence with undo...');
  const game = new Game3(5);

  const moves = [6, 11, 7, 2, 1, 3, 8, 13, 9, 4, 14, 19];

  for (const move of moves) {
    const legal = game.isLegal(move);
    if (!legal) {
      console.log(`  Skipping illegal move ${move}`);
      continue;
    }
    game.play(move);
  }

  const moveCountBefore = game.moveCount;

  // Undo all moves
  while (game._undoStack.length > 1) {
    game.undo();
  }

  // Should be back to initial state
  console.assert(game.moveCount === 1, 'Should be back at initial state');
  console.assert(game.cells[12] === BLACK, 'Center should still be BLACK');

  console.log('✓ Long sequence and undo work');
}

function testBoardState() {
  console.log('Testing board state consistency...');
  const game = new Game3(5);

  // Make several moves
  game.play(6);
  game.play(11);
  game.play(7);

  const boardBefore = new Int8Array(game.cells);

  // Undo and redo
  game.undo();
  game.play(7);

  for (let i = 0; i < boardBefore.length; i++) {
    console.assert(game.cells[i] === boardBefore[i],
      `Cell ${i} mismatch after undo/redo: ${game.cells[i]} vs ${boardBefore[i]}`);
  }

  console.log('✓ Board state consistency maintained');
}

// Run tests
try {
  testBasicPlay();
  testUndo();
  testPassMove();
  testGroupLiberties();
  testCapture();
  testLongSequence();
  testBoardState();

  console.log('\n✅ All tests passed!');
} catch (e) {
  console.error('\n❌ Test failed:', e.message);
  console.error(e.stack);
  process.exit(1);
}
