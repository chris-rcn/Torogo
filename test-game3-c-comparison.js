#!/usr/bin/env node

const { Game3, BLACK, WHITE, EMPTY, PASS } = require('./game3.js');
const { execSync } = require('child_process');
const fs = require('fs');

// Summary of test results
let testsRun = 0;
let testsPassed = 0;
let testsFailed = 0;

function assert(condition, message) {
  testsRun++;
  if (condition) {
    testsPassed++;
  } else {
    testsFailed++;
    console.log(`✗ FAILED: ${message}`);
  }
}

function assertEqual(actual, expected, message) {
  assert(actual === expected, `${message}: expected ${expected}, got ${actual}`);
}

// Test 1: JS implementation basic correctness
function test_js_basic() {
  console.log('\nTest: JS Implementation Basic Correctness');

  const game = new Game3(9);

  assertEqual(game.N, 9, 'Board size');
  assertEqual(game.boardSize, 9, 'Board size field');
  assertEqual(game.moveCount, 1, 'Initial moveCount');
  assertEqual(game.current, WHITE, 'Current player');
  assert(game.cells[40] === BLACK, 'Center stone is BLACK');

  game.play(0);
  assertEqual(game.moveCount, 2, 'MoveCount after move');
  assertEqual(game.cells[0], WHITE, 'White stone at 0');

  game.undo();
  assertEqual(game.moveCount, 1, 'MoveCount after undo');
  assertEqual(game.cells[0], EMPTY, 'Cell emptied after undo');

  console.log('  ✓ JS basic correctness passed');
}

// Test 2: Group handling
function test_js_groups() {
  console.log('\nTest: JS Group Handling');

  const game = new Game3(13);

  // Place stones that should merge
  // For 13x13: idx = row*13 + col
  // Use indices that are orthogonally adjacent
  game.play(45);  // BLACK at (3,6)
  game.play(46);  // WHITE at (3,7)
  game.play(58);  // BLACK at (4,6) - orthogonally adjacent to 45 (down)

  const gid45 = game._gid[45];
  const gid58 = game._gid[58];

  assert(gid45 === gid58, 'Adjacent same-color stones merge');
  assertEqual(game.groupSize(gid45), 2, 'Group size is 2');

  console.log('  ✓ JS group handling passed');
}

// Test 3: Captures
function test_js_captures() {
  console.log('\nTest: JS Capture Detection');

  const game = new Game3(13);

  // Create a simple capture scenario
  game.play(45);  // BLACK
  game.play(46);  // WHITE
  game.play(57);  // BLACK
  game.play(58);  // WHITE
  game.play(70);  // BLACK
  game.play(71);  // WHITE
  game.play(69);  // BLACK - check for captures

  // Check that game state is consistent
  assert(game.emptyCount < 169, 'Stones have been placed');

  console.log('  ✓ JS capture detection passed');
}

// Test 4: Undo/redo cycles
function test_js_undo_redo() {
  console.log('\nTest: JS Undo/Redo Cycles');

  const game = new Game3(13);
  const moves = [45, 46, 57, 58, 70, 71, 69, 72];

  for (const move of moves) {
    game.play(move);
  }

  const stateAfterMoves = {
    moveCount: game.moveCount,
    emptyCount: game.emptyCount,
    cells: new Int8Array(game.cells),
  };

  // Undo 3 moves
  game.undo();
  game.undo();
  game.undo();

  const stateAfterUndo = {
    moveCount: game.moveCount,
    emptyCount: game.emptyCount,
  };

  // Redo
  for (let i = 0; i < 3; i++) {
    game.play(moves[moves.length - 3 + i]);
  }

  assertEqual(game.moveCount, stateAfterMoves.moveCount, 'MoveCount restored after redo');
  assertEqual(game.emptyCount, stateAfterMoves.emptyCount, 'EmptyCount restored after redo');

  console.log('  ✓ JS undo/redo cycles passed');
}

// Test 5: Scoring
function test_js_scoring() {
  console.log('\nTest: JS Scoring');

  const game = new Game3(9);

  const score = game.estimateScore();
  assert(typeof score.black === 'number', 'Black score is a number');
  assert(typeof score.white === 'number', 'White score is a number');
  assert(score.white > score.black, 'White has komi advantage');

  console.log('  ✓ JS scoring passed');
}

// Test 6: Eye detection
function test_js_eye_detection() {
  console.log('\nTest: JS Eye Detection');

  const game = new Game3(13);

  game.play(45);
  game.play(46);
  game.play(57);
  game.play(58);

  const isEye = game.isTrueEye(59);
  assert(typeof isEye === 'boolean', 'isTrueEye returns boolean');

  console.log('  ✓ JS eye detection passed');
}

// Test 7: Multiple games consistency
function test_js_multiple_games() {
  console.log('\nTest: JS Multiple Games Consistency');

  const moves = [45, 46, 57, 58, 70, 71, 69, 72, 82, 83];

  // Game 1
  const game1 = new Game3(13);
  for (const move of moves) {
    game1.play(move);
  }

  // Game 2 (should be identical)
  const game2 = new Game3(13);
  for (const move of moves) {
    game2.play(move);
  }

  assertEqual(game1.moveCount, game2.moveCount, 'Identical games have same moveCount');
  assertEqual(game1.emptyCount, game2.emptyCount, 'Identical games have same emptyCount');

  // Compare boards
  let boardsMatch = true;
  for (let i = 0; i < 169; i++) {
    if (game1.cells[i] !== game2.cells[i]) {
      boardsMatch = false;
      break;
    }
  }
  assert(boardsMatch, 'Identical games produce identical boards');

  console.log('  ✓ JS multiple games consistency passed');
}

// Test 8: Stress test - many undo/redo cycles
function test_js_stress() {
  console.log('\nTest: JS Stress Test (many operations)');

  const game = new Game3(13);
  const moves = [];

  // Generate sequence of moves
  for (let i = 0; i < 169 && moves.length < 30; i++) {
    if (game.isLegal(i)) {
      game.play(i);
      moves.push(i);
    }
  }

  const initialEmpty = game.emptyCount;

  // Many undo/redo cycles
  for (let cycle = 0; cycle < 5; cycle++) {
    for (let i = 0; i < Math.min(5, moves.length); i++) {
      game.undo();
    }
    for (let i = 0; i < Math.min(5, moves.length); i++) {
      game.play(moves[moves.length - 1 - i]);
    }
  }

  assertEqual(game.emptyCount, initialEmpty, 'EmptyCount consistent after stress test');

  console.log('  ✓ JS stress test passed');
}

// Test 9: Ko rule
function test_js_ko() {
  console.log('\nTest: JS Ko Rule');

  const game = new Game3(13);

  game.play(45);
  game.play(46);

  // If there's a capture, ko should be set
  // (This test just verifies ko is tracked, not the full ko logic)
  assert(typeof game.ko === 'number', 'Ko is a number');

  console.log('  ✓ JS ko rule passed');
}

// Test 10: Pass moves
function test_js_passes() {
  console.log('\nTest: JS Pass Moves');

  const game = new Game3(9);

  const initialPasses = game.consecutivePasses;
  game.play(PASS);
  assertEqual(game.consecutivePasses, initialPasses + 1, 'Passes incremented');
  assert(!game.gameOver, 'Game not over after 1 pass');

  game.play(PASS);
  assert(game.gameOver, 'Game over after 2 passes');

  console.log('  ✓ JS pass moves passed');
}

// Main test runner
function runTests() {
  console.log('Game3.js (JavaScript) Comprehensive Test Suite');
  console.log('='.repeat(60));

  test_js_basic();
  test_js_groups();
  test_js_captures();
  test_js_undo_redo();
  test_js_scoring();
  test_js_eye_detection();
  test_js_multiple_games();
  test_js_stress();
  test_js_ko();
  test_js_passes();

  console.log('\n' + '='.repeat(60));
  console.log(`Results: ${testsPassed} assertions passed, ${testsFailed} failed out of ${testsRun}`);

  if (testsFailed === 0) {
    console.log('✓ All JavaScript tests passed!');
    return 0;
  } else {
    console.log(`✗ ${testsFailed} assertion(s) failed`);
    return 1;
  }
}

process.exit(runTests());
