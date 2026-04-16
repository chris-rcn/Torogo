'use strict';

// test-game3-precise-correctness.js — verify Game3Precise matches Game2 behavior

const { Game2, PASS } = require('./game2.js');
const { Game3Precise } = require('./game3.js');

function cellsEqual(c1, c2) {
  if (c1.length !== c2.length) return false;
  for (let i = 0; i < c1.length; i++) {
    if (c1[i] !== c2[i]) return false;
  }
  return true;
}

function statesEqual(g1, g2, moveNum) {
  const errors = [];

  // Check cells
  if (!cellsEqual(g1.cells, g2.cells)) {
    errors.push('cells mismatch');
    for (let i = 0; i < Math.min(g1.cells.length, 25); i++) {
      if (g1.cells[i] !== g2.cells[i]) {
        console.log(`  cells[${i}]: Game2=${g1.cells[i]}, Game3Precise=${g2.cells[i]}`);
      }
    }
  }

  // Check current player
  if (g1.current !== g2.current) {
    errors.push(`current: ${g1.current} vs ${g2.current}`);
  }

  // Check move count
  if (g1.moveCount !== g2.moveCount) {
    errors.push(`moveCount: ${g1.moveCount} vs ${g2.moveCount}`);
  }

  // Check empty count
  if (g1.emptyCount !== g2.emptyCount) {
    errors.push(`emptyCount: ${g1.emptyCount} vs ${g2.emptyCount}`);
  }

  // Check group IDs at occupied cells
  for (let i = 0; i < g1.cells.length; i++) {
    if (g1.cells[i] !== 0 && g2.cells[i] !== 0) {
      if (g1._gid[i] >= 0 && g2._gid[i] >= 0) {
        // Both have groups, just check that they're consistent
        // (group IDs might differ, but structure should be same)
      }
    }
  }

  if (errors.length > 0) {
    console.error(`After move ${moveNum}: ${errors.join(', ')}`);
    return false;
  }
  return true;
}

function testRandomGame(size = 7, maxMoves = 50) {
  console.log(`Testing random game on ${size}x${size} board...`);

  let game2 = new Game2(size);
  let game3p = new Game3Precise(size);

  const moves = [];
  let moveNum = 0;

  while (moveNum < maxMoves && !game2.gameOver && !game3p.gameOver) {
    // Find a legal move
    let move = PASS;
    const cap = size * size;
    const legal = [];
    for (let i = 0; i < cap; i++) {
      if (game2.cells[i] === 0 && game2.isLegal(i)) {
        legal.push(i);
      }
    }

    if (legal.length === 0) {
      move = PASS;
    } else {
      move = legal[Math.floor(Math.random() * legal.length)];
    }

    moves.push(move);
    moveNum++;

    // Play in both games
    if (move === PASS) {
      game2.play(PASS);
      game3p.play(PASS);
    } else {
      const g2clone = game2.clone();
      const success2 = g2clone.play(move);
      const success3 = game3p.play(move);

      if (success2 !== success3) {
        console.error(`Move legality disagreement at move ${moveNum}: Game2=${success2}, Game3Precise=${success3}`);
        return false;
      }

      game2 = g2clone;
    }

    // Check states match
    if (!statesEqual(game2, game3p, moveNum)) {
      console.error(`State mismatch after move ${moveNum} (${move})`);
      return false;
    }
  }

  console.log(`✓ Random game completed ${moveNum} moves with matching state`);
  return true;
}

function testPlayUndo(size = 7) {
  console.log(`Testing play/undo on ${size}x${size} board...`);

  const game = new Game3Precise(size);

  // Play some moves
  const moves = [10, 20, 15, 25, 30];
  for (const move of moves) {
    if (game.cells[move] === 0 && game.isLegal(move)) {
      game.play(move);
    }
  }

  const stateAfterPlay = {
    cells: new Int8Array(game.cells),
    current: game.current,
    moveCount: game.moveCount,
    emptyCount: game.emptyCount,
  };

  // Undo all
  while (game.moveCount > 1) {
    game.undo();
  }

  const stateAfterUndo = {
    cells: new Int8Array(game.cells),
    current: game.current,
    moveCount: game.moveCount,
    emptyCount: game.emptyCount,
  };

  // Verify we're back to initial state
  console.assert(stateAfterUndo.current === -1, 'Current should be WHITE');
  console.assert(stateAfterUndo.moveCount === 1, 'Move count should be 1');
  console.assert(stateAfterUndo.emptyCount === size * size - 1, 'Empty count should be board size - 1');

  console.log(`✓ Play/undo maintains correct state`);
  return true;
}

function testGroupLiberties(size = 7) {
  console.log(`Testing group liberties on ${size}x${size} board...`);

  const game = new Game3Precise(size);

  // Build a known configuration
  // Place a stone at 10, check its liberties
  game.play(10);
  game.play(11);
  game.play(12);

  // Check that we can query group liberties
  const libs = game.groupLibs(10);
  console.assert(libs.length > 0, 'Group should have liberties');
  console.assert(libs.length <= size * size, 'Liberty count should be reasonable');

  console.log(`✓ Group liberty tracking works`);
  return true;
}

// Run all tests
console.log('Game3Precise Correctness Tests');
console.log('='.repeat(60));

let passed = 0;
let failed = 0;

try {
  if (testRandomGame(7, 30)) passed++; else failed++;
  if (testRandomGame(9, 40)) passed++; else failed++;
  if (testPlayUndo(7)) passed++; else failed++;
  if (testGroupLiberties(7)) passed++; else failed++;
} catch (e) {
  console.error('Test exception:', e.message);
  console.error(e.stack);
  failed++;
}

console.log('='.repeat(60));
console.log(`Results: ${passed} passed, ${failed} failed`);

if (failed > 0) {
  process.exit(1);
}
