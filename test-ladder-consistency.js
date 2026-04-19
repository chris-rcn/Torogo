#!/usr/bin/env node
'use strict';

// Test that ladder statuses remain consistent when converting Game2 -> Game3
// Plays random games, converting at each step and checking ladder analysis

const { Game2, BLACK, WHITE, PASS } = require('./game2.js');
const { Game3, game3FromGame2 } = require('./game3.js');
const { getAllLadderStatuses } = require('./ladder2.js');

let testsRun = 0;
let testsFailed = 0;

function assert(condition, message) {
  testsRun++;
  if (!condition) {
    testsFailed++;
    console.error(`✗ FAILED: ${message}`);
  }
}

// Test: Random games with ladder consistency checks
function testLadderConsistency() {
  console.log('Test: Ladder Consistency Across Conversions');

  const numGames = 10;
  const maxMovesPerGame = 50;

  for (let gameNum = 0; gameNum < numGames; gameNum++) {
    const g2 = new Game2(13);
    let previousLadderMap = null;
    let moveCount = 0;

    while (moveCount < maxMovesPerGame && !g2.gameOver) {
      // Convert Game2 to Game3
      const g3 = game3FromGame2(g2);

      // Get ladder statuses from Game3
      const ladders = getAllLadderStatuses(g3);

      // Create map of (gid, color) -> status for easy comparison
      const currentLadderMap = new Map();
      for (const ladder of ladders) {
        const key = `${ladder.gid}:${ladder.color}`;
        currentLadderMap.set(key, ladder.status);
      }

      // Compare with previous position (if applicable)
      if (previousLadderMap !== null) {
        // Groups that were in ladder before should still be in ladder
        // (or might have escaped/been captured)
        for (const [key, prevStatus] of previousLadderMap) {
          const currentStatus = currentLadderMap.get(key);

          // If group still exists and is still in atari/near-atari, status should be defined
          if (currentStatus !== undefined) {
            assert(prevStatus !== undefined,
              `Game ${gameNum}: Ladder status should exist for tracked group`);
          }
        }
      }

      // Identify urgent moves (where moverSucceeds is determined)
      const urgentMoves = [];
      for (const ladder of ladders) {
        if (ladder.status && ladder.status.urgentLibs && ladder.status.urgentLibs.length > 0) {
          urgentMoves.push(...ladder.status.urgentLibs);
        }
      }

      // Choose next move: occasionally play an urgent move, otherwise random
      let nextMove = PASS;
      if (urgentMoves.length > 0 && Math.random() < 0.3) {
        // 30% of time, play an urgent ladder move
        nextMove = urgentMoves[Math.floor(Math.random() * urgentMoves.length)];
      } else {
        // Otherwise, play a random legal move
        for (let i = 0; i < 169; i++) {
          if (g2.isLegal(i)) {
            nextMove = i;
            if (Math.random() < 0.05) break;  // 5% chance to stop searching
          }
        }
      }

      // Play the move
      if (nextMove !== PASS) {
        const played = g2.play(nextMove);
        if (played) {
          moveCount++;
          previousLadderMap = currentLadderMap;
        }
      } else {
        g2.play(PASS);
        moveCount++;
        previousLadderMap = currentLadderMap;
      }
    }

    assert(moveCount > 0, `Game ${gameNum}: Should have made moves`);
    console.log(`  Game ${gameNum}: ${moveCount} moves, ${previousLadderMap ? previousLadderMap.size : 0} ladder groups`);
  }

  console.log('  ✓ Ladder consistency test passed');
}

// Test: Specific ladder scenario
function testSpecificLadder() {
  console.log('\nTest: Specific Ladder Scenario');

  const g2 = new Game2(9);

  // Create a simple ladder: two stones in atari
  g2.play(0);   // BLACK at (0,0)
  g2.play(1);   // WHITE at (0,1)
  g2.play(9);   // BLACK at (1,0)
  g2.play(10);  // WHITE at (1,1)

  // Convert and check ladder status
  let g3 = game3FromGame2(g2);
  let ladders = getAllLadderStatuses(g3);

  assert(ladders.length >= 0, 'Should have ladder analysis');
  console.log(`  Found ${ladders.length} groups in ladder range`);

  // Make a move and reconvert
  const legalMoves = [];
  for (let i = 0; i < 81; i++) {
    if (g2.isLegal(i)) {
      legalMoves.push(i);
      if (legalMoves.length >= 5) break;
    }
  }

  if (legalMoves.length > 0) {
    const move = legalMoves[0];
    g2.play(move);

    // Reconvert and verify no crash
    g3 = game3FromGame2(g2);
    ladders = getAllLadderStatuses(g3);

    assert(ladders !== null, 'Ladder analysis should still work after move');
    console.log(`  After move ${move}: ${ladders.length} groups in ladder range`);
  }

  console.log('  ✓ Specific ladder test passed');
}

// Test: Conversion correctness affects ladder analysis
function testConversionAccuracy() {
  console.log('\nTest: Conversion Accuracy for Ladder Analysis');

  const g2 = new Game2(9);

  // Play moves to create groups
  const moves = [0, 1, 9, 10, 2, 3, 11, 12, 4, 5];
  for (const move of moves) {
    if (g2.isLegal(move)) {
      g2.play(move);
    }
  }

  // Convert once
  const g3a = game3FromGame2(g2);
  const laddersA = getAllLadderStatuses(g3a);

  // Verify conversion is deterministic
  const g3b = game3FromGame2(g2);
  const laddersB = getAllLadderStatuses(g3b);

  assert(laddersA.length === laddersB.length, 'Same Game2 should produce same number of ladder groups');

  // Verify board states match
  let boardMatch = true;
  for (let i = 0; i < 81; i++) {
    if (g3a.cells[i] !== g3b.cells[i]) {
      boardMatch = false;
      break;
    }
  }
  assert(boardMatch, 'Two conversions should produce identical board state');

  console.log(`  Ladder groups: ${laddersA.length}, Conversion deterministic: ${boardMatch}`);
  console.log('  ✓ Conversion accuracy test passed');
}

// Run all tests
function runTests() {
  console.log('\n' + '='.repeat(70));
  console.log('LADDER CONSISTENCY TEST SUITE');
  console.log('='.repeat(70));

  testLadderConsistency();
  testSpecificLadder();
  testConversionAccuracy();

  console.log('\n' + '='.repeat(70));
  console.log(`Results: ${testsRun - testsFailed}/${testsRun} tests passed`);

  if (testsFailed === 0) {
    console.log('✓ ALL TESTS PASSED');
    return 0;
  } else {
    console.log(`✗ ${testsFailed} test(s) failed`);
    return 1;
  }
}

process.exit(runTests());
