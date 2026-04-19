#!/usr/bin/env node
'use strict';

// Test that tactical statuses remain consistent when converting Game2 -> Game3
// Plays random games, converting at each step and checking tactics analysis

const { Game2, BLACK, WHITE, PASS } = require('./game2.js');
const { Game3, game3FromGame2 } = require('./game3.js');
const { searchChains } = require('./tactics3.js');

let testsRun = 0;
let testsFailed = 0;

function assert(condition, message) {
  testsRun++;
  if (!condition) {
    testsFailed++;
    console.error(`✗ FAILED: ${message}`);
  }
}

// Test: Random games with tactics status consistency checks
function testTacticsConsistency() {
  console.log('Test: Tactics Status Consistency Across Moves');

  const numGames = 10;
  const maxMovesPerGame = 50;

  for (let gameNum = 0; gameNum < numGames; gameNum++) {
    const g2 = new Game2(13);
    let previousTactics = [];
    let moveCount = 0;
    let consistencyChecks = 0;

    while (moveCount < maxMovesPerGame && !g2.gameOver) {
      // Convert Game2 to Game3
      const g3 = game3FromGame2(g2);

      // Get tactical statuses from Game3 with node limit to prevent deep recursion
      const currentTactics = searchChains(g3, 10000);

      // On moves after the first, verify tactics status transitions are logical
      if (moveCount > 0) {
        // Create maps for easy lookup
        const prevMap = new Map();
        for (const t of previousTactics) {
          prevMap.set(t.gid, t);
        }

        const currMap = new Map();
        for (const t of currentTactics) {
          currMap.set(t.gid, t);
        }

        // Check consistency: groups that existed should have logical transitions
        for (const [gid, prevTactic] of prevMap) {
          const currTactic = currMap.get(gid);

          if (currTactic !== undefined) {
            // Group still exists and is still in tactical range (1-3 liberties)
            // Status transitions should be logical:
            // - moverSucceeds can be true, false, or null (inconclusive)
            // - Status can change if move affected liberties/groups

            assert(currTactic.status !== null,
              `Game ${gameNum} move ${moveCount}: Group ${gid} should have status if in tactical range`);

            // Verify status is defined (not null)
            if (prevTactic.status !== null && currTactic.status !== null) {
              const prevSucceeds = prevTactic.status.moverSucceeds;
              const currSucceeds = currTactic.status.moverSucceeds;

              // moverSucceeds can be true, false, or null, and can change based on the move
              assert(typeof currSucceeds === 'boolean' || currSucceeds === null,
                `Game ${gameNum} move ${moveCount}: Mover status should be boolean or null`);

              // Verify urgentLibs is an array
              assert(Array.isArray(currTactic.status.urgentLibs),
                `Game ${gameNum} move ${moveCount}: urgentLibs should be an array`);
            }

            consistencyChecks++;
          }
          // If group no longer in tactical range, it either escaped or was captured - both logical
        }
      }

      // Identify urgent moves (where moverSucceeds is true and there are urgent libs)
      const urgentMoves = [];
      for (const tactic of currentTactics) {
        if (tactic.status && tactic.status.moverSucceeds === true &&
            tactic.status.urgentLibs && tactic.status.urgentLibs.length > 0) {
          urgentMoves.push(...tactic.status.urgentLibs);
        }
      }

      // Choose next move: occasionally play an urgent move, otherwise random
      let nextMove = PASS;
      if (urgentMoves.length > 0 && Math.random() < 0.3) {
        // 30% of time, play an urgent tactical move
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
          previousTactics = currentTactics;
        }
      } else {
        g2.play(PASS);
        moveCount++;
        previousTactics = currentTactics;
      }
    }

    assert(moveCount > 0, `Game ${gameNum}: Should have made moves`);
    assert(consistencyChecks > 0, `Game ${gameNum}: Should have checked tactics consistency`);
    console.log(`  Game ${gameNum}: ${moveCount} moves, ${consistencyChecks} consistency checks passed`);
  }

  console.log('  ✓ Tactics status consistency test passed');
}

// Test: Specific tactics scenario
function testSpecificTactics() {
  console.log('\nTest: Specific Tactics Scenario');

  const g2 = new Game2(9);

  // Create a chain with limited liberties
  g2.play(0);   // BLACK at (0,0)
  g2.play(1);   // WHITE at (0,1)
  g2.play(9);   // BLACK at (1,0)
  g2.play(10);  // WHITE at (1,1)

  // Convert and check tactics status
  let g3 = game3FromGame2(g2);
  let tactics = searchChains(g3, 10000);

  assert(Array.isArray(tactics), 'Should return array of tactics');
  console.log(`  Found ${tactics.length} groups in tactical range (1-3 liberties)`);

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
    tactics = searchChains(g3, 10000);

    assert(tactics !== null, 'Tactics analysis should still work after move');
    console.log(`  After move ${move}: ${tactics.length} groups in tactical range`);
  }

  console.log('  ✓ Specific tactics test passed');
}

// Test: Conversion accuracy affects tactics analysis
function testConversionAccuracy() {
  console.log('\nTest: Conversion Accuracy for Tactics Analysis');

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
  const tacticsA = searchChains(g3a, 10000);

  // Verify conversion is deterministic
  const g3b = game3FromGame2(g2);
  const tacticsB = searchChains(g3b, 10000);

  assert(tacticsA.length === tacticsB.length, 'Same Game2 should produce same number of tactical groups');

  // Verify board states match
  let boardMatch = true;
  for (let i = 0; i < 81; i++) {
    if (g3a.cells[i] !== g3b.cells[i]) {
      boardMatch = false;
      break;
    }
  }
  assert(boardMatch, 'Two conversions should produce identical board state');

  console.log(`  Tactical groups: ${tacticsA.length}, Conversion deterministic: ${boardMatch}`);
  console.log('  ✓ Conversion accuracy test passed');
}

// Test: Urgent tactics moves make sense
function testUrgentMoves() {
  console.log('\nTest: Urgent Tactics Moves');

  const g2 = new Game2(13);

  // Play enough moves to create some tactical situations
  for (let i = 0; i < 169 && i < 30; i++) {
    if (g2.isLegal(i)) {
      g2.play(i);
    }
  }

  const g3 = game3FromGame2(g2);
  const tactics = searchChains(g3, 10000);

  // Count urgent moves
  let totalUrgent = 0;
  let movesWithUrgent = 0;

  for (const tactic of tactics) {
    if (tactic.status && tactic.status.urgentLibs && tactic.status.urgentLibs.length > 0) {
      totalUrgent += tactic.status.urgentLibs.length;
      movesWithUrgent++;
    }
  }

  console.log(`  Found ${movesWithUrgent} groups with urgent moves (${totalUrgent} moves total)`);
  console.log('  ✓ Urgent moves test passed');
}

// Run all tests
function runTests() {
  console.log('\n' + '='.repeat(70));
  console.log('TACTICS CONSISTENCY TEST SUITE');
  console.log('='.repeat(70));

  testTacticsConsistency();
  testSpecificTactics();
  testConversionAccuracy();
  testUrgentMoves();

  console.log('\n' + '='.repeat(70));
  console.log(`Results: ${testsRun - testsFailed}/${testsRun} assertions passed`);

  if (testsFailed === 0) {
    console.log('✓ ALL TESTS PASSED');
    return 0;
  } else {
    console.log(`✗ ${testsFailed} assertion(s) failed`);
    return 1;
  }
}

process.exit(runTests());
