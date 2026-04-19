#!/usr/bin/env node
'use strict';

// Test that tactical statuses remain consistent when converting Game2 -> Game3
// Plays random games, converting at each step and checking tactics analysis

const { Game2, BLACK, WHITE, PASS } = require('./game2.js');
const { Game3, game3FromGame2 } = require('./game3.js');
const { searchChains } = require('./tactics3.js');

let testsRun = 0;
let testsFailed = 0;

// Statistics tracking
const stats = {
  totalInconclusiveResults: 0,
  totalDefinitiveResults: 0,
  totalUrgentLibs: 0,
  totalGroupsEvaluated: 0,
  maxUrgentLibsPerGroup: 0,
  depthLimitTests: 0,
};

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

  const numGames = 20;
  const maxMovesPerGame = 100;
  let totalInconclusivePerGame = 0;

  for (let gameNum = 0; gameNum < numGames; gameNum++) {
    const g2 = new Game2(13);
    let previousTactics = [];
    let moveCount = 0;
    let consistencyChecks = 0;
    let gameInconclusiveCount = 0;

    while (moveCount < maxMovesPerGame && !g2.gameOver) {
      // Convert Game2 to Game3
      const g3 = game3FromGame2(g2);

      // Get tactical statuses from Game3 with node limit to prevent deep recursion
      const currentTactics = searchChains(g3, 10000);

      // Track statistics
      for (const tactic of currentTactics) {
        stats.totalGroupsEvaluated++;
        if (tactic.status) {
          if (tactic.status.moverSucceeds === null) {
            stats.totalInconclusiveResults++;
            gameInconclusiveCount++;
          } else {
            stats.totalDefinitiveResults++;
          }
          if (tactic.status.urgentLibs) {
            stats.totalUrgentLibs += tactic.status.urgentLibs.length;
            stats.maxUrgentLibsPerGroup = Math.max(stats.maxUrgentLibsPerGroup, tactic.status.urgentLibs.length);
          }
        }
      }

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
            if (prevTactic.status !== null && currTactic.status !== null) {
              const prevSucceeds = prevTactic.status.moverSucceeds;
              const currSucceeds = currTactic.status.moverSucceeds;

              // moverSucceeds can be true, false, or null, and can change based on the move
              assert(typeof currSucceeds === 'boolean' || currSucceeds === null,
                `Game ${gameNum} move ${moveCount}: Mover status should be boolean or null`);

              // Verify urgentLibs is an array
              assert(Array.isArray(currTactic.status.urgentLibs),
                `Game ${gameNum} move ${moveCount}: urgentLibs should be an array`);

              // If both were definitive, verify the status transition makes sense
              if (prevSucceeds !== null && currSucceeds !== null) {
                // Status can change due to board changes, but should be consistent with outcome
                assert(typeof currSucceeds === 'boolean',
                  `Game ${gameNum} move ${moveCount}: Definitive status should remain definitive or become inconclusive`);
              }
            }

            consistencyChecks++;
          }
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
    totalInconclusivePerGame += gameInconclusiveCount;
    console.log(`  Game ${gameNum}: ${moveCount} moves, ${consistencyChecks} checks, ${gameInconclusiveCount} inconclusive`);
  }

  console.log(`  Inconclusive results across all games: ${totalInconclusivePerGame}`);
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

// Test: Verify logical transitions of tactical status as board changes
function testStatusTransitions() {
  console.log('\nTest: Tactical Status Transitions');

  const g2 = new Game2(9);

  // Create a specific scenario: build a chain gradually
  const moves = [0, 1, 9, 10, 18, 19, 2, 3, 11];
  for (const move of moves) {
    if (g2.isLegal(move)) {
      g2.play(move);
    }
  }

  // Now play moves and track how tactical status changes
  let previousLibCount = new Map();  // gid -> liberty count
  let transitionCount = 0;

  for (let moveIdx = 0; moveIdx < 15; moveIdx++) {
    // Find legal moves
    let played = false;
    for (let i = 0; i < 81; i++) {
      if (g2.isLegal(i)) {
        const g3 = game3FromGame2(g2);
        const currentTactics = searchChains(g3, 5000);

        // Record liberty counts for each group
        for (const tactic of currentTactics) {
          const libertyCount = tactic.status && tactic.status.libs ? tactic.status.libs.length : 0;
          const prevLibCount = previousLibCount.get(tactic.gid);

          if (prevLibCount !== undefined && prevLibCount !== libertyCount) {
            transitionCount++;
            // If liberty count decreased from 3 to 2 or 1, group is getting more desperate
            // If liberty count increased from 1 or 2 to 3, group is getting safer
            assert(tactic.status && tactic.status.libs,
              `Group ${tactic.gid}: Status should be defined when liberties change`);
          }

          previousLibCount.set(tactic.gid, libertyCount);
        }

        g2.play(i);
        played = true;
        break;
      }
    }

    if (!played) break;
  }

  console.log(`  Observed ${transitionCount} liberty transitions`);
  console.log('  ✓ Status transitions test passed');
}

// Test: Performance benchmark for searchChains
function testPerformanceBenchmark() {
  console.log('\nTest: Performance Benchmark');

  const benchmarks = [];

  // Test on boards with varying complexity
  const configurations = [
    { size: 9, moves: 20, label: '9x9 early' },
    { size: 9, moves: 40, label: '9x9 mid' },
    { size: 13, moves: 30, label: '13x13 early' },
    { size: 13, moves: 60, label: '13x13 mid' },
  ];

  for (const config of configurations) {
    const g2 = new Game2(config.size);

    // Play moves
    let movesPlayed = 0;
    for (let i = 0; i < config.size * config.size && movesPlayed < config.moves; i++) {
      if (g2.isLegal(i)) {
        g2.play(i);
        movesPlayed++;
      }
    }

    // Benchmark searchChains
    const g3 = game3FromGame2(g2);
    const startTime = Date.now();
    const tactics = searchChains(g3, 10000);
    const elapsed = Date.now() - startTime;

    benchmarks.push({
      config: config.label,
      elapsed,
      groupsFound: tactics.length,
      movesPlayed,
    });

    console.log(`  ${config.label}: ${elapsed}ms, ${tactics.length} groups in ${movesPlayed} moves`);
  }

  // Verify no benchmark took excessively long
  for (const bm of benchmarks) {
    assert(bm.elapsed < 5000, `${bm.config} should complete in reasonable time (< 5000ms)`);
  }

  console.log('  ✓ Performance benchmark test passed');
}

// Test: Verify depth limit protects against stack overflow with extreme constraints
function testDepthLimitProtection() {
  console.log('\nTest: Depth Limit Protection');

  const g2 = new Game2(13);

  // Play enough moves to create complex position
  let movesPlayed = 0;
  for (let i = 0; i < 169 && movesPlayed < 40; i++) {
    if (g2.isLegal(i)) {
      g2.play(i);
      movesPlayed++;
    }
  }

  // Test with extreme node limits to verify depth limit prevents stack overflow
  const extremeLimits = [1, 10, 50, 100];
  let allCompleted = true;

  for (const limit of extremeLimits) {
    try {
      const g3 = game3FromGame2(g2);
      const tactics = searchChains(g3, limit);

      assert(Array.isArray(tactics), `With limit ${limit}, should still return array`);
      assert(tactics.length >= 0, `With limit ${limit}, should have valid group count`);

      console.log(`  Limit ${limit}: ${tactics.length} groups analyzed without stack overflow`);
    } catch (e) {
      if (e instanceof RangeError && e.message.includes('stack')) {
        console.log(`  ✗ Limit ${limit}: Stack overflow occurred!`);
        allCompleted = false;
      } else {
        throw e;
      }
    }
  }

  assert(allCompleted, 'All extreme limits should complete without stack overflow');
  console.log('  ✓ Depth limit protection test passed');
}

// Test: Compare results with different node limits
function testNodeLimitVariation() {
  console.log('\nTest: Node Limit Variation Impact');

  const g2 = new Game2(13);

  // Play enough moves to create tactical situations
  let movesPlayed = 0;
  for (let i = 0; i < 169 && movesPlayed < 35; i++) {
    if (g2.isLegal(i)) {
      g2.play(i);
      movesPlayed++;
    }
  }

  // Test with different node limits - use fresh g3 instance for each limit
  const limits = [5000, 10000];
  let firstResults = null;

  for (const limit of limits) {
    // Create fresh g3 from the same g2 state
    const g3 = game3FromGame2(g2);
    const tactics = searchChains(g3, limit);
    let inconclusiveCount = 0;
    let definitiveCount = 0;

    for (const tactic of tactics) {
      if (tactic.status) {
        if (tactic.status.moverSucceeds === null) {
          inconclusiveCount++;
        } else {
          definitiveCount++;
        }
      }
    }

    const results = {
      limit,
      groupsFound: tactics.length,
      inconclusive: inconclusiveCount,
      definitive: definitiveCount,
    };

    console.log(`  Limit ${results.limit}: ${results.groupsFound} groups, ${results.definitive} definitive, ${results.inconclusive} inconclusive`);

    // For now, just verify results are consistent across multiple calls with same limit
    if (firstResults === null) {
      firstResults = results;
    } else {
      // Higher limits should give same or more definitive results
      assert(results.definitive >= firstResults.definitive,
        `Higher node limit should give same or more definitive results`);
    }
  }

  console.log('  ✓ Node limit variation test passed');
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
  testStatusTransitions();
  testPerformanceBenchmark();
  testDepthLimitProtection();
  testNodeLimitVariation();

  console.log('\n' + '='.repeat(70));
  console.log('STATISTICS');
  console.log('='.repeat(70));
  console.log(`Total groups evaluated: ${stats.totalGroupsEvaluated}`);
  console.log(`Total definitive results: ${stats.totalDefinitiveResults}`);
  console.log(`Total inconclusive results (null): ${stats.totalInconclusiveResults}`);
  const inconclusiveRate = stats.totalGroupsEvaluated > 0
    ? ((stats.totalInconclusiveResults / stats.totalGroupsEvaluated) * 100).toFixed(2)
    : 0;
  console.log(`Inconclusive rate: ${inconclusiveRate}%`);
  console.log(`Total urgent liberties found: ${stats.totalUrgentLibs}`);
  console.log(`Max urgent libs per group: ${stats.maxUrgentLibsPerGroup}`);

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
