#!/usr/bin/env node
'use strict';

// Test tactical analysis using parseBoard for reproducible positions

const { Game2, BLACK, WHITE, parseBoard, coordStr } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');
const { searchChains } = require('./tactics3.js');

function testPosition(name, boardStr, toMove, expectedAnalysis) {
  console.log(`\n${'='.repeat(70)}`);
  console.log(`Test: ${name}`);
  console.log(`${'='.repeat(70)}`);

  // Parse the board
  const g2 = parseBoard(boardStr, toMove);
  const g3 = game3FromGame2(g2);

  console.log(`\nBoard (${g3.N}x${g3.N}):`);
  console.log(g3.toString(undefined, { axisLabels: true }));

  console.log(`\nCurrent player: ${g3.current === BLACK ? 'BLACK' : 'WHITE'}`);

  // Analyze at different depths
  const depths = [10, 15];
  const results = {};

  for (const depth of depths) {
    const tactics = searchChains(g3, 10000, depth);
    results[depth] = tactics;

    console.log(`\n--- Depth ${depth} ---`);
    console.log(`Groups in tactical range (1-3 liberties): ${tactics.length}`);

    for (const tactic of tactics) {
      const color = tactic.color === BLACK ? 'BLACK' : 'WHITE';
      const status = tactic.status ? (
        tactic.status.moverSucceeds === null ? 'inconclusive' :
        tactic.status.moverSucceeds ? 'wins' : 'loses'
      ) : 'null';
      const libs = tactic.status ? tactic.status.libs.length : '?';

      // Find a stone in this group
      let stoneCoord = '?';
      for (let i = 0; i < g3.N * g3.N; i++) {
        if (g3._gid[i] === tactic.gid) {
          stoneCoord = coordStr(i, g3.N);
          break;
        }
      }

      console.log(`  ${color} group at ${stoneCoord}: ${libs} liberties, ${status}`);
    }
  }

  // Verify expected analysis
  if (expectedAnalysis) {
    console.log(`\n--- Verification ---`);
    let passed = true;
    for (const groupCheck of expectedAnalysis) {
      const groupsAtDepth = results[groupCheck.depth] || [];
      const matchingGroup = groupsAtDepth.find(t =>
        t.gid === groupCheck.gid && t.color === groupCheck.color
      );

      if (matchingGroup) {
        const actualStatus = matchingGroup.status ? (
          matchingGroup.status.moverSucceeds === null ? 'inconclusive' :
          matchingGroup.status.moverSucceeds ? 'wins' : 'loses'
        ) : 'null';

        const match = actualStatus === groupCheck.expectedStatus;
        const checkMark = match ? '✓' : '✗';
        console.log(`${checkMark} Depth ${groupCheck.depth}, GID ${groupCheck.gid}: ${actualStatus} (expected ${groupCheck.expectedStatus})`);
        if (!match) passed = false;
      } else {
        console.log(`✗ Depth ${groupCheck.depth}, GID ${groupCheck.gid}: group not found`);
        passed = false;
      }
    }
    return passed;
  }
  return true;
}

// Test 1: Simple position - group with many liberties should escape
const simpleBoard = `
● ● ● · · · · · · ·
● ○ ● · · · · · · ·
● ○ ● · · · · · · ·
○ ○ ○ · · · · · · ·
· · · · · · · · · ·
· · · · · · · · · ·
· · · · · · · · · ·
· · · · · · · · · ·
· · · · · · · · · ·
· · · · · · · · · ·
`;

testPosition(
  'Simple: BLACK group with many liberties',
  simpleBoard,
  WHITE,
  [
    { depth: 10, color: BLACK, expectedStatus: 'wins' },
    { depth: 15, color: BLACK, expectedStatus: 'wins' }
  ]
);

// Test 2: Group in atari that can't escape
const ataribBoard = `
● · · · · · · · · ·
○ ○ ○ ○ ○ ○ ○ ○ ○ ○
· · · · · · · · · ·
· · · · · · · · · ·
· · · · · · · · · ·
· · · · · · · · · ·
· · · · · · · · · ·
· · · · · · · · · ·
· · · · · · · · · ·
· · · · · · · · · ·
`;

testPosition(
  'Atari: BLACK stone surrounded, cannot escape',
  ataribBoard,
  WHITE,
  [
    { depth: 10, color: BLACK, expectedStatus: 'loses' },
    { depth: 15, color: BLACK, expectedStatus: 'loses' }
  ]
);

// Test 3: Position from find-depth-15-chain requiring depth 15
// Game 1, Move 55: BLACK group at g9 with 3 liberties
// Depth 10: inconclusive, Depth 15: wins
// NOTE: parseBoard inverts rows, so first row in string becomes bottom row on board
const problemBoard = `
· · · · · · · · · · · · ·
· · · · · · · · · · · · ·
· · · · · · · · · · · · ·
● · · ○ · · ● · · · · · ·
· · · · · · ● ○ ○ · · · ·
○ · · · · · ● · · ○ ○ · ·
· ● · · · ● ● · · · · · ·
● · · · · · · · · ○ ○ · ○
○ · · · · · · ○ · ○ · · ·
○ ● · · · · · ● ○ ○ · · ○
· ● · ○ ○ ● · ● · · · ● ·
○ ○ · ○ · ● · · · · ○ · ○
· · ● ● · ● · · · · · · ●
`;

testPosition(
  'Found problematic position: BLACK group at g9, depth 10 inconclusive, depth 15 wins',
  problemBoard,
  WHITE
);

// Test 4: Position requiring EXACTLY depth 15
// Game 0, Move 62: WHITE group at b3 with 3 liberties
// Depth 14: inconclusive, Depth 15: wins
const depth15RequiredBoard = `
· · · · · · · · · · · · ·
· · · · · · · · · · · · ·
· · · · · · · · · · · · ·
· ● · · · · · · · · · · ·
· · ● ● · · · · · · ● ○ ·
○ · ● ○ ● ○ · · · · ○ · ·
○ · · ○ · ○ ○ · ○ · · · ○
· · ● ● · · · · · · ● ○ ·
· · ● · · ○ · ● ● · · · ○
· ● · · · · · · · · · · ·
· ● ● · · · · · · · · · ·
· (○) ○ · ○ ○ · · ○ · · ● ●
· · ● · · · ● ● ○ · · · ·
`;

console.log('\n--- Testing depth 14 vs 15 specific positions ---');
testPosition(
  'DEPTH 15 REQUIRED: WHITE group at b3, inconclusive at 14, wins at 15',
  depth15RequiredBoard,
  BLACK
);

console.log(`\n${'='.repeat(70)}`);
console.log('Tests completed');
console.log(`${'='.repeat(70)}\n`);
