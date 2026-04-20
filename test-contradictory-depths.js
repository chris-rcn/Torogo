#!/usr/bin/env node
'use strict';

// Test that demonstrates contradictory results at different depths
// Position from find-depth-15-chain: Game 0, Move 62

const { Game2, WHITE, BLACK, parseBoard, coordStr } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');
const { searchChains } = require('./tactics3.js');

// Position requiring exactly depth 15 (inconclusive at depth 14)
// Board is inverted because parseBoard inverts rows
const boardStr = `
· · · · · · · · · · · · ·
· · · · · · · · · · · · ·
· · · · · · · · · · · · ·
· ● · · · · · · · · · · ·
· · ● · · ○ · ● ● · · · ○
○ · · ○ · ○ ○ · ○ · · · ○
· · ● ● · · · · · · ● ○ ·
· · ● · · ○ · ● ● · · · ○
· ● · · · · · · · · · · ·
· ● ● · · · · · · · · · ·
· (○) ○ · ○ ○ · · ○ · · ● ●
· · ● · · · ● ● ○ · · · ·
`;

console.log('Testing position with contradictory results at different depths\n');
console.log('Position: Game 0, Move 62 - requires depth 15\n');

const g2 = parseBoard(boardStr, BLACK);
const g3 = game3FromGame2(g2);

console.log('Board:');
console.log(g3.toString(undefined, { axisLabels: true }));
console.log('\nCurrent player: BLACK\n');

// Analyze at depths 10 and 15
const results = {};
for (const depth of [10, 15]) {
  const tactics = searchChains(g3, 10000, depth);
  results[depth] = tactics;

  console.log(`--- Depth ${depth} ---`);
  console.log(`Groups in tactical range: ${tactics.length}`);

  for (const tactic of tactics) {
    const color = tactic.color === BLACK ? 'BLACK' : 'WHITE';
    const status = tactic.status ? (
      tactic.status.moverSucceeds === null ? 'inconclusive' :
      tactic.status.moverSucceeds ? 'wins' : 'loses'
    ) : 'null';
    const libs = tactic.status ? tactic.status.libs.length : '?';

    let stoneCoord = '?';
    for (let i = 0; i < g3.N * g3.N; i++) {
      if (g3._gid[i] === tactic.gid) {
        stoneCoord = coordStr(i, g3.N);
        break;
      }
    }

    console.log(`  ${color} at ${stoneCoord}: ${libs} libs, ${status}`);
  }
  console.log();
}

// Compare results
console.log('--- Contradiction Check ---\n');

const map10 = new Map();
const map15 = new Map();

for (const t of results[10]) {
  map10.set(t.gid, t);
}
for (const t of results[15]) {
  map15.set(t.gid, t);
}

let foundContradiction = false;

// Check groups that appear in both
for (const [gid, tactic10] of map10) {
  const tactic15 = map15.get(gid);
  if (tactic15) {
    const result10 = tactic10.status ? tactic10.status.moverSucceeds : null;
    const result15 = tactic15.status ? tactic15.status.moverSucceeds : null;

    // Both definitive but opposite
    if (result10 !== null && result15 !== null && result10 !== result15) {
      foundContradiction = true;
      const color = tactic10.color === BLACK ? 'BLACK' : 'WHITE';
      let coord = '?';
      for (let i = 0; i < g3.N * g3.N; i++) {
        if (g3._gid[i] === gid) {
          coord = coordStr(i, g3.N);
          break;
        }
      }
      console.log(`✗ CONTRADICTION: ${color} at ${coord}`);
      console.log(`    Depth 10: ${result10 ? 'wins' : 'loses'}`);
      console.log(`    Depth 15: ${result15 ? 'wins' : 'loses'}`);
    }
  }
}

if (!foundContradiction) {
  console.log('No direct contradictions (both definitive but opposite)');
}

// Check for results that become inconclusive at higher depth
console.log('\n--- Definitiveness Regression Check ---\n');

let hasRegression = false;
for (const [gid, tactic10] of map10) {
  const tactic15 = map15.get(gid);
  if (tactic15) {
    const result10 = tactic10.status ? tactic10.status.moverSucceeds : null;
    const result15 = tactic15.status ? tactic15.status.moverSucceeds : null;

    // Definitive at depth 10, inconclusive at depth 15
    if (result10 !== null && result15 === null) {
      hasRegression = true;
      const color = tactic10.color === BLACK ? 'BLACK' : 'WHITE';
      let coord = '?';
      for (let i = 0; i < g3.N * g3.N; i++) {
        if (g3._gid[i] === gid) {
          coord = coordStr(i, g3.N);
          break;
        }
      }
      console.log(`✗ REGRESSION: ${color} at ${coord}`);
      console.log(`    Depth 10: ${result10 ? 'wins' : 'loses'} (definitive)`);
      console.log(`    Depth 15: inconclusive (WORSE!)`);
    }
  }
}

if (!hasRegression) {
  console.log('No definitiveness regressions');
}

// Overall result
console.log('\n' + '='.repeat(70));
if (foundContradiction || hasRegression) {
  console.log('✗ TEST FAILED: Algorithm produces contradictory results');
  console.log('  Higher depth should not make results less certain or opposite!');
  process.exit(1);
} else {
  console.log('✓ No contradictions detected in this position');
  process.exit(0);
}
