'use strict';

// Test ladder detection with Game3-Precise

const { Game3Precise, PASS, BLACK, WHITE } = require('./game3.js');
const { getAllLadderStatuses, getLadderStatus } = require('./ladder2.js');

function testLadderDetection() {
  console.log('Testing ladder detection with Game3-Precise...');

  // Create a simple position
  const game = new Game3Precise(9);

  // Play some moves to create ladder scenarios
  game.play(10);  // W
  game.play(11);  // B
  game.play(12);  // W
  game.play(20);  // B
  game.play(21);  // W
  game.play(19);  // B

  // Test getAllLadderStatuses
  const statuses = getAllLadderStatuses(game);
  console.assert(Array.isArray(statuses), 'getAllLadderStatuses should return array');

  // Verify structure of returned objects
  for (const status of statuses) {
    console.assert(typeof status.gid === 'number', 'Should have gid');
    console.assert(typeof status.color === 'number', 'Should have color');
    console.assert(status.color === BLACK || status.color === WHITE, 'Color should be BLACK or WHITE');
    console.assert(status.status !== null, 'Status should not be null for valid groups');
    if (status.status) {
      console.assert(Array.isArray(status.status.libs), 'Status should have libs array');
      console.assert(typeof status.status.moverSucceeds === 'boolean', 'Should have moverSucceeds');
      console.assert(Array.isArray(status.status.urgentLibs), 'Should have urgentLibs array');
    }
  }

  console.log('✓ Ladder detection works with Game3-Precise');
}

function testGroupLibs2() {
  console.log('Testing groupLibs2() method...');

  const game = new Game3Precise(9);

  // Play some stones
  game.play(10);  // W
  game.play(11);  // B
  game.play(12);  // W
  game.play(20);  // B

  // Test that groupLibs2 works
  const libs2 = game.groupLibs2(10);
  console.assert(typeof libs2.count === 'number', 'Should have count');
  console.assert(typeof libs2.lib0 === 'number', 'Should have lib0');
  console.assert(typeof libs2.lib1 === 'number', 'Should have lib1');

  // Verify count matches actual liberty count
  const libsArray = game.groupLibs(10);
  console.assert(libs2.count === libsArray.length, `count (${libs2.count}) should match groupLibs length (${libsArray.length})`);

  // If count >= 1, lib0 should be in the liberties
  if (libs2.count >= 1 && libs2.lib0 >= 0) {
    console.assert(libsArray.includes(libs2.lib0), 'lib0 should be in liberties array');
  }

  // If count >= 2, lib1 should be in the liberties
  if (libs2.count >= 2 && libs2.lib1 >= 0) {
    console.assert(libsArray.includes(libs2.lib1), 'lib1 should be in liberties array');
  }

  console.log('✓ groupLibs2() works correctly');
}

try {
  testGroupLibs2();
  testLadderDetection();

  console.log('\n' + '='.repeat(60));
  console.log('All ladder tests passed! ✓');
} catch (e) {
  console.error('Test failed with error:');
  console.error(e);
  process.exit(1);
}
