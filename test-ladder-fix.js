'use strict';

// Test suite for the ladder detection bug fix

const { Game3Precise } = require('./game3-precise.js');
const { getAllLadderStatuses, getLadderStatus } = require('./ladder2.js');

let passCount = 0;
let failCount = 0;

function assert(condition, message) {
  if (condition) {
    passCount++;
    console.log(`✓ ${message}`);
  } else {
    failCount++;
    console.log(`✗ FAILED: ${message}`);
  }
}

console.log('Testing Ladder Detection Bug Fix\n');
console.log('=====================================\n');

// Test 1: Single getLadderStatus call preserves state
console.log('Test 1: Single getLadderStatus call preserves state');
{
  const game = new Game3Precise(13);
  const moves = [45, 46, 47, 57, 58, 59, 70, 71, 72, 82, 83, 84];
  for (const move of moves) game.play(move);

  const idx = 46;
  const libsBefore = game.groupLibs2(idx).count;
  const status = getLadderStatus(game, idx);
  const libsAfter = game.groupLibs2(idx).count;

  assert(libsBefore === libsAfter, `Liberty count stable after getLadderStatus: ${libsBefore} === ${libsAfter}`);
  assert(status !== null, 'getLadderStatus returned valid status');
}

// Test 2: Multiple getLadderStatus calls preserve all groups
console.log('\nTest 2: Multiple getLadderStatus calls preserve all groups');
{
  const game = new Game3Precise(13);
  const moves = [45, 46, 47, 57, 58, 59, 70, 71, 72, 82, 83, 84];
  for (const move of moves) game.play(move);

  // Get initial atari groups
  const getAtariGroups = () => {
    const cap = game.N * game.N;
    const visited = new Set();
    const groups = [];
    for (let i = 0; i < cap; i++) {
      if (game.cells[i] === 0) continue;
      const gid = game._gid[i];
      if (visited.has(gid)) continue;
      visited.add(gid);
      const { count: lc } = game.groupLibs2(i);
      if (lc === 1 || lc === 2) {
        groups.push({ gid, lc });
      }
    }
    return groups;
  };

  const groupsBefore = getAtariGroups();
  assert(groupsBefore.length === 2, `Found 2 atari groups initially: ${groupsBefore.length}`);

  // Call getLadderStatus for each group
  for (const idx of [46, 57]) {
    const { count: lc } = game.groupLibs2(idx);
    if (lc === 1 || lc === 2) {
      getLadderStatus(game, idx);
    }
  }

  const groupsAfter = getAtariGroups();
  assert(groupsAfter.length === 2, `Still have 2 atari groups after calls: ${groupsAfter.length}`);
  assert(groupsBefore[0].lc === groupsAfter[0].lc, `Group 0 liberty count preserved: ${groupsBefore[0].lc} === ${groupsAfter[0].lc}`);
  assert(groupsBefore[1].lc === groupsAfter[1].lc, `Group 1 liberty count preserved: ${groupsBefore[1].lc} === ${groupsAfter[1].lc}`);
}

// Test 3: getAllLadderStatuses preserves state
console.log('\nTest 3: getAllLadderStatuses preserves state');
{
  const game = new Game3Precise(13);
  const moves = [45, 46, 47, 57, 58, 59, 70, 71, 72, 82, 83, 84];
  for (const move of moves) game.play(move);

  const getAtariCount = () => {
    const cap = game.N * game.N;
    const visited = new Set();
    let count = 0;
    for (let i = 0; i < cap; i++) {
      if (game.cells[i] === 0) continue;
      const gid = game._gid[i];
      if (visited.has(gid)) continue;
      visited.add(gid);
      const { count: lc } = game.groupLibs2(i);
      if (lc === 1 || lc === 2) count++;
    }
    return count;
  };

  const countBefore = getAtariCount();
  const results = getAllLadderStatuses(game);
  const countAfter = getAtariCount();

  assert(countBefore === 2, `Found 2 atari groups: ${countBefore}`);
  assert(countAfter === 2, `Still have 2 atari groups after getAllLadderStatuses: ${countAfter}`);
  assert(results.length === 2, `getAllLadderStatuses returned 2 results: ${results.length}`);
}

// Test 4: Multiple getAllLadderStatuses calls are consistent
console.log('\nTest 4: Multiple getAllLadderStatuses calls are consistent');
{
  const game = new Game3Precise(13);
  const moves = [45, 46, 47, 57, 58, 59, 70, 71, 72, 82, 83, 84];
  for (const move of moves) game.play(move);

  const results1 = getAllLadderStatuses(game);
  const results2 = getAllLadderStatuses(game);
  const results3 = getAllLadderStatuses(game);

  assert(results1.length === results2.length, `First and second calls return same count: ${results1.length} === ${results2.length}`);
  assert(results2.length === results3.length, `Second and third calls return same count: ${results2.length} === ${results3.length}`);

  // Check that gids are the same in the same order
  if (results1.length === results2.length && results2.length === results3.length) {
    let same = true;
    for (let i = 0; i < results1.length; i++) {
      if (results1[i].gid !== results2[i].gid || results2[i].gid !== results3[i].gid) {
        same = false;
        break;
      }
    }
    assert(same, 'Multiple calls return same groups in same order');
  }
}

// Test 5: Complex position with many groups
console.log('\nTest 5: Complex position with many groups');
{
  const game = new Game3Precise(13);

  // Play a more complex game
  const moves = [
    66, 67, 78, 79, 90, 91, 102, 103,
    55, 56, 68, 69, 80, 81, 93, 94, 105, 106,
  ];
  for (const move of moves) {
    if (game.isLegal(move)) game.play(move);
  }

  const results1 = getAllLadderStatuses(game);
  const results2 = getAllLadderStatuses(game);

  assert(results1.length === results2.length, `Complex position: consistent results ${results1.length} === ${results2.length}`);
}

// Test 6: Ko rule is properly set
console.log('\nTest 6: Ko rule is properly set');
{
  const game = new Game3Precise(9);

  // Play moves to create a capture
  game.play(40);
  game.play(41);
  game.play(32);
  game.play(31);
  game.play(48);
  game.play(49);
  game.play(57);

  assert(typeof game.ko === 'number', `Ko is a number after play: ${typeof game.ko}`);
  assert(game.ko === -1 || game.ko >= 0, `Ko is valid: ${game.ko}`);
}

// Test 7: State consistency after deep recursion
console.log('\nTest 7: State consistency after deep recursion');
{
  const game = new Game3Precise(9);

  // Play a game to near end
  const moves = [40, 41, 49, 48, 32, 31, 23, 24, 33, 34, 42, 43, 50, 51];
  for (const move of moves) {
    if (game.isLegal(move)) game.play(move);
  }

  const moveCountBefore = game.moveCount;
  const koStatusBefore = game.ko;

  // Run ladder detection
  getAllLadderStatuses(game);

  assert(game.moveCount === moveCountBefore, `Move count unchanged: ${game.moveCount}`);
  assert(game.ko === koStatusBefore, `Ko status unchanged: ${game.ko}`);
}

console.log('\n=====================================');
console.log(`Results: ${passCount} passed, ${failCount} failed`);

if (failCount === 0) {
  console.log('✓ All tests passed!');
  process.exit(0);
} else {
  console.log(`✗ ${failCount} test(s) failed`);
  process.exit(1);
}
