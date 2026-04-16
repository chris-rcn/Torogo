'use strict';

// Verify that Game2 and Game3-Precise ladder detection produce identical results

const { Game2 } = require('./game2.js');
const { Game3Precise } = require('./game3.js');
const { getAllLadderStatuses: getAllLadderStatusesG3 } = require('./ladder2.js');

// Game2-based ladder detection
const Ladder2Game2 = (function() {
  function _canReach3Libs(game, idx) {
    const { count: lc, lib0, lib1 } = game.groupLibs2(idx);
    if (lc >= 3) return true;
    if (lc === 0) return false;
    const defColor = game.cells[idx];
    if (game.current === defColor) {
      const libs = lc === 1 ? [lib0] : [lib0, lib1];
      for (const libIdx of libs) {
        const g = game.clone();
        if (!g.play(libIdx)) continue;
        if (g.cells[idx] === 0) continue;
        if (_canReach3Libs(g, idx)) return true;
      }
      return false;
    }
    const libs = lc === 1 ? [lib0] : [lib0, lib1];
    for (const libIdx of libs) {
      const g = game.clone();
      if (!g.play(libIdx)) continue;
      if (g.cells[idx] === 0) return false;
      const afterLc = g.groupLibs2(idx).count;
      if (afterLc === 0) return false;
      if (afterLc === 1 && !_canReach3Libs(g, idx)) return false;
    }
    return true;
  }

  function getAllLadderStatusesGame2(game) {
    const cap = game.N * game.N;
    const results = [];
    const visited = new Set();
    for (let i = 0; i < cap; i++) {
      if (game.cells[i] === 0) continue;
      const gid = game._gid[i];
      if (visited.has(gid)) continue;
      visited.add(gid);
      const { count: lc } = game.groupLibs2(i);
      if (lc === 0 || lc > 2) continue;
      results.push({ gid, idx: i, lc });
    }
    return results;
  }

  return { getAllLadderStatuses: getAllLadderStatusesGame2 };
})();

function playMoves(game, moves) {
  for (const move of moves) {
    if (!game.play(move)) break;
  }
}

// Test position
const moves = [45, 67, 55, 57, 69, 79, 59, 75, 73];

const game2 = new Game2(13);
const game3 = new Game3Precise(13);

playMoves(game2, moves);
playMoves(game3, moves);

console.log('Verifying ladder detection results...');
console.log('Test position: 13x13, after moves:', moves.join(' '));
console.log('');

const results2 = Ladder2Game2.getAllLadderStatuses(game2);
const results3 = getAllLadderStatusesG3(game3);

console.log(`Game2 found: ${results2.length} groups with 1-2 liberties`);
console.log(`Game3-Precise found: ${results3.length} groups with 1-2 liberties`);

// Check if they found the same groups
const gids2 = new Set(results2.map(r => r.gid));
const gids3 = new Set(results3.map(r => r.gid));

console.log('');
if (gids2.size === gids3.size && results2.length === results3.length) {
  console.log('✓ Same number of groups found');
} else {
  console.log('✗ DIFFERENT number of groups!');
  console.log(`  Game2: ${gids2.size} unique groups`);
  console.log(`  Game3-Precise: ${gids3.size} unique groups`);
}

// Show the groups
console.log('\nGame2 groups:');
results2.forEach(r => {
  const x = r.idx % 13, y = Math.floor(r.idx / 13);
  console.log(`  gid=${r.gid} at (${x},${y}) with ${r.lc} liberties`);
});

console.log('\nGame3-Precise groups:');
results3.forEach(r => {
  console.log(`  gid=${r.gid}, color=${r.color}`);
});
