const { Game2 } = require('./game2.js');
const { Game3Precise } = require('./game3-precise.js');
const { getAllLadderStatuses: getAllLadderStatusesG3 } = require('./ladder2.js');

const Ladder2Game2 = (function() {
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

// Simple position with stones in atari
const moves = [
  // Create some groups
  45, 46, 47,      // White row
  57, 58, 59,      // Black row (next line)
  70, 71, 72,      // White
  82, 83, 84,      // Black
];

const game2 = new Game2(13);
const game3 = new Game3Precise(13);

for (const move of moves) {
  game2.play(move);
  game3.play(move);
}

const results2 = Ladder2Game2.getAllLadderStatuses(game2);
const results3 = getAllLadderStatusesG3(game3);

console.log(`Game2 found: ${results2.length} groups`);
console.log(`Game3-Precise found: ${results3.length} groups`);

console.log('\nGame2 groups:', results2.map(r => r.gid).sort((a,b) => a-b));
console.log('Game3-Precise gids:', results3.map(r => r.gid).sort((a,b) => a-b));

if (results2.length === results3.length) {
  console.log('\n✓ Same count');
} else {
  console.log('\n✗ DIFFERENT COUNTS!');
}
