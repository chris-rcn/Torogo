const { Game2 } = require('./game2.js');
const { Game3Precise } = require('./game3.js');
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
      if (game.groupSize(gid) < 1) continue;  // Added this!
      const { count: lc } = game.groupLibs2(i);
      if (lc === 0 || lc > 2) continue;
      results.push({ gid, idx: i, lc });
    }
    return results;
  }
  return { getAllLadderStatuses: getAllLadderStatusesGame2 };
})();

const moves = [
  45, 46, 47, 57, 58, 59, 70, 71, 72, 82, 83, 84,
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

if (results2.length === results3.length) {
  console.log('✓ Same count');
} else {
  console.log('✗ DIFFERENT COUNTS!');
}
