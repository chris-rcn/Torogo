const { Game3Precise, PASS } = require('./game3-precise.js');
const { getAllLadderStatuses } = require('./ladder2.js');

const game = new Game3Precise(9);

// Play a game to near end
const moves = [40, 41, 49, 48, 32, 31, 23, 24, 33, 34, 42, 43, 50, 51];
for (const move of moves) {
  if (game.isLegal(move)) game.play(move);
}

console.log(`Initial cells[24] = ${game.cells[24]}`);
console.log(`Initial cells[41] = ${game.cells[41]}`);

// Manually call getAllLadderStatuses to track what happens
const cap = game.N * game.N;
const results = [];
const visited = new Set();

for (let i = 0; i < cap; i++) {
  if (game.cells[i] === 0) continue;
  const gid = game._gid[i];
  if (visited.has(gid)) continue;
  visited.add(gid);
  if (game.groupSize(gid) < 1) continue;
  const { count: lc } = game.groupLibs2(i);
  if (lc === 0 || lc > 2) continue;

  console.log(`\nAnalyzing group at index ${i} (gid=${gid}):`);
  console.log(`  cells[24] before = ${game.cells[24]}`);
  console.log(`  cells[41] before = ${game.cells[41]}`);

  const { getLadderStatus } = require('./ladder2.js');
  const status = require('./ladder2.js').getLadderStatus(game, i);

  console.log(`  cells[24] after = ${game.cells[24]}`);
  console.log(`  cells[41] after = ${game.cells[41]}`);

  results.push({ gid, status });
}

console.log(`\n\nFinal cells[24] = ${game.cells[24]} (should be -1)`);
console.log(`Final cells[41] = ${game.cells[41]} (should be -1)`);
