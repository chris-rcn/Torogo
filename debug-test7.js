const { Game3Precise } = require('./game3-precise.js');
const { getAllLadderStatuses } = require('./ladder2.js');

const game = new Game3Precise(9);

// Play a game to near end
const moves = [40, 41, 49, 48, 32, 31, 23, 24, 33, 34, 42, 43, 50, 51];
for (const move of moves) {
  if (game.isLegal(move)) game.play(move);
}

const moveCountBefore = game.moveCount;
const cellsCopy = new Int8Array(game.cells);

console.log(`Before getAllLadderStatuses:`);
console.log(`  moveCount: ${game.moveCount}`);
console.log(`  cells hash: ${Array.from(game.cells).join(',')}`);

// Run ladder detection
const results = getAllLadderStatuses(game);
console.log(`\nallGetLadderStatuses found ${results.length} groups`);

console.log(`\nAfter getAllLadderStatuses:`);
console.log(`  moveCount: ${game.moveCount}`);
console.log(`  cells hash: ${Array.from(game.cells).join(',')}`);

console.log(`\nBoard changed:`);
let changed = false;
for (let i = 0; i < cellsCopy.length; i++) {
  if (cellsCopy[i] !== game.cells[i]) {
    console.log(`  cells[${i}]: ${cellsCopy[i]} -> ${game.cells[i]}`);
    changed = true;
  }
}

if (!changed) {
  console.log(`  (no changes)`);
}
