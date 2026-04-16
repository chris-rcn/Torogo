const { Game3Precise } = require('./game3.js');
const { getAllLadderStatuses } = require('./ladder2.js');

const game = new Game3Precise(13);

// Play some moves
const moves = [45, 46, 47, 57, 58, 59, 70, 71, 72, 82, 83, 84];
for (const move of moves) {
  game.play(move);
}

console.log('Initial state:');
console.log('  moveCount:', game.moveCount);
console.log('  current:', game.current);

// Call getAllLadderStatuses
const results1 = getAllLadderStatuses(game);
console.log('\nAfter first getAllLadderStatuses:');
console.log('  moveCount:', game.moveCount);
console.log('  current:', game.current);
console.log('  results:', results1.length);

// Call again
const results2 = getAllLadderStatuses(game);
console.log('\nAfter second getAllLadderStatuses:');
console.log('  moveCount:', game.moveCount);
console.log('  current:', game.current);
console.log('  results:', results2.length);

if (results1.length === results2.length) {
  console.log('\n✓ Consistent results');
} else {
  console.log('\n✗ INCONSISTENT RESULTS!');
}
