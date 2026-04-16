const { Game3Precise, PASS } = require('./game3-precise.js');

const game = new Game3Precise(9);

// Play a game to near end
const moves = [40, 41, 49, 48, 32, 31, 23, 24, 33, 34, 42, 43, 50, 51];
for (const move of moves) {
  if (game.isLegal(move)) game.play(move);
}

console.log('Initial state:');
console.log(`  cells[24] = ${game.cells[24]} (WHITE)`);
console.log(`  Group at 24: lib0=15, lib1=25`);

game.play(PASS);
console.log(`After PASS: current=${game.current}`);

game.play(15);
console.log(`\nAfter move 15:`);
console.log(`  cells[24] = ${game.cells[24]}`);
console.log(`  cells[15] = ${game.cells[15]}`);
console.log(`  current=${game.current}`);

// Check the group at 24 now
const { count: lc, lib0, lib1 } = game.groupLibs2(24);
console.log(`  Group at 24: libs=${lc} (${lib0}, ${lib1})`);

// Check neighbors of 24
const nbr = game._nbr;
const base = 24 * 4;
console.log(`  Neighbors of 24:`);
for (let i = 0; i < 4; i++) {
  const ni = nbr[base + i];
  const c = game.cells[ni];
  console.log(`    idx ${ni}: ${c}`);
}

console.log(`\nNow playing move 25:`);
const legalBefore = game.isLegal(25);
console.log(`  is Legal(25)? ${legalBefore}`);

game.play(25);
console.log(`After move 25:`);
console.log(`  cells[24] = ${game.cells[24]}`);
console.log(`  cells[25] = ${game.cells[25]}`);
console.log(`  current=${game.current}`);

console.log(`\nNow undoing move 25:`);
game.undo();
console.log(`After undo:`);
console.log(`  cells[24] = ${game.cells[24]} (should be -1)`);
console.log(`  cells[25] = ${game.cells[25]} (should be 0)`);
console.log(`  current=${game.current}`);
