const { Game3Precise, PASS } = require('./game3.js');

const game = new Game3Precise(9);

// Play a game to near end
const moves = [40, 41, 49, 48, 32, 31, 23, 24, 33, 34, 42, 43, 50, 51];
for (const move of moves) {
  if (game.isLegal(move)) game.play(move);
}

console.log(`Initial cells[24] = ${game.cells[24]} (should be -1)`);

// Now manually simulate what getLadderStatus does when analyzing the group at 24
const stoneIdx = 24;
const { count: lc, lib0, lib1 } = game.groupLibs2(stoneIdx);

console.log(`\nGroup at ${stoneIdx}:`);
console.log(`  lc=${lc}, lib0=${lib0}, lib1=${lib1}`);

// getLadderStatus plays PASS first
console.log(`\nPlay PASS:`);
game.play(PASS);
console.log(`  cells[24] = ${game.cells[24]}`);

// Then _canReach3Libs loop
const libs = lc === 1 ? [lib0] : [lib0, lib1];
for (let libIdx of libs) {
  console.log(`\nPlay liberty ${libIdx}:`);
  if (!game.play(libIdx)) {
    console.log(`  (illegal)`);
    continue;
  }
  console.log(`  cells[24] = ${game.cells[24]}`);

  // Then for attacker's turn in _canReach3Libs
  const afterLc = game.groupLibs2(stoneIdx).count;
  console.log(`  afterLc = ${afterLc}`);

  // Undo
  console.log(`  Undo:`);
  game.undo();
  console.log(`    cells[24] = ${game.cells[24]}`);
}

// Undo PASS
console.log(`\nUndo PASS:`);
game.undo();
console.log(`  cells[24] = ${game.cells[24]} (should be -1)`);

if (game.cells[24] !== -1) {
  console.log(`\nERROR: cells[24] is ${game.cells[24]}, should be -1!`);
}
