const { Game3Precise, PASS } = require('./game3.js');

const game = new Game3Precise(13);

const moves = [45, 46, 47, 57, 58, 59, 70, 71, 72, 82, 83, 84];
for (const move of moves) {
  game.play(move);
}

const stoneIdx = 46;
console.log(`Analyzing stone at index ${stoneIdx}`);
console.log(`Initial state: _ls[2] = ${game._ls[2]}`);

const { count: lc, lib0, lib1 } = game.groupLibs2(stoneIdx);
console.log(`  liberties: count=${lc}, lib0=${lib0}, lib1=${lib1}`);

console.log(`\nTrying each liberty:`);

// Play PASS first (like getLadderStatus does)
game.play(PASS);

for (const libIdx of [lib0, lib1]) {
  console.log(`\n  Playing liberty ${libIdx}...`);
  game.play(libIdx);
  console.log(`    Cell at stone index: ${game.cells[stoneIdx]}`);
  console.log(`    Cell at liberty: ${game.cells[libIdx]}`);
  const { count: afterLc } = game.groupLibs2(stoneIdx);
  console.log(`    After liberty count: ${afterLc}`);
  console.log(`    Would recurse: ${afterLc === 1}`);
  game.undo();
}

game.undo();  // Undo PASS
console.log(`\nFinal _ls[2] = ${game._ls[2]} (should be 2)`);
