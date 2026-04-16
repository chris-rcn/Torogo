const { Game3Precise, PASS } = require('./game3-precise.js');
const { getLadderStatus } = require('./ladder2.js');

const game = new Game3Precise(13);

const moves = [45, 46, 47, 57, 58, 59, 70, 71, 72, 82, 83, 84];
for (const move of moves) {
  game.play(move);
}

console.log('Before getLadderStatus:');
const { count: lcBefore, lib0: lib0Before, lib1: lib1Before } = game.groupLibs2(46);
console.log(`  groupLibs2(46) = {count: ${lcBefore}, lib0: ${lib0Before}, lib1: ${lib1Before}}`);
console.log(`  cells[46] = ${game.cells[46]}`);
console.log(`  _gid[46] = ${game._gid[46]}`);
console.log(`  _ls[2] = ${game._ls[2]}`);

console.log('\nCalling getLadderStatus(game, 46)...');
const status = getLadderStatus(game, 46);
console.log(`  Returned: ${JSON.stringify(status)}`);

console.log('\nAfter getLadderStatus:');
console.log(`  cells[46] = ${game.cells[46]}`);
console.log(`  _gid[46] = ${game._gid[46]}`);
console.log(`  _ls[2] = ${game._ls[2]}`);
const { count: lcAfter, lib0: lib0After, lib1: lib1After } = game.groupLibs2(46);
console.log(`  groupLibs2(46) = {count: ${lcAfter}, lib0: ${lib0After}, lib1: ${lib1After}}`);

if (lcAfter === 0) {
  console.log('\nERROR: Liberty count is 0 after getLadderStatus!');

  // Check what happened to the liberty bitset
  console.log(`\nDetailed group info for gid 2:`);
  const gid = 2;
  const W = game._W;
  const gb = gid * W;

  console.log(`  _ss[2] (stone count) = ${game._ss[2]}`);
  console.log(`  _ls[2] (liberty count) = ${game._ls[2]}`);
  console.log(`  _gc[2] (color) = ${game._gc[2]}`);

  // Check if stones are still there
  console.log(`  \nStones in bitset:`);
  let stoneCount = 0;
  for (let wi = 0; wi < W; wi++) {
    let w = game._sw[gb + wi];
    while (w) {
      const bit = 31 - Math.clz32(w & -w);
      const idx = wi * 32 + bit;
      if (idx < 169) {
        console.log(`    idx ${idx}: cells[${idx}] = ${game.cells[idx]}`);
        stoneCount++;
      }
      w &= w - 1;
    }
  }
  console.log(`  Total stones in bitset: ${stoneCount}`);

  // Check liberties
  console.log(`  \nLiberties in bitset:`);
  let libCount = 0;
  const lb = gid * W;
  for (let wi = 0; wi < W; wi++) {
    let w = game._lw[lb + wi];
    while (w && libCount < 5) {
      const bit = 31 - Math.clz32(w & -w);
      const idx = wi * 32 + bit;
      if (idx < 169) {
        console.log(`    idx ${idx}: cells[${idx}] = ${game.cells[idx]}`);
        libCount++;
      }
      w &= w - 1;
    }
  }
  console.log(`  Total liberties found: ${libCount} (expected at least 1)`);
}
