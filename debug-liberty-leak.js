const { Game3Precise, PASS } = require('./game3.js');

const game = new Game3Precise(13);

const moves = [45, 46, 47, 57, 58, 59, 70, 71, 72, 82, 83, 84];
for (const move of moves) {
  game.play(move);
}

console.log('Initial state:');
console.log(`  _ls[2] = ${game._ls[2]}`);
console.log(`  _ls[4] = ${game._ls[4]}`);

// Manually trace through what getLadderStatus does
const stoneIdx = 46;
const { count: lc, lib0, lib1 } = game.groupLibs2(stoneIdx);
const libs = [lib0, lib1];
const gColor = game.cells[stoneIdx];
const defending = gColor === game.current;

console.log(`\nGetting status for stone at ${stoneIdx} (gid 2):`);
console.log(`  defending=${defending}, libs=[${lib0}, ${lib1}]`);

console.log(`\nPlay PASS...`);
game.play(PASS);
console.log(`  _ls[2] = ${game._ls[2]}`);

console.log(`\nNow simulate _canReach3Libs for defender's turn...`);

// The attacker (WHITE) is defending, so try each liberty
for (const libIdx of libs) {
  console.log(`\n  Play liberty ${libIdx}...`);
  const playResult = game.play(libIdx);
  console.log(`    Result: ${playResult}`);
  console.log(`    _ls[2] = ${game._ls[2]} after play`);

  // Simulate _canReach3Libs recursively checking this position
  // For simplicity, just check if stone is still there
  if (game.cells[stoneIdx] !== 0) {
    console.log(`    Stone at ${stoneIdx} still exists`);

    // In _canReach3Libs, we would recursively call _canReach3Libs
    // But for now, let's just check neighbors
    // When BLACK plays at one of WHITE's liberties, WHITE stones will get liberties
    const nbr = game._nbr;
    const base = libIdx * 4;
    console.log(`    Checking neighbors of ${libIdx}:`);
    for (let i = 0; i < 4; i++) {
      const ni = nbr[base + i];
      const gid = game._gid[ni];
      if (gid === 2) {
        console.log(`      Found gid 2 neighbor at ${ni}`);
      }
    }
  }

  console.log(`  Undo...`);
  game.undo();
  console.log(`    _ls[2] = ${game._ls[2]} after undo`);
}

console.log(`\nUndo PASS...`);
game.undo();
console.log(`  _ls[2] = ${game._ls[2]} after undo PASS`);

console.log(`\nFinal state:`);
console.log(`  _ls[2] = ${game._ls[2]} (started at 2, should end at 2)`);
