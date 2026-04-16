const { Game3Precise, PASS } = require('./game3-precise.js');

const game = new Game3Precise(13);

const moves = [45, 46, 47, 57, 58, 59, 70, 71, 72, 82, 83, 84];
for (const move of moves) {
  game.play(move);
}

console.log(`Initial opStack.length = ${game._opStack.length}`);
console.log(`Initial _ls[2] = ${game._ls[2]}\n`);

// Manually trace the exact sequence that getLadderStatus does
const stoneIdx = 46;
const { count: lc, lib0, lib1 } = game.groupLibs2(stoneIdx);
const libs = lc === 1 ? [lib0] : [lib0, lib1];

console.log(`Calling play(PASS)...`);
game.play(PASS);
console.log(`  opStack.length = ${game._opStack.length}`);
console.log(`  _ls[2] = ${game._ls[2]}`);

// Now simulate just the first iteration of _canReach3Libs
const libIdx = lib0;
console.log(`\nCalling play(${libIdx}) for _canReach3Libs...`);
game.play(libIdx);
console.log(`  opStack.length = ${game._opStack.length}`);
console.log(`  _ls[2] = ${game._ls[2]}`);

// In _canReach3Libs, if defending, we would recursively call _canReach3Libs
// For now, just check if we would
const defending = game.cells[stoneIdx] !== 0 && game.cells[stoneIdx] === game.current;
console.log(`  defending = ${defending}`);

console.log(`\nCalling undo() from _canReach3Libs...`);
game.undo();
console.log(`  opStack.length = ${game._opStack.length}`);
console.log(`  _ls[2] = ${game._ls[2]}`);

// Try the second liberty
const libIdx2 = lib1;
console.log(`\nCalling play(${libIdx2}) for _canReach3Libs...`);
game.play(libIdx2);
console.log(`  opStack.length = ${game._opStack.length}`);
console.log(`  _ls[2] = ${game._ls[2]}`);

console.log(`\nCalling undo() from _canReach3Libs...`);
game.undo();
console.log(`  opStack.length = ${game._opStack.length}`);
console.log(`  _ls[2] = ${game._ls[2]}`);

// Now undo the PASS
console.log(`\nCalling undo() to undo the PASS...`);
game.undo();
console.log(`  opStack.length = ${game._opStack.length}`);
console.log(`  _ls[2] = ${game._ls[2]}`);

console.log(`\nFinal state:`);
console.log(`  _ls[2] = ${game._ls[2]} (should be 2)`);
console.log(`  opStack.length = ${game._opStack.length} (should be initial ${moves.length + 1})`);
