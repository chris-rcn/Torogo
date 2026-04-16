const { Game3Precise, PASS } = require('./game3-precise.js');

const game = new Game3Precise(13);

const moves = [45, 46, 47, 57, 58, 59, 70, 71, 72, 82, 83, 84];
for (const move of moves) {
  game.play(move);
}

console.log('Testing getLadderStatus on gid 2 (at index 46)...');

const stoneIdx = 46;
console.log(`\nInitial state at index ${stoneIdx}:`);
const { count: lc, lib0, lib1 } = game.groupLibs2(stoneIdx);
console.log(`  groupLibs2(${stoneIdx}) = {count: ${lc}, lib0: ${lib0}, lib1: ${lib1}}`);
console.log(`  cells[${stoneIdx}] = ${game.cells[stoneIdx]}`);
console.log(`  _gid[${stoneIdx}] = ${game._gid[stoneIdx]}`);

const atari = lc === 1;
const libs = atari ? [lib0] : [lib0, lib1];
const gColor = game.cells[stoneIdx];
const mover = game.current;
const defending = gColor === mover;

console.log(`\n  atari=${atari}, gColor=${gColor}, mover=${mover}, defending=${defending}`);

// Try opponent playing first
console.log(`\nStep 1: Try opponent playing first (play PASS)`);
console.log(`  Current player before PASS: ${game.current}`);
console.log(`  Operation stack length before PASS: ${game._opStack.length}`);

game.play(PASS);

console.log(`  Current player after PASS: ${game.current}`);
console.log(`  Operation stack length after PASS: ${game._opStack.length}`);

// Check if group still exists
console.log(`\n  After PASS, checking group at ${stoneIdx}:`);
console.log(`    cells[${stoneIdx}] = ${game.cells[stoneIdx]}`);
console.log(`    _gid[${stoneIdx}] = ${game._gid[stoneIdx]}`);
const gid46 = game._gid[stoneIdx];
if (gid46 !== -1) {
  const lc46 = game.groupLibs2(stoneIdx).count;
  console.log(`    groupLibs2(${stoneIdx}).count = ${lc46}`);
}

console.log(`\nStep 2: Simulating _canReach3Libs (call play on each liberty)`);
for (const libIdx of libs) {
  console.log(`\n  Playing liberty ${libIdx}...`);
  console.log(`    Current before: ${game.current}`);
  console.log(`    Operation stack length before: ${game._opStack.length}`);

  const playResult = game.play(libIdx);
  console.log(`    play(${libIdx}) returned: ${playResult}`);
  console.log(`    Current after: ${game.current}`);
  console.log(`    Operation stack length after: ${game._opStack.length}`);

  // Check if original group still exists
  console.log(`    After play(${libIdx}), checking group at ${stoneIdx}:`);
  console.log(`      cells[${stoneIdx}] = ${game.cells[stoneIdx]}`);
  console.log(`      _gid[${stoneIdx}] = ${game._gid[stoneIdx]}`);
  if (game.cells[stoneIdx] !== 0) {
    const lcAfter = game.groupLibs2(stoneIdx).count;
    console.log(`      groupLibs2(${stoneIdx}).count = ${lcAfter}`);
  }

  console.log(`    Undoing...`);
  game.undo();
  console.log(`    Current after undo: ${game.current}`);
  console.log(`    Operation stack length after undo: ${game._opStack.length}`);

  // Check state after undo
  console.log(`    After undo, checking group at ${stoneIdx}:`);
  console.log(`      cells[${stoneIdx}] = ${game.cells[stoneIdx]}`);
  console.log(`      _gid[${stoneIdx}] = ${game._gid[stoneIdx]}`);
  if (game.cells[stoneIdx] !== 0) {
    const lcAfterUndo = game.groupLibs2(stoneIdx).count;
    console.log(`      groupLibs2(${stoneIdx}).count = ${lcAfterUndo}`);
  }
}

console.log(`\n\nStep 3: Undo the PASS`);
console.log(`  Current before undo PASS: ${game.current}`);
console.log(`  Operation stack length before undo PASS: ${game._opStack.length}`);

game.undo();

console.log(`  Current after undo PASS: ${game.current}`);
console.log(`  Operation stack length after undo PASS: ${game._opStack.length}`);

console.log(`\n\nFinal state after getLadderStatus simulation:`);
console.log(`  cells[${stoneIdx}] = ${game.cells[stoneIdx]}`);
console.log(`  _gid[${stoneIdx}] = ${game._gid[stoneIdx]}`);
if (game.cells[stoneIdx] !== 0) {
  const { count: lcFinal } = game.groupLibs2(stoneIdx);
  console.log(`  groupLibs2(${stoneIdx}).count = ${lcFinal}`);
}
