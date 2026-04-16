const { Game3Precise, PASS } = require('./game3.js');

const game = new Game3Precise(13);

const moves = [45, 46, 47, 57, 58, 59, 70, 71, 72, 82, 83, 84];
for (const move of moves) {
  game.play(move);
}

function checkAllGroups(label) {
  const cap = game.N * game.N;
  const visited = new Set();
  console.log(`${label}:`);
  for (let i = 0; i < cap; i++) {
    if (game.cells[i] === 0) continue;
    const gid = game._gid[i];
    if (!visited.has(gid)) {
      visited.add(gid);
      const { count: lc } = game.groupLibs2(i);
      if (lc === 1 || lc === 2) {
        console.log(`  gid ${gid}: ${lc} liberties`);
      }
    }
  }
}

checkAllGroups('BEFORE ANY CALLS');

// Simulate first getLadderStatus call on gid 2 at index 46
console.log('\n=== First getLadderStatus call (gid 2 at index 46) ===');

const idx1 = 46;
let { count: lc1, lib0: lib0_1, lib1: lib1_1 } = game.groupLibs2(idx1);
const libs1 = lc1 === 1 ? [lib0_1] : [lib0_1, lib1_1];
const gColor1 = game.cells[idx1];
const defending1 = gColor1 === game.current;

console.log(`\nPlay PASS...`);
game.play(PASS);
console.log(`After PASS: current=${game.current}, opStack length=${game._opStack.length}`);

// Simulate one iteration of the loop in _canReach3Libs
console.log(`\nPlay liberty ${lib0_1}...`);
game.play(lib0_1);
console.log(`After play(${lib0_1}): opStack length=${game._opStack.length}`);
console.log(`  gid2 at idx46: cells=${game.cells[idx1]}, _gid=${game._gid[idx1]}, lc=${game.groupLibs2(idx1).count}`);

console.log(`Undo...`);
game.undo();
console.log(`After undo: opStack length=${game._opStack.length}`);
console.log(`  gid2 at idx46: cells=${game.cells[idx1]}, _gid=${game._gid[idx1]}, lc=${game.groupLibs2(idx1).count}`);

console.log(`\nUndo PASS...`);
game.undo();
console.log(`After undo PASS: opStack length=${game._opStack.length}, current=${game.current}`);

checkAllGroups('AFTER FIRST CALL');

// Simulate second getLadderStatus call on gid 4 at index 57
console.log('\n\n=== Second getLadderStatus call (gid 4 at index 57) ===');

const idx2 = 57;
console.log(`\nBefore second call, checking all atari groups...`);
const cap = game.N * game.N;
const visited2 = new Set();
for (let i = 0; i < cap; i++) {
  if (game.cells[i] === 0) continue;
  const gid = game._gid[i];
  if (visited2.has(gid)) continue;
  visited2.add(gid);
  const { count: lc } = game.groupLibs2(i);
  if (lc === 1 || lc === 2) {
    console.log(`  gid ${gid} at idx ${i}: ${lc} liberties`);
  }
}

let { count: lc2, lib0: lib0_2, lib1: lib1_2 } = game.groupLibs2(idx2);
console.log(`\ngroupLibs2(${idx2}) = {count: ${lc2}, lib0: ${lib0_2}, lib1: ${lib1_2}}`);
const libs2 = lc2 === 1 ? [lib0_2] : [lib0_2, lib1_2];
const gColor2 = game.cells[idx2];
const defending2 = gColor2 === game.current;

console.log(`\nPlay PASS...`);
game.play(PASS);
console.log(`After PASS: current=${game.current}, opStack length=${game._opStack.length}`);

// Simulate one iteration of the loop in _canReach3Libs
console.log(`\nPlay liberty ${lib0_2}...`);
game.play(lib0_2);
console.log(`After play(${lib0_2}): opStack length=${game._opStack.length}`);
console.log(`  gid4 at idx57: cells=${game.cells[idx2]}, _gid=${game._gid[idx2]}, lc=${game.groupLibs2(idx2).count}`);

// Check if gid 2 is still there
console.log(`\n  Checking gid 2 (should still exist):`);
for (let i = 0; i < cap; i++) {
  if (game.cells[i] === 1) { // BLACK stone
    const gid = game._gid[i];
    if (gid === 2) {
      console.log(`    Found gid 2 at index ${i}: lc=${game.groupLibs2(i).count}`);
      break;
    }
  }
}

console.log(`\nUndo...`);
game.undo();
console.log(`After undo: opStack length=${game._opStack.length}`);
console.log(`  gid4 at idx57: cells=${game.cells[idx2]}, _gid=${game._gid[idx2]}, lc=${game.groupLibs2(idx2).count}`);

console.log(`\nUndo PASS...`);
game.undo();
console.log(`After undo PASS: opStack length=${game._opStack.length}, current=${game.current}`);

checkAllGroups('AFTER SECOND CALL');
