const { Game3Precise } = require('./game3-precise.js');

const game = new Game3Precise(13);

const moves = [45, 46, 47, 57, 58, 59, 70, 71, 72, 82, 83, 84];
for (const move of moves) {
  game.play(move);
}

console.log('Initial state:');
function showGroups(label) {
  const visited = new Set();
  const groups = [];
  for (let i = 0; i < 169; i++) {
    if (game.cells[i] !== 0) {
      const gid = game._gid[i];
      if (!visited.has(gid)) {
        visited.add(gid);
        const { count: lc } = game.groupLibs2(i);
        if (lc === 1 || lc === 2) {
          groups.push({ gid, lc, idx: i });
        }
      }
    }
  }
  console.log(`${label}: ${groups.length} atari groups`, groups.map(g => `gid${g.gid}(${g.lc}libs)`).join(', '));
  return groups;
}

let groups = showGroups('Before');
const atariIdx = groups[0].idx;

console.log(`\nAnalyzing group at idx ${atariIdx}...`);
console.log(`Operation stack length before play(PASS): ${game._opStack.length}`);

// Simulate what getLadderStatus does
console.log('\nPlaying PASS...');
game.play(-1);  // PASS
console.log(`  Operation stack length after play(PASS): ${game._opStack.length}`);
console.log(`  Last operation:`, game._opStack[game._opStack.length - 1]);

// Now simulate what _canReach3Libs would do
console.log('\nSimulating _canReach3Libs with recursive play/undo...');
const lib = game.groupLibs2(atariIdx).lib0;
console.log(`  Stone at ${atariIdx} has lib0=${lib}`);

console.log(`  Operation stack before play(${lib}): ${game._opStack.length}`);
const playResult = game.play(lib);
console.log(`  play(${lib}) returned: ${playResult}`);
console.log(`  Operation stack after play(${lib}): ${game._opStack.length}`);

console.log(`  Now undoing...`);
game.undo();
console.log(`  Operation stack after undo: ${game._opStack.length}`);
console.log(`  Current after undo: ${game.current}`);

// Check group state after undo
console.log(`\nGroup state after undo in _canReach3Libs simulation:`);
showGroups('  During PASS (after internal undo)');

// Now undo the PASS
console.log(`\nUndoing the PASS...`);
game.undo();
console.log(`  Operation stack length after undo PASS: ${game._opStack.length}`);

// Check final state
console.log(`\nFinal group state:`);
showGroups('After');
