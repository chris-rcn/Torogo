const { Game3Precise } = require('./game3-precise.js');

const game = new Game3Precise(13);
const moves = [45, 46, 47, 57, 58, 59, 70, 71, 72, 82, 83, 84];
for (const move of moves) {
  game.play(move);
}

console.log('After initial moves:');
console.log('  moveCount:', game.moveCount);
console.log('  _opStack length:', game._opStack.length);

// Check the last move operation
const lastOp = game._opStack[game._opStack.length - 1];
console.log('  Last op type:', lastOp.type);
if (lastOp.type === 'move') {
  console.log('  Captured:', lastOp.captured);
}

// Manually do what getAllLadderStatuses does
console.log('\nManually calling play(PASS) and checking state:');
console.log('Before play(PASS):');
for (let i = 0; i < 169; i++) {
  if (game.cells[i] !== 0) {
    const gid = game._gid[i];
    if ((gid === 2 || gid === 4) && i < 100) {
      console.log(`  Cell ${i}: color=${game.cells[i]}, gid=${gid}`);
    }
  }
}

game.play(-1); // PASS
console.log('\nAfter play(PASS), opStack length:', game._opStack.length);

game.undo();
console.log('\nAfter undo(), opStack length:', game._opStack.length);

console.log('After undo:');
for (let i = 0; i < 169; i++) {
  if (game.cells[i] !== 0) {
    const gid = game._gid[i];
    if ((gid === 2 || gid === 4) && i < 100) {
      console.log(`  Cell ${i}: color=${game.cells[i]}, gid=${gid}`);
    }
  }
}

// Check if gid 2 and 4 still have stones
console.log('\nChecking group membership:');
let gid2count = 0, gid4count = 0;
for (let i = 0; i < 169; i++) {
  if (game._gid[i] === 2) gid2count++;
  if (game._gid[i] === 4) gid4count++;
}
console.log(`  gid 2: ${gid2count} stones`);
console.log(`  gid 4: ${gid4count} stones`);
