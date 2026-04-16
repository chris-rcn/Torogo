const { Game3Precise, PASS } = require('./game3.js');

const game = new Game3Precise(9);

// Play a game to near end
const moves = [40, 41, 49, 48, 32, 31, 23, 24, 33, 34, 42, 43, 50, 51];
for (const move of moves) {
  if (game.isLegal(move)) game.play(move);
}

console.log('Initial state: cells[24] = ' + game.cells[24] + ' (WHITE)');

// Hook undo to see what's being restored
const originalUndo = game.undo.bind(game);

game.undo = function() {
  const topOp = this._opStack[this._opStack.length - 1];
  if (topOp && topOp.type === 'move' && topOp.captured && topOp.captured.length > 0) {
    console.log(`\nUndoing move that captured stones:`);
    console.log(`  op.color = ${topOp.color}`);
    for (const stone of topOp.captured) {
      const idx = stone.idx || stone;
      const gid = stone.gid;
      const oppColor = -topOp.color;
      console.log(`    stone at ${idx}: gid=${gid}, will restore as color=${oppColor} (opposite of ${topOp.color})`);
    }
  }
  return originalUndo();
};

// Now play the problematic moves
console.log('\nPlaying PASS...');
game.play(PASS);

const { count: lc, lib0, lib1 } = game.groupLibs2(24);
console.log(`Group at 24 has liberties: ${lc} (${lib0}, ${lib1})`);

console.log('\nPlaying move 15...');
game.play(15);

console.log('\nUndoing move 15...');
game.undo();

console.log('\nPlaying move 15 again...');
game.play(15);

console.log('\nPlaying move 25 (this should capture the group at 24)...');
game.play(25);
console.log(`After playing 25: cells[24] = ${game.cells[24]}`);

console.log('\nUndoing move 25...');
game.undo();
console.log(`After undo: cells[24] = ${game.cells[24]} (should be -1)`);

if (game.cells[24] !== -1) {
  console.log('\nERROR: cells[24] is ' + game.cells[24] + ', should be -1!');
}
