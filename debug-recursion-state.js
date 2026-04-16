const { Game3Precise, PASS } = require('./game3-precise.js');

const game = new Game3Precise(9);

// Play a game to near end
const moves = [40, 41, 49, 48, 32, 31, 23, 24, 33, 34, 42, 43, 50, 51];
for (const move of moves) {
  if (game.isLegal(move)) game.play(move);
}

// Hook the undo function with full details
const originalUndo = game.undo.bind(game);
let undoDepth = 0;

game.undo = function() {
  undoDepth++;
  const indent = '  '.repeat(undoDepth);
  const topOp = this._opStack[this._opStack.length - 1];

  if (topOp && topOp.type === 'move' && topOp.captured && topOp.captured.length > 0) {
    console.log(`${indent}[UNDO move with capture]`);
    console.log(`${indent}  move=${topOp.move}, color=${topOp.color}`);
    for (const stone of topOp.captured) {
      const idx = stone.idx || stone;
      console.log(`${indent}    captured at ${idx}, will restore as ${-topOp.color}`);
    }
  }

  const before = this.cells[24];
  originalUndo();
  const after = this.cells[24];

  if (before !== after) {
    console.log(`${indent}  cells[24]: ${before} -> ${after}`);
  }

  undoDepth--;
};

// Manually simulate the problematic sequence
console.log('Initial: cells[24]=-1');

game.play(PASS);
console.log('After PASS: current=' + game.current);

// First time through the loop
game.play(15);
console.log('After play(15): cells[24]=' + game.cells[24]);

game.undo();
console.log('After undo: cells[24]=' + game.cells[24]);

// Second time through the loop - this is where the bug happens
console.log('\nSecond iteration:');
game.play(15);
console.log('After play(15): cells[24]=' + game.cells[24]);

game.play(25);
console.log('After play(25): cells[24]=' + game.cells[24] + ' (was -1, now 0?)');

game.undo();
console.log('After undo(25): cells[24]=' + game.cells[24] + ' (should be -1)');

game.undo();
console.log('After undo(15): cells[24]=' + game.cells[24]);

game.undo();
console.log('After undo(PASS): cells[24]=' + game.cells[24]);
