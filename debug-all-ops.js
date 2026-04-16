const { Game3Precise, PASS } = require('./game3.js');
const { getLadderStatus } = require('./ladder2.js');

const game = new Game3Precise(13);

const moves = [45, 46, 47, 57, 58, 59, 70, 71, 72, 82, 83, 84];
for (const move of moves) {
  game.play(move);
}

const initialOpCount = game._opStack.length;
let operationLog = [];

// Intercept _undoOperation to see what gets undone
const originalUndoOp = game._undoOperation.bind(game);
let undoCount = 0;

game._undoOperation = function(op) {
  if (op.type === 'mergeGroups' || op.gid === 2 || (op.idx !== undefined && op.gid === 2)) {
    console.log(`[UNDO] ${op.type}` + (op.gid === 2 ? ` gid=2` : '') + (op.idx !== undefined ? ` idx=${op.idx}` : '') + ` -> _ls[2]=${this._ls[2]}`);
  }
  undoCount++;
  return originalUndoOp(op);
};

console.log(`Initial state: _ls[2] = ${game._ls[2]}\n`);
console.log(`Calling getLadderStatus(game, 46)...\n`);

const status = getLadderStatus(game, 46);

console.log(`\nFinal state: _ls[2] = ${game._ls[2]} (expected 2)`);
console.log(`\nTotal operations undone: ${undoCount}`);
console.log(`Operation stack length before: ${initialOpCount}`);
console.log(`Operation stack length after: ${game._opStack.length}`);
