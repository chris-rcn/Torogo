const { Game3Precise, PASS } = require('./game3.js');

const game = new Game3Precise(9);

// Play a game to near end
const moves = [40, 41, 49, 48, 32, 31, 23, 24, 33, 34, 42, 43, 50, 51];
for (const move of moves) {
  if (game.isLegal(move)) game.play(move);
}

// Hook the play and undo functions to track moves
const originalPlay = game.play.bind(game);
const originalUndo = game.undo.bind(game);
let playStack = [];

game.play = function(move) {
  if (move !== PASS) {
    playStack.push(move);
    console.log(`[PLAY] move=${move}, stack depth=${playStack.length}, cells[24]=${this.cells[24]}`);
  } else {
    playStack.push(-1);
    console.log(`[PASS], stack depth=${playStack.length}, cells[24]=${this.cells[24]}`);
  }
  const result = originalPlay(move);
  if (result) {
    console.log(`       -> legal, cells[24]=${this.cells[24]}`);
  } else {
    console.log(`       -> ILLEGAL`);
    playStack.pop();
  }
  return result;
};

game.undo = function() {
  if (playStack.length > 0) {
    const move = playStack.pop();
    console.log(`[UNDO] move=${move}, stack depth=${playStack.length + 1}->${playStack.length}, cells[24]=${this.cells[24]}`);
  }
  const result = originalUndo();
  if (playStack.length < playStack.length + 1) {  // Just for logging
    console.log(`       -> undo done, cells[24]=${this.cells[24]}`);
  }
  return result;
};

// Now call getLadderStatus on the group at index 24
const { getLadderStatus } = require('./ladder2.js');

console.log('Starting getLadderStatus for group at index 24...\n');
const status = getLadderStatus(game, 24);
console.log(`\nFinished getLadderStatus, cells[24]=${game.cells[24]}`);
