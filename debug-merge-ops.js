const { Game3Precise, PASS } = require('./game3-precise.js');

const game = new Game3Precise(13);

const moves = [45, 46, 47, 57, 58, 59, 70, 71, 72, 82, 83, 84];
for (const move of moves) {
  game.play(move);
}

// Trap mergeGroups calls
const originalPlay = game.play.bind(game);
const mergeOps = [];

game.play = function(move) {
  const beforeOpCount = this._opStack.length;
  const result = originalPlay(move);
  const afterOpCount = this._opStack.length;

  // Check if any mergeGroups operations were added
  for (let i = beforeOpCount; i < afterOpCount; i++) {
    if (this._opStack[i].type === 'mergeGroups') {
      mergeOps.push({
        move,
        mainGid: this._opStack[i].mainGid,
        otherId: this._opStack[i].otherId,
      });
      console.log(`[play(${move})] Merged groups: ${this._opStack[i].mainGid} <- ${this._opStack[i].otherId}`);
    }
  }

  return result;
};

console.log('Testing getLadderStatus...\n');

// Use _canReach3Libs logic directly
const { getLadderStatus } = require('./ladder2.js');

const stoneIdx = 46;
console.log(`Analyzing stone at ${stoneIdx} (gid 2):`);
const status = getLadderStatus(game, stoneIdx);

console.log(`\nMerge operations during getLadderStatus: ${mergeOps.length}`);
for (const op of mergeOps) {
  console.log(`  move ${op.move}: merged ${op.mainGid} <- ${op.otherId}`);
}

console.log(`\nFinal state: _ls[2] = ${game._ls[2]} (should be 2)`);
