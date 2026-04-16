const { Game2 } = require('./game2.js');
const { Game3Precise } = require('./game3-precise.js');

function buildPosition(GameClass, N) {
  const game = new GameClass(N);
  const moves = [10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33];
  for (const move of moves) {
    if (game.isLegal(move)) game.play(move);
  }
  return game;
}

const game2 = buildPosition(Game2, 13);
const game3 = buildPosition(Game3Precise, 13);

console.log('Isolated play/undo overhead:');
console.log('='.repeat(60));

// Test 1: Clone + play (Game2)
console.log('\nGame2: clone() + play()');
let start = process.hrtime.bigint();
for (let i = 0; i < 10000; i++) {
  const g = game2.clone();
  g.play(15);
}
let end = process.hrtime.bigint();
let time2_clone_play = Number(end - start) / 1e6;
console.log(`10000x: ${time2_clone_play.toFixed(2)}ms, ${(time2_clone_play/10000).toFixed(4)}ms per op`);

// Test 2: Play + undo (Game3-Precise)
console.log('\nGame3-Precise: play() + undo()');
start = process.hrtime.bigint();
for (let i = 0; i < 10000; i++) {
  game3.play(15);
  game3.undo();
}
end = process.hrtime.bigint();
let time3_play_undo = Number(end - start) / 1e6;
console.log(`10000x: ${time3_play_undo.toFixed(2)}ms, ${(time3_play_undo/10000).toFixed(4)}ms per op`);

console.log('\n' + '='.repeat(60));
let ratio = (time2_clone_play / time3_play_undo).toFixed(2);
let improvement = ((time2_clone_play - time3_play_undo) / time2_clone_play * 100).toFixed(1);
console.log(`Speedup: ${ratio}x (${improvement}% improvement)`);
