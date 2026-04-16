const { Game2 } = require('./game2.js');
const { Game3Precise } = require('./game3.js');

const game2 = new Game2(13);
const game3 = new Game3Precise(13);

// Play moves until we hit a ko situation
let moveIdx = 0;
for (let move = 0; move < 169 && moveIdx < 35; move++) {
  if (game2.isLegal(move)) {
    const legal2 = game2.play(move);
    const legal3 = game3.play(move);
    moveIdx++;

    if (moveIdx === 28 || moveIdx === 30) {
      console.log(`\nAfter move ${moveIdx} at index ${move}:`);
      console.log(`  game2.ko = ${game2.ko}`);
      console.log(`  game3.ko = ${game3.ko}`);
    }
  }
}

console.log('\nDetails for moves 28-30:');
console.log(`game2.koStone = ${game2.koStone}`);
