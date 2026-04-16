const { Game3Precise } = require('./game3.js');

const game = new Game3Precise(9);

// Play a game to near end
const moves = [40, 41, 49, 48, 32, 31, 23, 24, 33, 34, 42, 43, 50, 51];
for (const move of moves) {
  if (game.isLegal(move)) game.play(move);
}

// Check the _gc array
console.log('_gc array values:');
for (let gid = 0; gid < 10; gid++) {
  const val = game._gc[gid];
  console.log(`  _gc[${gid}] = ${val} (raw bytes: ${game._gc.buffer.byteLength})`);
}

// Check how Int8Array interprets values
const testArray = new Int8Array(5);
testArray[0] = 255;
testArray[1] = -1;
testArray[2] = 1;
testArray[3] = 0;
console.log('\nTest Int8Array:');
console.log(`  testArray[0] = 255 -> ${testArray[0]}`);
console.log(`  testArray[1] = -1 -> ${testArray[1]}`);
console.log(`  testArray[2] = 1 -> ${testArray[2]}`);
console.log(`  testArray[3] = 0 -> ${testArray[3]}`);

// Check what _gc[6] actually is
console.log(`\n_gc[6] detailed:`);
console.log(`  typeof _gc[6] = ${typeof game._gc[6]}`);
console.log(`  _gc[6] === 255? ${game._gc[6] === 255}`);
console.log(`  _gc[6] === -1? ${game._gc[6] === -1}`);
console.log(`  _gc[6] & 0xFF = ${game._gc[6] & 0xFF}`);
