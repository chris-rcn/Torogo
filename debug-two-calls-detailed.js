const { Game3Precise, PASS } = require('./game3.js');
const { getLadderStatus } = require('./ladder2.js');

const game = new Game3Precise(13);

const moves = [45, 46, 47, 57, 58, 59, 70, 71, 72, 82, 83, 84];
for (const move of moves) {
  game.play(move);
}

console.log('Initial state:');
console.log(`  _ls[2] = ${game._ls[2]}`);
console.log(`  _ls[4] = ${game._ls[4]}`);
console.log(`  opStack.length = ${game._opStack.length}\n`);

console.log('First getLadderStatus call (gid 2 at index 46):');
console.log(`  before: _ls[2]=${game._ls[2]}, opStack=${game._opStack.length}`);
const status1 = getLadderStatus(game, 46);
console.log(`  after:  _ls[2]=${game._ls[2]}, opStack=${game._opStack.length}`);

console.log('\nSecond getLadderStatus call (gid 4 at index 57):');
console.log(`  before: _ls[4]=${game._ls[4]}, opStack=${game._opStack.length}`);
const status2 = getLadderStatus(game, 57);
console.log(`  after:  _ls[4]=${game._ls[4]}, opStack=${game._opStack.length}`);

console.log('\nFinal state:');
console.log(`  _ls[2] = ${game._ls[2]} (should be 2)`);
console.log(`  _ls[4] = ${game._ls[4]} (should be 2)`);

// Check if there are any atari groups left
const cap = game.N * game.N;
const visited = new Set();
let atariCount = 0;
for (let i = 0; i < cap; i++) {
  if (game.cells[i] === 0) continue;
  const gid = game._gid[i];
  if (visited.has(gid)) continue;
  visited.add(gid);
  const { count: lc } = game.groupLibs2(i);
  if (lc === 1 || lc === 2) {
    console.log(`  gid ${gid}: ${lc} liberties`);
    atariCount++;
  }
}
console.log(`  Total atari groups: ${atariCount}`);
