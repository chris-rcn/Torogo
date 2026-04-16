const { Game3Precise } = require('./game3-precise.js');

const game = new Game3Precise(9);

// Play a game to near end
const moves = [40, 41, 49, 48, 32, 31, 23, 24, 33, 34, 42, 43, 50, 51];
for (const move of moves) {
  if (game.isLegal(move)) game.play(move);
}

// Check all groups
console.log('Groups on board:');
const cap = game.N * game.N;
const visited = new Set();
for (let i = 0; i < cap; i++) {
  if (game.cells[i] === 0) continue;
  const gid = game._gid[i];
  if (visited.has(gid)) continue;
  visited.add(gid);
  const color = game._gc[gid];
  const size = game.groupSize(gid);
  const { count: lc } = game.groupLibs2(i);
  const colorStr = color === 1 ? 'BLACK' : color === -1 ? 'WHITE' : `???(${color})`;
  console.log(`  gid ${gid}: color=${colorStr}, size=${size}, libs=${lc}`);
}

// Check what happens during analysis of gid 6
console.log(`\n\nDetailed analysis of gid=6 at index 24:`);
console.log(`  cells[24] before = ${game.cells[24]}`);
console.log(`  _gc[6] = ${game._gc[6]}`);

// Check all stones in gid 6
console.log(`  Stones in group 6:`);
const W = game._W;
const gb = 6 * W;
let stoneCount = 0;
for (let wi = 0; wi < W; wi++) {
  let w = game._sw[gb + wi];
  while (w) {
    const bit = 31 - Math.clz32(w & -w);
    const idx = wi * 32 + bit;
    if (idx < 81) {
      console.log(`    idx ${idx}: cells[${idx}] = ${game.cells[idx]}`);
      stoneCount++;
    }
    w &= w - 1;
  }
}
console.log(`  Total stones: ${stoneCount}`);
