const { Game2 } = require('./game2.js');
const { Game3Precise } = require('./game3.js');

const game2 = new Game2(13);
const game3 = new Game3Precise(13);

console.log('Initial state comparison:\n');

// Check board
console.log('Board state:');
for (let i = 0; i < 169; i++) {
  if (game2.cells[i] !== 0) {
    console.log(`  cells[${i}]: game2=${game2.cells[i]}, game3=${game3.cells[i]}`);
  }
}

// Check the center stone groups
console.log('\nCenter stone (should be at 84 for 13x13):');
const center = 6 * 13 + 6;
console.log(`  Center index: ${center}`);
console.log(`  game2.cells[${center}]: ${game2.cells[center]}`);
console.log(`  game3.cells[${center}]: ${game3.cells[center]}`);

console.log('\nGroup info for center stone:');
const gid2 = game2._gid[center];
const gid3 = game3._gid[center];
console.log(`  game2: gid=${gid2}, color=${game2._gc[gid2]}, size=${game2.groupSize(gid2)}`);
console.log(`  game3: gid=${gid3}, color=${game3._gc[gid3]}, size=${game3.groupSize(gid3)}`);

const libs2 = game2.groupLibs2(center);
const libs3 = game3.groupLibs2(center);
console.log(`  game2: liberties count=${libs2.count}, lib0=${libs2.lib0}, lib1=${libs2.lib1}`);
console.log(`  game3: liberties count=${libs3.count}, lib0=${libs3.lib0}, lib1=${libs3.lib1}`);

console.log('\nAll groups:');
const visited2 = new Set();
const visited3 = new Set();

for (let i = 0; i < 169; i++) {
  if (game2.cells[i] === 0) continue;
  const gid = game2._gid[i];
  if (!visited2.has(gid)) {
    visited2.add(gid);
    const size = game2.groupSize(gid);
    const libCount = game2.groupLibs2(i).count;
    const color = game2._gc[gid];
    console.log(`  game2 gid ${gid}: color=${color}, size=${size}, libs=${libCount}`);
  }
}

for (let i = 0; i < 169; i++) {
  if (game3.cells[i] === 0) continue;
  const gid = game3._gid[i];
  if (!visited3.has(gid)) {
    visited3.add(gid);
    const size = game3.groupSize(gid);
    const libCount = game3.groupLibs2(i).count;
    const color = game3._gc[gid];
    console.log(`  game3 gid ${gid}: color=${color}, size=${size}, libs=${libCount}`);
  }
}
