const { Game3Precise } = require('./game3.js');
const { getLadderStatus } = require('./ladder2.js');

const game = new Game3Precise(9);

// Play a game to near end
const moves = [40, 41, 49, 48, 32, 31, 23, 24, 33, 34, 42, 43, 50, 51];
for (const move of moves) {
  if (game.isLegal(move)) game.play(move);
}

// Check the specific cells that change
const cells24Before = game.cells[24];
const cells41Before = game.cells[41];
const cells51Before = game.cells[51];

console.log('Before getAllLadderStatuses:');
console.log(`  cells[24] = ${cells24Before} (${cells24Before === 1 ? 'BLACK' : cells24Before === -1 ? 'WHITE' : 'EMPTY'})`);
console.log(`  cells[41] = ${cells41Before} (${cells41Before === 1 ? 'BLACK' : cells41Before === -1 ? 'WHITE' : 'EMPTY'})`);
console.log(`  cells[51] = ${cells51Before} (${cells51Before === 1 ? 'BLACK' : cells51Before === -1 ? 'WHITE' : 'EMPTY'})`);

// Check groups that own these cells
console.log('\nGroup ownership:');
for (const idx of [24, 41, 51]) {
  const gid = game._gid[idx];
  const color = game._gc[gid];
  if (gid !== -1) {
    const libCount = game.groupLibs2(idx).count;
    console.log(`  cells[${idx}]: gid=${gid}, color=${color}, libCount=${libCount}`);
  }
}

// Manually call getLadderStatus for one group
const cap = game.N * game.N;
const visited = new Set();
let processCount = 0;

for (let i = 0; i < cap; i++) {
  if (game.cells[i] === 0) continue;
  const gid = game._gid[i];
  if (visited.has(gid)) continue;
  visited.add(gid);
  const { count: lc } = game.groupLibs2(i);
  if (lc === 0 || lc > 2) continue;

  processCount++;
  console.log(`\nProcessing group ${gid} at index ${i}...`);

  // Check cells before
  const cb24 = game.cells[24];
  const cb41 = game.cells[41];
  const cb51 = game.cells[51];

  const status = getLadderStatus(game, i);

  // Check cells after
  const ca24 = game.cells[24];
  const ca41 = game.cells[41];
  const ca51 = game.cells[51];

  if (cb24 !== ca24 || cb41 !== ca41 || cb51 !== ca51) {
    console.log('  Cells changed!');
    if (cb24 !== ca24) console.log(`    cells[24]: ${cb24} -> ${ca24}`);
    if (cb41 !== ca41) console.log(`    cells[41]: ${cb41} -> ${ca41}`);
    if (cb51 !== ca51) console.log(`    cells[51]: ${cb51} -> ${ca51}`);
  } else {
    console.log('  No changes to tracked cells');
  }

  if (processCount >= 3) break;
}

console.log('\n\nFinal cells:');
console.log(`  cells[24] = ${game.cells[24]}`);
console.log(`  cells[41] = ${game.cells[41]}`);
console.log(`  cells[51] = ${game.cells[51]}`);
