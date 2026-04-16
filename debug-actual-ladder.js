const { Game3Precise } = require('./game3-precise.js');
const { getLadderStatus, getAllLadderStatuses } = require('./ladder2.js');

const game = new Game3Precise(13);

const moves = [45, 46, 47, 57, 58, 59, 70, 71, 72, 82, 83, 84];
for (const move of moves) {
  game.play(move);
}

console.log('Initial state:');
const cap = game.N * game.N;
const visited = new Set();
for (let i = 0; i < cap; i++) {
  if (game.cells[i] === 0) continue;
  const gid = game._gid[i];
  if (!visited.has(gid)) {
    visited.add(gid);
    const { count: lc } = game.groupLibs2(i);
    if (lc === 1 || lc === 2) {
      console.log(`  Found atari group: gid ${gid} at index ${i}, liberties=${lc}`);
    }
  }
}

console.log('\nNow calling getAllLadderStatuses step by step...');

const results = [];
const visited2 = new Set();
let callCount = 0;

for (let i = 0; i < cap; i++) {
  if (game.cells[i] === 0) continue;
  const gid = game._gid[i];
  if (visited2.has(gid)) continue;
  visited2.add(gid);

  console.log(`\nBefore group gid ${gid} at index ${i}:`);
  const { count: lc } = game.groupLibs2(i);
  console.log(`  groupLibs2(${i}).count = ${lc}`);

  if (lc === 0 || lc > 2) {
    console.log(`  Skipping (lc not in [1,2])`);
    continue;
  }

  callCount++;
  console.log(`Call ${callCount}: Calling getLadderStatus(game, ${i})`);
  const status = getLadderStatus(game, i);

  if (status) {
    results.push({ gid, status });
    console.log(`  ✓ Status: ${JSON.stringify(status)}`);
  }

  // Check state after each call
  console.log(`After getLadderStatus for gid ${gid}:`);
  const visited3 = new Set();
  let count = 0;
  for (let j = 0; j < cap; j++) {
    if (game.cells[j] === 0) continue;
    const gid2 = game._gid[j];
    if (!visited3.has(gid2)) {
      visited3.add(gid2);
      const { count: lc2 } = game.groupLibs2(j);
      if (lc2 === 1 || lc2 === 2) {
        console.log(`    gid ${gid2}: ${lc2} liberties`);
        count++;
      }
    }
  }
  console.log(`    Total atari groups: ${count}`);

  if (count === 0) {
    console.log('ERROR: All atari groups disappeared!');
    break;
  }
}

console.log(`\nFinal result: ${results.length} groups analyzed`);
