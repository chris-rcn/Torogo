const { Game3Precise } = require('./game3-precise.js');

const game = new Game3Precise(13);

const moves = [45, 46, 47, 57, 58, 59, 70, 71, 72, 82, 83, 84];
for (const move of moves) {
  game.play(move);
}

console.log('Initial state:');
const visited = new Set();
for (let i = 0; i < 169; i++) {
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

console.log('\nNow simulating getAllLadderStatuses step by step...');

function getLadderStatus(game, stoneIdx) {
  const { count: lc, lib0, lib1 } = game.groupLibs2(stoneIdx);
  if (lc < 1 || lc > 2) return null;

  const libs = lc === 1 ? [lib0] : [lib0, lib1];
  const gColor = game.cells[stoneIdx];
  const mover = game.current;
  const defending = gColor === mover;

  let escape;
  if (defending && lc === 1) {
    escape = false;
  } else {
    game.play(-1); // PASS
    // Just check if we can reach 3 libs (simplified)
    const gid = game._gid[stoneIdx];
    escape = gid !== -1;
    game.undo();
  }

  return { libs, defending };
}

const cap = game.N * game.N;
const results = [];
const visited2 = new Set();
let callCount = 0;

for (let i = 0; i < cap; i++) {
  if (game.cells[i] === 0) continue;
  const gid = game._gid[i];
  if (visited2.has(gid)) continue;
  visited2.add(gid);
  const { count: lc } = game.groupLibs2(i);
  if (lc === 0 || lc > 2) continue;

  callCount++;
  console.log(`\nCall ${callCount}: Analyzing group gid ${gid} at index ${i}`);
  const status = getLadderStatus(game, i);
  if (status) {
    results.push({ gid, status });
    console.log(`  ✓ Analyzed successfully`);
  }

  // Check state after each call
  console.log(`  Checking groups after call ${callCount}:`);
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
}

console.log(`\nFinal result: ${results.length} groups analyzed`);
