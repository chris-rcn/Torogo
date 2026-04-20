#!/usr/bin/env node
'use strict';

const { Game2, BLACK, WHITE, coordStr } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');
const { searchChains } = require('./tactics3.js');

console.log('Searching for chains requiring depth 21 to solve...\n');

let found = false;
let gamesChecked = 0;

for (let gameNum = 0; gameNum < 500 && !found; gameNum++) {
  const g2 = new Game2(13);
  let consecutivePasses = 0;
  let moveCount = 0;

  while (consecutivePasses < 1 && !g2.gameOver && moveCount < 200 && !found) {
    const g3 = game3FromGame2(g2);

    // Search with depth 20 and depth 21
    console.log(`Game ${gameNum}, Move ${moveCount}: Searching depth 20...`);
    const tactics20 = searchChains(g3, 10000, 20);

    console.log(`Game ${gameNum}, Move ${moveCount}: Searching depth 21...`);
    const tactics21 = searchChains(g3, 10000, 21);

    // Create maps for comparison
    const map20 = new Map();
    const map21 = new Map();

    for (const t of tactics20) {
      map20.set(t.gid, t);
    }
    for (const t of tactics21) {
      map21.set(t.gid, t);
    }

    // Find chains that are inconclusive at depth 20 but definitive at depth 21
    for (const [gid, tactic21] of map21) {
      const tactic20 = map20.get(gid);

      if (tactic20 && tactic21.status && tactic20.status) {
        const result20 = tactic20.status.moverSucceeds;
        const result21 = tactic21.status.moverSucceeds;

        // Check if depth 20 was inconclusive but depth 21 is definitive
        if (result20 === null && result21 !== null) {
          // Count stones in the group
          let groupSize = 0;
          for (let i = 0; i < 169; i++) {
            if (g3._gid[i] === gid) groupSize++;
          }

          console.log(`\n✓ Found chain requiring depth 21 (unsolvable at depth 20)!`);
          console.log(`Game ${gameNum}, Move ${moveCount}`);
          console.log(`Group ID: ${gid}, Color: ${tactic21.color === 1 ? 'BLACK' : 'WHITE'}, Size: ${groupSize} stones`);
          console.log(`Depth 20 result: inconclusive (null)`);
          console.log(`Depth 21 result: ${result21 ? 'wins' : 'loses'}\n`);

          // Save this position
          found = true;

          // Find a stone in the chain to mark
          let chainStoneIdx = null;
          for (let i = 0; i < 169; i++) {
            if (g3._gid[i] === gid) {
              chainStoneIdx = i;
              break;
            }
          }

          // Display the board with the group marked
          console.log('Board position:');
          console.log(g3.toString(chainStoneIdx));
          console.log();

          // Show all critical liberties for this group
          const chain = [];
          for (let i = 0; i < 169; i++) {
            if (g3._gid[i] === gid) {
              chain.push(i);
            }
          }

          console.log(`Chain stones at: ${chain.map(i => coordStr(i, 13)).join(', ')}`);

          // Get liberties
          const liberties = new Set();
          const N = 13;
          const nbr = g3._nbr;
          for (const idx of chain) {
            for (let i = 0; i < 4; i++) {
              const ni = nbr[idx * 4 + i];
              if (ni >= 0 && g3.cells[ni] === 0) {
                liberties.add(ni);
              }
            }
          }

          console.log(`Liberties (${liberties.size}): ${Array.from(liberties).map(i => coordStr(i, 13)).join(', ')}`);
          console.log();

          process.exit(0);
        }
      }
    }

    gamesChecked++;

    const move = g2.randomLegalMove();
    if (move !== -1) {
      g2.play(move);
      consecutivePasses = 0;
      moveCount++;
    } else {
      g2.play(-1);
      consecutivePasses++;
      moveCount++;
    }
  }
}

if (!found) {
  console.log(`\nNo depth-21 required chains found in ${gamesChecked} positions.`);
  process.exit(1);
}
