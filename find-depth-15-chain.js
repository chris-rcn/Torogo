#!/usr/bin/env node
'use strict';

// Find a chain that requires depth 15 to solve
// (inconclusive at depth 10, definitive at depth 15)

const { Game2, coordStr } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');
const { searchChains } = require('./tactics3.js');

console.log('Searching for chains requiring depth 15 to solve...\n');

let found = false;
let gamesChecked = 0;

for (let gameNum = 0; gameNum < 200 && !found; gameNum++) {
  const g2 = new Game2(13);
  let consecutivePasses = 0;
  let moveCount = 0;

  while (consecutivePasses < 1 && !g2.gameOver && moveCount < 200 && !found) {
    const g3 = game3FromGame2(g2);

    // Search with depth 10 and depth 15
    const tactics10 = searchChains(g3, 10000, 10);
    const tactics15 = searchChains(g3, 10000, 15);

    // Create maps for comparison
    const map10 = new Map();
    const map15 = new Map();

    for (const t of tactics10) {
      map10.set(t.gid, t);
    }
    for (const t of tactics15) {
      map15.set(t.gid, t);
    }

    // Search with depth 14 as well
    const tactics14 = searchChains(g3, 10000, 14);
    const map14 = new Map();
    for (const t of tactics14) {
      map14.set(t.gid, t);
    }

    // Find chains that are inconclusive at depth 14 but definitive at depth 15
    for (const [gid, tactic15] of map15) {
      const tactic14 = map14.get(gid);

      if (tactic14 && tactic15.status && tactic14.status) {
        const result14 = tactic14.status.moverSucceeds;
        const result15 = tactic15.status.moverSucceeds;

        // Check if depth 14 was inconclusive but depth 15 is definitive
        if (result14 === null && result15 !== null) {
          // Count stones in the group
          let groupSize = 0;
          for (let i = 0; i < 169; i++) {
            if (g3._gid[i] === gid) groupSize++;
          }

          // Prefer larger groups (size > 1)
          if (groupSize > 1 || gameNum > 150) {
            console.log(`✓ Found chain requiring depth 15 (unsolvable at depth 14)!`);
            console.log(`Game ${gameNum}, Move ${moveCount}`);
            console.log(`Group ID: ${gid}, Color: ${tactic15.color === 1 ? 'BLACK' : 'WHITE'}, Size: ${groupSize} stones`);
            console.log(`Depth 14 result: inconclusive (null)`);
            console.log(`Depth 15 result: ${result15 ? 'wins' : 'loses'}\n`);

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

            // Display the board with axis labels, marking the chain stone
            console.log('\nBoard:');
            console.log(g3.toString(chainStoneIdx, { axisLabels: true }));

            // Show whose turn it is
            console.log(`\nCurrent player: ${g3.current === 1 ? 'BLACK' : 'WHITE'}`);
            console.log(`Defending color: ${tactic15.color === 1 ? 'BLACK' : 'WHITE'}`);

            // Show the status
            console.log(`\nChain Analysis:`);
            console.log(`- Liberties: ${tactic15.status.libs.length}`);
            console.log(`- Mover succeeds: ${result15}`);
            console.log(`- Urgent liberties: ${tactic15.status.urgentLibs.join(', ') || 'none'}`);
            console.log(`- Color: ${tactic15.color === 1 ? 'BLACK' : 'WHITE'}`);

            // Save position for debugging
            const fs = require('fs');
            const cells = Array.from(g3.cells);
            const gids = Array.from(g3._gid);
            fs.writeFileSync('debug-position.json', JSON.stringify({
              gameNum, moveCount, gid, groupColor: tactic15.color,
              groupSize: Array.from(gids).filter(g => g === gid).length,
              cells, gids, board: g3.toString(undefined, { axisLabels: true })
            }, null, 2));

            // Find a stone in the group for details
            for (let i = 0; i < 169; i++) {
              if (g3._gid[i] === gid) {
                console.log(`\nChain position: ${coordStr(i, 13)}`);
                break;
              }
            }

            console.log(`\nPosition saved to debug-position.json for verification`);
            break;
          }
        }
      }
    }

    // Play a random move
    const move = g2.randomLegalMove();
    if (move !== -1) {
      g2.play(move);
      consecutivePasses = 0;
      moveCount++;
    } else {
      // No legal moves, play pass
      g2.play(-1);
      consecutivePasses++;
      moveCount++;
    }
  }

  if (!found) {
    gamesChecked++;
    if (gamesChecked % 10 === 0) {
      console.log(`Checked ${gamesChecked} games...`);
    }
  }
}


if (!found) {
  console.log(`\nNo chains found requiring depth 15 after checking ${gamesChecked} games`);
  process.exit(1);
}
