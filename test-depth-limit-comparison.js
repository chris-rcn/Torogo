#!/usr/bin/env node
'use strict';

// Compare solve ratio (definitive vs inconclusive) with different depth limits

const { Game2 } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');
const { searchChains } = require('./tactics3.js');

function runTestWithDepthLimit(depthLimit, numGames = 5) {
  let totalGroups = 0;
  let definitiveCount = 0;
  let inconclusiveCount = 0;
  let totalMoves = 0;

  for (let gameNum = 0; gameNum < numGames; gameNum++) {
    const g2 = new Game2(13);
    let consecutivePasses = 0;
    let movesThisGame = 0;

    // Play until first pass (game end)
    while (consecutivePasses < 1 && !g2.gameOver) {
      const g3 = game3FromGame2(g2);
      const tactics = searchChains(g3, 10000, depthLimit);

      for (const tactic of tactics) {
        totalGroups++;
        if (tactic.status && tactic.status.moverSucceeds === null) {
          inconclusiveCount++;
        } else {
          definitiveCount++;
        }
      }

      // Play a move (prefer legal moves, otherwise pass)
      let moved = false;
      for (let i = 0; i < 169; i++) {
        if (g2.isLegal(i)) {
          g2.play(i);
          consecutivePasses = 0;
          moved = true;
          movesThisGame++;
          totalMoves++;
          break;
        }
      }

      if (!moved) {
        // No legal moves, play pass
        g2.play(-1);
        consecutivePasses++;
        movesThisGame++;
        totalMoves++;
      }
    }

    console.log(`  Game ${gameNum}: ${movesThisGame} moves`);
  }

  return {
    depthLimit,
    totalGroups,
    definitive: definitiveCount,
    inconclusive: inconclusiveCount,
    totalMoves,
    definitiveRate: totalGroups > 0 ? ((definitiveCount / totalGroups) * 100).toFixed(2) : 0,
    inconclusiveRate: totalGroups > 0 ? ((inconclusiveCount / totalGroups) * 100).toFixed(2) : 0,
  };
}

console.log('Comparing solve ratio with different depth limits...\n');
console.log('Testing with 5 games, playing until first pass...\n');

const results20 = runTestWithDepthLimit(20);
const results15 = runTestWithDepthLimit(15);
const results10 = runTestWithDepthLimit(10);

console.log('\n====== DEPTH LIMIT 20 ======');
console.log(`Total groups: ${results20.totalGroups}`);
console.log(`Total moves: ${results20.totalMoves}`);
console.log(`Definitive: ${results20.definitive} (${results20.definitiveRate}%)`);
console.log(`Inconclusive: ${results20.inconclusive} (${results20.inconclusiveRate}%)`);

console.log('\n====== DEPTH LIMIT 15 ======');
console.log(`Total groups: ${results15.totalGroups}`);
console.log(`Total moves: ${results15.totalMoves}`);
console.log(`Definitive: ${results15.definitive} (${results15.definitiveRate}%)`);
console.log(`Inconclusive: ${results15.inconclusive} (${results15.inconclusiveRate}%)`);

console.log('\n====== DEPTH LIMIT 10 ======');
console.log(`Total groups: ${results10.totalGroups}`);
console.log(`Total moves: ${results10.totalMoves}`);
console.log(`Definitive: ${results10.definitive} (${results10.definitiveRate}%)`);
console.log(`Inconclusive: ${results10.inconclusive} (${results10.inconclusiveRate}%)`);

console.log('\n====== COMPARISON (15 vs 20) ======');
const defDiff15 = results15.definitive - results20.definitive;
const rateDiff15 = parseFloat(results15.definitiveRate) - parseFloat(results20.definitiveRate);
console.log(`Definitive change: ${defDiff15 > 0 ? '+' : ''}${defDiff15} (${rateDiff15 > 0 ? '+' : ''}${rateDiff15.toFixed(2)}%)`);

console.log('\n====== COMPARISON (10 vs 20) ======');
const defDiff10 = results10.definitive - results20.definitive;
const rateDiff10 = parseFloat(results10.definitiveRate) - parseFloat(results20.definitiveRate);
console.log(`Definitive change: ${defDiff10 > 0 ? '+' : ''}${defDiff10} (${rateDiff10 > 0 ? '+' : ''}${rateDiff10.toFixed(2)}%)`);
