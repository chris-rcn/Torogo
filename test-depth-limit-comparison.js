#!/usr/bin/env node
'use strict';

// Compare solve ratio and performance across depth limits 1-15

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

      // Play a random move
      const move = g2.randomLegalMove();
      if (move !== -1) {
        g2.play(move);
        consecutivePasses = 0;
        movesThisGame++;
        totalMoves++;
      } else {
        // No legal moves, play pass
        g2.play(-1);
        consecutivePasses++;
        movesThisGame++;
        totalMoves++;
      }
    }
  }

  return {
    depthLimit,
    totalGroups,
    definitive: definitiveCount,
    inconclusive: inconclusiveCount,
    totalMoves,
    solveRatio: totalGroups > 0 ? (definitiveCount / totalGroups) : 0,
  };
}

console.log('Comparing solve ratio and performance across depth limits 1-15...\n');
console.log('Testing with 5 games, playing until first pass...\n');

const results = [];
const startOverall = Date.now();

console.log('Depth | Time (ms) | Solve Ratio | Total Groups | Definitive | Inconclusive');
console.log('------|-----------|------------|--------------|------------|-------------');

for (let depth = 1; depth <= 15; depth++) {
  const startTime = Date.now();
  const result = runTestWithDepthLimit(depth);
  const elapsed = Date.now() - startTime;

  result.elapsed = elapsed;
  results.push(result);

  const solveRatioPercent = (result.solveRatio * 100).toFixed(2);
  console.log(
    `${String(depth).padStart(5)} | ${String(elapsed).padStart(9)} | ${solveRatioPercent.padStart(10)}% | ${String(result.totalGroups).padStart(12)} | ${String(result.definitive).padStart(10)} | ${String(result.inconclusive).padStart(12)}`
  );
}

const elapsedOverall = Date.now() - startOverall;
console.log(`\nTotal elapsed time: ${elapsedOverall}ms\n`);

// Analysis
console.log('====== ANALYSIS ======');
const fastest = results.reduce((a, b) => a.elapsed < b.elapsed ? a : b);
const slowest = results.reduce((a, b) => a.elapsed > b.elapsed ? a : b);
const bestSolve = results.reduce((a, b) => a.solveRatio > b.solveRatio ? a : b);
const worstSolve = results.reduce((a, b) => a.solveRatio < b.solveRatio ? a : b);

console.log(`Fastest: Depth ${fastest.depthLimit} (${fastest.elapsed}ms)`);
console.log(`Slowest: Depth ${slowest.depthLimit} (${slowest.elapsed}ms)`);
console.log(`Best solve: Depth ${bestSolve.depthLimit} (${(bestSolve.solveRatio * 100).toFixed(2)}%)`);
console.log(`Worst solve: Depth ${worstSolve.depthLimit} (${(worstSolve.solveRatio * 100).toFixed(2)}%)`);
