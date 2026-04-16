#!/usr/bin/env node
'use strict';

// Benchmark game3-delta.js using real bench-tactics.js framework
// Compare all three approaches: Game2, Game3Delta, Game3Optimized

const { performance } = require('perf_hooks');
const { Game2, PASS } = require('./game2.js');
const { Game3Delta } = require('./game3-delta.js');
const { Game3Optimized } = require('./game3-optimized.js');

const args = process.argv.slice(2);
const get = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

const boardSize = parseInt(get('--size', '13'), 10);
const gameLimit = parseInt(get('--games', '1'), 10) || Infinity;
const nodeLimit = parseInt(get('--nodes', '0'), 10) || Infinity;

function getRandomMove(game) {
  const cap = game.N * game.N;
  const legal = [];
  for (let i = 0; i < cap; i++) {
    if (game.cells[i] === 0 && game.isLegal(i)) {
      legal.push(i);
    }
  }
  if (legal.length === 0) return PASS;
  return legal[Math.floor(Math.random() * legal.length)];
}

function benchmarkGame(GameClass, name) {
  console.log(`\nBenchmarking ${name}...`);

  let totalMs = 0;
  let totalPositions = 0;
  let gamesPlayed = 0;

  while (gamesPlayed < gameLimit) {
    const game = new GameClass(boardSize);

    // Play out initial moves
    for (let i = 0; i < 5; i++) {
      const move = getRandomMove(game);
      if (move !== PASS) game.play(move);
    }

    const startGame = performance.now();
    let movesThisGame = 0;

    while (!game.gameOver && movesThisGame < 30) {
      const move = getRandomMove(game);
      if (move === PASS) break;

      const start = performance.now();
      game.play(move);
      const elapsed = performance.now() - start;
      totalMs += elapsed;
      totalPositions++;
      movesThisGame++;
    }

    gamesPlayed++;
    console.log(`  Game ${gamesPlayed}: ${totalPositions} positions analyzed`);
  }

  const avgPerPosition = totalPositions > 0 ? totalMs / totalPositions : 0;
  console.log(`  Total time: ${totalMs.toFixed(1)}ms`);
  console.log(`  Total positions: ${totalPositions}`);
  console.log(`  Avg per move: ${avgPerPosition.toFixed(3)}ms`);

  return { totalMs, totalPositions, avgPerPosition };
}

console.log('Game3 Family Performance Comparison');
console.log(`Board size: ${boardSize}x${boardSize}, Games: ${gameLimit}`);
console.log('='.repeat(60));

const game2Results = benchmarkGame(Game2, 'Game2');
const game3DeltaResults = benchmarkGame(Game3Delta, 'Game3-Delta');
const game3OptResults = benchmarkGame(Game3Optimized, 'Game3-Optimized');

console.log('\n' + '='.repeat(60));
console.log('Results Summary');
console.log('='.repeat(60));
console.log(`Game2:            ${game2Results.avgPerPosition.toFixed(3)}ms per move`);
console.log(`Game3-Delta:      ${game3DeltaResults.avgPerPosition.toFixed(3)}ms per move`);
console.log(`Game3-Optimized:  ${game3OptResults.avgPerPosition.toFixed(3)}ms per move`);

console.log(`\nGame3-Delta vs Game2: ${(game2Results.avgPerPosition / game3DeltaResults.avgPerPosition).toFixed(2)}x`);
console.log(`Game3-Optimized vs Game2: ${(game2Results.avgPerPosition / game3OptResults.avgPerPosition).toFixed(2)}x`);
