#!/usr/bin/env node
'use strict';

// bench-game3-precise-series.js — measure play() performance across many moves
// Compares Game2.clone() vs Game3Precise for a series of random moves

const { performance } = require('perf_hooks');
const { Game2, PASS } = require('./game2.js');
const { Game3Precise } = require('./game3-precise.js');

function getRandomLegalMove(game) {
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

function benchmarkSeries(GameClass, name, boardSize = 13, maxMoves = 100) {
  console.log(`\nBenchmarking ${name}...`);

  let totalMs = 0;
  let movesPlayed = 0;
  let games = 0;

  const startTime = performance.now();

  while (movesPlayed < maxMoves) {
    let game = new GameClass(boardSize);

    for (let moveNum = 0; moveNum < 50 && movesPlayed < maxMoves; moveNum++) {
      const move = getRandomLegalMove(game);
      if (move === PASS) break;

      const moveStart = performance.now();

      if (GameClass === Game2) {
        const clone = game.clone();
        clone.play(move);
        game = clone;
      } else {
        game.play(move);
      }

      const moveEnd = performance.now();
      totalMs += (moveEnd - moveStart);
      movesPlayed++;
    }

    games++;
  }

  const elapsed = performance.now() - startTime;
  const avgPerMove = totalMs / movesPlayed;
  const avgPerGame = elapsed / games;

  console.log(`  ${movesPlayed} moves in ${elapsed.toFixed(1)}ms (${games} games)`);
  console.log(`  Avg per move: ${avgPerMove.toFixed(3)}ms`);
  console.log(`  Avg per game: ${avgPerGame.toFixed(3)}ms`);

  return avgPerMove;
}

console.log('Game Series Performance Benchmark');
console.log('(100+ random moves on 13x13)');
console.log('='.repeat(60));

const game2Time = benchmarkSeries(Game2, 'Game2.clone()', 13, 150);
const game3PreciseTime = benchmarkSeries(Game3Precise, 'Game3Precise', 13, 150);

console.log('\n' + '='.repeat(60));
console.log('Results');
console.log('='.repeat(60));
console.log(`Game2:           ${game2Time.toFixed(3)}ms per move`);
console.log(`Game3Precise:    ${game3PreciseTime.toFixed(3)}ms per move`);

console.log(`\nGame3Precise speedup: ${(game2Time / game3PreciseTime).toFixed(2)}x`);
