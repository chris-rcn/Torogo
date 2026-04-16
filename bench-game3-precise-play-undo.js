#!/usr/bin/env node
'use strict';

// bench-game3-precise-play-undo.js — measure play() + undo() cycles
// Compares Game2.clone() vs Game3Precise.play/undo

const { performance } = require('perf_hooks');
const { Game2 } = require('./game2.js');
const { Game3Precise } = require('./game3-precise.js');

function benchmarkPlayUndo(GameClass, name, iterations = 100) {
  console.log(`\nBenchmarking ${name}...`);

  let game = new GameClass(13);

  // Setup: play some opening moves
  for (let i = 0; i < 10; i++) {
    const cap = game.N * game.N;
    for (let j = 0; j < cap; j++) {
      if (game.cells[j] === 0 && game.isLegal(j)) {
        if (GameClass === Game2) {
          const clone = game.clone();
          clone.play(j);
          game = clone;
        } else {
          game.play(j);
        }
        break;
      }
    }
  }

  const start = performance.now();

  for (let iter = 0; iter < iterations; iter++) {
    const cap = game.N * game.N;

    // Find a legal move
    let move = -1;
    for (let i = 0; i < cap; i++) {
      if (game.cells[i] === 0 && game.isLegal(i)) {
        move = i;
        break;
      }
    }

    if (move === -1) break;

    if (GameClass === Game2) {
      // Game2: clone, play, discard
      const clone = game.clone();
      clone.play(move);
      // Discard clone implicitly
    } else {
      // Game3Precise: play then undo
      game.play(move);
      game.undo();
    }
  }

  const elapsed = performance.now() - start;
  console.log(`  ${iterations} cycles in ${elapsed.toFixed(1)}ms`);
  console.log(`  Avg per cycle: ${(elapsed / iterations).toFixed(3)}ms`);
  return elapsed / iterations;
}

console.log('Play/Undo Cycle Benchmark');
console.log('(Simulates tactical search pattern)');
console.log('='.repeat(60));

const game2Time = benchmarkPlayUndo(Game2, 'Game2.clone()', 200);
const game3PreciseTime = benchmarkPlayUndo(Game3Precise, 'Game3Precise', 200);

console.log('\n' + '='.repeat(60));
console.log('Results');
console.log('='.repeat(60));
console.log(`Game2:           ${game2Time.toFixed(3)}ms per cycle`);
console.log(`Game3Precise:    ${game3PreciseTime.toFixed(3)}ms per cycle`);

console.log(`\nGame3Precise vs Game2: ${(game2Time / game3PreciseTime).toFixed(2)}x`);
