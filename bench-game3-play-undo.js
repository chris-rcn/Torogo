'use strict';

// Real test: measure play() + undo() cycles (tactical search pattern)

const { performance } = require('perf_hooks');
const { Game2 } = require('./game2.js');
const { Game3Delta } = require('./game3-delta.js');
const { Game3Optimized } = require('./game3-optimized.js');

function benchmarkPlayUndo(GameClass, name, iterations = 100) {
  console.log(`\nBenchmarking ${name}...`);

  const game = new GameClass(13);

  // Setup: play some opening moves
  for (let i = 0; i < 10; i++) {
    const cap = game.N * game.N;
    for (let j = 0; j < cap; j++) {
      if (game.cells[j] === 0 && game.isLegal(j)) {
        game.play(j);
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
      // Game3: play then undo
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
const game3DeltaTime = benchmarkPlayUndo(Game3Delta, 'Game3-Delta', 200);
const game3OptTime = benchmarkPlayUndo(Game3Optimized, 'Game3-Optimized', 200);

console.log('\n' + '='.repeat(60));
console.log('Results');
console.log('='.repeat(60));
console.log(`Game2:            ${game2Time.toFixed(3)}ms per cycle`);
console.log(`Game3-Delta:      ${game3DeltaTime.toFixed(3)}ms per cycle`);
console.log(`Game3-Optimized:  ${game3OptTime.toFixed(3)}ms per cycle`);

console.log(`\nGame3-Delta speedup:      ${(game2Time / game3DeltaTime).toFixed(2)}x`);
console.log(`Game3-Optimized speedup:  ${(game2Time / game3OptTime).toFixed(2)}x`);
