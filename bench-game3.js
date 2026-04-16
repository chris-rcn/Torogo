'use strict';

const { Game3 } = require('./game3.js');
const { Game2 } = typeof require !== 'undefined' ? require('./game2.js') : { Game2: null };

if (!Game2) {
  console.error('Game2 not available for benchmark');
  process.exit(1);
}

// Test tactical search performance: shallow branch exploration
function benchmarkShallowSearch(GameClass, name, iterations = 100) {
  console.log(`\nBenchmarking ${name}...`);

  const start = Date.now();

  for (let iter = 0; iter < iterations; iter++) {
    const game = new GameClass(5);

    // Make some initial moves to create a position
    game.play(6);
    game.play(11);
    game.play(7);
    game.play(2);
    game.play(1);

    // Simulate tactical search: many shallow explorations
    for (let depth = 0; depth < 5; depth++) {
      const emptyCount = game.emptyCount;
      const legalMoves = [];

      // Collect legal moves
      const cap = game.N * game.N;
      for (let i = 0; i < cap; i++) {
        if (game.cells[i] === 0 && game.isLegal(i)) {
          legalMoves.push(i);
        }
      }

      // Explore each legal move
      for (let i = 0; i < Math.min(legalMoves.length, 4); i++) {
        const moveIdx = legalMoves[i];

        if (GameClass === Game2) {
          const clone = game.clone();
          clone.play(moveIdx);
          // Use clone result (implicitly drops it)
        } else {
          game.play(moveIdx);
          game.undo();
        }
      }
    }
  }

  const elapsed = Date.now() - start;
  console.log(`  ${iterations} iterations: ${elapsed}ms (${(elapsed / iterations).toFixed(2)}ms per game)`);
  return elapsed;
}

// Test undo performance: verify undo is fast
function benchmarkUndo(GameClass, name, iterations = 1000) {
  console.log(`\nBenchmarking undo with ${name}...`);

  const game = new GameClass(5);

  // Setup: make some moves
  const moves = [6, 11, 7, 2, 1, 3, 8, 13, 9, 4, 14, 19];
  for (const move of moves) {
    if (game.isLegal(move)) {
      game.play(move);
    }
  }

  const start = Date.now();

  for (let iter = 0; iter < iterations; iter++) {
    // Play and undo a move repeatedly
    const move = 0; // Try same move each time (ko protection varies)
    if (game.isLegal(move) && game.cells[move] === 0) {
      game.play(move);
      game.undo();
    }
  }

  const elapsed = Date.now() - start;
  console.log(`  ${iterations} iterations: ${elapsed}ms (${(elapsed / iterations).toFixed(3)}ms per undo)`);
  return elapsed;
}

// Test memory usage (rough estimate)
function benchmarkMemory(GameClass, name) {
  console.log(`\nBenchmarking memory with ${name}...`);

  const count = 100;
  const games = [];

  // Create many game instances
  for (let i = 0; i < count; i++) {
    const game = new GameClass(5);
    game.play(6);
    game.play(11);
    games.push(game);
  }

  // Rough memory estimate: check object size
  let totalSize = 0;
  for (const game of games) {
    const g = game;
    totalSize +=
      g.cells.byteLength +
      g._gid.byteLength +
      g._gc.byteLength +
      g._sw.byteLength +
      g._ss.byteLength +
      g._lw.byteLength +
      g._ls.byteLength;
  }

  console.log(`  ${count} games: ~${(totalSize / 1024).toFixed(1)}KB (${(totalSize / count / 1024).toFixed(2)}KB per game)`);
}

console.log('='.repeat(50));
console.log('Game3 vs Game2 Tactical Search Benchmark');
console.log('='.repeat(50));

const game3Time = benchmarkShallowSearch(Game3, 'Game3 (incremental)', 100);
const game2Time = benchmarkShallowSearch(Game2, 'Game2 (clone-based)', 100);

console.log(`\nShallow search speedup: ${(game2Time / game3Time).toFixed(1)}x faster`);

benchmarkUndo(Game3, 'Game3');
// Game2 doesn't have undo; it uses clone/discard pattern

benchmarkMemory(Game3, 'Game3');
benchmarkMemory(Game2, 'Game2');

console.log('\n' + '='.repeat(50));
console.log('✅ Benchmark complete');
