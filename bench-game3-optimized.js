#!/usr/bin/env node
'use strict';

// Benchmark Game3-Optimized for tactical search workload

const { performance } = require('perf_hooks');
const { Game2 } = require('./game2.js');
const { Game3Optimized } = require('./game3-optimized.js');

function benchmarkTacticalSearch(GameClass, name, iterations = 100) {
  console.log(`\nBenchmarking ${name}...`);

  const start = performance.now();
  let totalPositions = 0;

  for (let iter = 0; iter < iterations; iter++) {
    const game = new GameClass(7);

    // Play out some position
    const moves = [
      { c: 1, m: 12 }, { c: -1, m: 17 }, { c: 1, m: 11 }, { c: -1, m: 13 },
      { c: 1, m: 7 }, { c: -1, m: 18 }, { c: 1, m: 19 }, { c: -1, m: 24 },
    ];

    for (const move of moves) {
      if (game.current === move.c && game.isLegal(move.m)) {
        game.play(move.m);
      }
    }

    // Simulate tactical search: many play/undo pairs at shallow depth
    for (let depth = 0; depth < 5; depth++) {
      const cap = game.N * game.N;
      const legalMoves = [];

      for (let i = 0; i < cap; i++) {
        if (game.cells[i] === 0 && game.isLegal(i)) {
          legalMoves.push(i);
        }
      }

      // Explore each legal move (typical tactical search pattern)
      for (let i = 0; i < Math.min(legalMoves.length, 3); i++) {
        const move = legalMoves[i];
        game.play(move);
        // "Evaluate position" (just read cells)
        for (let j = 0; j < cap; j++) {
          const _ = game.cells[j];
        }
        game.undo();
        totalPositions++;
      }
    }
  }

  const elapsed = performance.now() - start;
  console.log(`  Time: ${elapsed.toFixed(1)}ms for ${iterations} games`);
  console.log(`  Positions explored: ${totalPositions}`);
  console.log(`  Avg per position: ${(elapsed / totalPositions * 1000).toFixed(2)}µs`);
  return elapsed;
}

console.log('='.repeat(60));
console.log('Tactical Search Workload Benchmark');
console.log('(Optimized for play/undo pairs)');
console.log('='.repeat(60));

// Benchmark Game3-Optimized (Game2 doesn't have undo, so benchmark separately)
const game3OptimizedTime = benchmarkTacticalSearch(Game3Optimized, 'Game3-Optimized (dual-board)', 50);

// For comparison, benchmark Game2 with clone/discard pattern
console.log(`\nBenchmarking Game2 (clone-based)...`);
const start = performance.now();
let totalPositions2 = 0;

for (let iter = 0; iter < 50; iter++) {
  const game = new Game2(7);
  const moves = [
    12, 17, 11, 13,
    7, 18, 19, 24,
  ];

  for (const move of moves) {
    if (game.isLegal(move)) {
      game.play(move);
    }
  }

  // Simulate tactical search: many clone/play patterns
  for (let depth = 0; depth < 5; depth++) {
    const cap = game.N * game.N;
    const legalMoves = [];

    for (let i = 0; i < cap; i++) {
      if (game.cells[i] === 0 && game.isLegal(i)) {
        legalMoves.push(i);
      }
    }

    for (let i = 0; i < Math.min(legalMoves.length, 3); i++) {
      const move = legalMoves[i];
      const clone = game.clone();
      clone.play(move);
      for (let j = 0; j < cap; j++) {
        const _ = clone.cells[j];
      }
      totalPositions2++;
    }
  }
}

const game2Time = performance.now() - start;

console.log(`  Time: ${game2Time.toFixed(1)}ms`);
console.log(`  Positions explored: ${totalPositions2}`);
console.log(`  Avg per position: ${(game2Time / totalPositions2 * 1000).toFixed(2)}µs`);

console.log('\n' + '='.repeat(60));
console.log('Results');
console.log('='.repeat(60));
console.log(`Game2 (clone-based): ${game2Time.toFixed(1)}ms`);
console.log(`Game3-Optimized (dual-board): ${game3OptimizedTime.toFixed(1)}ms`);
console.log(`Speedup: ${(game2Time / game3OptimizedTime).toFixed(2)}x faster with Game3-Optimized`);

if (game3OptimizedTime < game2Time) {
  console.log('\n✅ Game3-Optimized is faster for tactical search!');
} else {
  console.log('\n⚠️  Game2 is still faster (likely due to V8 optimization differences)');
}
