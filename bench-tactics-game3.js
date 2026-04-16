#!/usr/bin/env node
'use strict';

// bench-tactics-game3.js — Benchmark Game3 tactical search vs Game2
//
// Compares Game3 (incremental) vs Game2 (clone-based) using the same
// tactical search patterns as bench-tactics.js
//
// Usage:
//   node bench-tactics-game3.js --size <n> [--games <n>] [--nodes <n>]

const { performance } = require('perf_hooks');
const path = require('path');
const { Game2, BLACK, WHITE, PASS } = require('./game2.js');
const { Game3 } = require('./game3.js');
const { searchChain: searchChainGame2 } = require('./tactics3.js');
const { searchChain: searchChainGame3 } = require('./tactics-game3.js');

const args = process.argv.slice(2);
const get = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

const boardSize = parseInt(get('--size', '9'), 10);
const gameLimit = parseInt(get('--games', '2'), 10) || Infinity;
const nodeLimit = parseInt(get('--nodes', '0'), 10) || Infinity;

if (!boardSize) {
  process.stderr.write('Usage: node bench-tactics-game3.js --size <n> [--games <n>] [--nodes <n>]\n');
  process.exit(1);
}

// Simple random move generator (no AI, just picks legal moves)
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

function benchmarkGame(GameClass, searchChainFunc, name) {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`Benchmarking ${name}`);
  console.log(`${'='.repeat(60)}`);

  let chainMax = 0;
  let totalMs = 0;
  let totalCalls = 0;
  let totalChains = 0;
  let definitive = 0;
  let gamesPlayed = 0;

  while (gamesPlayed < gameLimit) {
    const game = new GameClass(boardSize);

    while (!game.gameOver) {
      if (Math.random() < 0.1) {
        const N = game.N;
        const cap = N * N;
        const visited = new Set();
        let positionMs = 0;

        for (let i = 0; i < cap; i++) {
          if (game.cells[i] === 0) continue;
          const gid = game.groupIdAt(i);
          if (visited.has(gid)) continue;
          visited.add(gid);
          const lc = game.groupLibs(i).length;
          if (lc === 0 || lc > 3) continue;

          const t0 = performance.now();
          const result = searchChainFunc(game, i, nodeLimit);
          const elapsed = performance.now() - t0;
          positionMs += elapsed;
          totalChains++;
          if (result && result.moverSucceeds !== null) definitive++;
          if (elapsed > chainMax) chainMax = elapsed;
        }

        totalCalls++;
        totalMs += positionMs;
      }

      const move = getRandomMove(game);
      if (GameClass === Game2) {
        game.play(move);
      } else {
        game.play(move);
      }
    }

    gamesPlayed++;
    if (gamesPlayed % Math.max(1, Math.floor(gameLimit / 10)) === 0 || gamesPlayed === gameLimit) {
      const positionAvg = totalCalls > 0 ? totalMs / totalCalls : 0;
      const defRatio = totalChains > 0 ? definitive / totalChains : 0;
      console.log(`Games: ${gamesPlayed}/${gameLimit}  Positions: ${totalCalls}  Chains: ${totalChains}  Avg/pos: ${positionAvg.toFixed(3)}ms  Max: ${chainMax.toFixed(3)}ms  Definitive: ${(defRatio * 100).toFixed(1)}%`);
    }
  }

  const positionAvg = totalCalls > 0 ? totalMs / totalCalls : 0;
  console.log(`\nFinal stats for ${name}:`);
  console.log(`  Total time:      ${totalMs.toFixed(1)}ms`);
  console.log(`  Total positions: ${totalCalls}`);
  console.log(`  Total chains:    ${totalChains}`);
  console.log(`  Avg per position: ${positionAvg.toFixed(3)}ms`);
  console.log(`  Max chain time:  ${chainMax.toFixed(3)}ms`);

  return { totalMs, totalCalls, chainMax };
}

// Run both benchmarks
const game2Results = benchmarkGame(Game2, searchChainGame2, 'Game2 (clone-based)');
const game3Results = benchmarkGame(Game3, searchChainGame3, 'Game3 (incremental undo)');

// Compare results
console.log(`\n${'='.repeat(60)}`);
console.log('COMPARISON');
console.log(`${'='.repeat(60)}`);
console.log(`Total time:     Game2=${game2Results.totalMs.toFixed(1)}ms  Game3=${game3Results.totalMs.toFixed(1)}ms`);
const speedup = game2Results.totalMs / game3Results.totalMs;
console.log(`Speedup:        ${speedup.toFixed(2)}x faster with Game3`);
console.log(`Max chain time: Game2=${game2Results.chainMax.toFixed(3)}ms  Game3=${game3Results.chainMax.toFixed(3)}ms`);
console.log(`Benefit:        ${((1 - game3Results.chainMax / game2Results.chainMax) * 100).toFixed(1)}% reduction in worst-case`);
