#!/usr/bin/env node
'use strict';

// Benchmark tactics search: Game2 vs Game3-Optimized
// Uses the same bench-tactics.js framework

const { performance } = require('perf_hooks');
const path = require('path');
const { Game2, PASS } = require('./game2.js');
const { Game3Optimized } = require('./game3-optimized.js');
const { searchChain: searchChainGame2 } = require('./tactics3.js');
const { searchChain: searchChainGame3 } = require('./tactics-game3-optimized.js');

const args = process.argv.slice(2);
const get = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

const boardSize = parseInt(get('--size', '9'), 10);
const gameLimit = parseInt(get('--games', '2'), 10) || Infinity;
const nodeLimit = parseInt(get('--nodes', '0'), 10) || Infinity;

if (!boardSize) {
  process.stderr.write('Usage: node bench-tactics-optimized.js --size <n> [--games <n>] [--nodes <n>]\n');
  process.exit(1);
}

// Simple random move generator
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
      game.play(move);
    }

    gamesPlayed++;
    const progress = Math.max(1, Math.floor(gameLimit / 10));
    if (gamesPlayed % progress === 0 || gamesPlayed === gameLimit) {
      const positionAvg = totalCalls > 0 ? totalMs / totalCalls : 0;
      const defRatio = totalChains > 0 ? definitive / totalChains : 0;
      console.log(`Games: ${gamesPlayed}/${gameLimit}  Pos: ${totalCalls}  Chains: ${totalChains}  Avg: ${positionAvg.toFixed(2)}ms  Max: ${chainMax.toFixed(2)}ms  Def: ${(defRatio*100).toFixed(1)}%`);
    }
  }

  const positionAvg = totalCalls > 0 ? totalMs / totalCalls : 0;
  console.log(`\nFinal stats for ${name}:`);
  console.log(`  Total time:       ${totalMs.toFixed(1)}ms`);
  console.log(`  Total positions:  ${totalCalls}`);
  console.log(`  Total chains:     ${totalChains}`);
  console.log(`  Avg per position: ${positionAvg.toFixed(2)}ms`);
  console.log(`  Max chain time:   ${chainMax.toFixed(2)}ms`);
  console.log(`  Definitive:       ${(definitive/totalChains*100).toFixed(1)}%`);

  return { totalMs, totalCalls, chainMax, positionAvg };
}

console.log('Tactical Search Benchmark');
console.log('Using bench-tactics.js framework');
console.log(`Board size: ${boardSize}x${boardSize}`);
console.log(`Games: ${gameLimit}, Node limit: ${nodeLimit === Infinity ? 'unlimited' : nodeLimit}`);

const game2Results = benchmarkGame(Game2, searchChainGame2, 'Game2 (clone-based)');
const game3Results = benchmarkGame(Game3Optimized, searchChainGame3, 'Game3-Optimized');

console.log(`\n${'='.repeat(60)}`);
console.log('COMPARISON');
console.log(`${'='.repeat(60)}`);
console.log(`Total time:       Game2=${game2Results.totalMs.toFixed(1)}ms  Game3=${game3Results.totalMs.toFixed(1)}ms`);
console.log(`Avg per position: Game2=${game2Results.positionAvg.toFixed(2)}ms  Game3=${game3Results.positionAvg.toFixed(2)}ms`);
const speedup = game2Results.totalMs / game3Results.totalMs;
console.log(`Speedup:          ${speedup.toFixed(2)}x faster with Game3-Optimized`);
console.log(`Max chain time:   Game2=${game2Results.chainMax.toFixed(2)}ms  Game3=${game3Results.chainMax.toFixed(2)}ms`);
