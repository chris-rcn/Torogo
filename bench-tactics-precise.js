#!/usr/bin/env node
'use strict';

// bench-tactics-precise.js — benchmark searchChain with Game3Precise.
//
// Compares Game3Precise against Game2 for tactical search performance.
// For a random 10% of positions, calls searchChain on every chain with 1-3
// liberties, timing each call.

const { performance } = require('perf_hooks');
const path = require('path');
const { Game2, BLACK, WHITE, PASS } = require('./game2.js');
const { Game3Precise } = require('./game3.js');
const { searchChain } = require('./tactics3.js');

const args = process.argv.slice(2);
const get  = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

const boardSize  = parseInt(get('--size',   '13'), 10);
const agentName  =          get('--agent',  'random');
const gameLimit  = parseInt(get('--games',  '0'),  10) || Infinity;
const nodeLimit  = parseInt(get('--nodes',  '0'),  10) || Infinity;
const useGame3   = get('--game3', 'false') === 'true';

if (!boardSize) {
  process.stderr.write('Usage: node bench-tactics-precise.js --size <n> [--agent <name>] [--games <n>] [--nodes <n>] [--game3 true|false]\n');
  process.exit(1);
}

const { getMove } = require(path.join(__dirname, 'ai', agentName + '.js'));

let chainMax     = 0;
let totalMs      = 0;
let totalCalls   = 0;
let totalChains  = 0;
let definitive   = 0;
let gamesPlayed  = 0;
let nextPrint    = 1;

function printLine() {
  const positionAvg = totalCalls  > 0 ? totalMs / totalCalls : 0;
  const defRatio    = totalChains > 0 ? definitive / totalChains : 0;
  process.stdout.write(`calls=${totalCalls}  game=${gamesPlayed + 1}  positionAvg=${positionAvg.toFixed(3)}  chainMax=${chainMax.toFixed(3)}  definitive=${(defRatio * 100).toFixed(1)}%\n`);
}

function printChain(game, idx, result, ms) {
  const N = game.N;
  const color = game.cells[idx] === BLACK ? '●' : '○';
  const mover = game.current === BLACK ? '●' : '○';
  const wx = idx % N, wy = (idx / N) | 0;
  const lc = result ? result.libs.length : 0;
  const libs   = result ? Array.from(result.libs).map(i => `${i % N},${(i / N) | 0}`).join(' ') : '';
  const urgent = result ? result.urgentLibs.map(i => `${i % N},${(i / N) | 0}`).join(' ') : '';
  process.stdout.write(`  slowest chain: ${color} at (${wx},${wy})  libs=${lc}  ms=${ms.toFixed(3)}  ${mover} to play\n`);
  process.stdout.write(`    libs: [${libs}]\n`);
  process.stdout.write(`    moverSucceeds=${result ? result.moverSucceeds : '?'}  urgentLibs=[${urgent}]\n`);
  if (game.toString) {
    process.stdout.write(game.toString(idx, { centerAt: idx }) + '\n');
  }
}

const GameClass = useGame3 ? Game3Precise : Game2;
const gameLabel = useGame3 ? 'Game3Precise' : 'Game2';
console.log(`Benchmarking ${gameLabel} with searchChain`);
console.log(`Board size: ${boardSize}x${boardSize}, Games: ${gameLimit}`);
console.log('='.repeat(60));

while (gamesPlayed < gameLimit) {
  let game = new GameClass(boardSize);

  while (!game.gameOver) {
    if (Math.random() < 0.1) {
      const N = game.N;
      const cap = N * N;
      const visited = new Set();
      let positionMs = 0;
      let worstMs = 0, worstIdx = -1, worstResult = null;
      let newChainMax = false;

      for (let i = 0; i < cap; i++) {
        if (game.cells[i] === 0) continue;
        const gid = game._gid[i];
        if (visited.has(gid)) continue;
        visited.add(gid);
        const lc = game.groupLibs(i).length;
        if (lc === 0 || lc > 3) continue;

        const t0 = performance.now();
        const result = searchChain(game, i, nodeLimit);
        const elapsed = performance.now() - t0;
        positionMs += elapsed;
        totalChains++;
        if (result && result.moverSucceeds !== null) definitive++;
        if (elapsed > chainMax) { chainMax = elapsed; newChainMax = true; }

        if (elapsed > worstMs) {
          worstMs = elapsed;
          worstIdx = i;
          worstResult = result;
        }
      }

      totalCalls++;
      totalMs += positionMs;

      if (newChainMax && worstIdx !== -1) {
        printLine();
        printChain(game, worstIdx, worstResult, worstMs);
      }
    }

    const move = getMove(game);
    if (move === PASS) {
      game.play(PASS);
    } else {
      if (useGame3) {
        game.play(move);
      } else {
        const clone = game.clone();
        if (!clone.play(move)) break;
        game = clone;
      }
    }
  }

  gamesPlayed++;
  if (gamesPlayed % nextPrint === 0) {
    nextPrint *= 2;
    printLine();
  }
}

console.log('\n' + '='.repeat(60));
console.log('Final Results');
console.log('='.repeat(60));
const positionAvg = totalCalls  > 0 ? totalMs / totalCalls : 0;
const defRatio    = totalChains > 0 ? definitive / totalChains : 0;
console.log(`Games played: ${gamesPlayed}`);
console.log(`Total positions analyzed: ${totalCalls}`);
console.log(`Total chains analyzed: ${totalChains}`);
console.log(`Average time per position: ${positionAvg.toFixed(3)}ms`);
console.log(`Maximum chain time: ${chainMax.toFixed(3)}ms`);
console.log(`Definitive results: ${(defRatio * 100).toFixed(1)}%`);
console.log(`Total time: ${totalMs.toFixed(1)}ms`);
