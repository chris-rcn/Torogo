#!/usr/bin/env node
'use strict';

// bench-tactics.js — benchmark searchChain across game positions.
//
// For a random 10% of positions, calls searchChain on every chain with 1-3
// liberties, timing each call. Tracks maximum and average total time per
// position. When a new maximum is found, prints the position and slowest chain.
//
// Usage:
//   node bench-tactics.js --size <n> [--agent <name>] [--games <n>] [--nodes <n>]

const { performance } = require('perf_hooks');
const path = require('path');
const { Game2, BLACK, WHITE, PASS } = require('./game2.js');
const { searchChain } = require('./tactics3.js');

const args = process.argv.slice(2);
const get  = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

const boardSize  = parseInt(get('--size',   '13'), 10);
const agentName  =          get('--agent',  'random');
const gameLimit  = parseInt(get('--games',  '0'),  10) || Infinity;
const nodeLimit  = parseInt(get('--nodes',  '0'),  10) || Infinity;

if (!boardSize) {
  process.stderr.write('Usage: node bench-tactics.js --size <n> [--agent <name>] [--games <n>] [--nodes <n>]\n');
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
  process.stdout.write(game.toString(idx, { centerAt: idx }) + '\n');
}

while (gamesPlayed < gameLimit) {
  const game = new Game2(boardSize);

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
        nextPrint = Math.ceil(totalCalls * 1.5);
      } else if (totalCalls >= nextPrint) {
        printLine();
        nextPrint = Math.ceil(nextPrint * 1.5);
      }
    }

    const result = getMove(game, 1);
    const idx = result.type === 'place' ? result.y * boardSize + result.x : PASS;
    game.play(idx);
  }

  gamesPlayed++;
}

printLine();
process.stdout.write(`done  games=${gamesPlayed}  calls=${totalCalls}  chainMax=${chainMax.toFixed(3)}\n`);
