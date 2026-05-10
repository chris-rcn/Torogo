#!/usr/bin/env node
'use strict';

const { performance } = require('perf_hooks');
const { Game2 } = require('./game2.js');
const { getAllLadderStatuses } = require('./ladder2.js');

const N = 13;
const nGames = 100;
const minChain = 2;

let positions = 0;
let totalMs = 0;
let totalGroups = 0;

for (let g = 0; g < nGames; g++) {
  const game = new Game2(N);
  while (!game.gameOver) {
    const move = game.randomLegalMove();
    if (move < 0) break;
    positions++;
    const t0 = performance.now();
    const results = getAllLadderStatuses(game, minChain);
    totalMs += performance.now() - t0;
    totalGroups += results.length;
    game.play(move);
  }
}

const avgMs = (totalMs / positions).toFixed(3);
const avgGroups = (totalGroups / positions).toFixed(1);
console.log(`minChain=${minChain}:  ${positions} positions  ${totalMs.toFixed(0)}ms total  ${avgMs}ms/pos  ${avgGroups} groups/pos`);
