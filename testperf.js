#!/usr/bin/env node
'use strict';

const { performance } = require('perf_hooks');

const args = process.argv.slice(2);
const get  = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

if (!get('--size')) { console.error('--size is required'); process.exit(1); }
const boardSize = parseInt(get('--size'), 10);
if (isNaN(boardSize) || boardSize < 2) { console.error('--size must be >= 2'); process.exit(1); }

const engine = get('--engine', 'game');

let playGame;
const mod = require('./game2.js');
const Game2 = mod.Game2;
const PASS  = mod.PASS;
const N = boardSize;
const cap = N * N;
const _sharedGame = new Game2(N);
// Try random probes first; fall back to full candidate list
playGame = function () {
  const game = _sharedGame;
  game.reset();
  while (!game.gameOver) {
    game.play(game.randomLegalMove());
  }
};

let games = 0;
let nextPrint = 10000;
const startTime = performance.now();

for (;;) {
  playGame();
  games++;

  if (games >= nextPrint) {
    const elapsed = (performance.now() - startTime) / 1000;
    const msPerGame = (elapsed * 1000 / games).toFixed(3);
    console.log(`games: ${games}  ms/game: ${msPerGame}`);
    nextPrint += nextPrint / 2;
  }
}
