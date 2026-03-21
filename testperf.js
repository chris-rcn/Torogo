#!/usr/bin/env node
'use strict';

const { performance } = require('perf_hooks');

const args = process.argv.slice(2);
const get  = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

const boardSize = parseInt(get('--size', '9'), 10);
if (isNaN(boardSize) || boardSize < 2) { console.error('--size must be >= 2'); process.exit(1); }

const engine = get('--engine', 'game');

let playGame;
if (engine === 'game2' || engine === 'game3') {
  const mod  = engine === 'game2' ? require('./game2.js') : require('./game3.js');
  const Game2 = mod.Game2 || mod.Game3;
  const PASS  = mod.PASS;
  const N = boardSize;
  const cap = N * N;
  const _sharedGame = new Game2(N);
  // Try random probes first; fall back to full candidate list
  playGame = function () {
    const game = _sharedGame;
    game.reset();
    while (!game.gameOver) {
      let placed = false;
      for (let k = 0; k < 32 && !placed; k++) {
        const idx = Math.floor(Math.random() * cap);
        if (game.cells[idx] !== 0) continue;
        if (game.isTrueEye(idx)) continue;
        if (game.isLegal(idx)) { game.play(idx); placed = true; }
      }
      if (placed) continue;
      const cands = [];
      for (let idx = 0; idx < cap; idx++) {
        if (game.cells[idx] === 0) cands.push(idx);
      }
      while (cands.length > 0) {
        const i = Math.floor(Math.random() * cands.length);
        const idx = cands[i];
        cands[i] = cands[cands.length - 1];
        cands.pop();
        if (game.isTrueEye(idx)) continue;
        if (game.isLegal(idx)) { game.play(idx); placed = true; break; }
      }
      if (!placed) game.play(PASS);
    }
  };
} else {
  const { Game } = require('./game.js');
  const random = require('./ai/random.js');
  playGame = function () {
    const game = new Game(boardSize);
    while (!game.gameOver) {
      const move = random(game);
      if (move.type === 'place') game.placeStone(move.x, move.y);
      else game.pass();
    }
  };
}

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
