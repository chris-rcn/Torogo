#!/usr/bin/env node
'use strict';

const { Game2, BLACK, WHITE, parseBoard, PASS } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');

// Find first failing case
for (let test = 0; test < 200; test++) {
  const game2 = new Game2(13);
  let consecutivePasses = 0;

  while (consecutivePasses < 1 && !game2.gameOver) {
    const move = game2.randomLegalMove();
    if (move !== -1) {
      game2.play(move);
      consecutivePasses = 0;
    } else {
      game2.play(-1);
      consecutivePasses++;
    }
  }

  const original = game2.toString(PASS);
  const game3 = game3FromGame2(game2);
  const converted = game3.toString(PASS);

  if (converted !== original) {
    console.log(`Failing position (test ${test + 1}):\n`);
    console.log(original);
    process.exit(1);
  }
}

console.log('No failures found');
