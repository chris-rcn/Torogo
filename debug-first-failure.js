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
    console.log(`Found failure at test ${test + 1}\n`);

    const oLines = original.split('\n');
    const cLines = converted.split('\n');

    for (let i = 0; i < oLines.length; i++) {
      if (oLines[i] !== cLines[i]) {
        console.log(`Line ${i}:`);
        console.log(`  Game2: "${oLines[i]}"`);
        console.log(`  Game3: "${cLines[i]}"`);
      }
    }

    console.log('\nCell state comparison:');
    let diffs = 0;
    for (let i = 0; i < 169; i++) {
      if (game2.cells[i] !== game3.cells[i]) {
        if (diffs < 5) {
          const x = i % 13;
          const y = (i / 13) | 0;
          console.log(`  Cell ${i} (${x},${y}): Game2=${game2.cells[i]}, Game3=${game3.cells[i]}`);
        }
        diffs++;
      }
    }
    if (diffs > 5) console.log(`  ... and ${diffs - 5} more cell differences`);
    console.log(`Total: ${diffs} cell differences`);

    process.exit(1);
  }
}

console.log('No failures found');
