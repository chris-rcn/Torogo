#!/usr/bin/env node
'use strict';

const { Game2, BLACK, WHITE, parseBoard } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');

// Generate a simple position
const g2_original = new Game2(13);
let consecutivePasses = 0;
let moveCount = 0;

while (consecutivePasses < 1 && !g2_original.gameOver && moveCount < 50) {
  const move = g2_original.randomLegalMove();
  if (move !== -1) {
    g2_original.play(move);
    consecutivePasses = 0;
    moveCount++;
  } else {
    g2_original.play(-1);
    consecutivePasses++;
    moveCount++;
  }
}

const original = g2_original.toString();
const g3 = game3FromGame2(g2_original);
const converted = g3.toString();
const g2_parsed = parseBoard(converted, BLACK);
const parsed = g2_parsed.toString();

console.log('Original == Converted?', original === converted);
console.log('Original == Parsed?', original === parsed);

if (original !== converted) {
  console.log('\nOriginal vs Converted difference:');
  const oLines = original.split('\n');
  const cLines = converted.split('\n');
  for (let i = 0; i < oLines.length; i++) {
    if (oLines[i] !== cLines[i]) {
      console.log(`Line ${i}:`);
      console.log(`  O: "${oLines[i]}"`);
      console.log(`  C: "${cLines[i]}"`);
    }
  }
}

if (original !== parsed) {
  console.log('\nOriginal vs Parsed difference:');
  const oLines = original.split('\n');
  const pLines = parsed.split('\n');
  for (let i = 0; i < oLines.length; i++) {
    if (oLines[i] !== pLines[i]) {
      console.log(`Line ${i}:`);
      console.log(`  O: "${oLines[i]}"`);
      console.log(`  P: "${pLines[i]}"`);
    }
  }
}
