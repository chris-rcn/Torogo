#!/usr/bin/env node
'use strict';

const { Game2, BLACK } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');

// Generate one simple test position
const g2 = new Game2(13);

// Play a few moves
g2.play(g2.randomLegalMove());
g2.play(g2.randomLegalMove());
g2.play(g2.randomLegalMove());

const original = g2.toString();
const g3 = game3FromGame2(g2);
const converted = g3.toString();

console.log('Game2.toString():\n');
console.log(original);
console.log('\nGame3.toString():\n');
console.log(converted);

console.log('\n' + '='.repeat(50));
console.log('Checking line by line:');

const origLines = original.split('\n');
const convLines = converted.split('\n');

for (let i = 0; i < Math.max(origLines.length, convLines.length); i++) {
  const o = origLines[i] || '(missing)';
  const c = convLines[i] || '(missing)';
  const match = o === c ? '✓' : '✗';
  console.log(`${match} Line ${i}:`);
  if (o !== c) {
    console.log(`  Original: ${o}`);
    console.log(`  Converted: ${c}`);
  }
}
