'use strict';

const { Game3Precise, BLACK, WHITE } = require('./game3.js');
const { getAllLadderStatuses } = require('./ladder2.js');

// Test that ko rule works correctly after capturing with the fixed captured array format
const game = new Game3Precise(13);

// Build a simple position that will trigger a capture
const moves = [
  66, 67,  // white-black
  79, 80,  // white-black
  92, 93,  // white-black
];

console.log('Setting up test position...');
for (const move of moves) {
  if (!game.play(move)) {
    console.log(`Failed to play move ${move}`);
  }
}

console.log(`After setup: moveCount=${game.moveCount}, ko=${game.ko}`);

// Try to trigger a capture with ko
// Build a position where black captures white, then white can't immediately recapture
const testMoves = [
  78, 96, // Additional moves
  77,     // This should create a capture scenario
];

for (const move of testMoves) {
  const result = game.play(move);
  if (!result) {
    console.log(`Move ${move} failed (illegal)`);
  } else {
    console.log(`After move ${move}: ko=${game.ko}, type=${typeof game.ko}, moveCount=${game.moveCount}`);
  }
}

// Check if ko is an integer or an object
console.log('\nKo rule check:');
console.log(`Type of ko: ${typeof game.ko}`);
if (typeof game.ko === 'object') {
  console.log('ERROR: ko is an object, should be an integer!');
  console.log('ko =', game.ko);
} else if (game.ko >= 0) {
  console.log(`Ko is at index ${game.ko}`);
}

// Now test the ladder detection bug
console.log('\n\nTesting ladder detection multiple calls...');
const testGame = new Game3Precise(9);
const testMoves2 = [
  40, 41, 42, 43,
  49, 50, 51, 52,
];
for (const m of testMoves2) {
  testGame.play(m);
}

console.log('First call to getAllLadderStatuses:');
let results1 = getAllLadderStatuses(testGame);
console.log(`Found ${results1.length} ladder groups`);

console.log('\nSecond call to getAllLadderStatuses:');
let results2 = getAllLadderStatuses(testGame);
console.log(`Found ${results2.length} ladder groups`);

if (results1.length !== results2.length) {
  console.log('ERROR: Different results on second call!');
} else {
  console.log('✓ Consistent results');
}
