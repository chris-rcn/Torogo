'use strict';

// Count the distinct canonical pattern IDs that appear at legal non-eye moves
// across a large corpus of complete self-play games.

const { createState, extractFeatures, NUM_PATTERNS } = require('./ppat-lib.js');
const { Game2 } = require('./game2.js');

const N      = parseInt(process.argv[2], 10) || 9;
const GAMES  = parseInt(process.argv[3], 10) || 10000;

const seen = new Uint8Array(NUM_PATTERNS);
const st   = createState(N);

let totalMoves = 0;
for (let g = 0; g < GAMES; g++) {
  const game = new Game2(N);
  while (!game.gameOver) {
    extractFeatures(game, st);
    for (let i = 0; i < st.count; i++) seen[st.patIds[i]] = 1;
    totalMoves += st.count;
    const m = game.randomLegalMove();
    game.play(m >= 0 ? m : -1);
  }
}

let seenCount = 0;
for (let i = 0; i < NUM_PATTERNS; i++) if (seen[i]) seenCount++;

console.log(`board: ${N}×${N},  games: ${GAMES},  moves sampled: ${totalMoves}`);
console.log(`distinct patIds seen: ${seenCount} / ${NUM_PATTERNS}`);
