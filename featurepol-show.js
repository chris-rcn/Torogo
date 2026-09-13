'use strict';

// featurepol-show.js — render a position with the featurepol policy's move
// likelihoods.  Board size and a move sequence (comma-separated coordinate
// moves, e.g. "g7,h10,e7") replay from an EMPTY board: the list carries every
// stone, auto-stone included, so the first listed move is black's.  This is
// the full-move-list convention — this tool adopts it first; today's eval
// files still OMIT the free stone, so prepend the centre coord when pasting
// a sequence from one of them;
// the board prints stones as circles and every legal move as a 3-digit
// softmax per-mille (temperature 1, clamped to 999).  Empty cells that are
// not legal moves (illegal or true eyes) print a dot.
//
// Usage: node featurepol-show.js <size> [moves]
//   FPOL_DATA   featurepol model to load (same env the agent uses)
//
// Example: FPOL_DATA=out/featurepol-xyz.js node featurepol-show.js 9 c3,g7,e3

const path = require('path');
const FeaturePol = require('./featurepol-lib.js');
const { Game2, parseMove, PASS, BLACK } = require('./game2.js');

const [sizeArg, movesArg] = process.argv.slice(2);
if (!sizeArg) { console.error('usage: node featurepol-show.js <size> [moves]'); process.exit(1); }
const N = parseInt(sizeArg, 10);

const FPOL_DATA = process.env.FPOL_DATA || path.join(__dirname, 'featurepol-cbk7wa32.js');
const { weights, modelName } = FeaturePol.loadModel({ name: 'featurepol', path: FPOL_DATA });

const game = new Game2(N, false);   // empty board: the move list carries every stone
const moves = movesArg ? movesArg.split(',').map(s => s.trim()).filter(Boolean) : [];
for (const m of moves) {
  const idx = parseMove(m, N);
  if (idx !== PASS && (idx < 0 || idx >= N * N)) { console.error(`bad move "${m}" for size ${N}`); process.exit(1); }
  if (!game.play(idx)) { console.error(`illegal move "${m}" (move ${moves.indexOf(m) + 1})`); process.exit(1); }
}

const state = FeaturePol.createState(N, weights.spec);
FeaturePol.extractFeatures(game, state, weights);
FeaturePol.computeSoftmax(state, weights, 1);

// pct: per-mille, rounded, clamped to 999 (three digits of precision).
const pct = new Map();
for (let i = 0; i < state.count; i++) {
  pct.set(state.moves[i], String(Math.min(999, Math.round(state.probs[i] * 1000))).padStart(3));
}

console.log(`model: ${modelName}  spec: ${weights.spec.str}`);
console.log(`size: ${N}  moves: ${moves.length}`);
console.log();
console.log(game.current === BLACK ? 'black(●) to move' : 'white(○) to move');
// Column letters across the top, row numbers down the left; rows print
// top-to-bottom as N..1, matching parseBoard/coordStr orientation.
const colHdr = [];
for (let x = 0; x < N; x++) colHdr.push(String.fromCharCode(97 + x).padStart(3));
console.log('   ' + colHdr.join(' '));
for (let y = N - 1; y >= 0; y--) {
  const cells = [];
  for (let x = 0; x < N; x++) {
    const idx = y * N + x, c = game.cells[idx];
    if (c === BLACK) cells.push('  ●');
    else if (c !== 0) cells.push('  ○');
    else if (pct.has(idx)) cells.push(pct.get(idx));
    else cells.push('  ·');
  }
  console.log(String(y + 1).padStart(2) + ' ' + cells.join(' '));
}
