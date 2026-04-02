#!/usr/bin/env node
'use strict';

// build-book.js — build an opening book by iteratively extending it one position
// at a time.
//
// Usage: node build-book.js --agent <name> --size <n> --budget <ms> [options]
//
// Options:
//   --agent <name>   AI agent in ai/ folder (required)
//   --size  <n>      Board size (default: 13)
//   --budget <ms>    Time budget per agent call in ms (required)
//   --in    <path>   Book file to read from (default: none)
//   --out   <path>   Book file to write to (default: book.json)
//   --help
//
// Algorithm (per outer iteration):
//   1. Start a fresh game (center move pre-placed by Game2).
//   2. Inner loop:
//      a. Check if the current position is already in the book.
//      b. Run the agent and record the returned move.
//      c. If position was new → break (book extended by one position).
//      d. Otherwise → play the move and continue.
//   3. Save book to disk.
//
// Each iteration extends the book by exactly one new position.  Positions on
// the main line accumulate higher selection counts the more often they are
// traversed.

const fs   = require('fs');
const path = require('path');
const { performance } = require('perf_hooks');
const { Game2, PASS, BLACK, WHITE } = require('./game2.js');
const { addToBook, lookupBook, serializeBook, deserializeBook, canonicalHash, INV_T, applyTransform } = require('./book.js');

const args = process.argv.slice(2);
const get  = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };
const has  = (flag) => args.includes(flag);

if (has('--help') || has('-h')) {
  console.log('Usage: node build-book.js --agent <name> [--budget <ms>] [--size <n>] [--in <book.json>] [--out <book.json>]');
  process.exit(0);
}

const agentName = get('--agent', null);
if (!agentName) { console.error('--agent is required'); process.exit(1); }
const boardSize = parseInt(get('--size', 13), 10);
const budgetMs  = parseInt(get('--budget', 5000), 10);
const inFile    = get('--in',  null);
const outFile   = get('--out', path.join('out', `book-${Date.now()}.js`));

fs.mkdirSync(path.dirname(outFile), { recursive: true });

const { getMove } = require(path.join(__dirname, 'ai', agentName + '.js'));

// Load existing book from --in if provided.
let book = new Map();
book.minEmptyCount = Infinity;
if (inFile) {
  book = deserializeBook(require(path.resolve(inFile)));
  console.log(`Loaded ${book.size} positions from ${inFile}`);
} else {
  console.log('Starting new book.');
}
console.log(`Output: ${outFile}`);

const N = boardSize;

// Print the book entries for a given game state as a board with counts.
function printBookBoard(game, label) {
  console.log(`${game.current===BLACK?'●':'○'} to play:`);
  const cells = game.cells;
  const { hash, t } = canonicalHash(cells, N);
  const entry = book.get(hash);

  const counts = new Map();
  let passCount = 0;
  if (entry) {
    for (const [canonMove, count] of entry) {
      if (canonMove === PASS) { passCount += count; continue; }
      const cx = canonMove % N, cy = (canonMove / N) | 0;
      const [ax, ay] = applyTransform(INV_T[t], cx, cy, N);
      counts.set(ay * N + ax, count);
    }
  }

  const maxCount = counts.size > 0 ? Math.max(...counts.values()) : 0;
  const W = Math.max(3, String(maxCount).length + 1);

  if (label) console.log(`\n${label}`);
  else console.log();
//  const letters = Array.from({length: N}, (_, i) => String.fromCharCode(97 + i));
//  console.log('   ' + letters.map(l => l.padStart(W)).join(''));
  for (let y = 0; y < N; y++) {
    let row = '';
//    row += String.fromCharCode(97 + y) + '  ';
    for (let x = 0; x < N; x++) {
      const idx = y * N + x;
      const c = cells[idx];
      let s;
      if      (c === BLACK)      s = '●';
      else if (c === WHITE)      s = '○';
      else if (counts.has(idx))  s = String(counts.get(idx));
      else                       s = '·';
      row += s.padStart(W);
    }
    console.log(row);
  }
  if (passCount > 0) console.log(`   pass: ${passCount}`);
}

let iterations = 0;
let totalDepth = 0;
const startTime = performance.now();

while (true) {
  const game = new Game2(N);
  let depth = 0;
  const snapshots = [game.clone()];   // snapshot before each move

  while (!game.gameOver) {
    const inBook = lookupBook(book, game);
    const result = getMove(game, budgetMs);
    const moveIdx = result.type === 'place' ? result.move : PASS;
    addToBook(book, game, moveIdx);

    if (!inBook) break;   // new/low-experience position — book extended

    depth++;
    game.play(moveIdx);
    snapshots.push(game.clone());
  }

  iterations++;
  totalDepth += depth + 1;
  fs.writeFileSync(outFile, serializeBook(book), 'utf8');

  const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
  const avgDepth = (totalDepth / iterations).toFixed(2);
  console.log(`\niter=${iterations}  positions=${book.size}  avgDepth=${avgDepth}  minEmpty=${book.minEmptyCount}  elapsed=${elapsed}s  out=${outFile}`);
  for (const snap of snapshots) {
    printBookBoard(snap, '');
  }
}
