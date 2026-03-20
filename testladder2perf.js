#!/usr/bin/env node
'use strict';

// testladder2perf.js — throughput benchmark for getLadderStatus2.
//
// Usage: node testladder2perf.js [--size <n>] [--moves <m>] [--duration <s>]
//   --size      board size                  (default: 11)
//   --moves     random moves before probing (default: 30)
//   --duration  seconds to run              (default: run forever)
//
// Each iteration:
//   1. Play --moves random legal non-true-eye moves on a Game2.
//   2. Call getLadderStatus2 for every group with 1 or 2 liberties.
// Reports: iterations/s, getLadderStatus2 calls/s, and the ms cost per call.

const { performance } = require('perf_hooks');

const args = process.argv.slice(2);
const get  = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

const boardSize = parseInt(get('--size',     '11'), 10);
const moves     = parseInt(get('--moves',    '30'), 10);
const duration  = parseFloat(get('--duration', 'Infinity'));

if (isNaN(boardSize) || boardSize < 2) { console.error('--size must be >= 2');      process.exit(1); }
if (isNaN(moves)     || moves < 0)     { console.error('--moves must be >= 0');     process.exit(1); }

const { Game2, PASS } = require('./game2.js');
const { getLadderStatus2 } = require('./ai/ladder2.js');

const N   = boardSize;
const cap = N * N;

// Advance a Game2 by up to `n` random legal non-true-eye moves.
function playRandom(game2, n) {
  for (let m = 0; m < n && !game2.gameOver; m++) {
    let placed = false;
    // Fast path: random probes.
    for (let k = 0; k < 32 && !placed; k++) {
      const idx = Math.floor(Math.random() * cap);
      if (game2.cells[idx] !== 0) continue;
      if (game2.isTrueEye(idx))   continue;
      if (game2.isLegal(idx))     { game2.play(idx); placed = true; }
    }
    if (placed) continue;
    // Slow path: scan all empties.
    const cands = [];
    for (let i = 0; i < cap; i++) if (game2.cells[i] !== 0 && !game2.isTrueEye(i) && game2.isLegal(i)) cands.push(i);
    for (let i = cands.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const t = cands[i]; cands[i] = cands[j]; cands[j] = t;
    }
    if (cands.length > 0) { game2.play(cands[0]); placed = true; }
    if (!placed) game2.play(PASS);
  }
}

// Call getLadderStatus2 for every group with 1 or 2 liberties.
// Returns the number of calls made.
function probeAllGroups(game2) {
  const visitedGids = new Uint8Array(game2._nextGid + 1);
  let calls = 0;
  for (let i = 0; i < cap; i++) {
    if (game2.cells[i] === 0) continue;
    const gid = game2._gid[i];
    if (visitedGids[gid]) continue;
    visitedGids[gid] = 1;
    const lc = game2._ls[gid];
    if (lc < 1 || lc > 2) continue;
    getLadderStatus2(game2, i);
    calls++;
  }
  return calls;
}

let iters      = 0;
let totalCalls = 0;
let nextPrint  = 1000;
const deadline = performance.now() + duration * 1000;
const start    = performance.now();

const game2 = new Game2(N);

for (;;) {
  game2.reset();
  playRandom(game2, moves);
  totalCalls += probeAllGroups(game2);
  iters++;

  if (iters >= nextPrint) {
    const elapsed   = (performance.now() - start) / 1000;
    const itersPerS = (iters / elapsed).toFixed(0);
    const callsPerS = (totalCalls / elapsed).toFixed(0);
    const usPerCall = totalCalls > 0 ? ((elapsed * 1e6) / totalCalls).toFixed(2) : '-';
    console.log(
      `iters: ${iters}  iters/s: ${itersPerS}  ` +
      `ladder calls: ${totalCalls}  calls/s: ${callsPerS}  µs/call: ${usPerCall}`
    );
    nextPrint = Math.ceil(nextPrint * 1.5);
  }

  if (performance.now() >= deadline) break;
}
