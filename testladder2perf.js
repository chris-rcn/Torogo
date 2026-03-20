#!/usr/bin/env node
'use strict';

// testladder2perf.js — throughput benchmark for getLadderStatus2.
//
// Usage: node testladder2perf.js [--duration <s>]
//   --duration  seconds to run  (default: run forever)
//
// Each game iteration plays a full 13x13 random game to completion.
// After every single move, getLadderStatus2 is called for every group
// with 1 or 2 liberties.
// Reports: games/s, getLadderStatus2 calls/s, and µs/call.

const { performance } = require('perf_hooks');

const args = process.argv.slice(2);
const get  = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

const duration = parseFloat(get('--duration', 'Infinity'));

const { Game2, PASS } = require('./game2.js');
const { getLadderStatus2 } = require('./ai/ladder2.js');

const N   = 13;
const cap = N * N;

// Play one random legal non-true-eye move (or pass if none available).
function playOneRandom(game2) {
  // Fast path: random probes.
  for (let k = 0; k < 32; k++) {
    const idx = Math.floor(Math.random() * cap);
    if (game2.cells[idx] !== 0) continue;
    if (game2.isTrueEye(idx))   continue;
    if (game2.isLegal(idx))     { game2.play(idx); return; }
  }
  // Slow path: scan all empties in random order.
  const cands = [];
  for (let i = 0; i < cap; i++) {
    if (game2.cells[i] === 0 && !game2.isTrueEye(i) && game2.isLegal(i)) cands.push(i);
  }
  for (let i = cands.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const t = cands[i]; cands[i] = cands[j]; cands[j] = t;
  }
  if (cands.length > 0) { game2.play(cands[0]); return; }
  game2.play(PASS);
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

let games      = 0;
let totalCalls = 0;
let totalPos   = 0;  // number of positions probed (one per move + game start)
let nextPrint  = 10;
const deadline = performance.now() + duration * 1000;
const start    = performance.now();

const game2 = new Game2(N);

for (;;) {
  game2.reset();
  // Probe after the constructor's initial stone.
  totalCalls += probeAllGroups(game2);
  totalPos++;

  while (!game2.gameOver) {
    playOneRandom(game2);
    totalCalls += probeAllGroups(game2);
    totalPos++;
  }
  games++;

  if (games >= nextPrint) {
    const elapsed   = (performance.now() - start) / 1000;
    const gamesPerS = (games / elapsed).toFixed(1);
    const callsPerS = (totalCalls / elapsed).toFixed(0);
    const usPerCall = totalCalls > 0 ? ((elapsed * 1e6) / totalCalls).toFixed(2) : '-';
    const usPerPos  = totalPos   > 0 ? ((elapsed * 1e6) / totalPos).toFixed(1)   : '-';
    console.log(
      `games: ${games}  games/s: ${gamesPerS}  ` +
      `ladder calls: ${totalCalls}  calls/s: ${callsPerS}  µs/call: ${usPerCall}  ` +
      `positions: ${totalPos}  µs/pos: ${usPerPos}`
    );
    nextPrint = Math.ceil(nextPrint * 1.5);
  }

  if (performance.now() >= deadline) break;
}
