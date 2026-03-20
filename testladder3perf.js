#!/usr/bin/env node
'use strict';

// testladder3perf.js — throughput benchmark for getLadderStatus3.
//
// Usage: node testladder3perf.js [--duration <s>]
//   --duration  seconds to run  (default: run forever)
//
// Each game iteration plays a full 13x13 random game to completion.
// After every single move, getLadderStatus3 is called for every group
// with 1 or 2 liberties.
// Reports: games/s, getLadderStatus3 calls/s, and µs/call.

const { performance } = require('perf_hooks');

const args = process.argv.slice(2);
const get  = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

const duration = parseFloat(get('--duration', 'Infinity'));

const { Game3, PASS } = require('./game3.js');
const { getLadderStatus3 } = require('./ai/ladder3.js');

const N   = 13;
const cap = N * N;

// Play one random legal non-true-eye move (or pass if none available).
function playOneRandom(game3) {
  // Fast path: random probes.
  for (let k = 0; k < 32; k++) {
    const idx = Math.floor(Math.random() * cap);
    if (game3.cells[idx] !== 0) continue;
    if (game3.isTrueEye(idx))   continue;
    if (game3.isLegal(idx))     { game3.play(idx); return; }
  }
  // Slow path: scan all empties in random order.
  const cands = [];
  for (let i = 0; i < cap; i++) {
    if (game3.cells[i] === 0 && !game3.isTrueEye(i) && game3.isLegal(i)) cands.push(i);
  }
  for (let i = cands.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const t = cands[i]; cands[i] = cands[j]; cands[j] = t;
  }
  if (cands.length > 0) { game3.play(cands[0]); return; }
  game3.play(PASS);
}

// Call getLadderStatus3 for every group with 1 or 2 liberties.
function probeAllGroups(game3) {
  const visitedGids = new Uint8Array(game3._nextGid + 1);
  for (let i = 0; i < cap; i++) {
    if (game3.cells[i] === 0) continue;
    const gid = game3._gid[i];
    if (visitedGids[gid]) continue;
    visitedGids[gid] = 1;
    const lc = game3._ls[gid];
    if (lc < 1 || lc > 2) continue;
    getLadderStatus3(game3, i);
  }
}

let games    = 0;
let totalPos = 0;  // number of positions probed (one per move + game start)
let nextPrint  = 10;
const deadline = performance.now() + duration * 1000;
const start    = performance.now();

const game3 = new Game3(N);

for (;;) {
  game3.reset();
  // Probe after the initial empty board.
  probeAllGroups(game3);
  totalPos++;

  while (!game3.gameOver) {
    playOneRandom(game3);
    probeAllGroups(game3);
    totalPos++;
  }
  games++;

  if (games >= nextPrint) {
    const elapsed = (performance.now() - start) / 1000;
    const usPerPos = totalPos > 0 ? ((elapsed * 1e6) / totalPos).toFixed(1) : '-';
    console.log(`games: ${games}  positions: ${totalPos}  µs/pos: ${usPerPos}`);
    nextPrint = Math.ceil(nextPrint * 1.5);
  }

  if (performance.now() >= deadline) break;
}
