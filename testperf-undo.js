#!/usr/bin/env node
'use strict';
const { performance } = require('perf_hooks');
const { Game2 } = require('./game2.js');
const { Game3 } = require('./game3.js');

const args = process.argv.slice(2);
const get  = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };
const N    = parseInt(get('--size', '9'),  10);
const REPS = parseInt(get('--reps', '3000'), 10);

function makePrng(seed) {
  let s = seed >>> 0;
  return () => { s = Math.imul(s, 1664525) + 1013904223 >>> 0; return s; };
}

// Play until `target` stones placed; return move sequence.
function buildPosition(game, seed, target) {
  const rand = makePrng(seed);
  game.reset();
  const played = [];
  let attempts = 0;
  while (played.length < target && !game.gameOver && attempts < N * N * 20) {
    attempts++;
    const idx = rand() % (N * N);
    if (game.cells[idx] !== 0) continue;
    if (game.isLegal(idx)) { game.play(idx); played.push(idx); }
  }
  return played;
}

function replayMoves(game, moves) {
  game.reset();
  for (const idx of moves) game.play(idx);
}

function legalMoves(game) {
  const moves = [];
  for (let i = 0; i < N * N; i++) {
    if (game.cells[i] === 0 && game.isLegal(i)) moves.push(i);
  }
  return moves;
}

function bench(label, tryMoves, fn) {
  const t0 = performance.now();
  for (let r = 0; r < REPS; r++) {
    for (const idx of tryMoves) fn(idx);
  }
  const ms = performance.now() - t0;
  return { label, ms, usPerOp: ms / (REPS * tryMoves.length) * 1000 };
}

// ── Phases ────────────────────────────────────────────────────────────────────

const cap    = N * N;
const phases = [
  { name: 'opening', target: Math.floor(cap * 0.05) },
  { name: 'early',   target: Math.floor(cap * 0.20) },
  { name: 'mid',     target: Math.floor(cap * 0.45) },
  { name: 'late',    target: Math.floor(cap * 0.70) },
];

console.log(`Board: ${N}x${N}  |  reps: ${REPS}\n`);
console.log(`${'phase'.padEnd(10)} ${'depth'.padStart(5)} ${'legal'.padStart(6)}   ${'clone+play (µs)'.padStart(16)}   ${'play+undo (µs)'.padStart(15)}   ${'ratio'.padStart(7)}`);
console.log('─'.repeat(75));

for (const { name, target } of phases) {
  const g2 = new Game2(N);
  const seq = buildPosition(g2, 42, target);
  const tryMoves = legalMoves(g2);

  const g3 = new Game3(N);
  replayMoves(g3, seq);

  if (tryMoves.length === 0) {
    console.log(`${name.padEnd(10)} ${String(seq.length).padStart(5)} ${'0'.padStart(6)}   (no legal moves)`);
    continue;
  }

  // Warmup
  for (let r = 0; r < 100; r++) {
    for (const idx of tryMoves) { const c = g2.clone(); c.play(idx); }
  }
  for (let r = 0; r < 100; r++) {
    for (const idx of tryMoves) { g3.play(idx); g3.undo(); }
  }

  const rClone = bench('clone', tryMoves, idx => { const c = g2.clone(); c.play(idx); });
  const rUndo  = bench('undo',  tryMoves, idx => { g3.play(idx); g3.undo(); });

  const ratio = rClone.ms / rUndo.ms;
  console.log(
    `${name.padEnd(10)} ${String(seq.length).padStart(5)} ${String(tryMoves.length).padStart(6)}` +
    `   ${rClone.usPerOp.toFixed(3).padStart(16)}` +
    `   ${rUndo.usPerOp.toFixed(3).padStart(15)}` +
    `   ${ratio.toFixed(2).padStart(6)}x`
  );
}
