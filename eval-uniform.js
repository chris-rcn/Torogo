#!/usr/bin/env node
'use strict';

// eval-uniform.js — Measure |ΔV| = |V*(s) − V| under the uniform random policy.
//
// For each position in the file, V is estimated by averaging z ∈ {−1,+1} over
// N rollouts using randomLegalMove.  V*(s) is the best candidate's kwr/1000.
// Reports mean and stddev of |ΔV| across all positions.
//
// Usage:
//   node eval-uniform.js --file <datafile> [--N <n>] [--position-limit <n>]

const fs = require('fs');
const { Game2, PASS, parseMove } = require('./game2.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help']);
if (opts.help || !opts.file) {
  console.error('Usage: node eval-uniform.js --file <datafile> [--N <n>] [--position-limit <n>]');
  process.exit(1);
}

const dataFile      = opts.file;
const N             = opts.N             !== undefined ? parseInt(opts.N, 10)            : 50;
const positionLimit = opts['position-limit'] !== undefined ? parseInt(opts['position-limit'], 10) : Infinity;

const lines = fs.readFileSync(dataFile, 'utf8').split('\n').filter(l => l.trim()).slice(0, positionLimit);

function rollout(game, player) {
  const sim = game.clone();
  while (!sim.gameOver) {
    sim.play(sim.randomLegalMove());
  }
  return sim.estimateWinner() === player ? 1 : -1;
}

const deltas = [];

for (const line of lines) {
  const { boardSize, history, candidates } = JSON.parse(line);

  const game = new Game2(boardSize);
  let ok = true;
  for (const c of history) {
    if (!game.play(parseMove(c, boardSize))) { ok = false; break; }
  }
  if (!ok || game.gameOver) continue;

  const best = candidates.find(c => c.kwr !== null);
  if (!best) continue;
  const vStar = 2 * (best.kwr / 1000) - 1;

  const player = game.current;
  let V = 0;
  for (let i = 0; i < N; i++) V += rollout(game, player);
  V /= N;

  deltas.push(Math.abs(vStar - V));
}

const n = deltas.length;
if (n === 0) { console.error('No positions evaluated.'); process.exit(1); }

const mean = deltas.reduce((s, d) => s + d, 0) / n;
const variance = deltas.reduce((s, d) => s + (d - mean) ** 2, 0) / n;
const stddev = Math.sqrt(variance);

console.log(`positions: ${n},  N: ${N}`);
console.log(`mean |ΔV|: ${mean.toFixed(4)}  stddev: ${stddev.toFixed(4)}`);
