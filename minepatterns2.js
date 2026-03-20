#!/usr/bin/env node
'use strict';

// minepatterns2.js — mine 3×3 pattern value statistics from gengamedata output.
//
// Usage: node minepatterns2.js --file <path> [--size <n>]
//   --file  path to newline-delimited JSON produced by gengamedata.js (required)
//   --size  board size (default: 11)
//
// For every position in the input file, for every legal non-true-eye candidate
// move (excluding pass), the pattern value sample is computed as:
//
//   value = (kwr_move - kwr_pass) / 1000
//
// where kwr is winRatio × 1000.  Values can be negative.
//
// Output: one line per observed pattern hash:
//   <hash>,<mean_value>,<count>

const fs = require('fs');
const { Game, DEFAULT_KOMI } = require('./game.js');
const { patternHash } = require('./patterns.js');

const args = process.argv.slice(2);
const get  = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

const file = get('--file', null);
const size = parseInt(get('--size', '11'), 10);

if (!file) {
  console.error('Usage: node minepatterns2.js --file <path> [--size <n>]');
  process.exit(1);
}

if (isNaN(size) || size < 2) { console.error('--size must be >= 2'); process.exit(1); }

const lines = fs.readFileSync(file, 'utf8').trim().split('\n').filter(l => l.trim());

// Map from patternHash → { sum: number, count: number }
const stats = new Map();

function addSample(hash, value) {
  let s = stats.get(hash);
  if (!s) { s = { sum: 0, count: 0 }; stats.set(hash, s); }
  s.sum += value;
  s.count++;
}

// Parse a move string ("a1", "k11", "pass") into { type, x?, y? }.
function parseMove(m) {
  if (m === 'pass') return { type: 'pass' };
  const x = m.charCodeAt(0) - 97;
  const y = parseInt(m.slice(1), 10) - 1;
  return { type: 'place', x, y };
}

// Normalise a candidate entry to { m: string, kwr: number|null }.
// gengamedata.js uses { m, kwr } for normal moves, { move, winRatio: null }
// for the game-ending double-pass case.
function normalise(c) {
  if (c.m !== undefined) return { m: c.m, kwr: c.kwr ?? null };
  return { m: c.move, kwr: null };
}

for (const line of lines) {
  let entry;
  try { entry = JSON.parse(line); } catch (e) { continue; }

  const { position, candidates } = entry;
  if (!Array.isArray(position) || !Array.isArray(candidates)) continue;

  // Find the pass candidate's kwr.  Skip this position if it's unavailable.
  const passEntry = candidates.map(normalise).find(c => c.m === 'pass');
  if (!passEntry || passEntry.kwr == null) continue;
  const passKwr = passEntry.kwr;

  // Reconstruct the game state by replaying position moves.
  const game = new Game(size, DEFAULT_KOMI);
  for (const m of position) {
    const move = parseMove(m);
    if (move.type === 'pass') game.pass();
    else game.placeStone(move.x, move.y);
  }

  // Record a sample for every non-pass, non-true-eye candidate.
  for (const raw of candidates) {
    const c = normalise(raw);
    if (c.m === 'pass') continue;
    if (c.kwr == null) continue;

    const move = parseMove(c.m);
    const { x, y } = move;

    if (game.board.isTrueEye(x, y, game.current)) continue;

    const hash  = patternHash(game, x, y, game.current);
    const value = (c.kwr - passKwr) / 1000;
    addSample(hash, value);
  }
}

// Output: hash,mean,count
for (const [hash, { sum, count }] of stats) {
  const mean = sum / count;
  console.log(`${hash},${mean.toFixed(6)},${count}`);
}
