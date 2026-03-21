#!/usr/bin/env node
'use strict';

// minepatternsB2.js — mine 3×3 pattern value statistics from createmovedetails output.
// Like minepatternsB.js but uses Game2 and patternHash2 for speed.
//
// Usage: node minepatternsB2.js --file <path>
//   --file  path to newline-delimited JSON produced by createmovedetails.js (required)
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
const { Game2, PASS } = require('./game2.js');
const { patternHash2 } = require('./patterns2.js');

const args = process.argv.slice(2);
const get  = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

const file = get('--file', null);

if (!file) {
  console.error('Usage: node minepatternsB2.js --file <path>');
  process.exit(1);
}

const lines = fs.readFileSync(file, 'utf8').trim().split('\n').filter(l => l.trim());

// Map from patternHash2 → { sum: number, count: number }
const stats = new Map();

function addSample(hash, value) {
  let s = stats.get(hash);
  if (!s) { s = { sum: 0, count: 0 }; stats.set(hash, s); }
  s.sum += value;
  s.count++;
}

// Parse a move string ("a1", "k11", "pass") into an integer index or PASS.
function parseMove(m, N) {
  if (m === 'pass') return PASS;
  const x = m.charCodeAt(0) - 97;
  const y = parseInt(m.slice(1), 10) - 1;
  return y * N + x;
}

// Normalise a candidate entry to { m: string, kwr: number|null }.
function normalise(c) {
  if (c.m !== undefined) return { m: c.m, kwr: c.kwr ?? null };
  return { m: c.move, kwr: null };
}

for (const line of lines) {
  let entry;
  try { entry = JSON.parse(line); } catch (e) { continue; }

  const { boardSize, history, candidates } = entry;
  if (!boardSize || !Array.isArray(history) || !Array.isArray(candidates)) continue;

  // Find the pass candidate's kwr.  Skip this position if it's unavailable.
  const passEntry = candidates.map(normalise).find(c => c.m === 'pass');
  if (!passEntry || passEntry.kwr == null) continue;
  const passKwr = passEntry.kwr;

  // Reconstruct the game state by replaying history moves.
  const game = new Game2(boardSize);
  for (const m of history) {
    game.play(parseMove(m, boardSize));
  }

  // Record a sample for every non-pass, non-true-eye candidate.
  for (const raw of candidates) {
    const c = normalise(raw);
    if (c.m === 'pass') continue;
    if (c.kwr == null) continue;

    const idx = parseMove(c.m, boardSize);
    if (game.isTrueEye(idx)) continue;

    const hash  = patternHash2(game, idx, game.current);
    const value = (c.kwr - passKwr) / 1000;
    addSample(hash, value);
  }
}

// Compute means, then normalize to [0, 1].
const means = new Map();
for (const [hash, { sum, count }] of stats) means.set(hash, sum / count);

let minVal = Infinity, maxVal = -Infinity;
for (const v of means.values()) {
  if (v < minVal) minVal = v;
  if (v > maxVal) maxVal = v;
}
const range = maxVal - minVal || 1;

// Output: hash,normalized_mean,count
for (const [hash, mean] of means) {
  const { count } = stats.get(hash);
  const normalized = (mean - minVal) / range;
  console.log(`${hash},${normalized.toFixed(6)},${count}`);
}
