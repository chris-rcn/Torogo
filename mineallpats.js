#!/usr/bin/env node
'use strict';

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
const { Game2, PASS, parseMove } = require('./game2.js');
const { patternHashes2 } = require('./patterns2.js');

const args = process.argv.slice(2);
const get  = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

const file = get('--file', null);

if (!file) {
  console.error('Usage: node minepatternsB2.js --file <path>');
  process.exit(1);
}

const lines = fs.readFileSync(file, 'utf8').trim().split('\n').filter(l => l.trim());

// Map from patternHashes2 → { sum: number, count: number }
const stats = new Map();

function addSample(hash, value) {
  let s = stats.get(hash);
  if (!s) { s = { sum: 0, count: 0 }; stats.set(hash, s); }
  s.sum += value;
  s.count++;
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

  // Collect valid candidates, then batch-hash them.
  const valid = [];
  for (const raw of candidates) {
    const c = normalise(raw);
    if (c.m === 'pass' || c.kwr == null) continue;
    const idx = parseMove(c.m, boardSize);
    if (game.isTrueEye(idx)) continue;
    valid.push({ idx, value: (c.kwr - passKwr) / 1000 });
  }

  const hashes = patternHashes2(game, valid.map(c => c.idx));
  for (let i = 0; i < hashes.length; i++) {
    addSample(hashes[i].pHash, valid[i].value);
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
