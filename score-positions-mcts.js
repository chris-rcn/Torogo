#!/usr/bin/env node
'use strict';

// score-positions-mcts.js — sample random mid-game positions from recorded games
// and estimate their value via MCTS (ai/mcts-terr.js).
//
// For each input game, two random positions are sampled: one from the first
// half of the move sequence and one from the second half.  The position is
// then scored by running MCTS for --budget milliseconds and reading the
// territory estimate from the search tree.
//
// Usage:
//   node score-positions-mcts.js --file <games-file> --budget <ms>
//
// Input format (one game per line):
//   <size>,<move>,<move>,...
//   where each move is two letters (column then row, a=0), pass = ..
//   e.g.  11,ff,ba,bg,...
//
// Output (one line per game, space-separated):
//   <move-list> <avg-black-territory> <budget-ms>
//   move-list uses the same encoding as the input

const fs      = require('fs');
const { Game2, PASS } = require('./game2.js');
const { getMove } = require('./ai/mcts-terr.js');

const args   = process.argv.slice(2);
const get    = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

const file   = get('--file', null);
const budget = parseInt(get('--budget', '1'), 10);

if (!file || budget <= 0) {
  process.stderr.write('Usage: node score-positions-mcts.js --file <path> --budget <ms>\n');
  process.exit(1);
}

function mulberry32(seed) {
  return function() {
    seed |= 0;
    seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

const random = mulberry32(42); // 42 is the seed

// ── Move encoding ─────────────────────────────────────────────────────────────

const colChar = x => String.fromCharCode(97 + x);
const rowChar = y => String.fromCharCode(97 + y);

function decodeMove(s, N) {
  if (s === '..') return PASS;
  const x = s.charCodeAt(0) - 97;
  const y = s.charCodeAt(1) - 97;
  return y * N + x;
}

function encodeMove(idx, N) {
  if (idx === PASS) return '..';
  return colChar(idx % N) + rowChar((idx / N) | 0);
}

// ── Main ──────────────────────────────────────────────────────────────────────

const lines = fs.readFileSync(file, 'utf8').trim().split('\n');

for (const line of lines) {
  if (!line.trim()) continue;

  const fields = line.split(',');
  const N      = parseInt(fields[0], 10);

  // fields[1] is the center stone placed automatically by the Game2 constructor
  // — skip it when replaying (it's already applied).  fields[2..] are the moves.
  const movesInFile = fields.slice(2).map(s => decodeMove(s.trim(), N));

  const n = movesInFile.length;
  if (n < 20) continue;

  const minMoves = 4;
  const half = Math.floor((n - minMoves) / 2);

  // Pick a single offset in [minMoves, half] and apply it from both the start
  // and the midpoint, guaranteeing the two positions are exactly half apart.
  const offset = minMoves + Math.floor(random() * (half - minMoves + 1));
  const prefixLens = [offset, half + offset];

  const center    = (N >> 1);
  const centerStr = colChar(center) + rowChar(center);

  for (const prefixLen of prefixLens) {
    const selectedMoves = movesInFile.slice(0, prefixLen);

    // Replay the selected moves to reconstruct the position.
    const g = new Game2(N);  // constructor places black at center
    for (const idx of selectedMoves) {
      if (g.gameOver) break;
      g.play(idx);
    }
    if (g.gameOver) continue;

    // Score the position via MCTS.
    const result = getMove(g, budget);
    const blackTerritory = result.blackTerritory;

    // Emit the position as a move list (same encoding as input), then the score.
    // Re-include the center stone so the output is self-contained.
    const moveParts = [N, centerStr, ...selectedMoves.map(idx => encodeMove(idx, N))];
    process.stdout.write(`${moveParts.join(',')} ${blackTerritory.toFixed(3)} ${budget}\n`);
  }
}
