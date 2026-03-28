#!/usr/bin/env node
'use strict';

// score-positions.js — sample random mid-game positions from recorded games
// and estimate their value via uniform random playouts.
//
// For each input game, two random positions are sampled: one from the first
// half of the move sequence and one from the second half.  The position is
// then scored by running many lightweight random playouts to completion and
// averaging the resulting black territory (area score, komi not included).
//
// This produces ground-truth position scores for use as a training baseline
// when optimizing a pattern-based playout policy.
//
// Usage:
//   node score-positions.js --file <games-file> --playouts <n>
//
// Input format (one game per line):
//   <size>,<move>,<move>,...
//   where each move is two letters (column then row, a=0), pass = ..
//   e.g.  11,ff,ba,bg,...
//
// Output (one line per game, space-separated):
//   <move-list> <avg-black-territory> <playouts-used>
//   move-list uses the same encoding as the input

const fs   = require('fs');
const path = require('path');
const { Game2, PASS, BLACK } = require('./game2.js');

const args  = process.argv.slice(2);
const get   = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

const file     = get('--file', null);
const playouts = parseInt(get('--playouts', '0'), 10);

if (!file || playouts <= 0) {
  process.stderr.write('Usage: node score-positions.js --file <path> --playouts <n>\n');
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

// ── Playout ───────────────────────────────────────────────────────────────────

// Play out game2 to completion using uniform random moves (no passes until
// no legal non-eye move exists).  Returns black's area score (komi excluded).
// game2 is mutated — clone before calling if you need the original.
function runPlayout(game2) {
  const N   = game2.N;
  const cap = N * N;
  const cells = game2.cells;

  // Maintain a list of empty cells; shuffle-and-pop to pick random moves.
  const empty = [];
  for (let i = 0; i < cap; i++) if (cells[i] === 0) empty.push(i);

  const moveLimit = empty.length + 20;  // safety cap against infinite loops
  let moves = 0;

  while (!game2.gameOver && moves < moveLimit) {
    let placed = false;
    let end = empty.length;

    // Partial Fisher-Yates: try candidates in random order until a legal
    // non-eye move is found.
    while (end > 0) {
      const ri  = Math.floor(Math.random() * end);
      const idx = empty[ri];
      empty[ri] = empty[end - 1];
      empty[end - 1] = idx;
      end--;

      if (game2.isTrueEye(idx)) continue;
      if (!game2.isLegal(idx))  continue;

      const { capturedCount } = game2.playInfo(idx);
      empty[end] = empty[empty.length - 1];
      empty.pop();

      // Captures free up cells not in our empty list — rebuild from scratch.
      if (capturedCount) {
        empty.length = 0;
        for (let i = 0; i < cap; i++) if (cells[i] === 0) empty.push(i);
      }

      placed = true;
      moves++;
      break;
    }

    if (!placed) { game2.play(PASS); moves++; }  // no legal move → pass
  }

  return game2.calcScore().black;
}

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

  // The input file's fields[1] is the center stone placed automatically by
  // the Game2 constructor — skip it when replaying (it's already applied).
  // fields[2..] are the remaining moves in order.
  const movesInFile = fields.slice(2).map(s => decodeMove(s.trim(), N));

  const n = movesInFile.length;
  if (n < 20) continue;

  const minMoves = 4;
  const half = Math.floor((n - minMoves) / 2);

  // Pick a single offset in [minMoves, half] and apply it from both the start
  // and the midpoint, guaranteeing the two positions are exactly half apart.
  const offset = minMoves + Math.floor(Math.random() * (half - minMoves + 1));
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
    if (g.gameOver) continue;  // skip if the prefix itself ended the game

    // Score the position by averaging black territory over many random playouts.
    let totalBlack = 0;
    for (let p = 0; p < playouts; p++) {
      totalBlack += runPlayout(g.clone());
    }
    const avgBlack = totalBlack / playouts;

    // Emit the position as a move list (same encoding as input), then the score.
    // Re-include the center stone so the output is self-contained.
    const moveParts = [N, centerStr, ...selectedMoves.map(idx => encodeMove(idx, N))];
    process.stdout.write(`${moveParts.join(',')} ${avgBlack.toFixed(3)} ${playouts}\n`);
  }
}
