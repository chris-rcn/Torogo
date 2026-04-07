#!/usr/bin/env node
'use strict';

// gen-evals.js — play full games, pick a random position, re-evaluate with a longer dwell,
// and emit one line per sample: <size> <move-sequence> <rootWinRatio>
//
// Usage:
//   node gen-evals.js --agent <name> --size <n> --budget <ms> --dwell <ms>
//
//   --agent   agent name under ai/ (e.g. rave)
//   --size    board size (e.g. 9)
//   --budget  ms per move when playing the full game
//   --dwell   ms budget for the re-evaluation at the sampled position

const path = require('path');
const { Game2, PASS } = require('./game2.js');

const args = process.argv.slice(2);
const get  = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

const agentName = get('--agent', null);
const size      = parseInt(get('--size',   '9'),    10);
const budgetMs  = parseInt(get('--budget', '100'),  10);
const dwellMs   = parseInt(get('--dwell',  '1000'), 10);

if (!agentName) {
  process.stderr.write('Usage: node gen-evals.js --agent <name> --size <n> --budget <ms> --dwell <ms>\n');
  process.exit(1);
}

const { getMove } = require(path.join(__dirname, 'ai', agentName + '.js'));

function moveToStr(idx, N) {
  if (idx === PASS) return '..';
  return String.fromCharCode(97 + idx % N) + String.fromCharCode(97 + (idx / N | 0));
}

function getMoveIdx(result, N) {
  if (result.move !== undefined) return result.move;
  return result.type === 'place' ? result.y * N + result.x : PASS;
}

for (;;) {
  // Play a full game recording every move index.
  const game = new Game2(size);
  const moves = [];
  while (!game.gameOver) {
    const r = getMove(game, budgetMs);
    const idx = getMoveIdx(r, size);
    game.play(idx);
    moves.push(idx);
  }

  // Find the index of the first pass.
  const firstPass = moves.indexOf(PASS);

  // Valid sample range: at least 5 moves in, at least 10 before the first pass.
  const lo = 5;
  const hi = (firstPass === -1 ? moves.length : firstPass) - 10;
  if (hi < lo) continue;

  const pos = lo + Math.floor(Math.random() * (hi - lo + 1));

  // Replay the game up to that position.
  const replay = new Game2(size);
  for (let i = 0; i < pos; i++) replay.play(moves[i]);

  // Re-evaluate with the dwell budget.
  const r = getMove(replay, dwellMs);
  const winRatio = r.rootWinRatio !== undefined ? r.rootWinRatio : 0.5;

  // Emit: size <space> move-sequence <space> winRatio
  const seq = moves.slice(0, pos).map(m => moveToStr(m, size)).join('');
  process.stdout.write(`${size} ${seq} ${winRatio}\n`);
}
