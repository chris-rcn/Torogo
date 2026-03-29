#!/usr/bin/env node
'use strict';

// learn-patterns.js — online pattern learning via self-play.
//
// Plays training games (ravepatlearn vs ravepatlearn) and after each game
// updates an in-memory pattern table using the winner's moves (mirroring
// mine-pats-selection.js).  Every 4th game is an evaluation game against
// pure rave.  Prints summary on an exponential interval.
//
// Usage:
//   node learn-patterns.js --size <n> --budget <ms>
//   [--rave <ai>]     rave baseline ai name (default: rave)

const { performance } = require('perf_hooks');
const path = require('path');
const { Game2, BLACK, WHITE, PASS } = require('./game2.js');
const { getPatternHashes } = require('./pattern1.js');
const { makeGetMove } = require('./ai/ravepat.js');

const args = process.argv.slice(2);
const get  = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

const boardSize   = parseInt(get('--size',   '0'), 10);
const budgetMs    = parseInt(get('--budget', '0'), 10);
const raveName    =          get('--rave',   'rave');
const winnerOnly  = !args.includes('--all-moves');

if (!boardSize || !budgetMs) {
  process.stderr.write('Usage: node learn-patterns.js --size <n> --budget <ms> [--rave <ai>] [--all-moves]\n');
  process.exit(1);
}

const { getMove: refGetMove } = require(path.join(__dirname, 'ai', raveName + '.js'));

// ── Pattern table ──────────────────────────────────────────────────────────────

// Internal accumulators: hash → { wins, total }
const stats = new Map();

// Live pattern table read by ravepat via makeGetMove: hash → [ratio, count]
const table = new Map();

const CONF_K = parseFloat(process.env.PATTERN_CONF_K || '20');

function bumpPattern(hash, isWin, weight) {
  let s = stats.get(hash);
  if (!s) { s = { wins: 0, total: 0 }; stats.set(hash, s); }
  if (isWin) s.wins += weight;
  s.total += weight;
  table.set(hash, [s.wins / s.total, s.total]);
}

// ── Learn from a completed game ────────────────────────────────────────────────

function learnFromGame(moves, winner, N) {
  const cap  = N * N;
  const game = new Game2(N);

  for (let mi = 0; mi < moves.length; mi++) {
    const selected = moves[mi];
    const color    = game.current;

    if (selected !== PASS && (!winnerOnly || color === winner)) {
      const nonSelected = [];
      for (let i = 0; i < cap; i++) {
        if (i === selected) continue;
        if (game.cells[i] !== 0) continue;
        if (game.isTrueEye(i)) continue;
        if (!game.isLegal(i)) continue;
        nonSelected.push(i);
      }
      const nCandidates = nonSelected.length + 1;
      const hashes = getPatternHashes(game, [selected, ...nonSelected]);
      bumpPattern(hashes[0].hash, true, nCandidates - 1);
      for (let i = 1; i < hashes.length; i++) bumpPattern(hashes[i].hash, false, 1);
    }

    game.play(selected);
  }
}

// ── Play one game, return { winner, moves } ────────────────────────────────────

function playGame(blackPolicy, whitePolicy) {
  const game  = new Game2(boardSize);
  const moves = [];

  while (!game.gameOver) {
    const policy = game.current === BLACK ? blackPolicy : whitePolicy;
    const result = policy(game, budgetMs);
    const idx    = result.type === 'place' ? result.y * boardSize + result.x : PASS;
    game.play(idx);
    moves.push(idx);
  }

  return { winner: game.calcWinner(), moves };
}

// ── Main loop ──────────────────────────────────────────────────────────────────

const subjectGetMove = makeGetMove(table);

const evalHistory = [];   // 1 = subject won, 0 = ref won
const startTime = performance.now();

let nextPrint  = 1;
let totalGames = 0;

const GW = 7;   // games
const PW = 9;   // subjectWin%
const NW = 9;   // patterns
const LW = 9;   // elapsed

const hdr =
  `${'games'.padStart(GW)}` +
  `  ${'subjectWin%'.padStart(PW)}` +
  `  ${'patterns'.padStart(NW)}` +
  `  ${'elapsed'.padStart(LW)}`;
console.log(hdr);

function printStats() {
  const elapsed  = (((performance.now() - startTime) / 1000).toFixed(1) + 's').padStart(LW);
  const N        = evalHistory.length;
  const window   = evalHistory.slice(Math.floor(N / 2));
  const wins     = window.reduce((s, v) => s + v, 0);
  const pct      = (N > 0 ? (100 * wins / window.length).toFixed(1) + '%' : '—').padStart(PW);
  console.log(
    `${String(totalGames).padStart(GW)}` +
    `  ${pct}` +
    `  ${String(table.size).padStart(NW)}` +
    `  ${elapsed}`
  );
}

while (true) {
  const isEval = (totalGames % 2 === 0);
  if (isEval) {
    const evalGames = totalGames / 2;
    const subjectIsBlack = (evalGames % 2 === 0);
    const { winner, moves } = playGame(
      subjectIsBlack ? subjectGetMove : refGetMove,
      subjectIsBlack ? refGetMove     : subjectGetMove
    );
    const subjectColor = subjectIsBlack ? BLACK : WHITE;
    evalHistory.push(winner === subjectColor ? 1 : 0);
    learnFromGame(moves, winner, boardSize);
  } else {
    const { winner, moves } = playGame(
      subjectGetMove,
      subjectGetMove
    );
    learnFromGame(moves, winner, boardSize);
  }

  totalGames++;
  if (totalGames >= nextPrint) {
    printStats();
    nextPrint = Math.ceil(nextPrint * 1.5);
  }
}
