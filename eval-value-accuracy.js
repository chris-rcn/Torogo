#!/usr/bin/env node
'use strict';

// eval-value-accuracy.js — measure how often the value function predicts the winner.
//
// Reads a games file (one game per line: N,move,move,...[,b|w]),
// replays each game, samples positions at the given rate, and checks whether
// V(s) ≥ 0.5 ↔ BLACK wins.
//
// Usage:
//   node eval-value-accuracy.js --file <path> --model <path.js>
//                               [--games <n>] [--sample <rate>]
//
//   --file    path to games file (e.g. games-1000.tmp)
//   --model   path to weights JS file (as written by saveWeights)
//   --games   number of games to evaluate (random selection; default: all)
//   --sample  fraction of positions to evaluate per game (default: 0.1)

const fs   = require('fs');
const path = require('path');
const { Game2, BLACK, WHITE, PASS } = require('./game2.js');
const { extractFeatures, evaluateFeatures, loadWeights } = require('./vpatterns.js');
const Util = require('./util.js');

// ── Core ──────────────────────────────────────────────────────────────────────

// Parse a two-letter move token ("gg") to flat index, or PASS for "..".
function parseMoveToken(token, N) {
  if (token === '..') return PASS;
  const x = token.charCodeAt(0) - 97;
  const y = token.charCodeAt(1) - 97;
  return y * N + x;
}

// Evaluate accuracy of a model over a selection of games from filePath.
// Returns { correct, total, accuracy }.
//   nGames    — number of games to sample (default: all)
//   sampleRate — fraction of positions per game to evaluate (default: 0.1)
function evalValueAccuracy(filePath, model, { nGames = Infinity, sampleRate = 0.1 } = {}) {
  const lines = fs.readFileSync(filePath, 'utf8').split('\n').filter(l => l.trim());

  // Random selection of games.
  const selected = nGames < lines.length
    ? Util.shuffle(lines.slice()).slice(0, nGames)
    : lines;

  let correct = 0, total = 0;

  for (const line of selected) {
    const fields    = line.split(',');
    const N         = parseInt(fields[0], 10);
    const lastField = fields[fields.length - 1];
    const hasWinner = lastField === 'b' || lastField === 'w';
    const tokens    = hasWinner ? fields.slice(1, -1) : fields.slice(1);

    // Replay to determine winner (or use pre-computed).
    let winner;
    if (hasWinner) {
      winner = lastField === 'b' ? BLACK : WHITE;
      // Also need the game state at each sampled position, so replay below.
    }

    const sampledVs = [];
    const game = new Game2(N, false);
    for (const token of tokens) {
      if (game.gameOver) break;
      if (Math.random() < sampleRate)
        sampledVs.push(evaluateFeatures(extractFeatures(game, model.preparedSpecs), model.weights));
      game.play(parseMoveToken(token, N));
    }

    if (!hasWinner) winner = game.calcWinner();

    if (winner !== null) {
      for (const v of sampledVs) {
        if ((v >= 0.5) === (winner === BLACK)) correct++;
        total++;
      }
    }
  }

  return { correct, total, accuracy: total > 0 ? correct / total : 0 };
}

module.exports = { evalValueAccuracy };

// ── CLI ───────────────────────────────────────────────────────────────────────

if (require.main === module) {
  const opts = Util.parseArgs(process.argv.slice(2), ['help']);

  if (opts.help || !opts.file || !opts.model) {
    console.error('Usage: node eval-value-accuracy.js --file <path> --model <path.js> [--games <n>] [--sample <rate>]');
    process.exit(opts.help ? 0 : 1);
  }

  const nGames    = opts.games  ? parseInt(opts.games, 10)   : Infinity;
  const sampleRate = opts.sample ? parseFloat(opts.sample)   : 0.1;
  const model     = loadWeights(opts.model);

  const { correct, total, accuracy } = evalValueAccuracy(opts.file, model, { nGames, sampleRate });
  console.log(`correct=${correct}  total=${total}  accuracy=${(100 * accuracy).toFixed(2)}%`);
}
