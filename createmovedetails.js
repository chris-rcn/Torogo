#!/usr/bin/env node
'use strict';

// Generate per-position move-value data by self-play.
// At each position every legal move is enumerated.  For each candidate move the
// game is cloned, the move is made, then agent.genMove is called with the full
// budget.  The winRatio is flipped to the original player's perspective and
// recorded.  The game advances by playing the highest-rated move.
// Output is newline-delimited JSON, one object per position.

const path = require('path');
const { Game2, PASS, coordStr } = require('./game2.js');

const args = process.argv.slice(2);
const get  = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

if (args.includes('--help') || args.includes('-h')) {
  console.error('Usage: node createmovedetails.js [--agent <name>] --budget <ms> --size <n>');
  process.exit(0);
}

const agentName  = get('--agent',  'production');
const budget     = parseInt(get('--budget', '2000'), 10);
const boardSize  = parseInt(get('--size', 13),    10);

if (isNaN(budget) || budget < 1)       { console.error('--budget must be a positive integer'); process.exit(1); }
if (isNaN(boardSize) || boardSize < 2) { console.error('--size must be >= 2'); process.exit(1); }

const { getMove: agent } = require(path.join(__dirname, 'ai', agentName + '.js'));

const SKIP = 21;

function legalMoves(game2) {
  const N   = game2.N;
  const cap = N * N;
  const moves = [];
  for (let i = 0; i < cap; i++) {
    if (game2.cells[i] === 0 && game2.isLegal(i)) moves.push(i);
  }
  moves.push(PASS);
  return moves;
}


while (true) {
  const game = new Game2(boardSize);
  const N = boardSize;

  const history = [];
  let nextAnalysis = Math.floor(Math.random() * SKIP) + 1;

  while (!game.gameOver) {
    if (game.moveCount === nextAnalysis) {
      nextAnalysis += SKIP;
      const moves = legalMoves(game);
      const moveInfos = [];

      for (const move of moves) {
        const clone = game.clone();
        clone.play(move);

        if (clone.gameOver) {
          moveInfos.push({ m: coordStr(move, N), kwr: null });
          continue;
        }

        const oppResponseMove = agent(clone, budget);
        if (oppResponseMove.rootWinRatio === undefined) {
          console.error('agent did not return rootWinRatio');
          process.exit(1);
        }
        const wr = 1 - oppResponseMove.rootWinRatio;
        moveInfos.push({ m: coordStr(move, N), kwr: Math.round(1000 * wr) });
      }

      moveInfos.sort((a, b) => (b.kwr ?? -Infinity) - (a.kwr ?? -Infinity));

      process.stdout.write(JSON.stringify({
        boardSize,
        history,
        candidates: moveInfos,
      }) + '\n');
    }

    const advancingMove = agent(game, budget);
    if (Math.abs(advancingMove.rootWinRatio - 0.5) > 0.3) break;
    const advIdx = advancingMove.type === 'pass' ? PASS : advancingMove.y * N + advancingMove.x;
    game.play(advIdx);
    history.push(coordStr(advIdx, N));
  }
}
