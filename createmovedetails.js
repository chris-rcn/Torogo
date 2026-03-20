#!/usr/bin/env node
'use strict';

// Generate per-position move-value data by self-play.
//
// Usage: node createmovedetails.js [--agent <name>] [--budget <ms>] [--size <n>]
//   --agent    AI policy name   (default: rave)
//   --budget   budget per move  (default: 100)
//   --size     board size       (default: 11)
//
// At each position every legal move is enumerated.  For each candidate move the
// game is cloned, the move is made, then agent.genMove is called with the full
// budget.  The winRatio is flipped to the original player's perspective and
// recorded.  The game advances by playing the highest-rated move.
// Output is newline-delimited JSON, one object per position.

const path = require('path');
const { Game, DEFAULT_KOMI } = require('./game.js');

const args = process.argv.slice(2);
const get  = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

if (args.includes('--help') || args.includes('-h')) {
  console.error('Usage: node gengamedata.js [--agent <name>] [--budget <ms>] [--size <n>]');
  process.exit(0);
}

const agentName  = get('--agent',  'rave');
const budget     = parseInt(get('--budget', '100'), 10);
const boardSize  = parseInt(get('--size',    '11'), 10);

if (isNaN(budget) || budget < 1)     { console.error('--budget must be a positive integer'); process.exit(1); }
if (isNaN(boardSize) || boardSize < 2) { console.error('--size must be >= 2'); process.exit(1); }

const agent = require(path.join(__dirname, 'ai', agentName + '.js'));

// Returns an array of all legal moves {type, x?, y?} for the current player.
function legalMoves(game) {
  const moves = [];
  const { board, current, koFlag } = game;
  const N = board.size;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      if (board.get(x, y) === null &&
          !board.isSuicide(x, y, current) &&
          !board.isKo(x, y, current, koFlag)) {
        moves.push({ type: 'place', x, y });
      }
    }
  }
  moves.push({ type: 'pass' });
  return moves;
}

function applyMove(game, move) {
  if (move.type === 'place') game.placeStone(move.x, move.y);
  else game.pass();
}

function coordStr(move) {
  if (move.type === 'pass') return 'pass';
  return String.fromCharCode(97 + move.x) + (move.y + 1);
}

while (true) {
  const game = new Game(boardSize, DEFAULT_KOMI);

  const history = [];

  while (!game.gameOver) {
    if (Math.random() < 0.05) {
      const moves = legalMoves(game);
      const moveInfos = [];

      for (const move of moves) {
        const clone = game.clone();
        applyMove(clone, move);

        let oppResponseMove;
        if (clone.gameOver) {
          // Game ended immediately (double-pass); no genMove needed.
          moveInfos.push({ move: coordStr(move), winRatio: null });
          continue;
        }

        oppResponseMove = agent(clone, budget);
        if (oppResponseMove.rootWinRatio === undefined) {
          console.error('agent did not return rootWinRatio');
          process.exit(1);
        }
        const wr = 1 - oppResponseMove.rootWinRatio;
        moveInfos.push({ m: coordStr(move), kwr: Math.round(1000 * wr) });
      }

      moveInfos.sort((a, b) => (b.kwr ?? -Infinity) - (a.kwr ?? -Infinity));

      process.stdout.write(JSON.stringify({
        boardSize: boardSize,
        history: history,
        candidates: moveInfos,
      }) + '\n');
    }
  
    const advancingMove = agent(game, budget);
    if (Math.abs(advancingMove.rootWinRatio - 0.5) > 0.3) {  // Game is not balanced.
      break;
    }
    applyMove(game, advancingMove);
    history.push(coordStr(advancingMove));
  }
}
