'use strict';

/**
 * Auto-generate puzzle positions from agent self-play.
 *
 * Strategy:
 *   1. Play self-play games using the specified agent until `--max` puzzles
 *      have been collected.
 *   2. At each ply, call the agent twice on the same position.  If both calls
 *      return the same place move, the position is "forced" — emit it as a
 *      puzzle candidate.
 *   3. Deduplicate by Zobrist hash and emit puzzle objects to stdout.
 *
 * Usage:
 *   node genpuzzles.js [--agent <name>] [--size <n>] [--budget <ms>] [--max <n>]
 *
 *   --agent      AI agent to use (default: mcts)
 *   --size       board size (default: 7)
 *   --budget     time budget per agent call in ms (default: 500)
 *   --max        max puzzles to emit (default: 20)
 */

const path = require('path');
const { Game } = require('./game.js');
const randomAgent = require('./ai/random.js');

// ─── Parse CLI args ───────────────────────────────────────────────────────────

const args = process.argv.slice(2);
let agentName  = 'mcts';
let boardSize  = 7;
let budgetMs   = 500;
let maxPuzzles = 1000;

for (let i = 0; i < args.length; i++) {
  if      (args[i] === '--agent'  && args[i+1]) agentName  = args[++i];
  else if (args[i] === '--size'   && args[i+1]) boardSize  = Number(args[++i]);
  else if (args[i] === '--budget' && args[i+1]) budgetMs   = Number(args[++i]);
  else if (args[i] === '--max'    && args[i+1]) maxPuzzles = Number(args[++i]);
}

const agent = require(path.join(__dirname, 'ai', agentName + '.js'));

// ─── Self-play harvest ────────────────────────────────────────────────────────

const seenHashes = new Set();
let puzzleCount = 0;

console.log('// Auto-generated puzzles — paste into PUZZLES array in testpuzzles.js');
console.log(`// Generated: agent=${agentName} size=${boardSize} budget=${budgetMs}ms`);
console.log('');

while (puzzleCount < maxPuzzles) {
  const game = new Game(boardSize, 0);

  // Two random moves after the constructor's centre stone, to diversify openings.
  for (let r = 0; r < 2 && !game.gameOver; r++) {
    const move = randomAgent(game);
    if (move.type === 'place') game.placeStone(move.x, move.y);
    else game.pass();
  }

  const boardCells = boardSize * boardSize;
  let gamePuzzleFound = false;

  while (!game.gameOver && puzzleCount < maxPuzzles && game.moveCount < boardCells * 0.9) {
    const hashKey = game.hash.toString();
    const first = agent(game, budgetMs);
    if (first.type !== 'place') break;

    const second = agent(game, budgetMs);

    const agreed = second.type === 'place' &&
                   first.x === second.x   && first.y === second.y;

    if (agreed && !seenHashes.has(hashKey)) {
      seenHashes.add(hashKey);
      puzzleCount++;
      gamePuzzleFound = true;
      const indented = game.board.toAscii(first).split('\n').map(r => '      ' + r).join('\n');
      const toPlayChar = game.current === 'black' ? '●' : '○';
      const comment = `${new Date().toISOString()} agent=${agentName} budget=${budgetMs}ms`;
      console.log(`  {`);
      console.log(`    toPlay: '${toPlayChar}',`);
      console.log(`    comment: '${comment}',`);
      console.log(`    answers: [[${first.x}, ${first.y}]],`);
      console.log(`    board: \``);
      console.log(indented);
      console.log(`    \`,`);
      console.log(`  },`);
      break;
    }

    // Advance the game using the first call's move.
    if (first.type === 'place') game.placeStone(first.x, first.y);
    else game.pass();
  }

  if (!gamePuzzleFound) {
    console.log(`// Game ended with no move agreement found`);
  }

}
