'use strict';

/**
 * Auto-generate puzzle positions from MCTS self-play.
 *
 * Strategy:
 *   1. Play self-play games using MCTS with `--budget` ms per move until
 *      `--max` puzzles have been collected.
 *   2. At each ply, check whether the top root-child visit fraction >= `--threshold`.
 *      If so, the position is "forced" — save it as a puzzle candidate.
 *   3. Deduplicate by Zobrist hash and emit puzzle objects to stdout.
 *
 * Usage:
 *   node genpuzzles.js [--size <n>] [--budget <ms>] [--threshold <ratio>] [--max <n>]
 *
 *   --size       board size (default: 7)
 *   --budget     MCTS budget per move in ms (default: 500)
 *   --threshold  min fraction of visits on the top move to qualify (default: 0.65)
 *   --max        max puzzles to emit (default: 20)
 */

const { Game } = require('./game.js');
const randomAgent = require('./ai/random.js');
const mcts = require('./ai/mcts.js');

// ─── Parse CLI args ───────────────────────────────────────────────────────────

const args = process.argv.slice(2);
let boardSize  = 7;
let budgetMs   = 500;
let threshold  = 0.65;
let maxPuzzles = 20;

for (let i = 0; i < args.length; i++) {
  if      (args[i] === '--size'      && args[i+1]) boardSize  = Number(args[++i]);
  else if (args[i] === '--budget'    && args[i+1]) budgetMs   = Number(args[++i]);
  else if (args[i] === '--threshold' && args[i+1]) threshold  = Number(args[++i]);
  else if (args[i] === '--max'       && args[i+1]) maxPuzzles = Number(args[++i]);
}


// ─── Visit ratio from MCTS children ──────────────────────────────────────────
// Returns fraction of place-move visits captured by the top move, or 0.
function visitRatio(children) {
  const place = children.filter(c => c.move.type === 'place');
  const total = place.reduce((s, c) => s + c.visits, 0);
  return total > 0 ? place[0].visits / total : 0;
}

// ─── Self-play harvest ────────────────────────────────────────────────────────

const seenHashes = new Set();
let puzzleCount = 0;

console.log('// Auto-generated puzzles — paste into PUZZLES array in testpuzzles.js');
console.log(`// Generated: size=${boardSize} budget=${budgetMs}ms threshold=${threshold}`);
console.log('');

while (puzzleCount < maxPuzzles) {
  const game = new Game(boardSize, 0);

  // One random move after the constructor's centre stone, to diversify openings.
  if (!game.gameOver) {
    const move = randomAgent(game);
    if (move.type === 'place') game.placeStone(move.x, move.y);
    else game.pass();
  }

  while (!game.gameOver && puzzleCount < maxPuzzles) {
    const hashKey = game.hash.toString();
    const result = mcts(game, budgetMs);

    if (result.type === 'place') {
      if (!seenHashes.has(hashKey) && visitRatio(result.children) >= threshold) {
        seenHashes.add(hashKey);
        puzzleCount++;
        const indented = game.board.toAscii(result).split('\n').map(r => '      ' + r).join('\n');
        console.log(`  {`);
        console.log(`    name: 'Auto #${puzzleCount}',`);
        console.log(`    toPlay: '${game.current}',`);
        console.log(`    answer: [[${result.x}, ${result.y}]],`);
        console.log(`    board: \``);
        console.log(indented);
        console.log(`    \`,`);
        console.log(`  },`);
      }
      game.placeStone(result.x, result.y);
    } else {
      game.pass();
    }
  }

}

