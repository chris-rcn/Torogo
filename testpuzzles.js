'use strict';

const { Board, Game, ZOBRIST } = require('./game.js');

// ─── Board parser / serializer (thin wrappers around Board methods) ──────────

// boardToString(board, toPlay?) — serialize board to ASCII; if toPlay ('●'/'○')
// is given, prepend it as the first line so parseBoard can recover it.
function boardToString(board, toPlay) {
  const body = board.toAscii();
  return toPlay ? toPlay + '\n' + body : body;
}

// parseBoard(str) — returns { size, stones, toPlay? }.
// toPlay is '●' or '○' if the string was produced with boardToString(board, toPlay).
function parseBoard(boardStr) {
  const lines = boardStr.trim().split('\n').map(r => r.trim());
  let toPlay;
  if (lines[0] === '●' || lines[0] === '○') {
    toPlay = lines.shift();
  }
  const result = Board.parse(lines.join('\n'));
  if (toPlay !== undefined) result.toPlay = toPlay;
  return result;
}

// ─── Position builder ─────────────────────────────────────────────────────────

function buildPosition(puzzle) {
  const { size, stones } = parseBoard(puzzle.board);
  const game = new Game(size, 0);
  const c = size >> 1;
  game.board.set(c, c, null);
  game.hash = 0n;
  game.moveCount = 0;
  game.current = puzzle.toPlay === '●' ? 'black' : puzzle.toPlay === '○' ? 'white' : puzzle.toPlay;
  game.consecutivePasses = 0;
  game.koFlag = null;
  for (const [x, y, color] of stones) {
    game.board.set(x, y, color);
    game.hash ^= ZOBRIST[y][x][color];
  }
  game.board._rebuildGroups();
  return game;
}

// ─── Puzzles ──────────────────────────────────────────────────────────────────
// Each puzzle: white group(s) in atari (1 liberty), or black in atari needing
// to extend.  All positions verified: answer is legal and the group being
// captured/saved has exactly 1 liberty at that point.

const PUZZLES = [
  {
    id: 141947139,
    toPlay: '○',
    comment: '2026-03-15T16:21:32.736Z budget=1000ms preference=0.207',
    answers: [[3, 3]],
    board: `
      ○ · · ○ ○ · · · ●
      · ○ ○ ○ ● · · ● ●
      ● ● ● ○ ○ ○ · · ●
      ○ ○ ○(·)· ○ ○ ● ●
      ● ● ○ ● ● · ○ ● ○
      ○ ● ○ · ● ○ ○ ○ ○
      · ● ● · · ● · ○ ●
      ● ● ○ ○ ● ● ○ ○ ●
      ● ○ ● · ○ ● ● ● ●
    `,
  },
  {
    id: 807180432,
    toPlay: '●',
    comment: '2026-03-15T16:21:35.737Z budget=1000ms preference=0.286',
    answers: [[3, 5]],
    board: `
      ○ · · ○ ○ · · · ●
      · ○ ○ ○ ● · · ● ●
      ● ● ● ○ ○ ○ · · ●
      ○ ○ ○ ○ · ○ ○ ● ●
      ● ● ○ ● ● · ○ ● ○
      ○ ● ○(·)● ○ ○ ○ ○
      · ● ● ● · ● · ○ ●
      ● ● ○ ○ ● ● ○ ○ ●
      ● ○ ● ○ ○ ● ● ● ●
    `,
  },
  {
    id: 883551055,
    toPlay: '●',
    comment: '2026-03-15T16:21:37.738Z budget=1000ms preference=0.412',
    answers: [[6, 1]],
    board: `
      ○ · · ○ ○ · ○ · ●
      · ○ ○ ○ ● ·(·)● ●
      ● ● ● ○ ○ ○ · · ●
      ○ ○ ○ ○ · ○ ○ ● ●
      ● ● ○ ● ● · ○ ● ○
      ○ ● ○ ● ● ○ ○ ○ ○
      · ● ● ● · ● · ○ ●
      ● ● ○ ○ ● ● ○ ○ ●
      ● ○ ● ○ ○ ● ● ● ●
    `,
  },
  {
    id: 236378154,
    toPlay: '●',
    comment: '2026-03-15T16:21:39.739Z budget=1000ms preference=0.256',
    answers: [[1, 0]],
    board: `
      ○(·)· ○ ○ ○ ○ · ●
      · ○ ○ ○ ● · ● ● ●
      ● ● ● ○ ○ ○ · · ●
      ○ ○ ○ ○ · ○ ○ ● ●
      ● ● ○ ● ● · ○ ● ○
      ○ ● ○ ● ● ○ ○ ○ ○
      · ● ● ● · ● · ○ ●
      ● ● ○ ○ ● ● ○ ○ ●
      ● ○ ● ○ ○ ● ● ● ●
    `,
  },
  {
    id: 921929810,
    toPlay: '○',
    comment: '2026-03-15T16:21:40.739Z budget=1000ms preference=0.226',
    answers: [[6, 2]],
    board: `
      ○ ● · ○ ○ ○ ○ · ●
      · ○ ○ ○ ● · ● ● ●
      ● ● ● ○ ○ ○(·)· ●
      ○ ○ ○ ○ · ○ ○ ● ●
      ● ● ○ ● ● · ○ ● ○
      ○ ● ○ ● ● ○ ○ ○ ○
      · ● ● ● · ● · ○ ●
      ● ● ○ ○ ● ● ○ ○ ●
      ● · ● ○ ○ ● ● ● ●
    `,
  },
  {
    id: 718719598,
    toPlay: '●',
    comment: '2026-03-15T16:21:41.739Z budget=1000ms preference=0.725',
    answers: [[5, 1]],
    board: `
      ○ ● · ○ ○ ○ ○ · ●
      · ○ ○ ○ ●(·)● ● ●
      ● ● ● ○ ○ ○ ○ · ●
      ○ ○ ○ ○ · ○ ○ ● ●
      ● ● ○ ● ● · ○ ● ○
      ○ ● ○ ● ● ○ ○ ○ ○
      · ● ● ● · ● · ○ ●
      ● ● ○ ○ ● ● ○ ○ ●
      ● · ● ○ ○ ● ● ● ●
    `,
  },
];

// ─── CLI ─────────────────────────────────────────────────────────────────────

if (require.main === module) {
  const args = process.argv.slice(2);
  let agentName = null;
  let budgetMs = 200;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--agent' && args[i + 1]) agentName = args[++i];
    else if (args[i] === '--budget' && args[i + 1]) budgetMs = Number(args[++i]);
  }

  if (!agentName) {
    console.error('Usage: node testpuzzles.js --agent <name> [--budget <ms>]');
    process.exit(1);
  }

  const agent = require(`./ai/${agentName}.js`);

  console.log(`\n── Puzzle Benchmark: ${agentName} (${budgetMs}ms/move) ──\n`);
  console.log('  #  Size  ID                             Result');

  let correct = 0;
  for (let i = 0; i < PUZZLES.length; i++) {
    const puzzle = PUZZLES[i];
    const game = buildPosition(puzzle);
    const move = agent(game, budgetMs);

    const passed = move.type === 'place' &&
      puzzle.answers.some(([ax, ay]) => move.x === ax && move.y === ay);

    if (passed) correct++;

    const num = String(i + 1).padStart(3);
    const size = `${game.boardSize}x${game.boardSize}`.padEnd(4);
    const name = (puzzle.comment || '').slice(0, 30).padEnd(30);
    let result = passed ? '+' : '-';
    if (!passed) {
      const played = move.type === 'place' ? `${move.x},${move.y}` : 'pass';
      const expected = puzzle.answers.map(([ax, ay]) => `${ax},${ay}`).join(' or ');
      result += `  (played ${played}; expected ${expected})`;
    }
    console.log(`  ${num}  ${size}  ${name}  ${result}`);
  }

  console.log('  ──────────────────────────────────────────────────────');
  console.log(`     Total: ${correct}/${PUZZLES.length}`);
  process.exit(0);
}

// ─── Exports (for testfast.js round-trip test) ───────────────────────────────

if (typeof module !== 'undefined') module.exports = { parseBoard, boardToString };
