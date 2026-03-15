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
  // ── 5×5 puzzles ─────────────────────────────────────────────────────────────

  {
    // White at (2,1): neighbors B@(1,1),B@(3,1),B@(2,2) — only liberty (2,0).
    id: 1,
    toPlay: '●',
    answers: [[2, 0]],
    board: `
      · · · · ·
      · ● ○ ● ·
      · · ● · ·
      · · · · ·
      · · · · ·
    `,
  },
  {
    // White group {(1,2),(2,2)}: only liberty (2,1).
    id: 2,
    toPlay: '●',
    answers: [[2, 1]],
    board: `
      · · · · ·
      · ● · · ·
      ● ○ ○ ● ·
      · ● ● · ·
      · · · · ·
    `,
  },
  {
    // Black at (1,2): neighbors W@(0,2),W@(2,2),W@(1,3) — only liberty (1,1).
    // Extend to (1,1) to save.
    id: 3,
    toPlay: '●',
    answers: [[1, 1]],
    board: `
      · · · · ·
      · · · · ·
      ○ ● ○ · ·
      · ○ · · ·
      · · · · ·
    `,
  },
  {
    // White group {(1,1),(2,1),(3,1)}: only liberty (2,2).
    id: 4,
    toPlay: '●',
    answers: [[2, 2]],
    board: `
      · ● ● ● ·
      ● ○ ○ ○ ●
      · ● · ● ·
      · · · · ·
      · · · · ·
    `,
  },
  {
    // White group {(2,1),(2,2)}: only liberty (2,3).
    id: 5,
    toPlay: '●',
    answers: [[2, 3]],
    board: `
      · · ● · ·
      · ● ○ ● ·
      · ● ○ ● ·
      · · · · ·
      · · · · ·
    `,
  },
  {
    // B@(1,2) in atari at (2,2); B@(3,2) in atari at (2,2).
    // Connect both groups by playing (2,2).
    id: 6,
    toPlay: '●',
    answers: [[2, 2]],
    board: `
      · · · · ·
      · ○ · ○ ·
      ○ ● · ● ○
      · ○ · ○ ·
      · · · · ·
    `,
  },
  {
    // White group {(0,1),(0,2),(0,3)}: neighbors wrap to (4,2) on left.
    // Only liberty is (4,2) via toroidal wrap.
    id: 7,
    toPlay: '●',
    answers: [[4, 2]],
    board: `
      ● · · · ·
      ○ ● · · ●
      ○ ● · · ·
      ○ ● · · ●
      ● · · · ·
    `,
  },

  // ── 7×7 puzzles ─────────────────────────────────────────────────────────────

  {
    // White L-group {(2,2),(2,3),(2,4)}: only liberty (1,3).
    id: 8,
    toPlay: '●',
    answers: [[1, 3]],
    board: `
      · · · · · · ·
      · ● ● · · · ·
      · ● ○ ● · · ·
      · · ○ ● · · ·
      · ● ○ ● · · ·
      · · ● · · · ·
      · · · · · · ·
    `,
  },
  {
    // Black at (3,3): only liberty (3,4). Extend to save.
    id: 9,
    toPlay: '●',
    answers: [[3, 4]],
    board: `
      · · · · · · ·
      · · · · · · ·
      · · · ○ · · ·
      · · ○ ● ○ · ·
      · · · · · · ·
      · · · · · · ·
      · · · · · · ·
    `,
  },
  {
    // W@(2,3) liberty=(3,3); W@(4,3) liberty=(3,3). Fork: one move captures both.
    id: 10,
    toPlay: '●',
    answers: [[3, 3]],
    board: `
      · · · · · · ·
      · · · · · · ·
      · · ● · ● · ·
      · ● ○ · ○ ● ·
      · · ● · ● · ·
      · · · · · · ·
      · · · · · · ·
    `,
  },
  {
    // White column {(3,2),(3,3),(3,4),(3,5)}: only liberty (3,6).
    id: 11,
    toPlay: '●',
    answers: [[3, 6]],
    board: `
      · · · · · · ·
      · · · ● · · ·
      · · ● ○ ● · ·
      · · ● ○ ● · ·
      · · ● ○ ● · ·
      · · ● ○ ● · ·
      · · · · · · ·
    `,
  },
  {
    // White group {(0,3),(6,3)} connected via toroidal wrap.
    // Only liberty is (6,2).
    id: 12,
    toPlay: '●',
    answers: [[6, 2]],
    board: `
      · · · · · · ·
      · · · · · · ·
      ● · · · · · ·
      ○ ● · · · ● ○
      ● · · · · · ●
      · · · · · · ·
      · · · · · · ·
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
    const name = String(puzzle.id).padEnd(30);
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
