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
    toPlay: '○',
    comment: '2026-03-15T16:54:11.936Z agent=mcts budget=1000ms',
    answers: [[6, 4]],
    board: `
      · · · ● · ○ · · ●
      · · · · · · · · ·
      · · · · · · · · ·
      · ● · · · · ○ ● ○
      · · · · ● ·(·)· ·
      · · · · · · · · ·
      · · · · · · · · ·
      · · · · · · · · ·
      · · · · · · · ○ ·
    `,
  },
  {
    toPlay: '●',
    comment: '2026-03-15T16:54:33.943Z agent=mcts budget=1000ms',
    answers: [[2, 0]],
    board: `
      ● ·(·)● · ○ · · ●
      · · · · · ○ ○ · ·
      · · · · · · ○ · ·
      · ● ● · · · ○ ● ○
      · · · ● ● · ○ · ·
      ● · · · ○ · · · ·
      · · · · · · · · ·
      · · · · · · · · ●
      · · ○ · · · · ○ ·
    `,
  },
  {
    toPlay: '●',
    comment: '2026-03-15T16:56:44.727Z agent=mcts budget=1000ms',
    answers: [[7, 6]],
    board: `
      ○ · · · · ○ ● · ·
      · ● ● · ○ · · · ·
      ● · ● · ○ ● ○ ● ●
      ○ · · · ○ ○ ○ ○ ○
      · ● · · ● ● ● · ·
      · · · · ● ○ ● · ○
      · · · · · ○ ·(·)○
      · ○ · ● · ● · · ○
      · · · ● · ● · ○ ·
    `,
  },
  {
    toPlay: '●',
    comment: '2026-03-15T16:56:52.729Z agent=mcts budget=1000ms',
    answers: [[7, 0]],
    board: `
      ○ · · · · ○ ●(·)●
      · ● ● · ○ · · · ·
      ● · ● · ○ ● ○ ● ●
      ○ ○ · · ○ ○ ○ ○ ○
      · ● · · ● ● ● · ·
      · · · · ● ○ ● · ○
      · · · · · ○ · ● ○
      · ○ ○ ● · ● · · ○
      · · · ● · ● · ○ ·
    `,
  },
  {
    toPlay: '●',
    comment: '2026-03-15T16:56:56.730Z agent=mcts budget=1000ms',
    answers: [[3, 6]],
    board: `
      ○ · · · · ○ ● ● ●
      · ● ● · ○ · · · ·
      ● · ● · ○ ● ○ ● ●
      ○ ○ · · ○ ○ ○ ○ ○
      · ● · · ● ● ● · ·
      · · · · ● ○ ● · ○
      · · ·(·)○ ○ · ● ○
      · ○ ○ ● · ● · · ○
      · · · ● · ● · ○ ·
    `,
  },
  {
    toPlay: '○',
    comment: '2026-03-15T16:56:58.730Z agent=mcts budget=1000ms',
    answers: [[8, 8]],
    board: `
      ○ · · · · ○ ● ● ●
      · ● ● · ○ · · · ·
      ● · ● · ○ ● ○ ● ●
      ○ ○ · · ○ ○ ○ ○ ○
      · ● · · ● ● ● · ·
      · · · · ● ○ ● · ○
      · · · ● ○ ○ · ● ○
      · ○ ○ ● · ● · · ○
      · · · ● · ● · ○(·)
    `,
  },
  {
    toPlay: '●',
    comment: '2026-03-15T16:57:00.731Z agent=mcts budget=1000ms',
    answers: [[3, 0]],
    board: `
      ○ · ·(·)· ○ ● ● ●
      · ● ● · ○ · · · ·
      ● · ● · ○ ● ○ ● ●
      ○ ○ · · ○ ○ ○ ○ ○
      · ● · · ● ● ● · ·
      · · · · ● ○ ● · ○
      · · · ● ○ ○ · ● ○
      · ○ ○ ● · ● · · ○
      · · · ● · ● · ○ ○
    `,
  },
  {
    toPlay: '○',
    comment: '2026-03-15T16:57:06.731Z agent=mcts budget=1000ms',
    answers: [[4, 0]],
    board: `
      ○ · · ●(·)○ ● ● ●
      · ● ● · ○ · · · ·
      ● · ● · ○ ● ○ ● ●
      ○ ○ · · ○ ○ ○ ○ ○
      · ● · · ● ● ● · ·
      · · · · ● ○ ● · ○
      · · · ● ○ ○ · ● ○
      · ○ ○ ● ● ● · · ○
      · · · ● · ● ○ ○ ○
    `,
  },
  {
    toPlay: '○',
    comment: '2026-03-15T16:57:10.732Z agent=mcts budget=1000ms',
    answers: [[0, 1]],
    board: `
      ○ · · ● ○ ○ ● ● ●
      ·)● ● · ○ · · ● ·
      ● · ● · ○ ● ○ ● ●
      ○ ○ · · ○ ○ ○ ○ ○
      · ● · · ● ● ● · ·
      · · · · ● ○ ● · ○
      · · · ● ○ ○ · ● ○
      · ○ ○ ● ● ● · · ○
      · · · ● · ● ○ ○ ○
    `,
  },
  {
    toPlay: '●',
    comment: '2026-03-15T16:57:12.733Z agent=mcts budget=1000ms',
    answers: [[1, 2]],
    board: `
      ○ · · ● ○ ○ ● ● ●
      ○ ● ● · ○ · · ● ·
      ●(·)● · ○ ● ○ ● ●
      ○ ○ · · ○ ○ ○ ○ ○
      · ● · · ● ● ● · ·
      · · · · ● ○ ● · ○
      · · · ● ○ ○ · ● ○
      · ○ ○ ● ● ● · · ○
      · · · ● · ● ○ ○ ○
    `,
  },
  {
    toPlay: '○',
    comment: '2026-03-15T16:57:14.734Z agent=mcts budget=1000ms',
    answers: [[6, 7]],
    board: `
      ○ · · ● ○ ○ ● ● ●
      ○ ● ● · ○ · · ● ·
      ● ● ● · ○ ● ○ ● ●
      ○ ○ · · ○ ○ ○ ○ ○
      · ● · · ● ● ● · ·
      · · · · ● ○ ● · ○
      · · · ● ○ ○ · ● ○
      · ○ ○ ● ● ●(·)· ○
      · · · ● · ● ○ ○ ○
    `,
  },
  {
    toPlay: '●',
    comment: '2026-03-15T16:57:16.734Z agent=mcts budget=1000ms',
    answers: [[6, 6]],
    board: `
      ○ · · ● ○ ○ ● ● ●
      ○ ● ● · ○ · · ● ·
      ● ● ● · ○ ● ○ ● ●
      ○ ○ · · ○ ○ ○ ○ ○
      · ● · · ● ● ● · ·
      · · · · ● ○ ● · ○
      · · · ● ○ ○(·)● ○
      · ○ ○ ● ● ● ○ · ○
      · · · ● · ● ○ ○ ○
    `,
  },
  {
    toPlay: '○',
    comment: '2026-03-15T16:57:22.735Z agent=mcts budget=1000ms',
    answers: [[1, 8]],
    board: `
      ○ · · ● ○ ○ ● ● ●
      ○ ● ● · ○ · · ● ·
      ● ● ● · ○ ● ○ ● ●
      ○ ○ · · ○ ○ ○ ○ ○
      · ● · · ● ● ● · ·
      · · · · ● · ● · ○
      · · · ● · ○ ● ● ○
      · ○ ○ ● ● ● ○ · ○
      ●(·)· ● · ● ○ ○ ○
    `,
  },
  {
    toPlay: '○',
    comment: '2026-03-15T16:57:30.736Z agent=mcts budget=1000ms',
    answers: [[3, 4]],
    board: `
      ○ ● · ● ○ ○ ● ● ●
      ○ ● ● · ○ · · ● ·
      ● ● ● · ○ ● ○ ● ●
      ○ ○ · · ○ ○ ○ ○ ○
      · ● ○(·)● ● ● · ·
      · · ● · ● · ● · ○
      · · · ● · ○ ● ● ○
      · ○ ○ ● ● ● ○ · ○
      ● ○ · ● · ● ○ ○ ○
    `,
  },
  {
    toPlay: '○',
    comment: '2026-03-15T16:57:34.737Z agent=mcts budget=1000ms',
    answers: [[3, 3]],
    board: `
      ○ ● · ● ○ ○ ● ● ●
      ○ ● ● · ○ · · ● ·
      ● ● ● · ○ ● ○ ● ●
      ○ ○ ●(·)○ ○ ○ ○ ○
      · ● ○ ○ ● ● ● · ·
      · · ● · ● · ● · ○
      · · · ● · ○ ● ● ○
      · ○ ○ ● ● ● ○ · ○
      ● ○ · ● · ● ○ ○ ○
    `,
  },
  {
    toPlay: '●',
    comment: '2026-03-15T16:57:36.737Z agent=mcts budget=1000ms',
    answers: [[1, 5]],
    board: `
      ○ ● · ● ○ ○ ● ● ●
      ○ ● ● · ○ · · ● ·
      ● ● ● · ○ ● ○ ● ●
      ○ ○ ● ○ ○ ○ ○ ○ ○
      · ● ○ ○ ● ● ● · ·
      ·(·)● · ● · ● · ○
      · · · ● · ○ ● ● ○
      · ○ ○ ● ● ● ○ · ○
      ● ○ · ● · ● ○ ○ ○
    `,
  },
  {
    toPlay: '○',
    comment: '2026-03-15T16:57:38.738Z agent=mcts budget=1000ms',
    answers: [[3, 5]],
    board: `
      ○ ● · ● ○ ○ ● ● ●
      ○ ● ● · ○ · · ● ·
      ● ● ● · ○ ● ○ ● ●
      ○ ○ ● ○ ○ ○ ○ ○ ○
      · ● ○ ○ ● ● ● · ·
      · ● ●(·)● · ● · ○
      · · · ● · ○ ● ● ○
      · ○ ○ ● ● ● ○ · ○
      ● ○ · ● · ● ○ ○ ○
    `,
  },
  {
    toPlay: '●',
    comment: '2026-03-15T16:57:40.738Z agent=mcts budget=1000ms',
    answers: [[2, 6]],
    board: `
      ○ ● · ● ○ ○ ● ● ●
      ○ ● ● · ○ · · ● ·
      ● ● ● · ○ ● ○ ● ●
      ○ ○ ● ○ ○ ○ ○ ○ ○
      · ● ○ ○ ● ● ● · ·
      · ● ● ○ ● · ● · ○
      · ·(·)● · ○ ● ● ○
      · ○ ○ ● ● ● ○ · ○
      ● ○ · ● · ● ○ ○ ○
    `,
  },
  {
    toPlay: '○',
    comment: '2026-03-15T16:57:42.739Z agent=mcts budget=1000ms',
    answers: [[3, 1]],
    board: `
      ○ ● · ● ○ ○ ● ● ●
      ○ ● ●(·)○ · · ● ·
      ● ● ● · ○ ● ○ ● ●
      ○ ○ ● ○ ○ ○ ○ ○ ○
      · ● ○ ○ ● ● ● · ·
      · ● ● ○ ● · ● · ○
      · · ● ● · ○ ● ● ○
      · ○ ○ ● ● ● ○ · ○
      ● ○ · ● · ● ○ ○ ○
    `,
  },
  {
    toPlay: '●',
    comment: '2026-03-15T16:57:44.739Z agent=mcts budget=1000ms',
    answers: [[8, 1]],
    board: `
      ○ ● · ● ○ ○ ● ● ●
      ○ ● ● ○ ○ · · ●(·)
      ● ● ● · ○ ● ○ ● ●
      ○ ○ ● ○ ○ ○ ○ ○ ○
      · ● ○ ○ ● ● ● · ·
      · ● ● ○ ● · ● · ○
      · · ● ● · ○ ● ● ○
      · ○ ○ ● ● ● ○ · ○
      ● ○ · ● · ● ○ ○ ○
    `,
  },
  {
    toPlay: '○',
    comment: '2026-03-15T16:57:50.739Z agent=mcts budget=1000ms',
    answers: [[8, 4]],
    board: `
      ○ ● · ● ○ ○ ● ● ●
      · ● ● ○ ○ · · ● ●
      ● ● ● · ○ ● ○ ● ●
      ○ ○ ● ○ ○ ○ ○ ○ ○
      · ● ○ ○ ● ● ● ●(·)
      · ● ● ○ ● · ● · ○
      · · ● ● · ○ ● ● ○
      · ○ ○ ● ● ● ○ · ○
      ● ○ · ● · ● ○ ○ ○
    `,
  },
  {
    toPlay: '●',
    comment: '2026-03-15T16:57:52.740Z agent=mcts budget=1000ms',
    answers: [[4, 6]],
    board: `
      ○ ● · ● ○ ○ ● ● ●
      · ● ● ○ ○ · · ● ●
      ● ● ● · ○ ● ○ ● ●
      ○ ○ ● ○ ○ ○ ○ ○ ○
      · ● ○ ○ ● ● ● ● ○
      · ● ● ○ ● · ● · ○
      · · ● ●(·)○ ● ● ○
      · ○ ○ ● ● ● ○ · ○
      ● ○ · ● · ● ○ ○ ○
    `,
  },
  {
    toPlay: '●',
    comment: '2026-03-15T17:05:24.743Z agent=mcts budget=2000ms',
    answers: [[8, 7]],
    board: `
      ● ○ · · · ○ ● · · · ·
      ● · ○ ○ ○ ● ○ · · ● ●
      · ○ · · · ● ○ · ○ ● ·
      · ○ ● · · · · · · · ·
      · ● · ● ○ · · ○ ○ · ·
      · ○ ● ● ● ● ○ · · · ·
      ○ ○ · ● ○ ○ ○ · · ○ ●
      ● ○ · ● ○ · · ●(·)● ●
      · · · ● ○ · · · ○ · ·
      · ○ ● ● ● ● ● ○ · · ●
      ● ○ · · · ○ ● · · ○ ·
    `,
  },
  {
    toPlay: '○',
    comment: '2026-03-15T17:06:00.750Z agent=mcts budget=2000ms',
    answers: [[0, 9]],
    board: `
      ● ○ ● · · ○ ● ● · · ·
      ● ● ○ ○ ○ ● ○ · · ● ●
      · ○ · · · ● ○ · ○ ● ·
      · ○ ● · · · · · · · ·
      · ● · ● ○ · · ○ ○ · ·
      · ○ ● ● ● ● ○ · · · ○
      ○ ○ · ● ○ ○ ○ · · ○ ●
      ● ○ · ● ○ · · ● ● ● ●
      ● · · ● ○ · · ○ ○ · ○
      ·)○ ● ● ● ● ● ○ · · ●
      ● ○ · · · ○ ● · ○ ○ ·
    `,
  },
  {
    toPlay: '●',
    comment: '2026-03-15T17:06:12.752Z agent=mcts budget=2000ms',
    answers: [[1, 8]],
    board: `
      ● ○ ● · · ○ ● ● · · ·
      ● ● ○ ○ ○ ● ○ · · ● ●
      · ○ · · · ● ○ · ○ ● ·
      · ○ ● · · · · · · · ·
      · ● · ● ○ · · ○ ○ · ·
      · ○ ● ● ● ● ○ · · · ○
      ○ ○ · ● ○ ○ ○ · · ○ ●
      ● ○ · ● ○ · · ● ● ● ●
      ●(·)· ● ○ · ○ ○ ○ · ○
      ○ ○ ● ● ● ● ● ○ · · ●
      ● ○ · · · ○ ● · ○ ○ ●
    `,
  },
  {
    toPlay: '●',
    comment: '2026-03-15T17:06:20.753Z agent=mcts budget=2000ms',
    answers: [[2, 10]],
    board: `
      ● ○ ● · · ○ ● ● · · ·
      ● ● ○ ○ ○ ● ○ · · ● ●
      · ○ · · · ● ○ · ○ ● ·
      · ○ ● · · · · · · · ·
      · ● · ● ○ · · ○ ○ · ·
      · ○ ● ● ● ● ○ · · · ○
      ○ ○ · ● ○ ○ ○ · · ○ ●
      ● ○ · ● ○ · · ● ● ● ●
      ● ● · ● ○ · ○ ○ ○ ○ ○
      ○ ○ ● ● ● ● ● ○ · · ●
      ● ○(·)· · ○ ● · ○ ○ ●
    `,
  },
  {
    toPlay: '○',
    comment: '2026-03-15T17:06:56.755Z agent=mcts budget=2000ms',
    answers: [[8, 1]],
    board: `
      ● ● ● · · ○ ● ● · · ·
      ● ● ○ ○ ○ · ○ ·(·)● ●
      · ○ · · ○ · ○ · ○ ● ·
      · ○ ● · · ○ · · · · ·
      · ● · ● ○ · · ○ ○ · ○
      · ○ ● ● ● ● ○ · · · ○
      ○ ○ · ● ○ ○ ○ · ○ ○ ●
      ● ○ · ● ○ · ● ● ● ● ●
      ● ● · ● ○ ● ○ ○ ○ ○ ○
      · ● ● ● ● ● ● ○ · · ●
      ● · ● · · ○ ● · ○ ○ ●
    `,
  },
  {
    toPlay: '●',
    comment: '2026-03-15T17:07:00.755Z agent=mcts budget=2000ms',
    answers: [[8, 0]],
    board: `
      ● ● ● · · ○ ● ●(·)· ·
      ● ● ○ ○ ○ · ○ · ○ ● ●
      · ○ · · ○ · ○ · ○ ● ·
      · ○ ● · · ○ · · · · ·
      · ● · ● ○ · · ○ ○ · ○
      · ○ ● ● ● ● ○ · · · ○
      ○ ○ · ● ○ ○ ○ · ○ ○ ●
      ● ○ · ● ○ · ● ● ● ● ●
      ● ● · ● ○ ● ○ ○ ○ ○ ○
      · ● ● ● ● ● ● ○ · · ●
      ● · ● · · ○ ● · ○ ○ ●
    `,
  },
  {
    toPlay: '●',
    comment: '2026-03-15T17:07:16.756Z agent=mcts budget=2000ms',
    answers: [[9, 9]],
    board: `
      ● ● ● · · ○ ● ● ● · ·
      ● ● ○ ○ ○ · ○ · ○ ● ●
      · ○ · · ○ · ○ · ○ ● ·
      · ○ ● · · ○ ● · · · ○
      · ● · ● ○ · · ○ ○ · ○
      ○ ○ ● ● ● ● ○ · · · ○
      ○ ○ · ● ○ ○ ○ · ○ ○ ●
      ● ○ · ● ○ · ● ● ● ● ●
      ● ● · ● ○ ● ○ ○ ○ ○ ○
      · ● ● ● ● ● ● ○ ·(·)●
      ● · ● · · ○ ● · ○ ○ ●
    `,
  },
  {
    toPlay: '○',
    comment: '2026-03-15T17:08:46.768Z agent=mcts budget=2000ms',
    answers: [[3, 10]],
    board: `
      · · ● · · · · · · · ·
      · · · · · · · · · · ·
      · · · · · · · · · · ·
      · · · · ○ · · ● · · ·
      ○ · · · · · · · · ● ·
      ● · ● · · ● · · · · ●
      · · · · · · · · · · ·
      · ○ · · ○ · · · · · ○
      ○ · · ● ○ · · · · · ·
      · · · · · · · · · · ·
      · · ·(·)· · · · · · ·
    `,
  },
  {
    toPlay: '○',
    comment: '2026-03-15T17:10:22.794Z agent=mcts budget=2000ms',
    answers: [[3, 7]],
    board: `
      · ● ● · · · · · · · ·
      ○ · ○ · · ● · ● · · ·
      · ● · · · · · · · · ·
      · · ○ ● ○ · · ● · · ·
      ○ · · · ● ● · · · ● ·
      ● ○ ● · · ● ● ○ · · ●
      · · · · · · ○ · · ● ·
      · ○ ○(·)○ · · · · · ○
      ○ · ○ ● ○ · · · · · ·
      · · · ● ● · · · · · ○
      ○ · · ○ ● · · · · ○ ·
    `,
  },
  {
    toPlay: '○',
    comment: '2026-03-15T17:15:13.491Z agent=mcts budget=3000ms',
    answers: [[0, 9]],
    board: `
      ○ · · · ○ · · · ● ○ ·
      ○ · · ● · ○ · · · ○ ·
      · · · ● · · · · ○ · ·
      · ○ · ○ ● ○ · ○ · · ·
      · · · · · ● · · · ● ·
      · · · · · ● · · · · ·
      · · ○ · · ● · · ● · ·
      · · · · ● · · · ● ○ ·
      · ○ · ○ · · · · ● ● ·
      (·)● · · · ● · · ○ ● ○
      ○ · ● · ○ · · · ● ● ●
    `,
  },
  {
    toPlay: '●',
    comment: '2026-03-15T17:20:37.548Z agent=mcts budget=3000ms',
    answers: [[9, 3]],
    board: `
      · · ● · · ● · ○ ○ ○ ·
      ● · ○ · ● · · ● · ○ ·
      · · ○ · · · · ○ ● · ·
      · ○ ○ · · ○ · ● ○(·)○
      · ● ○ ● ● · ● · ● · ·
      ● · · ● · ● ● · · · ·
      · ○ ○ · · · · · ○ · ·
      · ○ · · ● · ○ ○ · ○ ○
      · · · ○ ● · ● · ○ ○ ●
      · · · · ● · ● ● ● ● ·
      ○ · · ● ○ ● · ○ · · ○
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
