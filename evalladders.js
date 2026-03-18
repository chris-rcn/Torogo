'use strict';
const { performance } = require('perf_hooks');
const path = require('path');
const { Game, ZOBRIST, parseBoard } = require('./game.js');

/**
 * Ladder evaluation script — run hardcoded ladder positions against an AI agent.
 *
 * Usage:
 *   node evalladders.js [options]
 *
 * Options:
 *   --agent   <name>   AI policy to evaluate (default: random)
 *   --budget  <ms>     Time budget per move in ms (default: 500)
 *   --trials  <n>      Trials per position (default: 10)
 *   --help             Show this help message
 *
 * Each position uses the same format as predictmoves.js:
 *   toPlay     '●' or '○'
 *   board      ASCII board string (· empty, ● black, ○ white; (x) marks answer)
 *   answers    [[x, y], ...]   agent must play one of these  (optional)
 *   prohibited [[x, y], ...]   agent must not play any of these (optional)
 *   comment    free-form string (optional)
 */

// Boolean flags that take no value.
const BOOL_FLAGS = new Set(['help']);

function parseArgs(argv) {
  const opts = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') { opts.help = true; continue; }
    if (arg.startsWith('--')) {
      const key = arg.slice(2);
      if (BOOL_FLAGS.has(key)) { opts[key] = true; continue; }
      const val = argv[i + 1];
      if (val === undefined || val.startsWith('--')) {
        console.error(`Missing value for ${arg}`);
        process.exit(1);
      }
      opts[key] = val;
      i++;
    } else {
      console.error(`Unknown argument: ${arg}`);
      process.exit(1);
    }
  }
  return opts;
}

const opts = parseArgs(process.argv.slice(2));

if (opts.help) {
  console.log('Usage: node evalladders.js [--agent <name>] [--budget <ms>] [--trials <n>]');
  process.exit(0);
}

const agentName = opts.agent  || 'random';
const budgetMs  = parseInt(opts.budget || '500', 10);
const trials    = parseInt(opts.trials || '10',  10);

if (isNaN(budgetMs) || budgetMs < 1) { console.error('--budget must be a positive integer'); process.exit(1); }
if (isNaN(trials)   || trials   < 1) { console.error('--trials must be a positive integer'); process.exit(1); }

const agent = require(path.join(__dirname, 'ai', agentName + '.js'));

// ── Position builder (mirrors predictmoves.js) ─────────────────────────────

function buildPosition(pos) {
  const { size, stones } = parseBoard(pos.board);
  const game = new Game(size, 0);
  const c = size >> 1;
  game.board.set(c, c, null);
  game.hash             = 0n;
  game.moveCount        = 0;
  game.current          = pos.toPlay === '●' ? 'black' : 'white';
  game.consecutivePasses = 0;
  game.koFlag           = null;
  for (const [x, y, color] of stones) {
    game.board.set(x, y, color);
    game.hash ^= ZOBRIST[y][x][color];
  }
  game.board._rebuildGroups();
  return game;
}

// ── Positions ──────────────────────────────────────────────────────────────

const POSITIONS = [
//  {
//    toPlay:     '○',
//    comment:    'Handcrafted escapable 11 stones',
//    answers: [[3, 5]],
//    board: `
//      · · ○ · · · · ○ · ● · · ·
//      · ○ · · · · ○ ○ ● · · ● ○
//      · ○ ○ ○ ○ ○ ○ · · · · · ·
//      · ○ · · · · · · · ● · · ●
//      · ○ · · ● ● · · ● · · · ·
//      · ○ ·(·)○ ○ ● ● · ○ · · ·
//      · ○ · ● ● ○ ○ ● · ● ○ ○ ·
//      · ○ · · ● ● ○ ○ ● ● · · ○
//      ○ · ○ ○ ● · ● ○ ○ ● · · ·
//      · · ● ○ · · ● ● ○ ● · ○ ·
//      · · · ○ · ○ ● ● ○ ○ ● · ·
//      · · ○ · · · ○ · ● ● · ● ·
//      ○ ○ · · ○ · ○ · · · ● · ·
//    `,
//  },
  {
    toPlay:     '○',
    comment:    'Handcrafted doomed 11 stones',
    prohibited: [[3, 5]],
    board: `
      · · ○ · · · · ○ · ● · · ·
      · ○ · · · · ○ ○ ● · · ● ○
      · ○ ○ ○ ○ ○ ○ · · · · · ·
      · ○ · · · · · · · ● · · ●
      · ● · · ● ● · · ● · · · ·
      · ○ ·(·)○ ○ ● ● · ○ · · ·
      · ○ · ● ● ○ ○ ● · ● ○ ○ ·
      · ○ · · ● ● ○ ○ ● ● · · ○
      ○ · ○ ○ ● · ● ○ ○ ● · · ·
      · · ● ○ · · ● ● ○ ● · ○ ·
      · · · ○ · ○ ● ● ○ ○ ● · ·
      · · ○ · · · ○ · ● ● · ● ·
      ○ ○ · · ○ · ○ · · · ● · ·
    `,
  },
//  {
//    toPlay:     '○',
//    comment:    'Game 35, move 155  white plays em — extends dead white group (3 stones, liberty at em)',
//    prohibited: [[4, 12]],
//    board: `
//      ○ ○ ○ ● · ● ● ○ ○ ○ · ○ ·
//      ○ ○ ○ ● ● ● ● ● · ● ○ ○ ·
//      ○ ● ● ○ ○ ● ● ● ● ● ● ○ ○
//      ○ ○ ● · ● · ● · ● ○ ○ ○ ○
//      ○ ○ ● ● ● ● · ● ● ● ● ○ ●
//      ○ ● ● ● ● ● ● ○ ○ ● · ● ●
//      ● ● ○ ○ ● · ● ● ○ ● ● ● ○
//      ○ ○ ○ · ○ ● · ● ○ ● ○ ○ ○
//      · ○ · ○ ○ ● ● ○ ○ ○ ○ ○ ●
//      · · ○ · ● ● ● ● ● ● ● ○ ·
//      · · ○ ○ ● ● ● · ● ○ ○ · ○
//      ● · ○ ● ○ ● ○ ● ● ○ ○ · ○
//      · · ○ ●(·)○ ○ ● ● ● ○ ● ·
//    `,
//  },
//  {
//    toPlay:     '●',
//    comment:    'Game 60, move 164  black plays ai — extends dead black group (3 stones, liberty at ai)',
//    prohibited: [[0, 8]],
//    board: `
//      ○ ○ · ○ ● ○ ● · ● ● ● ○ ●
//      ● ○ ○ ● ● · ● ● · ● ● ● ●
//      ○ · ○ ● · ● ● · ● ● ○ ● ○
//      ○ ○ ○ ● ● ● · ● ● ● ○ ● ●
//      · ○ ○ ○ ● ● ● · ● ○ ○ ○ ○
//      ○ · ○ ● ● · ● ● ● ● ○ ○ ·
//      ○ · ○ ○ ○ ● ● · ● ● ● ○ ·
//      · ○ ○ ○ ● ● ● ● · ● ○ · ○
//     (·)● ● ○ ○ ● ○ ○ ● ● ● ○ ○
//      · ○ ● ○ ○ ○ ○ ● ● · ● ● ○
//      ○ · ○ · ○ ● ● · ● · ● ○ ·
//      · ○ ○ ○ ● ● · ● ○ ● ● ○ ○
//      ○ · · ○ ● · ● ● · ● ○ ○ ○
//    `,
//  },
//  {
//    toPlay:     '●',
//    comment:    'Game 79, move 174  black plays fi — extends dead black group (3 stones, liberty at fi)',
//    prohibited: [[5, 8]],
//    board: `
//      ● · ● ● · ● ● ● ● ● ● · ●
//      · ● · ● · ● ● ○ ● ● ● ● ●
//      ○ ○ ● · ● ● ○ ○ ○ ● · ● ○
//      ● ○ ● ● ● ● ● ● ○ · ● ● ●
//      ● ○ ● ○ ○ ● ○ ○ ○ ○ · ○ ·
//      ● · ● ○ ○ ○ ○ · · ● ○ ○ ○
//      ● ● ● ○ · ○ ● ● ○ · ○ · ○
//      ● ○ ○ ○ ● ○ ○ ○ ○ ○ ○ ○ ●
//      ○ · ○ ○ ●(·)● ● ○ ○ ○ · ○
//      · ○ ○ ○ ○ ○ ● ○ ○ · ○ ○ ○
//      ○ ● ○ ○ ○ ○ ○ ○ ○ ○ ● ○ ·
//      ● · ● ○ ● ○ · ○ ○ ● ● ● ○
//      · ● · ● ● ○ ○ ● ○ ○ ○ ● ●
//    `,
//  },
//  {
//    toPlay:     '○',
//    comment:    'Game 131, move 181  white plays gf — extends dead white group (3 stones, liberty at gf)',
//    prohibited: [[6, 5]],
//    board: `
//      · ○ ○ ● · ● · ● ● ● ○ ○ ○
//      ○ ○ ○ ○ ● · ● · ● ○ ○ ○ ·
//      ○ ○ ○ ● · ● · ○ ● ● ○ · ○
//      ○ · ○ ○ ● ● ● ● · ● ○ ○ ○
//      ○ ○ ● ● · ● · ○ ● ○ ○ ○ ·
//      ○ ● ● · ● ○(·)○ ● ● · ○ ○
//      · ○ ○ ● ○ ○ ● ● · ● ● ○ ○
//      ● ○ ● ● ● ● · ● ● · ● ○ ○
//      ○ ○ ● ● ○ ● ○ ○ ● ● ○ · ○
//      · ○ ○ ○ ○ ● · ○ ● ● ○ ○ ○
//      ○ ● ○ ● ○ · ● ● ● ○ ○ ○ ○
//      ● · ○ ● ○ ○ ○ ○ ● ● ● ○ ○
//      ○ ○ ● ● ● ● ● ● ○ ○ ○ · ○
//    `,
//  },
//  {
//    toPlay:     '●',
//    comment:    'Game 154, move 154  black plays jk — extends dead black group (3 stones, liberty at jk)',
//    prohibited: [[9, 10]],
//    board: `
//      ● ○ ○ ● ○ ○ ○ · ○ ● ● · ●
//      ● ● ● · ○ ○ · ○ ○ ○ ○ ● ·
//      · ● · ● ● ○ ○ ○ ○ ● ● · ●
//      · ● · ● · ● ● ● ● ○ · ● ●
//      · ● ○ · · ○ ● ○ ○ · ● ○ ·
//      · ○ · ● · ● ● ○ ● ● ● ○ ●
//      ● · · ○ ● ○ ● ○ ● · ● ● ●
//      ○ ● ● ● ● ○ ● ○ ○ ● ○ ○ ●
//      ○ ○ ● ○ ○ ○ ○ · ○ ○ · ○ ○
//      ○ ● ● ● ○ ○ ○ ○ · ○ ○ ● ○
//      · ○ ● ○ ○ ○ ○ · ○(·)● ● ○
//      ○ ○ ○ · ● ● ○ ● ● ● ○ ○ ○
//      ○ · ○ ○ ● ○ ○ ○ ○ ○ ● ● ●
//    `,
//  },
//  {
//    toPlay:     '○',
//    comment:    'Game 195, move 171  white plays ma — extends dead white group (3 stones, liberty at ma)',
//    prohibited: [[12, 0]],
//    board: `
//      ○ · · · ● ○ ○ ● ○ ● ● ·(·)
//      ○ ● ○ ● ○ ○ · ○ ○ ○ ● ○ ○
//      ● · ○ ● ● ○ ○ · ○ ● ● ● ○
//      · ○ · ● ○ ○ ○ · ○ ○ ○ · ●
//      ○ ● ● · ● ○ ○ · ○ ● ● ● ●
//      ● · ○ ● · ○ · · ○ ○ ○ ● ·
//      · ○ ○ ● ● ○ ● ○ ○ · · ○ ●
//      · ● ○ ○ ○ ○ ○ · ● ○ ○ ○ ●
//      ● ○ ○ ○ ○ ○ · ○ ○ ○ ● ● ·
//      ○ ○ ○ ● ○ · ○ ○ ○ ○ · ● ●
//      ● ○ ● ● ● ○ · ○ ○ ○ ○ ● ○
//      ● ● · ● ○ ○ ○ ○ · ○ ○ ● ○
//      ● · ● ● ● ○ · · ○ ● · ● ○
//    `,
//  },
//  {
//    toPlay:     '●',
//    comment:    'Game 199, move 186  black plays kh — extends dead black group (3 stones, liberty at kh)',
//    prohibited: [[10, 7]],
//    board: `
//      ○ ● · ● ● ● · ● ○ · ○ ○ ·
//      ● ● ● ● ● ● ● ● ○ ● ● · ○
//      ○ ● ○ ● · ● ● ○ ○ ○ ○ ○ ·
//      · ○ ○ ● ● ● ● ● ● ● ○ ● ○
//      ○ ○ ● ● ● · ○ ● ● · ○ · ○
//      ● ● ● · · ● ● ● ○ ○ · ○ ●
//      ○ ○ ● ● ● ● ● ○ ○ ● ● ○ ●
//      ○ ● ● · ● ● ● ● ○ ●(·)● ○
//      · ○ ● ● ● · ● ○ ○ ○ ● ● ○
//      ○ · ● · ● ● ● ● ● ● ○ ○ ·
//      ● ● ● ● · ● · ● · ○ · ○ ○
//      ○ ● ● ● ○ ● ● ● · ○ ○ · ●
//      · · ● ● ○ ● ● ○ ○ ○ · ○ ○
//    `,
//  },
//  {
//    toPlay:     '●',
//    comment:    'Game 226, move 160  black plays ia — extends dead black group (3 stones, liberty at ia)',
//    prohibited: [[8, 0]],
//    board: `
//      ● ● ● ● ● ● ● ○(·)● ○ ● ○
//      ○ ● ● ● ○ ○ ○ ○ ● ○ ● ● ○
//      ● · ● ○ ○ · ○ ● ● ○ ○ ● ●
//      ● ● ● ● ○ ○ · ○ ○ ○ ● ● ·
//      · ○ ● · ● ● ○ · ○ ○ ○ ● ●
//      ● · ○ ● ● ○ ○ · ○ · ○ ○ ○
//      ● ○ ● · ● ○ · ○ ○ ○ · ○ ○
//      ● ● ● ● · ● ○ ● ● ● ○ ○ ○
//      ● · ● ● ● ● · ● ○ ● ● ● ●
//      ○ ● ● · ● ● ● ○ · ○ · ○ ·
//      ○ ● · ● · ● ○ ○ ○ ○ ○ ○ ○
//      ● ○ ○ ● ● ● · ○ · ○ · ○ ●
//      ● ○ ● ○ · ● · ○ ● · ○ ○ ○
//    `,
//  },
//  {
//    toPlay:     '○',
//    comment:    'Game 493, move 55  white plays hi — extends dead white group (4 stones, liberty at hi)',
//    prohibited: [[7, 8]],
//    board: `
//      · · ○ · · · · · · ● · · ·
//      · ○ · · · · · · ● · · ● ○
//      · · · · · · ● · · · · · ·
//      · · · · · · · · · ● · · ●
//      · · · · ● · · · ● · · · ·
//      · · · · · · · ● · ○ · · ·
//      · · · ● · · ● · · ● ○ ○ ·
//      · ○ · · ● · ● · ● ● · · ○
//      ○ · ○ ○ ● · ·(·)○ ● · · ·
//      · · ● ○ · · ○ ● ○ ● · ○ ·
//      · · · ○ · ○ ● ● ○ ○ ● · ·
//      · · ○ · · · ○ · ● ● · ● ·
//      ○ ○ · · ○ · ○ · · · ● · ·
//    `,
//  },
//  {
//    toPlay:     '○',
//    comment:    'Game 275, move 167  white plays ea — extends dead white group (4 stones, liberty at ea)',
//    prohibited: [[4, 0]],
//    board: `
//      ○ ● ● ·(·)● ● ● · ● ○ · ●
//      ○ ○ ● ● ○ ○ ● ○ ● ● ○ · ○
//      ● ○ ○ ● ○ ○ ● ○ ● ○ ○ ○ ●
//      ● ● ● · ● ● ● ○ ○ ○ · ○ ○
//      ● ○ ○ ● · ○ ● ○ ○ · ○ ○ ●
//      ● ○ · ● ● ○ ○ ● ● ○ ○ · ○
//      ○ ○ ○ ● ● ● ● ● ○ ○ ○ ○ ○
//      ○ ○ ● · ○ ● · ● ● ○ · ○ ○
//      ● ● ● ● ● ● · · ● ● ○ ○ ○
//      ○ ○ ○ ○ ○ ○ ● ○ ● ○ · · ○
//      · ○ · ○ ○ ○ ● ● ● ○ ○ · ○
//      ○ ○ ○ ● ● ○ ○ ○ ○ ● ○ ○ ·
//      ○ ● ○ ● ● ● ○ ● ● ● ○ · ○
//    `,
//  },
//  {
//    toPlay:     '○',
//    comment:    'Game 285, move 187  white plays gi — extends dead white group (5 stones, liberty at gi)',
//    prohibited: [[6, 8]],
//    board: `
//      ○ · ○ ○ · ○ ○ ○ ○ ○ ○ · ●
//      ○ ○ · ○ ● ○ ○ ● · ○ ○ ○ ○
//      ● ○ ○ ○ ○ ○ ○ ○ ○ · · ○ ○
//      · ○ ● ● ● ○ ● ● ○ ○ ○ · ○
//      ● ○ ● · ● ● ● · ● ● ● ○ ○
//      ○ ○ ○ ● · ○ · ● ● · ○ ● ○
//      ○ ○ ● · ○ ● ● · ● ○ ● · ○
//      ● ○ ○ ○ ● · ○ ● ○ ○ ● ● ●
//      ● ○ ● ● · ●(·)● · ● ● ○ ○
//      ○ · ○ · ● ○ ○ ● ● ○ ○ ○ ·
//      ○ ○ ● ● ○ ○ ○ ● ● ○ ○ ○ ○
//      · ○ ○ ● ● ● ● ○ ● ● ○ ○ ○
//      ○ ○ ○ ○ ○ ● ○ ○ ○ ○ · ○ ·
//    `,
//  },
//  {
//    toPlay:     '○',
//    comment:    'Game 68, move 163  white plays ce — extends dead white group (6 stones, liberty at ce)',
//    prohibited: [[2, 4]],
//    board: `
//      ● ● ● ● ● ○ · · ○ ● ● ○ ●
//      ○ ● ● ○ ○ · ○ ● · ● ● ○ ○
//      ○ ● ● ● ○ · ○ · ○ · · ○ ·
//      ● ● ● ○ ○ ○ · ○ · ○ ○ · ○
//      ● ·(·)● ● ● ○ · ○ ○ · ○ ●
//      · ● ○ ○ ○ ● ○ ○ · ○ ○ ○ ●
//      ● · ● ● ○ ● ● ○ ○ ● ● ● ·
//      · ● · ● ○ ● · ● ● ● ● · ●
//      ● ● ● ● ○ ● ● · ● ● ● · ●
//      ● · ● · ● ● · ○ ● ○ ● ● ●
//      ○ ● ● ● ○ ○ ● ● ● ○ ○ ○ ○
//      ○ ○ ● ○ ○ · ○ ● ○ · ● ● ○
//      ○ ● ● ○ ○ ○ ● ○ ○ ○ ○ ○ ○
//    `,
//  },
//  {
//    toPlay:     '○',
//    comment:    'Game 14, move 173  white plays cg — extends dead white group (8 stones, liberty at cg)',
//    prohibited: [[2, 6]],
//    board: `
//      ○ ● ● · ● ○ ● ○ · ○ ○ ○ ○
//      ○ ○ ● ● ● ○ ○ ○ · ○ ○ ○ ○
//      ○ ○ ● ● ○ ○ ○ ○ ○ ○ ● ● ●
//      ● ● ○ ● ● ○ · ○ · ○ ○ ○ ●
//      ● · ○ ● ● ● ○ ○ ○ ● ○ ○ ●
//      ○ · ● ● ● ● ○ ○ ○ ● ● ● ●
//      ● ·(·)● ● ● ● ○ ● ● ● ● ●
//      ○ ● ○ ● ○ ○ ● ● ● ○ ○ ● ○
//      ○ ○ ○ ● ● ○ ○ ○ ○ ○ ○ ● ○
//      ● ○ ● ● ○ ○ ● ○ ○ ○ · ○ ●
//      · ● ○ ○ ○ ● · ○ · · ○ ○ ●
//      · ● ● ○ ○ ○ ○ ○ ○ · ○ ○ ●
//      ● ● ● ● ● ● ● ● ○ ○ ○ · ○
//    `,
//  },
//  {
//    toPlay:     '○',
//    comment:    'Game 14, move 177  white plays bg — extends dead white group (9 stones, liberty at bg)',
//    prohibited: [[1, 6]],
//    board: `
//      ○ ● ● · ● ○ ● ○ · ○ ○ ○ ○
//      ○ ○ ● ● ● ○ ○ ○ · ○ ○ ○ ○
//      ○ ○ ● ● ○ ○ ○ ○ ○ ○ ● ● ●
//      ● ● · ● ● ○ · ○ · ○ ○ ○ ●
//      ● ● ○ ● ● ● ○ ○ ○ ● ○ ○ ●
//      ○ · ● ● ● ● ○ ○ ○ ● ● ● ●
//      ●(·)○ ● ● ● ● ○ ● ● ● ● ●
//      ○ ● ○ ● ○ ○ ● ● ● ○ ○ ● ○
//      ○ ○ ○ ● ● ○ ○ ○ ○ ○ ○ ● ○
//      ● ○ ● ● ○ ○ ● ○ ○ ○ · ○ ●
//      ● ● ○ ○ ○ ● · ○ · · ○ ○ ●
//      · ● ● ○ ○ ○ ○ ○ ○ · ○ ○ ●
//      ● ● ● ● ● ● ● ● ○ ○ ○ · ○
//    `,
//  },
];

// ── Evaluation ─────────────────────────────────────────────────────────────

const NW = Math.max('position'.length, ...POSITIONS.map(p => p.comment.length));
const TW = 2 * String(trials).length + 1;   // e.g. "10/10"

const startTime = performance.now();
console.log(`Agent: ${agentName}  budget: ${budgetMs}ms  trials: ${trials}\n`);
const RW = '100.0%'.length;
console.log(` ${'position'.padEnd(NW)}  ${'pass'.padStart(TW)}  ${'ratio'.padStart(RW)}`);
console.log(` ${'-'.repeat(NW)}  ${'-'.repeat(TW)}  ------`);

let totalTrials = 0;
let totalPassed = 0;

for (const pos of POSITIONS) {
  let passed = 0;

  for (let t = 0; t < trials; t++) {
    const game = buildPosition(pos);
    const move = agent(game, budgetMs);

    let ok = true;

    if (pos.answers && pos.answers.length > 0) {
      ok &&= move.type === 'place' &&
             pos.answers.some(([ax, ay]) => ax === move.x && ay === move.y);
    }
    if (pos.prohibited && pos.prohibited.length > 0) {
      ok &&= !(move.type === 'place' &&
               pos.prohibited.some(([ax, ay]) => ax === move.x && ay === move.y));
    }

    if (ok) passed++;
  }

  totalTrials += trials;
  totalPassed += passed;

  const fraction = `${passed}/${trials}`;
  const ratio    = (100 * passed / trials).toFixed(1) + '%';
  console.log(` ${pos.comment.padEnd(NW)}  ${fraction.padStart(TW)}  ${ratio.padStart(RW)}`);
}

const overallPct = (100 * totalPassed / totalTrials).toFixed(1);
const elapsed    = ((performance.now() - startTime) / 1000).toFixed(1);
console.log(`\nOverall: ${totalPassed}/${totalTrials} (${overallPct}%)  elapsed: ${elapsed}s`);
