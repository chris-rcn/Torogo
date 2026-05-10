'use strict';
const { performance } = require('perf_hooks');
const path = require('path');
const { Game2, BLACK, WHITE, parseBoard, parseMove } = require('./game2.js');
const Util = require('./util.js');

/**
 * Ladder evaluation script — run hardcoded ladder positions against an AI agent.
 *
 * Usage:
 *   node evalladders.js [options]
 *
 * Options:
 *   --agent   <name>   AI policy to evaluate (default: random)
 *   --budget  <ms>     Time budget per move in ms (required)
 *   --trials  <n>      Trials per position (default: 10)
 *   --help             Show this help message
 *
 */

const opts = Util.parseArgs(process.argv.slice(2), ['help']);

if (opts.help) {
  console.log('Usage: node evalladders.js [--agent <name>] [--budget <ms>] [--trials <n>]');
  process.exit(0);
}

const agentName = opts.agent  || 'random';
if (!opts.budget) { console.error('--budget is required'); process.exit(1); }
const budgetMs  = parseInt(opts.budget, 10);
const trials    = parseInt(opts.trials || '10',  10);

if (isNaN(budgetMs) || budgetMs < 1) { console.error('--budget must be a positive integer'); process.exit(1); }
if (isNaN(trials)   || trials   < 1) { console.error('--trials must be a positive integer'); process.exit(1); }

const { getMove: agent } = require(path.join(__dirname, 'ai', agentName + '.js'));

// ── Coordinate helpers ─────────────────────────────────────────────────────

// Returns true when the agent move matches the spec string ('pass' or 'f4'-style coord).
function matchesMove(s, move, N) {
  return move.move === parseMove(s, N);
}

// ── Position builder ────────────────────────────────────────────────────────

function buildPosition(pos) {
  return parseBoard(pos.board, pos.toPlay === '●' ? BLACK : WHITE);
}

// ── Positions ──────────────────────────────────────────────────────────────
// Row labels are 1-based from bottom (row 1 = y=0, bottom of board).

const POSITIONS = [

//  {
//    toPlay:     '●',
//    comment:    'Attack 2 stones',
//    require: ['f4'],
//    board: `
//         a b c d e f g
//       7 · · · · ○ · ○
//       6 · ● · · ○ ○ ·
//       5 · · · · · · ○
//       4 · · · · ·(·)·
//       3 ○ ○ · · · ○ ●
//       2 · ○ · · ● ○ ●
//       1 ○ · ○ · ● ● ●
//    `,
//  },
//
//  {
//    toPlay:     '●',
//    comment:    'Attack 2 stones (fail)',
//    prohibit: ['f4','e3'],
//    board: `
//         a b c d e f g
//       7 · · · · · · ·
//       6 · ○ · · · · ·
//       5 · · · · · · ·
//       4 · · · · ·(·)·
//       3 · · · ·(·)○ ●
//       2 · · · · ● ○ ●
//       1 · · · · ● ● ●
//    `,
//  },

  {
    toPlay:     '○',
    comment:    'Extend 2 stones',
    require: ['e3'],
    board: `
         a b c d e f g
       7 · · · · · · ·
       6 · ○ · · · · ·
       5 · · · · · · ·
       4 · · · · · ● ·
       3 · · · ·(·)○ ●
       2 · · · · ● ○ ●
       1 · · · · ● ● ●
    `,
  },

  {
    toPlay:     '○',
    comment:    'Extend 2 stones (fail)',
    prohibit: ['e3'],
    board: `
         a b c d e f g
       7 · · · · ○ · ·
       6 · ● · · · ○ ·
       5 · · · · · · ○
       4 · · · · · ● ·
       3 ○ · · ·(·)○ ●
       2 · ○ · · ● ○ ●
       1 · · ○ · ● ● ●
    `,
  },

  {
    toPlay:     '●',
    comment:    'Attack 3 stones',
    require: ['d3'],
    board: `
         a b c d e f g
       7 · · · · · ○ ·
       6 · ● · · · ○ ·
       5 · · · · · · ·
       4 · · · · · ● ·
       3 · · ·(·)○ ○ ●
       2 ○ ○ · · ● ○ ●
       1 · · · · · ● ●
    `,
  },

  {
    toPlay:     '●',
    comment:    'Attack 3 stones (fail)',
    prohibit: ['d3','e4'],
    board: `
         a b c d e f g
       7 · · · · · · ·
       6 · ○ · · · · ·
       5 · · · · · · ·
       4 · · · ·(·)● ·
       3 · · ·(·)○ ○ ●
       2 · · · · ● ○ ●
       1 · · · · · ● ●
    `,
  },

  {
    toPlay:     '○',
    comment:    'Extend 3 stones',
    require: ['e4'],
    board: `
         a b c d e f g
       7 · · · · · · ·
       6 · ○ · · · · ·
       5 · · · · · · ·
       4 · · · ·(·)● ·
       3 · · · ● ○ ○ ●
       2 · · · · ● ○ ●
       1 · · · · · ● ●
    `,
  },

  {
    toPlay:     '○',
    comment:    'Extend 3 stones (fail)',
    prohibit: ['e4'],
    board: `
         a b c d e f g
       7 · · · · ○ · ○
       6 · ● · · · ○ ·
       5 · · · · · · ○
       4 · · · ·(·)● ·
       3 ○ · · ● ○ ○ ●
       2 · ○ · · ● ○ ●
       1 ○ · ○ · · ● ●
    `,
  },

  {
    toPlay:     '●',
    comment:    'Attack 4 stones',
    require: ['e5'],
    board: `
         a b c d e f g
       7 · · · · · ○ ·
       6 · ● · · · ○ ·
       5 · · · ·(·)· ·
       4 · · · · ○ ● ·
       3 · · · ● ○ ○ ●
       2 ○ ○ · · ● ○ ●
       1 · · · · · ● ●
    `,
  },

  {
    toPlay:     '●',
    comment:    'Attack 4 stones (fail)',
    prohibit: ['e5','d4'],
    board: `
         a b c d e f g
       7 · · · · · ○ ·
       6 · ○ · · · · ·
       5 · · · ·(·)· ·
       4 · · ·(·)○ ● ·
       3 · · · ● ○ ○ ●
       2 ○ · · · ● ○ ●
       1 · · · · · ● ●
    `,
  },

  {
    toPlay:     '○',
    comment:    'Extend 4 stones',
    require: ['d4'],
    board: `
         a b c d e f g
       7 · · · · · · ·
       6 · ○ · · · · ·
       5 · · · · ● · ·
       4 · · ·(·)○ ● ·
       3 · · · ● ○ ○ ●
       2 · · · · ● ○ ●
       1 · · · · · ● ●
    `,
  },

  {
    toPlay:     '○',
    comment:    'Extend 4 stones (fail)',
    prohibit: ['f4'],
    board: `
         a b c d e f g h i
       9 · · · · · · ○ · ·
       8 · ● · · · · · ○ ·
       7 · · · · · · · · ○
       6 · · · · · · · · ·
       5 · · · · · · ● · ·
       4 · · · · ·(·)○ ● ●
       3 ○ · · · · ● ○ ○ ●
       2 · ○ · · · · ● ○ ●
       1 · · ○ · · · ● ● ●
    `,
  },

  {
    toPlay:     '●',
    comment:    'Attack 5 stones A',
    require: ['e4'],
    board: `
         a b c d e f g h i
       9 · · · · · · ○ · ·
       8 · ● · · · · · ○ ·
       7 · · · · · · · · ○
       6 · · · · · · · · ·
       5 · · · · · · ● · ·
       4 · · · ·(·)○ ○ ● ●
       3 ○ · · · · ● ○ ○ ●
       2 · ○ · · · · ● ○ ●
       1 · · ○ · · · ● ● ●
    `,
  },

  {
    toPlay:     '●',
    comment:    'Attack 5 stones B',
    require: ['c4'],
    board: `
         a b c d e f g
       7 · · · · · · ·
       6 · ● · · · · ·
       5 · · · · ● · ·
       4 · ·(·)○ ○ ● ·
       3 · · · ● ○ ○ ●
       2 · · · · ● ○ ●
       1 · · · · · ● ●
    `,
  },

  {
    toPlay:     '●',
    comment:    'Attack 5 stones (fail)',
    prohibit: ['c4','d5'],
    board: `
         a b c d e f g
       7 · · · · · · ·
       6 · ○ · · · · ·
       5 · · ·(·)● · ·
       4 · ·(·)○ ○ ● ●
       3 · · · ● ○ ○ ●
       2 · · · · ● ○ ●
       1 · · · · ● ● ●
    `,
  },

  {
    toPlay:     '○',
    comment:    'Extend 5 stones',
    require: ['f5'],
    board: `
         a b c d e f g h i
       9 · · · · · · · · ·
       8 · ○ · · · · · · ·
       7 · · · · · · · · ·
       6 · · · · · · · · ·
       5 · · · · ·(·)● · ·
       4 · · · · ● ○ ○ ● ·
       3 · · · · · ● ○ ○ ●
       2 · · · · · · ● ○ ●
       1 · · · · · · · ● ●
    `,
  },

  {
    toPlay:     '○',
    comment:    'Extend 5 stones (fail)',
    prohibit: ['f5'],
    board: `
         a b c d e f g h i
       9 · · · · · · ○ · ○
       8 · ● · · · · ○ ○ ·
       7 · · · · · · · · ○
       6 · · · · · · · · ·
       5 · · · · ·(·)● · ·
       4 · · · · ● ○ ○ ● ·
       3 ○ ○ · · · ● ○ ○ ●
       2 · ○ · · · · ● ○ ●
       1 ○ · ○ · · · · ● ●
    `,
  },

  {
    toPlay:     '●',
    comment:    'Attack 6 stones',
    require: ['f6'],
    board: `
         a b c d e f g h i
       9 · · · · · · ○ · ○
       8 · ● · · · · · ○ ·
       7 · · · · · · · · ○
       6 · · · · ·(·)· · ·
       5 · · · · · ○ ● · ·
       4 · · · · ● ○ ○ ● ·
       3 ○ · · · · ● ○ ○ ●
       2 · ○ · · · · ● ○ ●
       1 ○ · ○ · · · · ● ●
    `,
  },

  {
    toPlay:     '●',
    comment:    'Attack 6 stones (fail)',
    prohibit: ['f6','e5'],
    board: `
         a b c d e f g h i
       9 · · · · · · · · ·
       8 · ○ · · · · · · ·
       7 · · · · · · · · ·
       6 · · · · ·(·)· · ·
       5 · · · ·(·)○ ● · ·
       4 · · · · ● ○ ○ ● ●
       3 · · · · · ● ○ ○ ●
       2 · · · · · · ● ○ ●
       1 · · · · · · ● ● ●
    `,
  },

  {
    toPlay:     '○',
    comment:    'Extend 6 stones',
    require: ['e5'],
    board: `
         a b c d e f g h i
       9 · · · · · · · · ·
       8 · ○ · · · · · · ·
       7 · · · · · · · · ·
       6 · · · · · ● · · ·
       5 · · · ·(·)○ ● · ·
       4 · · · · ● ○ ○ ● ●
       3 · · · · · ● ○ ○ ●
       2 · · · · · · ● ○ ●
       1 · · · · · · ● ● ●
    `,
  },

  {
    toPlay:     '○',
    comment:    'Extend 6 stones (fail)',
    prohibit: ['e5'],
    board: `
         a b c d e f g h i
       9 · · · · · · ○ · ○
       8 · ● · · · · ○ ○ ·
       7 · · · · · · · ○ ○
       6 · · · · · ● · · ·
       5 · · · ·(·)○ ● · ·
       4 · · · · ● ○ ○ ● ●
       3 ○ ○ · · · ● ○ ○ ●
       2 · ○ ○ · · · ● ○ ●
       1 ○ · ○ · · · ● ● ●
    `,
  },

];

// ── Evaluation ─────────────────────────────────────────────────────────────

const NW = Math.max('position'.length, ...POSITIONS.map(p => p.comment.length));
const TW = 2 * String(trials).length + 1;   // e.g. "10/10"

const startTime = performance.now();
console.log(`Agent: ${agentName}  budget: ${budgetMs}ms  trials: ${trials}\n`);
const RW = '100.0%'.length;
console.log(` ${'position'.padEnd(NW)}  ${'pass'.padStart(TW)}  ${'ratio'.padStart(RW)}`);
console.log(` ${'-'.repeat(NW)}  ${'-'.repeat(TW)}  ------`);

let totalPassed = 0, totalTrials = 0;
for (const pos of POSITIONS) {
  let passed = 0;

  for (let t = 0; t < trials; t++) {
    const game = buildPosition(pos);
    const move = agent(game, budgetMs);

    let ok = true;

    if (pos.require  && pos.require.length  > 0) {
      ok &&= pos.require.some(s => matchesMove(s, move, game.N));
    }
    if (pos.prohibit && pos.prohibit.length > 0) {
      ok &&= !pos.prohibit.some(s => matchesMove(s, move, game.N));
    }

    if (ok) passed++;
  }

  totalPassed += passed;
  totalTrials += trials;
  const pct = (100 * passed / trials).toFixed(1) + '%';
  const frac = `${passed}/${trials}`;
  console.log(` ${pos.comment.padEnd(NW)}  ${frac.padStart(TW)}  ${pct.padStart(RW)}`);
}

const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
console.log(`\nOverall: ${totalPassed}/${totalTrials} (${(100 * totalPassed / totalTrials).toFixed(1)}%)  elapsed: ${elapsed}s`);
