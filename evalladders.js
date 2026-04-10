'use strict';
const { performance } = require('perf_hooks');
const path = require('path');
const { Game2, BLACK, WHITE, parseBoard } = require('./game2.js');
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

// Parse a coordinate string like 'j8' → { x: 9, y: 7 }.
// Column letter (a=0, b=1, …) gives x; row number (1-based) gives y.
function parseCoord(str) {
  return { x: str.charCodeAt(0) - 97, y: parseInt(str.slice(1), 10) - 1 };
}

// Returns true when the agent move matches the spec string ('pass' or 'j8'-style coord).
function matchesMove(s, move) {
  if (s === 'pass') return move.type === 'pass';
  const c = parseCoord(s);
  return move.type === 'place' && c.x === move.x && c.y === move.y;
}

// ── Position builder ────────────────────────────────────────────────────────

function buildPosition(pos) {
  return parseBoard(pos.board, pos.toPlay === '●' ? BLACK : WHITE);
}

// ── Positions ──────────────────────────────────────────────────────────────

const POSITIONS = [

  {
    toPlay:     '●',
    comment:    'Attack 2 stones',
    require: ['f4'],
    board: `
         a b c d e f g 
       1 · · · · · · · 
       2 · ● · · · · · 
       3 · · · · · · · 
       4 · · · · ·(·)· 
       5 · · · · · ○ ● 
       6 · · · · ● ○ ● 
       7 · · · · ● ● ● 
    `,
  },

  {
    toPlay:     '●',
    comment:    'Attack 2 stones (fail)',
    prohibit: ['f4','e5'],
    board: `
         a b c d e f g 
       1 · · · · · · · 
       2 · ○ · · · · · 
       3 · · · · · · · 
       4 · · · · ·(·)· 
       5 · · · ·(·)○ ● 
       6 · · · · ● ○ ● 
       7 · · · · ● ● ● 
    `,
  },

  {
    toPlay:     '○',
    comment:    'Extend 2 stones',
    require: ['e5'],
    board: `
         a b c d e f g 
       1 · · · · · · · 
       2 · ○ · · · · · 
       3 · · · · · · · 
       4 · · · · · ● · 
       5 · · · ·(·)○ ● 
       6 · · · · ● ○ ● 
       7 · · · · ● ● ● 
    `,
  },

  {
    toPlay:     '○',
    comment:    'Extend 2 stones (fail)',
    prohibit: ['e5'],
    board: `
         a b c d e f g 
       1 · · · · ○ · · 
       2 · ● · · · ○ · 
       3 · · · · · · ○ 
       4 · · · · · ● · 
       5 ○ · · ·(·)○ ● 
       6 · ○ · · ● ○ ● 
       7 · · ○ · ● ● ● 
    `,
  },

  {
    toPlay:     '●',
    comment:    'Attack 3 stones',
    require: ['d5'],
    board: `
         a b c d e f g 
       1 · · · · · · · 
       2 · ● · · · · · 
       3 · · · · · · · 
       4 · · · · · ● · 
       5 · · ·(·)○ ○ ● 
       6 · · · · ● ○ ● 
       7 · · · · · ● ● 
    `,
  },

  {
    toPlay:     '●',
    comment:    'Attack 3 stones (fail)',
    prohibit: ['d5','e4'],
    board: `
         a b c d e f g 
       1 · · · · · · · 
       2 · ○ · · · · · 
       3 · · · · · · · 
       4 · · · ·(·)● · 
       5 · · ·(·)○ ○ ● 
       6 · · · · ● ○ ● 
       7 · · · · · ● ● 
    `,
  },

  {
    toPlay:     '○',
    comment:    'Extend 3 stones',
    require: ['e4'],
    board: `
         a b c d e f g 
       1 · · · · · · · 
       2 · ○ · · · · · 
       3 · · · · · · · 
       4 · · · ·(·)● · 
       5 · · · ● ○ ○ ● 
       6 · · · · ● ○ ● 
       7 · · · · · ● ● 
    `,
  },

  {
    toPlay:     '○',
    comment:    'Extend 3 stones (fail)',
    prohibit: ['e4'],
    board: `
         a b c d e f g 
       1 · · · · ○ · ○ 
       2 · ● · · · ○ · 
       3 · · · · · · ○ 
       4 · · · ·(·)● · 
       5 ○ · · ● ○ ○ ● 
       6 · ○ · · ● ○ ● 
       7 ○ · ○ · · ● ● 
    `,
  },

  {
    toPlay:     '●',
    comment:    'Attack 4 stones',
    require: ['e3'],
    board: `
         a b c d e f g 
       1 · · · · · · · 
       2 · ● · · · · · 
       3 · · · ·(·)· · 
       4 · · · · ○ ● · 
       5 · · · ● ○ ○ ● 
       6 · · · · ● ○ ● 
       7 · · · · · ● ● 
    `,
  },

  {
    toPlay:     '●',
    comment:    'Attack 4 stones (fail)',
    prohibit: ['e3','d4'],
    board: `
         a b c d e f g 
       1 · · · · · · · 
       2 · ○ · · · · · 
       3 · · · ·(·)· · 
       4 · · ·(·)○ ● · 
       5 · · · ● ○ ○ ● 
       6 · · · · ● ○ ● 
       7 · · · · · ● ● 
    `,
  },

  {
    toPlay:     '○',
    comment:    'Extend 4 stones',
    require: ['d4'],
    board: `
         a b c d e f g 
       1 · · · · · · · 
       2 · ○ · · · · · 
       3 · · · · ● · · 
       4 · · ·(·)○ ● · 
       5 · · · ● ○ ○ ● 
       6 · · · · ● ○ ● 
       7 · · · · · ● ● 
    `,
  },

  {
    toPlay:     '○',
    comment:    'Extend 4 stones (fail)',
    pohibit: ['f6'],
    board: `
         a b c d e f g h i
       1 · · · · · · ○ · · 
       2 · ● · · · · · ○ · 
       3 · · · · · · · · ○ 
       4 · · · · · · · · · 
       5 · · · · · · ● · · 
       6 · · · · ·(·)○ ● ● 
       7 ○ · · · · ● ○ ○ ● 
       8 · ○ · · · · ● ○ ● 
       9 · · ○ · · · ● ● ● 
    `,
  },

  {
    toPlay:     '●',
    comment:    'Attack 5 stones',
    require: ['c4'],
    board: `
         a b c d e f g 
       1 · · · · · · · 
       2 · ● · · · · · 
       3 · · · · ● · · 
       4 · ·(·)○ ○ ● · 
       5 · · · ● ○ ○ ● 
       6 · · · · ● ○ ● 
       7 · · · · · ● ● 
    `,
  },

  {
    toPlay:     '●',
    comment:    'Attack 5 stones (fail)',
    prohibit: ['c4','d3'],
    board: `
         a b c d e f g 
       1 · · · · · · · 
       2 · ○ · · · · · 
       3 · · ·(·)● · · 
       4 · ·(·)○ ○ ● ● 
       5 · · · ● ○ ○ ● 
       6 · · · · ● ○ ● 
       7 · · · · ● ● ● 
    `,
  },

  {
    toPlay:     '○',
    comment:    'Extend 5 stones',
    require: ['f5'],
    board: `
         a b c d e f g h i
       1 · · · · · · · · · 
       2 · ○ · · · · · · · 
       3 · · · · · · · · · 
       4 · · · · · · · · · 
       5 · · · · ·(·)● · · 
       6 · · · · ● ○ ○ ● · 
       7 · · · · · ● ○ ○ ● 
       8 · · · · · · ● ○ ● 
       9 · · · · · · · ● ● 
    `,
  },

  {
    toPlay:     '○',
    comment:    'Extend 5 stones (fail)',
    prohibit: ['f5'],
    board: `
         a b c d e f g h i
       1 · · · · · · · · ○ 
       2 · ● · · · · · ○ · 
       3 · · · · · · · · · 
       4 · · · · · · · · · 
       5 · · · · ·(·)● · · 
       6 · · · · ● ○ ○ ● · 
       7 · · · · · ● ○ ○ ● 
       8 · ○ · · · · ● ○ ● 
       9 ○ · · · · · · ● ● 
    `,
  },

  {
    toPlay:     '●',
    comment:    'Attack 6 stones',
    require: ['f4'],
    board: `
         a b c d e f g h i
       1 · · · · · · ○ · ○ 
       2 · ● · · · · · ○ · 
       3 · · · · · · · · ○ 
       4 · · · · ·(·)· · · 
       5 · · · · · ○ ● · · 
       6 · · · · ● ○ ○ ● · 
       7 ○ · · · · ● ○ ○ ● 
       8 · ○ · · · · ● ○ ● 
       9 ○ · ○ · · · · ● ● 
    `,
  },

  {
    toPlay:     '●',
    comment:    'Attack 6 stones (fail)',
    prohibit: ['f4','e5'],
    board: `
         a b c d e f g h i
       1 · · · · · · ○ · · 
       2 · ○ · · · · · ○ · 
       3 · · · · · · · · ○ 
       4 · · · · ·(·)· · · 
       5 · · · ·(·)○ ● · · 
       6 · · · · ● ○ ○ ● ● 
       7 ○ · · · · ● ○ ○ ● 
       8 · ○ · · · · ● ○ ● 
       9 · · ○ · · · ● ● ● 
    `,
  },

  {
    toPlay:     '○',
    comment:    'Extend 6 stones',
    require: ['e5'],
    board: `
         a b c d e f g h i
       1 · · · · · · · · · 
       2 · ○ · · · · · · · 
       3 · · · · · · · · · 
       4 · · · · · ● · · · 
       5 · · · ·(·)○ ● · · 
       6 · · · · ● ○ ○ ● ● 
       7 · · · · · ● ○ ○ ● 
       8 · · · · · · ● ○ ● 
       9 · · · · · · ● ● ● 
    `,
  },

  {
    toPlay:     '○',
    comment:    'Extend 6 stones (fail)',
    prohibit: ['e5'],
    board: `
         a b c d e f g h i
       1 · · · · · · ○ · ○ 
       2 · ● · · · · ○ ○ · 
       3 · · · · · · · ○ ○ 
       4 · · · · · ● · · · 
       5 · · · ·(·)○ ● · · 
       6 · · · · ● ○ ○ ● ● 
       7 ○ ○ · · · ● ○ ○ ● 
       8 · ○ ○ · · · ● ○ ● 
       9 ○ · ○ · · · ● ● ● 
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

let totalTrials = 0;
let totalPassed = 0;

for (const pos of POSITIONS) {
  let passed = 0;

  for (let t = 0; t < trials; t++) {
    const game = buildPosition(pos);
    const move = agent(game, budgetMs);

    let ok = true;

    if (pos.require  && pos.require.length  > 0) {
      ok &&= pos.require.some(s => matchesMove(s, move));
    }
    if (pos.prohibit && pos.prohibit.length > 0) {
      ok &&= !pos.prohibit.some(s => matchesMove(s, move));
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
