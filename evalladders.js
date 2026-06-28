'use strict';
const { performance } = require('perf_hooks');
const path = require('path');
const { Game2, BLACK, WHITE, parseBoard, parseMove } = require('./game2.js');
const Util = require('./util.js');

/**
 * Ladder evaluation — run hardcoded ladder positions against an AI agent.
 *
 * Usage:
 *   node evalladders.js [options]
 *
 * Options:
 *   --agent   <name>   AI policy to evaluate (default: random)
 *   --budget  <ms>     Time budget per move in ms (default: 1)
 *   --oversample <n>   Evaluations per position (default: 10)
 *   --help             Show this help message
 *
 * Also used as a library (record-npats.js runs it as a per-checkpoint gate):
 *   evalLadders(agent, { budgetMs, oversample }) → { passed, total, rows }
 */

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

// Run every position `oversample` times against `agent` (getMove-style function).
// Returns { passed, total, rows: [{ comment, passed, oversample }] }.
function evalLadders(agent, { budgetMs = 1, oversample = 10 } = {}) {
  const rows = [];
  let passed = 0, total = 0;
  for (const pos of POSITIONS) {
    let posPassed = 0;
    for (let t = 0; t < oversample; t++) {
      const game = buildPosition(pos);
      const move = agent(game, budgetMs);

      let ok = true;
      if (pos.require  && pos.require.length  > 0) {
        ok &&= pos.require.some(s => matchesMove(s, move, game.N));
      }
      if (pos.prohibit && pos.prohibit.length > 0) {
        ok &&= !pos.prohibit.some(s => matchesMove(s, move, game.N));
      }
      if (ok) posPassed++;
    }
    rows.push({ comment: pos.comment, passed: posPassed, oversample });
    passed += posPassed;
    total += oversample;
  }
  return { passed, total, rows };
}

module.exports = { POSITIONS, evalLadders };

// ── CLI ──────────────────────────────────────────────────────────────────────

if (require.main === module) {

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['agent', 'budget', 'oversample']);

if (opts.help) {
  console.log('Usage: node evalladders.js [--agent <name>] [--budget <ms>] [--oversample <n>]');
  process.exit(0);
}

const agentName  = opts.agent  || 'random';
const budgetMs   = parseInt(opts.budget || '1', 10);
const oversample = parseInt(opts.oversample || '10',  10);

if (isNaN(budgetMs)   || budgetMs   < 1) { console.error('--budget must be a positive integer'); process.exit(1); }
if (isNaN(oversample) || oversample < 1) { console.error('--oversample must be a positive integer'); process.exit(1); }

const { getMove: agent } = require(path.join(__dirname, 'ai', agentName + '.js'));

const NW = Math.max('position'.length, ...POSITIONS.map(p => p.comment.length));
const TW = 2 * String(oversample).length + 1;   // e.g. "10/10"
const RW = '100.0%'.length;

const startTime = performance.now();
console.log(`Agent: ${agentName}  budget: ${budgetMs}ms  oversample: ${oversample}\n`);
console.log(` ${'position'.padEnd(NW)}  ${'pass'.padStart(TW)}  ${'ratio'.padStart(RW)}`);
console.log(` ${'-'.repeat(NW)}  ${'-'.repeat(TW)}  ------`);

const { passed, total, rows } = evalLadders(agent, { budgetMs, oversample });
for (const row of rows) {
  const pct = (100 * row.passed / row.oversample).toFixed(1) + '%';
  const frac = `${row.passed}/${row.oversample}`;
  console.log(` ${row.comment.padEnd(NW)}  ${frac.padStart(TW)}  ${pct.padStart(RW)}`);
}

const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
console.log(`\nOverall: ${passed}/${total} (${(100 * passed / total).toFixed(1)}%)  elapsed: ${elapsed}s`);

}
