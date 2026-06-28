'use strict';

// evalladders2.js — evaluate an agent against a GENERATED ladder-test file
// (produced by gen-ladders.js --out).  Independent of evalladders.js.
//
// File format: blank-line-separated blocks, each
//     type=kill chainSize=10 toPlay=W require=g1 by=puct-static
//        a b c d e f g h i j k l m         <- labels (ignored by parseBoard)
//     13 ○ · ● ...                          <- N rows of the position
//      1 ...
//   require=<coord>  → agent must play it;  prohibit=<c1,c2,...> → must avoid all.
//
// Usage: node evalladders2.js --file cases.txt [--agent npat] [--budget 1] [--oversample 1]

const fs = require('fs');
const path = require('path');
const { performance } = require('perf_hooks');
const { BLACK, WHITE, PASS, parseBoard, parseMove, coordStr } = require('./game2.js');
const Util = require('./util.js');

// Parse the text-block file into cases.
function loadCases(file) {
  const cases = [];
  for (const block of fs.readFileSync(file, 'utf8').split(/\n\s*\n/)) {
    const lines = block.split('\n');
    const metaLine = lines.find(l => l.includes('='));
    if (!metaLine) continue;                     // blank / torn block
    const meta = {};
    for (const tok of metaLine.trim().split(/\s+/)) {
      const eq = tok.indexOf('=');
      if (eq > 0) meta[tok.slice(0, eq)] = tok.slice(eq + 1);
    }
    cases.push({
      board:     lines.filter(l => l !== metaLine).join('\n'),   // parseBoard strips the labels
      toPlay:    meta.toPlay === 'B' ? BLACK : WHITE,
      require:   meta.require  ? meta.require.split(',')  : null,
      prohibit:  meta.prohibit ? meta.prohibit.split(',') : null,
      type:      meta.type || '?',
      chainSize: meta.chainSize ? parseInt(meta.chainSize, 10) : 0,
    });
  }
  return cases;
}

function evalCases(cases, agent, { budgetMs, oversample, verbose = false }) {
  let passed = 0, total = 0;
  const byType = new Map();   // type -> { passed, total }
  for (let i = 0; i < cases.length; i++) {
    const c = cases[i];
    let p = 0, lastGame = null, lastMove = PASS;
    for (let t = 0; t < oversample; t++) {
      const game = parseBoard(c.board, c.toPlay);
      const mv = agent(game, budgetMs);
      lastGame = game; lastMove = mv.move;
      let ok = true;
      if (c.require)  ok = ok &&  c.require.some(s => mv.move === parseMove(s, game.N));
      if (c.prohibit) ok = ok && !c.prohibit.some(s => mv.move === parseMove(s, game.N));
      if (ok) p++;
    }
    if (verbose) {
      const ans = c.require ? `require=${c.require.join(',')}` : `prohibit=${c.prohibit.join(',')}`;
      const res = p === oversample ? 'PASS' : p === 0 ? 'FAIL' : `${p}/${oversample}`;
      console.log(`\n#${i + 1} [${c.type} size${c.chainSize}] toPlay=${c.toPlay === BLACK ? 'B' : 'W'}  ${ans}  played=${coordStr(lastMove, lastGame.N)}  -> ${res}`);
      console.log(lastGame.toString(lastMove, { labels: true }));   // board with the agent's move marked
    }
    passed += p; total += oversample;
    const agg = byType.get(c.type) || { passed: 0, total: 0 };
    agg.passed += p; agg.total += oversample; byType.set(c.type, agg);
  }
  return { passed, total, byType };
}

module.exports = { loadCases, evalCases };

// ── CLI ───────────────────────────────────────────────────────────────────────
if (require.main === module) {
  const opts = Util.parseArgs(process.argv.slice(2), ['help', 'verbose'], ['agent', 'budget', 'file', 'oversample', 'verbose']);
  if (opts.help || !opts.file) {
    console.log('Usage: node evalladders2.js --file cases.txt [--agent npat] [--budget 1] [--oversample 1] [--verbose]');
    process.exit(opts.help ? 0 : 1);
  }
  const agentName  = opts.agent || 'random';
  const budgetMs   = parseInt(opts.budget || '1', 10);
  const oversample = parseInt(opts.oversample || '1', 10);
  const verbose    = opts.verbose !== undefined;
  const { getMove: agent } = require(path.join(__dirname, 'ai', agentName + '.js'));

  const cases = loadCases(opts.file);
  console.log(`file: ${opts.file}  cases: ${cases.length}  agent: ${agentName}  budget: ${budgetMs}ms  oversample: ${oversample}\n`);
  const t0 = performance.now();
  const { passed, total, byType } = evalCases(cases, agent, { budgetMs, oversample, verbose });
  for (const [type, a] of byType) {
    console.log(`  ${type.padEnd(14)} ${String(a.passed).padStart(5)}/${String(a.total).padEnd(5)}  ${(100 * a.passed / a.total).toFixed(1)}%`);
  }
  console.log(`\nOverall: ${passed}/${total} (${(100 * passed / total).toFixed(1)}%)  elapsed: ${((performance.now() - t0) / 1000).toFixed(1)}s`);
}
