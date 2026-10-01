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
// A result line (the problem line plus " result=... played=...", written by
// --verbose) is ignored, so verbose output loads as a case file.
//
// Usage: node evalladders2.js --file cases.txt [--agent npat] [--budget 1] [--oversample 1]

const fs = require('fs');
const path = require('path');
const { performance } = require('perf_hooks');
const { BLACK, WHITE, parseBoard, parseMove, coordStr } = require('./game2.js');
const Util = require('./util.js');

// Parse the text-block file into cases.
function loadCases(file) {
  const cases = [];
  for (const block of fs.readFileSync(file, 'utf8').split(/\n\s*\n/)) {
    const lines = block.split('\n').filter(l => !/ result=/.test(l));
    const metaLine = lines.find(l => l.includes('='));
    if (!metaLine) continue;                     // blank / torn block
    const meta = {};
    for (const tok of metaLine.trim().split(/\s+/)) {
      const eq = tok.indexOf('=');
      if (eq > 0) meta[tok.slice(0, eq)] = tok.slice(eq + 1);
    }
    cases.push({
      text:      lines.join('\n').replace(/^\n+|\n+$/g, ''),      // the block as read, for --verbose
      metaLine:  metaLine.trimEnd(),
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
    let p = 0;
    const played = [];
    for (let t = 0; t < oversample; t++) {
      const game = parseBoard(c.board, c.toPlay);
      const mv = agent(game, budgetMs);
      played.push(coordStr(mv.move, game.N));
      let ok = true;
      if (c.require)  ok = ok &&  c.require.some(s => mv.move === parseMove(s, game.N));
      if (c.prohibit) ok = ok && !c.prohibit.some(s => mv.move === parseMove(s, game.N));
      if (ok) p++;
    }
    if (verbose) {
      // The case block as read, then its problem line plus the result.
      const res = p === oversample ? 'PASS' : p === 0 ? 'FAIL' : `${p}/${oversample}`;
      console.log(`${c.text}\n${c.metaLine} result=${res}  played=${played.join(',')}\n`);
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
  const opts = Util.parseArgs(process.argv.slice(2), ['help', 'verbose'], ['agent', 'budget', 'file', 'limit', 'oversample', 'verbose']);
  if (opts.help || !opts.file) {
    console.log(`Usage: node evalladders2.js --file <cases.txt> [options]

Runs a ladder case file against an AI agent and reports, per case, how often
the agent played one of the required moves.  Unlike evalladders.js the cases
come from a file rather than being hardcoded.

  --file FILE       case file to run                            (required)
  --agent NAME      ai/<name>.js to evaluate                    (default random)
  --budget MS       time budget per move, milliseconds          (default 1)
  --limit N         run only the first N cases                  (default: all)
  --oversample N    evaluations per case; >1 is worth it for a
                    stochastic agent, whose answer varies       (default 1)
  --verbose         reprint each case block as read, then its problem line
                    plus " result=PASS|FAIL|p/n  played=<moves>"
  --help            show this message

Also usable as a library — the trainers' \`ladr\` column runs it per status
print: loadCases(file) then evalCases(cases, getMove, { budgetMs, oversample })`);
    process.exit(opts.help ? 0 : 1);
  }
  const agentName  = opts.agent || 'random';
  const budgetMs   = parseInt(opts.budget || '1', 10);
  const oversample = parseInt(opts.oversample || '1', 10);
  const limit      = opts.limit !== undefined ? parseInt(opts.limit, 10) : Infinity;
  if (!(limit >= 1)) { console.error('--limit must be a positive integer'); process.exit(1); }
  const verbose    = opts.verbose !== undefined;
  const _agentMod = require(path.join(__dirname, 'ai', agentName + '.js'));
// create(cfg)-style agents (phase-mux, the puct family) instantiate with a
// plain env reader; bare { getMove } modules are used directly.
const agent = (typeof _agentMod.create === 'function'
    ? _agentMod.create(Util.makeCfg(null)) : _agentMod).getMove;

  const all   = loadCases(opts.file);
  const cases = all.slice(0, limit);
  console.log(`file: ${opts.file}  cases: ${cases.length}${cases.length < all.length ? `/${all.length}` : ''}  agent: ${agentName}  budget: ${budgetMs}ms  oversample: ${oversample}\n`);
  const t0 = performance.now();
  const { passed, total, byType } = evalCases(cases, agent, { budgetMs, oversample, verbose });
  for (const [type, a] of byType) {
    console.log(`  ${type.padEnd(14)} ${String(a.passed).padStart(5)}/${String(a.total).padEnd(5)}  ${(100 * a.passed / a.total).toFixed(1)}%`);
  }
  console.log(`\nOverall: ${passed}/${total} (${(100 * passed / total).toFixed(1)}%)  elapsed: ${((performance.now() - t0) / 1000).toFixed(1)}s`);
}
