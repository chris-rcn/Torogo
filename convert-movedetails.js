'use strict';
// Convert the old movedetails JSONL format to the new *.md format
// (movedetails-format.js).  The old history OMITS the free centre stone (it
// relied on Game2(N,true)); the new history holds EVERY stone and replays from
// an empty board, so the centre stone is prepended as an explicit first move.
//
// Usage: node convert-movedetails.js --in <old> [--out <new.md>]

const fs = require('fs');
const { Game2, parseMove } = require('./game2.js');
const MD = require('./movedetails-format.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['in', 'out']);
if (opts.help || !opts.in) {
  console.error('Usage: node convert-movedetails.js --in <old> [--out <new.md>]');
  process.exit(opts.help ? 0 : 1);
}
const outPath = opts.out || opts.in.replace(/\.[^.]+$/, '') + '.md';

const lines = fs.readFileSync(opts.in, 'utf8').split('\n');
const out = [];
let n = 0;
for (const line of lines) {
  const l = line.trim();
  if (!l) continue;
  if (l[0] === '#') { out.push(l); continue; }        // pass headers through
  const p = JSON.parse(l);                            // { boardSize, history, candidates:[{m,kwr}] }
  const N = p.boardSize;
  const g = new Game2(N, true);                        // OLD reconstruction (free centre stone)
  for (const h of p.history) g.play(parseMove(h, N));
  const phase = 1 - g.emptyCount / (N * N);
  const history = [MD.centerMove(N), ...p.history];    // include the centre stone
  // old JSON stores kwr (x1000); in memory it is a winRatio.
  const candidates = p.candidates.map(c => ({ m: c.m, winRatio: c.kwr == null ? null : c.kwr / 1000 }));
  out.push(MD.formatRow({ boardSize: N, phase, history, candidates }));
  n++;
}
fs.writeFileSync(outPath, out.join('\n') + '\n');
console.error(`converted ${n} positions: ${opts.in} -> ${outPath}`);
