'use strict';

// convert-to-ndjson.js — convert old point-diffs CSV data files to the new gen-deltas NDJSON format.
//
// The old per-line format was:  <prefix coord csv> ; <coord>=<diff> <coord>=<diff> ...
// where <diff> = candidate's mean playout point score minus the (shared) pass baseline — i.e. only
// the point DELTA was stored.  So the converted record carries that delta as p[] with pp=0, which
// preserves the point target exactly (point-delta_i = p[i] − pp = p[i]).  The absolute point means
// and the win-ratio fields can't be recovered, so converted records have NO w/pw — the point
// target is available on them, the win target is not.
//
// Multiple inputs (comma-separated) are concatenated; board size is read from each file's header,
// so mixed-size outputs are fine (the trainer reads size per record).
//
// Usage:  node convert-to-ndjson.js --in OLD[,OLD...] --out NEW.ndjson

const fs = require('fs');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['in', 'out']);
if (opts.help || !opts.in || !opts.out) {
  console.log('Usage: node convert-to-ndjson.js --in OLD[,OLD...] --out NEW.ndjson');
  process.exit(opts.help ? 0 : 1);
}

const FILES = opts.in.split(',').map(s => s.trim()).filter(Boolean);
const q2 = x => Math.round(x * 100) / 100;
const fd = fs.openSync(opts.out, 'w');
let total = 0;

for (const file of FILES) {
  if (!fs.existsSync(file)) { console.error(`Error: input not found: ${file}`); process.exit(1); }
  const lines = fs.readFileSync(file, 'utf8').split('\n');
  let size = 0;
  for (const l of lines) { const m = /^#.*\bsize=(\d+)/.exec(l); if (m) { size = parseInt(m[1], 10); break; } }
  if (!size) { console.error(`Error: no "size=N" header in ${file}`); process.exit(1); }

  let n = 0;
  for (const line of lines) {
    if (!line || line[0] === '#') continue;
    const semi = line.indexOf(';');
    if (semi < 0) continue;
    const pos = line.slice(0, semi);
    const toks = line.slice(semi + 1).trim().split(/\s+/).filter(Boolean);
    const moves = [], p = [];
    for (const t of toks) {
      const eq = t.lastIndexOf('=');
      if (eq < 0) continue;
      moves.push(t.slice(0, eq));
      p.push(q2(parseFloat(t.slice(eq + 1))));
    }
    if (!moves.length) continue;
    fs.writeSync(fd, JSON.stringify({ size, pos, moves: moves.join(','), p, pp: 0 }) + '\n');
    n++; total++;
  }
  console.log(`converted ${n} records (size ${size}) from ${file}`);
}

fs.closeSync(fd);
console.log(`done: ${total} records -> ${opts.out}`);
