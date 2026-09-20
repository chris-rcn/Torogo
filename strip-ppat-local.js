#!/usr/bin/env node
'use strict';

// strip-ppat-local.js — zero the 7 hand-coded previous-move ("local") feature
// weights in a ppat data file.
//
// Layout of a ppat weight vector (see ppat-lib.js totalWeights):
//   [ phase 0 patterns | ... | phase P-1 patterns | phase 0 local x7 | ... ]
// i.e. phases*numPatterns pattern weights followed by phases*7 local weights.
// This zeroes the local block, leaving every pattern weight and the file's
// length untouched — so the result loads everywhere the original did and the
// local features simply contribute nothing.
//
// Zeroing rather than deleting is deliberate: the slots are addressed by fixed
// offset (prevOffset + b), so removing them would shift nothing but would break
// totalWeights and every consumer's length check.  A zero weight is exactly
// "this feature has no effect", which is what the C build with the features
// commented out produces at runtime.
//
// Usage:
//   node strip-ppat-local.js --in <file.js> [--out <file.js>] [--dry-run]
//
//   --in FILE     ppat weights file to read                        (required)
//   --out FILE    where to write (default: alongside, "-nolocal.js")
//   --dry-run     report what would change; write nothing
//   --help        show this message

const fs   = require('fs');
const path = require('path');
const Util = require('./util.js');
const PPat = require('./ppat-lib.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help', 'dry-run'], ['in', 'out']);
if (opts.help || !opts.in) {
  console.error('Usage: node strip-ppat-local.js --in <file.js> [--out <file.js>] [--dry-run]');
  process.exit(opts.help ? 0 : 1);
}

const inPath = path.resolve(opts.in);
const raw = require(inPath);
if (!raw || !raw.weights) { console.error(`${opts.in}: not a ppat weights file`); process.exit(1); }

const phases  = raw.phases || 1;
const adjLib  = raw.adjLib != null ? raw.adjLib : (raw.libCap != null ? raw.libCap : 2);   // legacy field: libCap
const nPat    = raw.numPatterns != null ? raw.numPatterns : (PPat.totalWeights(phases, adjLib) - phases * 7) / phases;
const expected = PPat.totalWeights(phases, adjLib);
if (raw.weights.length !== expected) {
  console.error(`${opts.in}: weights length ${raw.weights.length} but ${expected} expected ` +
                `(phases ${phases}, adjLib ${adjLib}, numPatterns ${nPat}) — refusing to guess the layout`);
  process.exit(1);
}

const localStart = phases * nPat;
const w = Float32Array.from(raw.weights);
let nonZero = 0, maxAbs = 0;
for (let i = localStart; i < w.length; i++) {
  const v = w[i];
  if (v !== 0) nonZero++;
  if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
  w[i] = 0;
}

console.log(`${opts.in}: ${w.length} weights = ${phases} phase(s) x ${nPat} patterns + ${phases} x 7 local`);
console.log(`  local block [${localStart}..${w.length - 1}]: ${nonZero} non-zero, max |w| ${maxAbs.toFixed(4)}`);
for (let p = 0; p < phases; p++) {
  const vals = Array.from(raw.weights.slice(localStart + p * 7, localStart + p * 7 + 7));
  console.log(`    phase ${p}: ` + vals.map(v => v.toFixed(3).padStart(8)).join(''));
}

if (opts['dry-run']) { console.log('  --dry-run: nothing written'); process.exit(0); }

const outPath = opts.out
  ? path.resolve(opts.out)
  : inPath.replace(/(\.js)?$/, '-nolocal.js').replace('.js-nolocal.js', '-nolocal.js');
const meta = [`phases: ${phases}`, `numPatterns: ${nPat}`, `adjLib: ${adjLib}`];
const src = [
  "'use strict';",
  `// Local (previous-move) features zeroed by strip-ppat-local.js from ${path.basename(inPath)}.`,
  `// Pattern weights unchanged; the 7 local slots per phase are 0, so those`,
  `// features contribute nothing while the file stays the expected length.`,
  `const _w = { weights: new Float32Array([${Array.from(w).join(',')}]), ${meta.join(', ')} };`,
  "if (typeof module !== 'undefined') module.exports = _w;",
  'else window.PPATWeights = _w;',
].join('\n') + '\n';
fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, src);
console.log(`  wrote ${outPath}`);
