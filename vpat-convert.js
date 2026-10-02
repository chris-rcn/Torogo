'use strict';

// vpat-convert.js — rewrite a vpatterns file's `weights: new Map([[k, v], ...])`
// literal as `weightsQ6: { count, b64 }` (see vpatterns.js), leaving every other
// line and field as written.  The file is read as text, never executed, so it
// also converts files whose Map exceeds V8's 2^24-entry limit and no longer load.
//
// Usage: node vpat-convert.js --in FILE [--out FILE]

const fs = require('fs');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['in', 'out']);
if (opts.help || !opts.in) {
  console.log(`Usage: node vpat-convert.js --in FILE [--out FILE]

Rewrites a vpatterns model's literal weights Map as weightsQ6 (base64 int32
keys and millionth values), which has no Map size limit and loads faster.

  --in FILE    vpatterns model in the old format (required)
  --out FILE   output path (default: <in>-q6.js)
  --help       show this message`);
  process.exit(opts.help ? 0 : 1);
}
const OUT = opts.out || opts.in.replace(/\.js$/, '') + '-q6.js';

const src = fs.readFileSync(opts.in, 'utf8');
const OPEN = 'weights: new Map([', CLOSE = '])';
const start = src.indexOf(OPEN);
if (start < 0) { console.error(`${opts.in}: no 'weights: new Map([' literal (already weightsQ6?)`); process.exit(1); }
const end = src.indexOf(CLOSE, start + OPEN.length);
if (end < 0) { console.error(`${opts.in}: unterminated weights Map`); process.exit(1); }

// Count the [k,v] pairs, then parse them into typed arrays.
let n = 0;
for (let i = start + OPEN.length; i < end; i++) if (src.charCodeAt(i) === 91) n++;   // '['
const keys = new Int32Array(n), q = new Int32Array(n);
let i = start + OPEN.length, got = 0, kept = 0;
while (got < n) {
  const a = src.indexOf('[', i), comma = src.indexOf(',', a), b = src.indexOf(']', comma);
  const k = Number(src.slice(a + 1, comma)), v = Number(src.slice(comma + 1, b));
  if (!Number.isInteger(k) || !Number.isFinite(v)) { console.error(`${opts.in}: bad weight entry near offset ${a}`); process.exit(1); }
  const m = Math.round(v * 1e6);
  if (m > 2147483647 || m < -2147483648) { console.error(`${opts.in}: weight ${v} does not fit weightsQ6`); process.exit(1); }
  got++;
  i = b + 1;
  if (m === 0) continue;   // reads back as 0 anyway; saveWeights skips these too
  keys[kept] = k | 0; q[kept] = m; kept++;
}
if (new Uint8Array(new Uint32Array([1]).buffer)[0] !== 1) { console.error('vpat-convert: assumes a little-endian host'); process.exit(1); }
const b64 = Buffer.concat([Buffer.from(keys.buffer, 0, kept * 4), Buffer.from(q.buffer, 0, kept * 4)]).toString('base64');

Util.writeFileAtomic(OUT, src.slice(0, start) + `weightsQ6: { count: ${kept}, b64: '${b64}' }` + src.slice(end + CLOSE.length));
console.log(`${opts.in}: ${n} weights (${n - kept} zero, dropped) -> ${kept} in ${OUT} (${fs.statSync(opts.in).size} -> ${fs.statSync(OUT).size} bytes)`);
