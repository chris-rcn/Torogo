'use strict';

// filter-evals.js — standalone data shaping for ppat eval files.
//
// Reads an eval file — either format, discriminated per line: the current
// gen-agent-evals one ("<size> <phase> <moves> <winRatio>") or the legacy
// gen_evals one ("<size> <moves> <winRatio> [best]") — and writes a
// filtered/balanced, shuffled subset to stdout.  Keeps the main training
// code (train_ppat.c) simple by doing data shaping here.  Lines pass
// through unmodified; the phase filters read the current format's phase
// column directly and replay only legacy lines.  See --help for options.

const fs = require('fs');
const { Game2, PASS } = require('./game2.js');
const { makeRng } = require('./xorshift.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'],
  ['file', 'no-extreme', 'min-phase', 'max-phase', 'value-cap', 'value-buckets', 'seed']);
if (opts.help) {
  console.log(`Usage: node filter-evals.js [options] > out.txt

Standalone data shaping for ppat eval files: reads an eval file, applies the
filters below in order, and writes the surviving records — always shuffled —
to stdout.  Keeps the training code (train_ppat.c) simple by doing data
shaping here.  A '#' header recording the filter settings is prepended;
input '#' lines are dropped.  Per-filter drop counts go to stderr.

Input: either eval format, discriminated per line — the current
gen-agent-evals one ("<size> <phase> <moves> <winRatio>") or the legacy
gen_evals one ("<size> <moves> <winRatio> [best]").  Lines pass through
unmodified, so a mixed input yields a mixed output.  The phase filters
read the current format's phase column directly; legacy lines must be
replayed, which costs far more.

  --file PATH        input file (default: stdin)
  --no-extreme F     keep only winRatio in [F, 1-F] (drop value extremes;
                     same margin as train_ppat's --filter)     (default 0 = off)
  --min-phase P      drop positions with phase < P             (default 0 = off)
  --max-phase P      drop positions with phase > P             (default 1 = off)
                     (phase = board fullness, 1 - empty/area; read from the
                     current format's phase column, replayed for legacy lines)
  --value-cap M      balance the value distribution: bucket survivors into
                     --value-buckets bins by winRatio, then cap each bin to
                     M x (count of the smallest non-empty bin), random-
                     subsampling any over-full bin              (default 0 = off)
  --value-buckets N  number of value bins for --value-cap       (default 10)
  --seed N           RNG seed (subsampling + output shuffle)    (default 1)

Examples:
  node filter-evals.js --file evals.txt --no-extreme 0.01 --value-cap 3 --value-buckets 20 > out.txt
  cat evals.txt | node filter-evals.js --value-cap 2 > out.txt`);
  process.exit(0);
}

const noExtreme    = opts['no-extreme']    !== undefined ? parseFloat(opts['no-extreme'])    : 0;
const minPhase     = opts['min-phase']     !== undefined ? parseFloat(opts['min-phase'])     : 0;
const maxPhase     = opts['max-phase']     !== undefined ? parseFloat(opts['max-phase'])     : 1;
const valueCap     = opts['value-cap']     !== undefined ? parseFloat(opts['value-cap'])     : 0;
const nBuckets     = opts['value-buckets'] !== undefined ? parseInt(opts['value-buckets'], 10) : 10;
const rng = makeRng(opts.seed !== undefined ? parseInt(opts.seed, 10) : 1);
const needPhase = minPhase > 0 || maxPhase < 1;   // only the phase filters require a replay

// Parse a move token ("e3" or "pass") to a flat board index.
function parseMoveTok(t, N) {
  if (t[0] === 'p') return PASS;
  return (parseInt(t.slice(1), 10) - 1) * N + (t.charCodeAt(0) - 97);
}

let total = 0, dropExtreme = 0, dropPhase = 0, skipped = 0;
const kept = [];   // { line, w }

function processLine(line) {
  if (!line || line[0] === '#') return;
  const p = line.split(/\s+/);
  if (p.length < 3) { skipped++; return; }
  // Current format's second token is a phase ("0.345"); a move list starts
  // with a coordinate letter.
  const cur = /^[0-9]/.test(p[1]);
  if (cur && p.length < 4) { skipped++; return; }
  const size = +p[0], moves = p[cur ? 2 : 1], w = +p[cur ? 3 : 2];
  if (!Number.isFinite(size) || !Number.isFinite(w) || !moves) { skipped++; return; }
  total++;

  // Value extremeness (cheap: no replay).
  if (noExtreme > 0 && (w < noExtreme || w > 1 - noExtreme)) { dropExtreme++; return; }

  // Phase filter: current-format lines carry the phase in the file (as
  // train-vpat-supervised trusts it); legacy lines must be replayed.
  if (needPhase) {
    let phase;
    if (cur) {
      phase = parseFloat(p[1]);
    } else {
      const g = new Game2(size, true);
      let ok = true;
      for (const t of moves.split(',')) { if (!g.play(parseMoveTok(t, size))) { ok = false; break; } }
      if (!ok) { skipped++; return; }
      phase = 1 - g.emptyCount / (size * size);
    }
    if (phase < minPhase || phase > maxPhase) { dropPhase++; return; }
  }

  kept.push({ line, w });
}

const fd = opts.file ? fs.openSync(opts.file, 'r') : 0;   // default: stdin
const buf = Buffer.alloc(1 << 22);   // 4MB chunks (whole-file reads overflow
let rem = '';                        // Node's string cap on multi-GB inputs)
for (;;) {
  const n = fs.readSync(fd, buf, 0, buf.length, null);
  if (n === 0) break;
  const lines = (rem + buf.toString('utf8', 0, n)).split('\n');
  rem = lines.pop();
  for (const line of lines) processLine(line);
}
if (fd !== 0) fs.closeSync(fd);
if (rem) processLine(rem);

// Per-value-bucket cap: cap each value bin to M × (smallest non-empty bin's count).
let out = kept;
let capMsg = '';
if (valueCap > 0) {
  const bins = Array.from({ length: nBuckets }, () => []);
  for (const k of kept) {
    let b = Math.floor(k.w * nBuckets);
    if (b >= nBuckets) b = nBuckets - 1;
    if (b < 0) b = 0;
    bins[b].push(k);
  }
  let minCount = Infinity;
  for (const b of bins) if (b.length > 0 && b.length < minCount) minCount = b.length;
  if (!Number.isFinite(minCount)) minCount = 0;
  const cap = Math.round(valueCap * minCount);

  out = [];
  let capped = 0;
  for (const b of bins) {
    // Partial Fisher-Yates when over cap: select `cap` uniformly at random.
    // Element-wise pushes: spreading a multi-million-entry bin overflows the
    // call stack.
    const take = Math.min(b.length, cap);
    for (let i = 0; i < take; i++) {
      if (b.length > cap) {
        const j = i + Math.floor(rng.random() * (b.length - i));
        const t = b[i]; b[i] = b[j]; b[j] = t;
      }
      out.push(b[i]);
    }
    capped += b.length - take;
  }
  capMsg = `, value-cap ${valueCap}×min(${minCount})=${cap}/bin over ${nBuckets} bins (capped ${capped})`;
}

// Always shuffle the output: downstream training gets a random order, and the
// train_ppat head/tail test split stays representative without a separate step.
for (let i = out.length - 1; i > 0; i--) {
  const j = Math.floor(rng.random() * (i + 1));
  const t = out[i]; out[i] = out[j]; out[j] = t;
}

process.stderr.write(
  `filter-evals: ${total} read → ${out.length} kept  ` +
  `(no-extreme=${noExtreme} min-phase=${minPhase} max-phase=${maxPhase} value-cap=${valueCap})  ` +
  `dropped: extreme=${dropExtreme} phase=${dropPhase} skipped=${skipped}\n`);

process.stdout.write(
  `# filter-evals: no-extreme=${noExtreme} min-phase=${minPhase} max-phase=${maxPhase}${capMsg}; ` +
  `kept ${out.length} of ${total}\n`);
// Batched writes: one syscall per ~64K lines, not per line.
for (let i = 0; i < out.length; i += 65536) {
  let s = '';
  const end = Math.min(i + 65536, out.length);
  for (let j = i; j < end; j++) s += out[j].line + '\n';
  process.stdout.write(s);
}
