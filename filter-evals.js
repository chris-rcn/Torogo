'use strict';

// filter-evals.js — standalone data shaping for ppat eval files.
//
// Reads an eval file (lines: "<size> <move1,move2,...> <winRatio> [best]") and
// writes a filtered/balanced, shuffled subset to stdout.  Keeps the main
// training code (train_ppat.c) simple by doing data shaping here.
//
// Filters, applied in order:
//   --no-extreme <f>     keep only winRatio in [f, 1-f] (drop value extremes);
//                        same margin as train_ppat's --filter.    (default 0 = off)
//   --min-phase <p>      drop positions with phase (= 1 - empty/area) < p;
//                        requires replaying each position.         (default 0 = off)
//   --max-phase <p>      drop positions with phase (= 1 - empty/area) > p;
//                        requires replaying each position.         (default 1 = off)
//   --value-cap <M>      balance the value distribution: bucket survivors into
//                        --value-buckets bins by winRatio, then cap each bin to
//                        M × (count of the smallest non-empty bin), random-
//                        subsampling any over-full bin.           (default 0 = off)
//   --value-buckets <N>  number of value bins for --value-cap.    (default 10)
//   --seed <n>           RNG seed (subsampling + shuffle).        (default 1)
//   --file <path>        input file.                              (default stdin)
//
// Output is always shuffled.
//
// Usage:
//   node filter-evals.js --file evals.txt --filter 0.01 --value-cap 3 --value-buckets 20 > out.txt
//   cat evals.txt | node filter-evals.js --value-cap 2 > out.txt

const fs = require('fs');
const { Game2, PASS } = require('./game2.js');
const { makeRng } = require('./xorshift.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'],
  ['file', 'no-extreme', 'min-phase', 'max-phase', 'value-cap', 'value-buckets', 'seed']);
if (opts.help) {
  process.stderr.write(
    'Usage: node filter-evals.js [--file F] [--no-extreme f] [--min-phase p] [--max-phase p] ' +
    '[--value-cap M] [--value-buckets N] [--seed n]  > out.txt\n');
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

const src   = fs.readFileSync(opts.file || 0, 'utf8');
const lines = src.split('\n');

let total = 0, dropExtreme = 0, dropPhase = 0, skipped = 0;
const kept = [];   // { line, w }

for (const line of lines) {
  if (!line || line[0] === '#') continue;
  const p = line.split(/\s+/);
  if (p.length < 3) { skipped++; continue; }
  const size = +p[0], moves = p[1], w = +p[2];
  if (!Number.isFinite(size) || !Number.isFinite(w) || !moves) { skipped++; continue; }
  total++;

  // Value extremeness (cheap: no replay).
  if (noExtreme > 0 && (w < noExtreme || w > 1 - noExtreme)) { dropExtreme++; continue; }

  // Phase filter (replay only when active).
  if (needPhase) {
    const g = new Game2(size);
    let ok = true;
    for (const t of moves.split(',')) { if (!g.play(parseMoveTok(t, size))) { ok = false; break; } }
    if (!ok) { skipped++; continue; }
    const phase = 1 - g.emptyCount / (size * size);
    if (phase < minPhase || phase > maxPhase) { dropPhase++; continue; }
  }

  kept.push({ line, w });
}

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
    if (b.length <= cap) { out.push(...b); continue; }
    // Partial Fisher-Yates: select `cap` uniformly at random, no full shuffle.
    for (let i = 0; i < cap; i++) {
      const j = i + Math.floor(rng.random() * (b.length - i));
      const t = b[i]; b[i] = b[j]; b[j] = t;
    }
    out.push(...b.slice(0, cap));
    capped += b.length - cap;
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
for (const k of out) process.stdout.write(k.line + '\n');
