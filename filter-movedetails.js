#!/usr/bin/env node
'use strict';

// filter-movedetails.js — filter a movedetails (ndjson) file by sample.
//
// Reads --in, applies filters to each position sample, and writes the surviving
// samples to stdout (original line text, no reserialization).  '#' header/comment
// lines are passed through unchanged, and a provenance comment recording the
// filter is appended.  Filter stats go to stderr so stdout stays pure data:
//   node filter-movedetails.js --in f.ndjson --min-phase 0.3 > out.ndjson
//
// Phase = board fullness (1 − emptyCount/area) ∈ [0,1] — the codebase's
// canonical game phase — computed by replaying each sample's history (captures
// make this differ from move count).
//
// Usage:
//   node filter-movedetails.js --in <file> [--min-phase F] [--max-phase F]
//                              [--phase-cap K] [--phase-buckets N]
//
//   --in             input movedetails file (required)
//   --min-phase      keep samples with phase >= F   (default 0)
//   --max-phase      keep samples with phase <= F   (default 1)
//   --phase-cap      then keep at most K samples per phase band, chosen evenly
//                    spaced through the band's samples in file order so they
//                    span the games rather than the first few (default 0 = off)
//   --phase-buckets  number of equal-width phase bands for --phase-cap (default 10)

const fs = require('fs');
const Util = require('./util.js');
const MD = require('./movedetails-format.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['in', 'max-phase', 'min-phase', 'phase-cap', 'phase-buckets']);
if (opts.help || !opts.in) {
  console.error(`Usage: node filter-movedetails.js --in <file> [options] > out.ndjson

Filter a movedetails (ndjson) file by sample: surviving lines go to stdout
verbatim (no reserialization), '#' header lines pass through, and a
provenance comment recording the filter is appended.  Stats go to stderr so
stdout stays pure data.  Phase = board fullness (1 - emptyCount/area),
computed by replaying each sample's history.

  --in PATH        input movedetails file (required)
  --min-phase F    keep samples with phase >= F (default 0)
  --max-phase F    keep samples with phase <= F (default 1)
  --phase-cap K    then keep at most K samples per phase band, evenly spaced
                   through the band's samples in file order (default 0 = off)
  --phase-buckets N  equal-width phase bands for --phase-cap (default 10)
  --help           show this message`);
  process.exit(opts.help ? 0 : 1);
}

const minPhase = opts['min-phase'] !== undefined ? parseFloat(opts['min-phase']) : 0;
const maxPhase = opts['max-phase'] !== undefined ? parseFloat(opts['max-phase']) : 1;
if (isNaN(minPhase) || isNaN(maxPhase)) { console.error('--min-phase/--max-phase must be numbers'); process.exit(1); }
if (minPhase > maxPhase)                { console.error('--min-phase must be <= --max-phase'); process.exit(1); }
const phaseCap     = opts['phase-cap']     !== undefined ? parseInt(opts['phase-cap'], 10)     : 0;
const phaseBuckets = opts['phase-buckets'] !== undefined ? parseInt(opts['phase-buckets'], 10) : 10;
if (isNaN(phaseCap) || phaseCap < 0)       { console.error('--phase-cap must be a non-negative integer'); process.exit(1); }
if (isNaN(phaseBuckets) || phaseBuckets < 1) { console.error('--phase-buckets must be a positive integer'); process.exit(1); }

const lines = fs.readFileSync(opts.in, 'utf8').split('\n');
const out = [];
const rows = [];   // { line, phase } surviving the phase filter, in file order
let total = 0;

for (const line of lines) {
  if (!line.trim())        continue;                 // drop blank lines
  if (line.startsWith('#')) { out.push(line); continue; }   // pass through headers
  total++;
  const { phase } = MD.parseRow(line);   // stored in the file, no replay
  if (phase < minPhase || phase > maxPhase) continue;
  rows.push({ line, phase });
}

// --phase-cap: bucket the survivors by phase and keep K per band, evenly
// spaced through the band in file order (the file is game-ordered, so the
// first K would come from a few games); output stays in file order.
let capMsg = '';
if (phaseCap > 0) {
  const bands = Array.from({ length: phaseBuckets }, () => []);
  for (const r of rows) {
    let b = Math.floor(r.phase * phaseBuckets);
    if (b >= phaseBuckets) b = phaseBuckets - 1;   // phase === 1 lands in the last band
    bands[b].push(r);
  }
  const chosen = new Set();
  const counts = [];
  for (const band of bands) {
    const take = Math.min(band.length, phaseCap);
    for (let i = 0; i < take; i++) chosen.add(band[Math.floor((i + 0.5) * band.length / take)]);
    counts.push(take);
  }
  rows.splice(0, rows.length, ...rows.filter(r => chosen.has(r)));
  capMsg = ` phase-cap=${phaseCap} phase-buckets=${phaseBuckets} per-band=[${counts.join(',')}]`;
}
for (const r of rows) out.push(r.line);   // emit the original line text unchanged
const kept = rows.length;

out.push(`# filter-movedetails: in=${opts.in} min-phase=${minPhase} max-phase=${maxPhase}${capMsg}  kept ${kept}/${total}`);
process.stdout.write(out.join('\n') + '\n');
console.error(`kept ${kept}/${total} samples  (phase in [${minPhase}, ${maxPhase}]${capMsg})`);
