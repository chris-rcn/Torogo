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
//
//   --in         input movedetails file (required)
//   --min-phase  keep samples with phase >= F   (default 0)
//   --max-phase  keep samples with phase <= F   (default 1)

const fs = require('fs');
const { Game2, parseMove } = require('./game2.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help']);
if (opts.help || !opts.in) {
  console.error('Usage: node filter-movedetails.js --in <file> [--min-phase F] [--max-phase F] > out.ndjson');
  process.exit(opts.help ? 0 : 1);
}

const minPhase = opts['min-phase'] !== undefined ? parseFloat(opts['min-phase']) : 0;
const maxPhase = opts['max-phase'] !== undefined ? parseFloat(opts['max-phase']) : 1;
if (isNaN(minPhase) || isNaN(maxPhase)) { console.error('--min-phase/--max-phase must be numbers'); process.exit(1); }
if (minPhase > maxPhase)                { console.error('--min-phase must be <= --max-phase'); process.exit(1); }

// Phase of a sample: replay its history, then board fullness = 1 − empty/area.
function phaseOf(sample) {
  const { boardSize, history } = sample;
  const game = new Game2(boardSize);
  for (const h of history) game.play(parseMove(h, boardSize));
  return 1 - game.emptyCount / (boardSize * boardSize);
}

const lines = fs.readFileSync(opts.in, 'utf8').split('\n');
const out = [];
let total = 0, kept = 0;

for (const line of lines) {
  if (!line.trim())        continue;                 // drop blank lines
  if (line.startsWith('#')) { out.push(line); continue; }   // pass through headers
  total++;
  const phase = phaseOf(JSON.parse(line));
  if (phase < minPhase || phase > maxPhase) continue;
  kept++;
  out.push(line);          // emit the original line text unchanged
}

out.push(`# filter-movedetails: in=${opts.in} min-phase=${minPhase} max-phase=${maxPhase}  kept ${kept}/${total}`);
process.stdout.write(out.join('\n') + '\n');
console.error(`kept ${kept}/${total} samples  (phase in [${minPhase}, ${maxPhase}])`);
