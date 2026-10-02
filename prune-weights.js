'use strict';

// prune-weights.js — keep only the most extreme |weight| fraction of a
// featurepol or vpatterns checkpoint (the filter-hpat-extreme.js operation,
// generalised to the other two families).  Output is the family's normal save
// format, so it loads anywhere the original did (spec/komi/ema metadata carried
// over).
//
// Usage: node prune-weights.js --in <model.js> --out <model.js> --keep F

const path = require('path');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['in', 'out', 'keep']);
if (opts.help || !opts.in) {
  console.error(`Usage: node prune-weights.js --in <model.js> [--out <model.js>] [--keep F]

Keep the most extreme |weight| fraction of a featurepol or vpatterns checkpoint
(family auto-detected).  For hpatterns files use filter-hpat-extreme.js.

  --in PATH       checkpoint to prune (required)
  --out PATH      pruned checkpoint to write (default: <in>-pruned.js,
                  echoed on start)
  --keep F        keep the most extreme |weight| fraction F, 0 < F < 1 (default 0.5)
  --help          show this message`);
  process.exit(opts.help ? 0 : 1);
}
const IN = path.resolve(opts.in);
const OUT = opts.out || opts.in.replace(/\.js$/, '') + '-pruned.js';
const KEEP = opts.keep !== undefined ? parseFloat(opts.keep) : 0.5;
if (!(KEEP > 0 && KEEP < 1)) { console.error('--keep must be in (0, 1)'); process.exit(1); }

const raw = require(IN);
const isVpat = raw && raw.specs !== undefined && (raw.weightsQ6 !== undefined || raw.weights instanceof Map);
const isFp   = raw && typeof raw.spec === 'string' && raw.keys !== undefined;
if (!isVpat && !isFp) {
  console.error(`unrecognised checkpoint format (expected a vpatterns or featurepol save; hpatterns -> filter-hpat-extreme.js)`);
  process.exit(1);
}

// Resolve the keep-fraction rule to a concrete magnitude floor.
function floorForKeep(mags) {
  mags.sort();   // Float64Array ascending
  const dropCount = Math.floor(mags.length * (1 - KEEP));
  return dropCount === 0 ? 0 : mags[dropCount - 1];   // drop everything <= this
}

if (isVpat) {
  const VPat = require('./vpatterns.js');
  const model = VPat.loadWeights(IN);
  const total = model.weights.size;
  const mags = new Float64Array(total);
  let i = 0;
  model.weights.forEach((k, v) => { mags[i++] = Math.abs(v); });
  const cut = floorForKeep(mags);
  const pruned = VPat.makeWeights();
  let kept = 0, dropped = 0, maxDropped = 0;
  model.weights.forEach((k, v) => {
    const a = Math.abs(v);
    if (a <= cut) { dropped++; if (a > maxDropped) maxDropped = a; return; }
    pruned.set(k, v);
    kept++;
  });
  VPat.saveWeights(OUT, { specs: model.specs, weights: pruned, komi: model.komi });
  console.log(`vpatterns: ${total} weights -> kept ${kept}, dropped ${dropped} (max dropped |w| ${maxDropped.toFixed(6)})`);
} else {
  const FP = require('./featurepol-lib.js');
  const { weights, ema, totalUpdates, komi } = FP.loadModel({ name: 'prune', path: IN });
  const total = weights.map.size;
  const mags = new Float64Array(total);
  let i = 0;
  weights.map.forEach((k, d) => { mags[i++] = Math.abs(weights.vals[d]); });
  const cut = floorForKeep(mags);
  const out = FP.createWeights({ spec: weights.spec, initialCapacity: Math.max(1024, total), ladderMinChain: weights.ladderMinChain, t3MinChain: weights.t3MinChain,
    t3DepthLimit: weights.t3DepthLimit, t3NodeLimit: weights.t3NodeLimit });
  let kept = 0, dropped = 0, maxDropped = 0;
  weights.map.forEach((key, d) => {
    const a = Math.abs(weights.vals[d]);
    if (a <= cut) { dropped++; if (a > maxDropped) maxDropped = a; return; }
    out.vals[FP.internKey(out, key)] = weights.vals[d];
    kept++;
  });
  out.vpatModel = weights.vpatModel;   // carry the embedded vpat<n> model through the prune
  require('fs').writeFileSync(OUT, FP.serialize(out, { spec: weights.spec.str, ema, totalUpdates, komi }));
  console.log(`featurepol: ${total} weights -> kept ${kept}, dropped ${dropped} (max dropped |w| ${maxDropped.toFixed(6)})`);
}
console.log('out: ' + OUT);
