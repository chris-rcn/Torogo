'use strict';

// featurepol-subtract.js — remove orphaned weights from a featurepol model.
//
// A --load retrain that drops feature spaces imports the whole saved weight
// table, so the dropped spaces' weights ride along unused ("orphans").  A
// key's owning space is not recoverable from the key (space tags are mixed
// one-way), but a space's hashes depend only on its own spec substring — so
// a standalone model whose spec is JUST the removed components, with its
// keys interned by extraction over a corpus, enumerates the orphan keys
// directly.  This tool subtracts that model's key set from the golden model.
//
// Only keys present in the orphans model are ever deleted, so the golden
// model's rare-but-live keys cannot be harmed.  Residual risks (both
// measured tiny): orphan keys the enumeration pass never emitted survive as
// dead bytes, and a cross-space hash collision (~2^-32 per pair) could
// delete one live key.
//
// Usage:
//   node featurepol-subtract.js --golden <model.js> --orphans <model.js> --save <out.js>

const FeaturePol = require('./featurepol-lib.js');
const Util = require('./util.js');
const fs = require('fs');

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['golden', 'orphans', 'save']);
if (opts.help || !opts.golden || !opts.orphans || !opts.save) {
  console.log(`Usage: node featurepol-subtract.js --golden <model.js> --orphans <model.js> --save <out.js>

Removes from the golden model every weight whose key appears in the orphans
model, and writes the result (spec/ema/totalUpdates/komi carried over from
the golden model).  The orphans model is typically a fresh model whose spec
is exactly the feature spaces REMOVED from the golden model's lineage, with
keys interned by an extraction pass over a large corpus — its weights are
ignored, only its key set matters.

  --golden PATH   model to clean (unchanged on disk)
  --orphans PATH  model enumerating the keys to delete
  --save PATH     cleaned model output`);
  process.exit(0);
}

const golden  = FeaturePol.loadModel({ name: 'golden',  path: opts.golden });
const orphans = FeaturePol.loadModel({ name: 'orphans', path: opts.orphans });

const before = golden.weights.map.size;
let removed = 0;
for (const key of orphans.weights.map.keys()) {
  if (golden.weights.map.delete(key)) removed++;
}

console.log(`golden: ${before} weights (${golden.spec.str})`);
console.log(`orphans: ${orphans.weights.map.size} keys (${orphans.spec.str})`);
console.log(`removed: ${removed}  kept: ${golden.weights.map.size}`);

fs.writeFileSync(opts.save, FeaturePol.serialize(golden.weights, {
  spec: golden.spec.str,
  ema: golden.ema,
  totalUpdates: golden.totalUpdates,
  komi: golden.komi === null ? undefined : golden.komi,
}));
console.log(`saved: ${opts.save}`);
