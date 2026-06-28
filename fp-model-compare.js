'use strict';

// fp-model-compare.js — compare two featurepol model files (JS-saved vs C-saved).
// Verifies the C serializer produces a JS-loadable model with matching spec, scale,
// key set, and quantized weights (allowing ±1 qval from rounding-boundary FP noise).
//
//   node fp-model-compare.js model_a.js model_b.js

const path = require('path');
const A = require(path.resolve(process.argv[2]));
const B = require(path.resolve(process.argv[3]));

function map(m) { const o = new Map(); for (let i = 0; i < m.count; i++) o.set(m.keys[i] | 0, m.qvals[i]); return o; }
const ma = map(A), mb = map(B);

let keyMiss = 0, qOver1 = 0, maxQ = 0;
for (const [k, qa] of ma) {
  if (!mb.has(k)) { keyMiss++; continue; }
  const d = Math.abs(qa - mb.get(k));
  if (d > maxQ) maxQ = d;
  if (d > 1) qOver1++;
}
for (const [k] of mb) if (!ma.has(k)) keyMiss++;

console.log(`spec match:   ${A.spec === B.spec}`);
console.log(`count:        ${A.count} vs ${B.count}`);
console.log(`scale:        ${A.scale.toPrecision(8)} vs ${B.scale.toPrecision(8)}  (reldiff ${(Math.abs(A.scale - B.scale) / A.scale).toExponential(2)})`);
console.log(`key set:      ${keyMiss === 0 ? 'identical' : keyMiss + ' missing'}`);
console.log(`qval: max|Δ|=${maxQ}  count|Δ|>1: ${qOver1}`);
process.exit((A.spec === B.spec && keyMiss === 0 && qOver1 === 0) ? 0 : 1);
