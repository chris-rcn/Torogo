'use strict';

// pair-agent-evals.js — paired comparison of evalagentvalues --verbose logs.
//
//   node pair-agent-evals.js <arm.log> <arm.log> [<arm.log> ...]
//
// Every log must come from the SAME position file in the same order (the
// sweep scripts snapshot the eval file, so this holds).  For each arm we
// compute, per position i, the difference of its squared error against the
// LEAVE-ONE-OUT mean of the other arms' squared errors, and report mean(d)
// with its standard error and z.  All arms score the identical position with
// the same label, so the per-position difficulty that dominates a pooled RMS
// cancels in the pair, and the leave-one-out reference keeps any single
// run's private noise out of the comparison.  mean(d) < 0 means the arm is
// MORE accurate than the rest of the sweep.  (Pairing fixes resolution, not
// label bias — the label still referees per-position disagreements.)

const fs = require('fs');

function loadErrs(path) {
  // verbose rows: idx hist phase pred target err  (err signed, e.g. +0.0947)
  const errs = [];
  for (const line of fs.readFileSync(path, 'utf8').split('\n')) {
    const m = /^\s*(\d+)\s+\d+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([+-][\d.]+)\s*$/.exec(line);
    if (m) errs[parseInt(m[1], 10)] = parseFloat(m[2]);
  }
  return errs;
}

function stats(d) {
  const n = d.length;
  const mean = d.reduce((a, x) => a + x, 0) / n;
  const varD = d.reduce((a, x) => a + (x - mean) * (x - mean), 0) / (n - 1);
  return { n, mean, se: Math.sqrt(varD / n) };
}
function row(label, st) {
  console.log(`${label.padEnd(38)} ${String(st.n).padStart(4)}  ${st.mean.toExponential(3).padStart(10)}  ${st.se.toExponential(2).padStart(8)}  ${(st.mean / st.se).toFixed(2).padStart(6)}`);
}

const paths = process.argv.slice(2);
if (paths.length < 2) {
  console.error('usage: node pair-agent-evals.js <arm.log> <arm.log> [...]');
  process.exit(1);
}
const errs = paths.map(loadErrs);

// Positions every log has — the common paired set.
const nPos = Math.min(...errs.map(e => e.length));
const common = [];
for (let i = 0; i < nPos; i++) if (errs.every(e => e[i] !== undefined)) common.push(i);
const sq = errs.map(e => common.map(i => e[i] * e[i]));
const K = paths.length;

// Each log vs the leave-one-out mean of the others: the reference carries
// no single run's private noise.  mean(d) < 0 = more accurate than the rest.
console.log(`${common.length} paired positions, ${K} logs`);
console.log('vs LEAVE-ONE-OUT MEAN                    n     mean(d)        SE       z');
const tot = common.map((_, i) => sq.reduce((a, s2) => a + s2[i], 0));
for (let k = 0; k < K; k++)
  row(paths[k], stats(common.map((_, i) => sq[k][i] - (tot[i] - sq[k][i]) / (K - 1))));
