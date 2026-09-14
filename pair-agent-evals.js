'use strict';

// pair-agent-evals.js — paired comparison of evalagentvalues --verbose logs.
//
//   node pair-agent-evals.js <control.log> <arm.log> [<arm.log> ...]
//
// Every log must come from the SAME position file in the same order (the
// sweep scripts snapshot the eval file, so this holds).  For each arm we
// compute, per position i, the difference of squared errors against the
// control, d(i) = errArm(i)^2 - errCtl(i)^2, and report mean(d) with its
// standard error and z.  Both arms score the identical position with the
// same label, so the per-position difficulty that dominates a pooled RMS
// cancels in d — the paired read resolves arm effects an order of magnitude
// below the pooled comparison, from the same runs.  mean(d) < 0 means the
// arm is MORE accurate than the control.  (Pairing fixes resolution, not
// label bias: d = (vA-vB)(vA+vB-2L), so the label still referees per-position
// disagreements.)

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

const [ctlPath, ...armPaths] = process.argv.slice(2);
if (!ctlPath || armPaths.length === 0) {
  console.error('usage: node pair-agent-evals.js <control.log> <arm.log> [...]');
  process.exit(1);
}
const ctl = loadErrs(ctlPath);
console.log(`control: ${ctlPath} (${ctl.length} positions)`);
console.log('arm                                      n     mean(d)        SE       z');
for (const p of armPaths) {
  const arm = loadErrs(p);
  const d = [];
  for (let i = 0; i < Math.min(ctl.length, arm.length); i++)
    if (ctl[i] !== undefined && arm[i] !== undefined)
      d.push(arm[i] * arm[i] - ctl[i] * ctl[i]);
  if (d.length === 0) { console.log(`${p}: no paired positions`); continue; }
  const n = d.length;
  const mean = d.reduce((a, x) => a + x, 0) / n;
  const varD = d.reduce((a, x) => a + (x - mean) * (x - mean), 0) / (n - 1);
  const se = Math.sqrt(varD / n);
  console.log(`${p.padEnd(38)} ${String(n).padStart(4)}  ${mean.toExponential(3).padStart(10)}  ${se.toExponential(2).padStart(8)}  ${(mean / se).toFixed(2).padStart(6)}`);
}
