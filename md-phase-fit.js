'use strict';

// md-phase-fit.js — map per-phase movedetails MAE to CGOS Elo.
//
// Reads every finished md-phase/<agent>.txt (written by md-phase-sweep.sh),
// takes each agent's Elo and game count from cgos/standings.js, and fits
//
//   elo = a - sum_b w_b * x_b        with w_b = A * exp(k * phase_b), A >= 0
//
// where x_b is the band's mae, and the band weights lie on one log-linear
// curve over phase, w_b = A * exp(k * mid_b), so neighbouring bands cannot
// take unrelated weights: two parameters plus the intercept.  (An mse term was
// dropped: the md-trunc30k-827 fit gave it zero weight.)  Fit by a grid over k with a non-negative linear
// solve for A at each point.  Every agent evaluated the same positions, so
// every file must carry the same per-band counts (a mismatch is a mixed sweep
// and an error); every band with positions is used.  Prints the input table,
// each agent's fitted Elo and residual, and the curves as Elo lost per 0.01
// of the regressor in each band.
//
// --save PATH writes the fitted mapping as JSON for evalmovedetails --elo-map.
//
// The saved map records the MD file it was fitted on (path and an order-free
// fingerprint of its positions); evalmovedetails shows Elo only on that file,
// unfiltered.  The file comes from the outputs' file= header, or --md-file for
// outputs that predate it.
//
// Usage: node md-phase-fit.js [--dir md-phase] [--save PATH] [--md-file F]

const fs   = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['dir', 'save', 'md-file']);
if (opts.help) {
  console.log(`Usage: node md-phase-fit.js [--dir md-phase] [--save PATH] [--md-file F]

Fit elo = a - sum_b w_b * mae_b, with the band weights on one log-linear
curve w_b = A * exp(k * phase_b), over the agents whose
md-phase/<agent>.txt is complete, then print the mapping.  All files must
share one per-band position count (same positions for every agent).
  --dir    directory of md-phase-sweep.sh outputs (default md-phase)
  --save   write the fitted mapping as JSON, for evalmovedetails --elo-map;
           it records the MD file the outputs were measured on
  --md-file  that MD file, for outputs whose header has no file= (older runs)`);
  process.exit(0);
}
const dir  = opts.dir || 'md-phase';

// --- standings: name -> { elo, games }
const standings = new Map();
for (const line of execFileSync('node', [path.join(__dirname, 'cgos', 'standings.js')], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).split('\n')) {
  const m = line.match(/^(\S+)\s+(-?\d+)\??\s+(\d+)\s/);
  if (m) standings.set(m[1], { elo: +m[2], games: +m[3] });
}

// --- per-agent band table
const agents = [];
for (const f of fs.readdirSync(dir).sort()) {
  if (!f.endsWith('.txt')) continue;
  const text = fs.readFileSync(path.join(dir, f), 'utf8');
  if (!/^SUMMARY/m.test(text)) continue;
  const name = f.slice(0, -4);
  const st = standings.get(name);
  if (!st) { console.error(`${name}: not in standings, skipped`); continue; }
  const bands = [];
  // band rows: phase, n, mae (x10^4); older outputs also carry an mse column
  for (const m of text.matchAll(/^(\d\.\d\d)-(\d\.\d\d)\s+(\d+)\s+(\d+|-)(?:\s+(?:[\d.]+|-))?$/mg))
    bands.push({ lo: +m[1], hi: +m[2], n: +m[3], mae: m[4] === '-' ? NaN : +m[4] / 10000 });
  const mae = +text.match(/mae=([\d.]+)/)[1];
  const fm = text.match(/^agent=\S+\s+file=(\S+)/m);
  const pm = text.match(/positions=(\d+)\/(\d+)/);
  agents.push({ name, elo: st.elo, games: st.games, bands, mae,
                file: fm ? fm[1] : null, positions: pm ? +pm[1] : NaN });
}
if (agents.length === 0) { console.error(`no finished files in ${dir}`); process.exit(1); }

const nb = agents[0].bands.length;
const counts = agents[0].bands.map(b => b.n).join(',');
for (const a of agents) {
  const c = a.bands.map(b => b.n).join(',');
  if (c !== counts) { console.error(`${a.name}: band counts [${c}] differ from ${agents[0].name}'s [${counts}]; the files are not from one sweep`); process.exit(1); }
}
const use = [];
for (let b = 0; b < nb; b++) if (agents[0].bands[b].n > 0) use.push(b);

// --- model: elo = a - sum_kind A_kind * sum_b exp(k_kind * mid_b) * x_{b,kind}
// Each kind's band weights lie on one log-linear curve over the band midpoint
// phase, so neighbouring bands cannot get unrelated weights.  For fixed k the
// model is linear in (a, A): grid over k per kind, and at each grid point
// solve for the intercept and the non-negative amplitudes by coordinate
// descent; keep the grid point with the smallest rms residual.
const kinds = ['mae'];
const rows = agents.length;
const mids = use.map(b => (agents[0].bands[b].lo + agents[0].bands[b].hi) / 2);
const y = agents.map(a => a.elo);

function fitFor(ks) {   // ks: k per kind -> { a0, A, rms }
  const Z = agents.map(a => kinds.map((t, j) => use.reduce((z, b, i) => z + Math.exp(ks[j] * mids[i]) * a.bands[b][t], 0)));
  const A = kinds.map(() => 0);
  let a0 = 0;
  const resid = i => { let r = y[i] - a0; for (let j = 0; j < kinds.length; j++) r += A[j] * Z[i][j]; return r; };
  for (let it = 0; it < 200; it++) {
    let sum = 0; for (let i = 0; i < rows; i++) sum += y[i] + (resid(i) - y[i] + a0); a0 = sum / rows;   // mean(y + ZA)
    for (let j = 0; j < kinds.length; j++) {
      let num = 0, den = 0;
      for (let i = 0; i < rows; i++) { const r = resid(i) - A[j] * Z[i][j]; num -= Z[i][j] * r; den += Z[i][j] * Z[i][j]; }
      A[j] = den > 0 ? Math.max(0, num / den) : 0;
    }
  }
  let ss = 0; for (let i = 0; i < rows; i++) ss += resid(i) ** 2;
  return { a0, A, rms: Math.sqrt(ss / rows), resid };
}

const K_LO = -12, K_HI = 12, K_STEP = 0.25;
const grid = [];
for (let k = K_LO; k <= K_HI + 1e-9; k += K_STEP) grid.push(k);
let best = null, bestKs = null;
const walk = (ks) => {
  if (ks.length === kinds.length) { const f = fitFor(ks); if (!best || f.rms < best.rms) { best = f; bestKs = ks.slice(); } return; }
  for (const k of grid) walk([...ks, k]);
};
walk([]);
const { a0, A, rms } = best;
const residual = best.resid;
// weight of band i for kind j, in Elo per unit of the regressor
const weight = (j, i) => A[j] * Math.exp(bestKs[j] * mids[i]);

// --- display
const bandName = b => `${agents[0].bands[b].lo.toFixed(1)}-${agents[0].bands[b].hi.toFixed(1)}`;
const bandHdr = use.map(bandName);
console.log(`agents: ${rows}  bands: ${use.length} of ${nb} (${agents[0].bands[use[0]].n} positions each)  curves: ${kinds.join(', ')}  parameters: ${1 + 2 * kinds.length}  intercept: ${a0.toFixed(0)}  rms residual: ${rms.toFixed(0)} Elo`);
console.log('');
for (const kind of kinds) {
  const dp = 4;
  console.log(`${kind} per band:`);
  console.log(`${'agent'.padEnd(28)} ${'elo'.padStart(5)} ${'games'.padStart(5)}  ${bandHdr.map(h => h.padStart(7)).join(' ')}   ${'all'.padStart(7)} ${'fit'.padStart(5)} ${'resid'.padStart(5)}`);
  for (let i = 0; i < rows; i++) {
    const a = agents[i];
    const cells = use.map(b => a.bands[b][kind].toFixed(dp).padStart(7)).join(' ');
    console.log(`${a.name.padEnd(28)} ${String(a.elo).padStart(5)} ${String(a.games).padStart(5)}  ${cells}   ${a[kind].toFixed(dp).padStart(7)} ${(y[i] - residual(i)).toFixed(0).padStart(5)} ${residual(i).toFixed(0).padStart(5)}`);
  }
  console.log('');
}
// the curves: weight per band as Elo lost per 0.01 of the regressor
console.log(`Elo lost per 0.01 of band ${kinds.join(' / ')}  (weight = A * exp(k * phase)):`);
console.log(`  ${'band'.padEnd(7)} ${kinds.map(t => t.padStart(7)).join(' ')}`);
for (let i = 0; i < use.length; i++)
  console.log(`  ${bandName(use[i]).padEnd(7)} ${kinds.map((t, j) => (weight(j, i) * 0.01).toFixed(0).padStart(7)).join(' ')}`);
console.log(`  ${'k'.padEnd(7)} ${kinds.map((t, j) => bestKs[j].toFixed(2).padStart(7)).join(' ')}`);
for (let j = 0; j < kinds.length; j++)
  if (bestKs[j] <= K_LO || bestKs[j] >= K_HI) console.log(`  note: ${kinds[j]} k is at the grid bound [${K_LO}, ${K_HI}]`);

if (opts.save) {
  // The MD file: every output that names one must name the same; --md-file
  // supplies it for outputs that predate the header field.
  const named = [...new Set(agents.map(a => a.file).filter(Boolean))];
  if (named.length > 1) { console.error(`outputs name different MD files: ${named.join(', ')}; not one sweep`); process.exit(1); }
  const mdFile = opts['md-file'] || named[0];
  if (!mdFile) { console.error('--save: the outputs name no MD file (file= header); pass --md-file'); process.exit(1); }
  if (named.length && opts['md-file'] && path.resolve(named[0]) !== path.resolve(opts['md-file'])) {
    console.error(`--md-file ${opts['md-file']} differs from the outputs' ${named[0]}`); process.exit(1);
  }
  const { loadPositions, mdFingerprint } = require('./evalmovedetails.js');
  const nPos = loadPositions(mdFile).length;
  const bad = agents.filter(a => a.positions !== nPos);
  if (bad.length) { console.error(`${mdFile} has ${nPos} positions but ${bad.map(a => `${a.name} evaluated ${a.positions}`).join(', ')}`); process.exit(1); }
  const map = {
    fitted: new Date().toISOString(), agents: rows, rmsResidual: Math.round(rms),
    phaseBuckets: nb, bands: use.map(b => ({ lo: agents[0].bands[b].lo, hi: agents[0].bands[b].hi })),
    intercept: a0, curves: Object.fromEntries(kinds.map((t, j) => [t, { A: A[j], k: bestKs[j] }])),
    mdFile, mdPositions: nPos, mdFingerprint: mdFingerprint(mdFile),
  };
  fs.writeFileSync(opts.save, JSON.stringify(map, null, 1) + '\n');
  console.log(`saved: ${opts.save}`);
}
