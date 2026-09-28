'use strict';

// md-phase-fit.js — map per-phase movedetails MAE to CGOS Elo.
//
// Reads every finished md-phase/<agent>.txt (written by md-phase-sweep.sh),
// takes each agent's Elo and game count from cgos/standings.js, and fits
//
//   elo = a - sum_b w_b * mae_b        with every w_b >= 0
//
// by non-negative least squares (coordinate descent).  Bands are used only
// if every agent has at least --min-n positions in them.  Prints the input
// table, the fitted weights (Elo lost per 0.01 of band MAE) and each
// agent's fitted Elo and residual.
//
// Usage: node md-phase-fit.js [--dir md-phase] [--min-n 20]

const fs   = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['dir', 'min-n']);
if (opts.help) {
  console.log(`Usage: node md-phase-fit.js [--dir md-phase] [--min-n 20]

Fit elo = a - sum_b w_b * mae_b (w_b >= 0) over the agents whose
md-phase/<agent>.txt is complete, then print the mapping.
  --dir    directory of md-phase-sweep.sh outputs (default md-phase)
  --min-n  a band is used only if every agent has this many positions in it (default 20)`);
  process.exit(0);
}
const dir  = opts.dir || 'md-phase';
const minN = parseInt(opts['min-n'] || '20', 10);

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
  for (const m of text.matchAll(/^(\d\.\d\d)-(\d\.\d\d)\s+(\d+)\s+(\d+|-)$/mg))
    bands.push({ lo: +m[1], hi: +m[2], n: +m[3], mae: m[4] === '-' ? NaN : +m[4] / 10000 });
  const mae = +text.match(/mae=([\d.]+)/)[1];
  agents.push({ name, elo: st.elo, games: st.games, bands, mae });
}
if (agents.length === 0) { console.error(`no finished files in ${dir}`); process.exit(1); }

const nb = agents[0].bands.length;
const use = [];
for (let b = 0; b < nb; b++) if (agents.every(a => a.bands[b].n >= minN)) use.push(b);

// --- NNLS by coordinate descent on w (>= 0) with a free intercept a
const rows = agents.length, k = use.length;
const X = agents.map(a => use.map(b => a.bands[b].mae));
const y = agents.map(a => a.elo);
const w = new Array(k).fill(0);
let a0 = 0;
function residual(i) { let r = y[i] - a0; for (let j = 0; j < k; j++) r += w[j] * X[i][j]; return r; }
for (let it = 0; it < 20000; it++) {
  let s = 0; for (let i = 0; i < rows; i++) s += residual(i) + a0; a0 = s / rows;   // intercept = mean(y + Xw)
  for (let j = 0; j < k; j++) {
    // minimise sum_i (y_i - a0 + sum_l w_l x_il)^2 over w_j alone, clamped at 0
    let num = 0, den = 0;
    for (let i = 0; i < rows; i++) {
      const r = residual(i) - w[j] * X[i][j];   // residual with w_j removed
      num -= X[i][j] * r; den += X[i][j] * X[i][j];
    }
    w[j] = den > 0 ? Math.max(0, num / den) : 0;
  }
}
let ss = 0; for (let i = 0; i < rows; i++) ss += residual(i) ** 2;
const rms = Math.sqrt(ss / rows);

// --- display
const bandHdr = use.map(b => `${agents[0].bands[b].lo.toFixed(1)}-${agents[0].bands[b].hi.toFixed(1)}`);
console.log(`agents: ${rows}  bands used: ${k} of ${nb} (min-n ${minN})  intercept: ${a0.toFixed(0)}  rms residual: ${rms.toFixed(0)} Elo`);
if (rows <= k + 1) console.log(`note: ${rows} agents for ${k + 1} parameters; the fit is underdetermined`);
console.log('');
console.log(`${'agent'.padEnd(28)} ${'elo'.padStart(5)} ${'games'.padStart(5)}  ${bandHdr.map(h => h.padStart(7)).join(' ')}   ${'mae'.padStart(6)} ${'fit'.padStart(5)} ${'resid'.padStart(5)}`);
for (let i = 0; i < rows; i++) {
  const a = agents[i];
  const cells = use.map(b => a.bands[b].mae.toFixed(4).padStart(7)).join(' ');
  console.log(`${a.name.padEnd(28)} ${String(a.elo).padStart(5)} ${String(a.games).padStart(5)}  ${cells}   ${a.mae.toFixed(4).padStart(6)} ${(y[i] - residual(i)).toFixed(0).padStart(5)} ${residual(i).toFixed(0).padStart(5)}`);
}
console.log('');
console.log(`Elo lost per 0.01 of band MAE:`);
for (let j = 0; j < k; j++) console.log(`  ${bandHdr[j]}: ${(w[j] * 0.01).toFixed(1)}`);
