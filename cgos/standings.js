'use strict';

// standings.js — print the rating table for a toroidal CGOS server.
//
// Usage:
//   node cgos/standings.js [--size <n>] [--data <dir>]
//
// Two ratings are shown for each engine:
//   cgosElo — the server's own incremental Elo (K decays as games accrue;
//             a trailing ? marks provisional ratings with K > 16).
//   mleElo  — anchored Bradley–Terry maximum-likelihood fit over ALL games
//             in the database (elo-lib.js).  This converges much faster and
//             is the number to trust for a quick estimate; ±95 shows the
//             half-width of a 95% confidence interval computed against the
//             other engines' fitted ratings held fixed.

const path = require('path');
const { DatabaseSync } = require('node:sqlite');
const Util = require('../util.js');
const { fitRatings, mleRating } = require('../elo-lib.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help']);
if (opts.help) {
  console.log('Usage: node cgos/standings.js [--size <n>] [--data <dir>]');
  process.exit(0);
}
const size    = opts.getInt('size', 9);
const dataDir = path.resolve(opts.get('data', path.join(__dirname, 'data', `${size}x${size}`)));
const db = new DatabaseSync(path.join(dataDir, 'cgos.state'), { readOnly: true });

const players = db.prepare('SELECT name, rating, K, games FROM password ORDER BY rating DESC').all();
const anchors = new Map(db.prepare('SELECT name, rating FROM anchors').all()
  .map(r => [r.name, r.rating]));
const games = db.prepare('SELECT w, b, res FROM games').all();
db.close();

// Aggregate pairwise results.  res is like "B+3.5", "W+Resign", "Draw".
const records = new Map();   // "w|b" -> { a: w, b: b, wa, wb }
for (const g of games) {
  const key = g.w + '|' + g.b;
  let rec = records.get(key);
  if (!rec) { rec = { a: g.w, b: g.b, wa: 0, wb: 0 }; records.set(key, rec); }
  const c = (g.res || '')[0];
  if (c === 'W')      rec.wa += 1;
  else if (c === 'B') rec.wb += 1;
  else { rec.wa += 0.5; rec.wb += 0.5; }   // Draw
}
const recList = [...records.values()];

const fitted = fitRatings(recList, { anchors });

// Per-player results vs fitted opponent ratings, for the 1-D CI.
const vs = new Map();        // name -> [{ r, w, l }]
for (const rec of recList) {
  if (fitted.has(rec.b)) {
    if (!vs.has(rec.a)) vs.set(rec.a, []);
    vs.get(rec.a).push({ r: fitted.get(rec.b), w: rec.wa, l: rec.wb });
  }
  if (fitted.has(rec.a)) {
    if (!vs.has(rec.b)) vs.set(rec.b, []);
    vs.get(rec.b).push({ r: fitted.get(rec.a), w: rec.wb, l: rec.wa });
  }
}

const rows = players.map(p => {
  const m = fitted.get(p.name);
  const ci = vs.has(p.name) ? mleRating(vs.get(p.name)) : null;
  let ciStr = '';
  if (ci) {
    if (Number.isFinite(ci.lo) && Number.isFinite(ci.hi))
      ciStr = `±${Math.round((ci.hi - ci.lo) / 2)}`;
    else if (Number.isFinite(ci.lo)) ciStr = `≥${Math.round(ci.lo)}`;   // won every game so far
    else if (Number.isFinite(ci.hi)) ciStr = `≤${Math.round(ci.hi)}`;   // lost every game so far
  }
  return {
    name: p.name,
    games: p.games,
    cgos: `${Math.round(p.rating)}${p.K > 16 ? '?' : ''}`,
    mle: m !== undefined && Number.isFinite(m) ? String(Math.round(m)) : '-',
    ci: ciStr,
    anchor: anchors.has(p.name) ? '⚓' : '',
  };
}).sort((x, y) => (y.mle === '-' ? -Infinity : +y.mle) - (x.mle === '-' ? -Infinity : +x.mle));

const W = Math.max(6, ...rows.map(r => r.name.length));
console.log(`${'engine'.padEnd(W)}  ${'games'.padStart(5)}  ${'cgosElo'.padStart(7)}  ${'mleElo'.padStart(6)}  ${'95%'.padStart(5)}`);
for (const r of rows) {
  console.log(`${r.name.padEnd(W)}  ${String(r.games).padStart(5)}  ${r.cgos.padStart(7)}  ${r.mle.padStart(6)}  ${r.ci.padStart(5)}  ${r.anchor}`);
}
console.log(`\n${games.length} games in database.  ⚓ = rating anchor.`);
