'use strict';

// opponents.js — show a player's most-played opponents on a toroidal CGOS
// ladder: games and win% from the player's perspective, most games first.
//
// Usage:
//   node cgos/opponents.js --p <player> [--top 10] [--size <n>] [--data <dir>]

const path = require('path');
const { DatabaseSync } = require('node:sqlite');
const Util = require('../util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['p', 'top', 'size', 'data']);
if (opts.help || !opts.p) {
  console.log('Usage: node cgos/opponents.js --p <player> [--top 10] [--size <n>] [--data <dir>]');
  process.exit(opts.help ? 0 : 1);
}
const player  = opts.p;
const top     = opts.getInt('top', 10);
const size    = opts.getInt('size', 13);
const dataDir = path.resolve(opts.get('data', path.join(__dirname, 'data', `${size}x${size}`)));
const db = new DatabaseSync(path.join(dataDir, 'cgos.state'), { readOnly: true });

const games = db.prepare('SELECT w, b, res FROM games WHERE w = ? OR b = ?').all(player, player);
db.close();
if (games.length === 0) {
  console.error(`no games for "${player}" in ${dataDir}`);
  process.exit(1);
}

// Wins from the player's perspective.  res is like "B+3.5", "W+Resign", "Draw".
const vs = new Map();   // opponent -> { games, wins }
for (const g of games) {
  const opp = g.w === player ? g.b : g.w;
  let rec = vs.get(opp);
  if (!rec) { rec = { games: 0, wins: 0 }; vs.set(opp, rec); }
  rec.games += 1;
  const c = (g.res || '')[0];
  if (c === 'D')                             rec.wins += 0.5;
  else if ((c === 'W') === (g.w === player)) rec.wins += 1;
}

const rows = [...vs.entries()]
  .sort((x, y) => y[1].games - x[1].games)
  .slice(0, top);

console.log(`player: ${player}  games: ${games.length}  opponents: ${vs.size}`);
console.log();
const W = Math.max(8, ...rows.map(([name]) => name.length));
console.log(`${'opponent'.padEnd(W)}  ${'games'.padStart(5)}  ${'win%'.padStart(5)}`);
for (const [name, r] of rows) {
  const pct = (100 * r.wins / r.games).toFixed(1);
  console.log(`${name.padEnd(W)}  ${String(r.games).padStart(5)}  ${pct.padStart(5)}`);
}
