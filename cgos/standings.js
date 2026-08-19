'use strict';

// standings.js — print the rating table for a toroidal CGOS server.
//
// Usage:
//   node cgos/standings.js [--size <n>] [--data <dir>]
//
// elo is the server's stored rating.  The server refits every rating from
// the full games table with the anchored Bradley–Terry MLE every
// mleInterval finalized games (see server/cgos/app/rating.py), applying
// incremental Elo updates in between, so the stored rating is never more
// than a few K-steps from the full-history fit.  A trailing ? marks
// provisional ratings (K > 16).

const path = require('path');
const { DatabaseSync } = require('node:sqlite');
const Util = require('../util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help']);
if (opts.help) {
  console.log('Usage: node cgos/standings.js [--size <n>] [--data <dir>]');
  process.exit(0);
}
const size    = opts.getInt('size', 13);
const dataDir = path.resolve(opts.get('data', path.join(__dirname, 'data', `${size}x${size}`)));
const db = new DatabaseSync(path.join(dataDir, 'cgos.state'), { readOnly: true });

let players;
try {
  players = db.prepare(
    'SELECT name, rating, K, games, total_move_ms, total_moves FROM password ORDER BY rating DESC'
  ).all();
} catch {
  // database predates the move-time columns (migrated on next server start)
  players = db.prepare('SELECT name, rating, K, games FROM password ORDER BY rating DESC')
    .all().map(p => ({ ...p, total_move_ms: 0, total_moves: 0 }));
}
const anchors = new Map(db.prepare('SELECT name, rating FROM anchors').all()
  .map(r => [r.name, r.rating]));
const games = db.prepare('SELECT w, b, res FROM games').all();
db.close();

// Count games from the games table itself — the password.games counter
// is cumulative and survives game deletions.
const gamesOf = new Map();
for (const g of games) {
  gamesOf.set(g.w, (gamesOf.get(g.w) || 0) + 1);
  gamesOf.set(g.b, (gamesOf.get(g.b) || 0) + 1);
}

const rows = players.map(p => ({
  name: p.name,
  games: gamesOf.get(p.name) || 0,
  elo: `${Math.round(p.rating)}${p.K > 16 ? '?' : ''}`,
  msMove: p.total_moves > 0 ? Util.fmt4(p.total_move_ms / p.total_moves) : '-',
  anchor: anchors.has(p.name) ? '⚓' : '',
}));

const W = Math.max(6, ...rows.map(r => r.name.length));
console.log(`${'engine'.padEnd(W)}  ${'games'.padStart(5)}  ${'elo'.padStart(5)}  ${'ms/mv'.padStart(5)}`);
for (const r of rows) {
  console.log(`${r.name.padEnd(W)}  ${String(r.games).padStart(5)}  ${r.elo.padStart(5)}  ${r.msMove.padStart(5)}  ${r.anchor}`);
}
console.log(`\n${games.length} games in database.  ⚓ = rating anchor.`);
