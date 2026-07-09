'use strict';

// purge.js — remove an engine and every trace of its games from a ladder.
//
// Usage:
//   node cgos/purge.js --name <engine> [--size <n>] [--data <dir>] [--dry-run]
//
// Deletes the engine's account, all games it played (results table, game
// archive, and per-game SGF/bin records), and its house/anchor entries,
// then refits the remaining players' ratings with the anchored
// Bradley-Terry MLE so the stored ratings no longer reflect the purged
// games.
//
// Run this while the server is stopped: a live server holds stale
// ratings in memory and may be writing the database concurrently.

const fs = require('fs');
const path = require('path');
const { DatabaseSync } = require('node:sqlite');
const Util = require('../util.js');
const { fitRatings } = require('../elo-lib.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help', 'dry-run']);
if (opts.help || !opts.name) {
  console.log('Usage: node cgos/purge.js --name <engine> [--size <n>] [--data <dir>] [--dry-run]');
  process.exit(opts.help ? 0 : 1);
}

const name    = opts.name;
const size    = opts.getInt('size', 13);
const dataDir = path.resolve(opts.get('data', path.join(__dirname, 'data', `${size}x${size}`)));
const dryRun  = !!opts['dry-run'];

const db = new DatabaseSync(path.join(dataDir, 'cgos.state'));

const exists = db.prepare('SELECT 1 FROM password WHERE name = ?').get(name)
  || db.prepare('SELECT 1 FROM games WHERE w = ? OR b = ? LIMIT 1').get(name, name);
if (!exists) {
  console.error(`no engine "${name}" in ${dataDir}`);
  db.close();
  process.exit(1);
}

const gids = db.prepare('SELECT gid FROM games WHERE w = ? OR b = ?')
  .all(name, name).map(r => r.gid);
const isAnchor = !!db.prepare('SELECT 1 FROM anchors WHERE name = ?').get(name);

console.log(`${dryRun ? '[dry-run] would purge' : 'purging'} "${name}" from ${dataDir}:`);
console.log(`  ${gids.length} games, account row${isAnchor ? ', ANCHOR entry (the scale loses this pin!)' : ''}`);

if (dryRun) { db.close(); process.exit(0); }

// ── Main database ─────────────────────────────────────────────────────────────

db.exec('BEGIN');
db.prepare('DELETE FROM games WHERE w = ? OR b = ?').run(name, name);
db.prepare('DELETE FROM password WHERE name = ?').run(name);
db.prepare('DELETE FROM house WHERE name = ?').run(name);
db.prepare('DELETE FROM anchors WHERE name = ?').run(name);
db.exec('COMMIT');

// ── Game archive ──────────────────────────────────────────────────────────────

try {
  const arc = new DatabaseSync(path.join(dataDir, 'archive.db'));
  const del = arc.prepare('DELETE FROM games WHERE gid = ?');
  for (const gid of gids) del.run(gid);
  arc.close();
} catch { /* no archive configured */ }

// ── SGF / bin records (server names them <gid>.sgf under html/SGF/Y/M/D) ────

const wanted = new Set();
for (const gid of gids) {
  wanted.add(`${gid}.sgf`); wanted.add(`${gid}.sgf.gz`); wanted.add(`${gid}.bin`);
}
let sgfCount = 0;
(function walk(dir) {
  let entries;
  try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return; }
  for (const e of entries) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) walk(p);
    else if (wanted.has(e.name)) { fs.unlinkSync(p); sgfCount++; }
  }
})(path.join(dataDir, 'html'));

// ── Refit remaining ratings without the purged games ─────────────────────────

const anchors = new Map(db.prepare('SELECT name, rating FROM anchors').all()
  .map(r => [r.name, r.rating]));
const records = [];
for (const g of db.prepare('SELECT w, b, res FROM games').all()) {
  const c = (g.res || '?')[0];
  if (c === 'W')      records.push({ a: g.w, b: g.b, wa: 1, wb: 0 });
  else if (c === 'B') records.push({ a: g.w, b: g.b, wa: 0, wb: 1 });
  else if (c === 'D') records.push({ a: g.w, b: g.b, wa: 0.5, wb: 0.5 });
}
let refitted = 0;
if (records.length) {
  const fit = fitRatings(records, { anchors });
  const upd = db.prepare('UPDATE password SET rating = ? WHERE name = ?');
  for (const [n, r] of fit) {
    if (Number.isFinite(r)) { upd.run(anchors.has(n) ? anchors.get(n) : r, n); refitted++; }
  }
}
db.close();

console.log(`done: ${gids.length} games, ${sgfCount} game records deleted; ` +
  `${refitted} remaining players refit over ${records.length} games`);
