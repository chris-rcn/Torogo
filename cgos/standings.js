'use strict';

// standings.js — print the rating table for a toroidal CGOS server.
//
// Usage:
//   node cgos/standings.js [--live] [--size <n>] [--data <dir>] [--refs <file>] [--ini <file>]
//
// --live shows only participating engines: house references and connected
// guests.  Retired engines (history in the database but no current role)
// are hidden.
//
// elo is the server's stored rating.  The server refits every rating from
// the full games table with the anchored Bradley–Terry MLE every
// mleInterval finalized games (see server/cgos/app/rating.py), applying
// incremental Elo updates in between, so the stored rating is never more
// than a few K-steps from the full-history fit.  A trailing ? marks
// provisional ratings (K > 16).
//
// 🏠 marks house engines (the reference fleet).  🔗 marks guest engines
// currently connected to the server (trials being
// rated, e.g. via join.js).  Guest connectivity is fetched live
// over the admin protocol ("who"); when the server is down the column is
// omitted.

const fs   = require('fs');
const net  = require('net');
const path = require('path');
const { DatabaseSync } = require('node:sqlite');
const Util = require('../util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help', 'live']);
if (opts.help) {
  console.log('Usage: node cgos/standings.js [--live] [--size <n>] [--data <dir>] [--refs <file>] [--ini <file>]');
  process.exit(0);
}
const size    = opts.getInt('size', 13);
const dataDir = path.resolve(opts.get('data', path.join(__dirname, 'data', `${size}x${size}`)));
const refsFile = path.resolve(opts.get('refs', path.join(__dirname, 'refs.json')));
const iniFile  = path.resolve(opts.get('ini', path.join(__dirname, `torogo${size}.ini`)));

// Ask the live server who is connected: log in as an admin ("admin*" names
// skip the engine path) and issue "who".  Resolves to a Set of names, or
// null if the server is unreachable (offline standings still work).
function queryConnected() {
  let host = '127.0.0.1', port = 1919, password = 'torogo';
  try {
    const fleet = JSON.parse(fs.readFileSync(refsFile, 'utf8'));
    if (fleet.host) host = fleet.host;
    if (fleet.port) port = fleet.port;
    if (fleet.password) password = fleet.password;
  } catch {}
  try {
    const m = fs.readFileSync(iniFile, 'utf8').match(/^\s*portNumber\s*=\s*(\d+)/m);
    if (m) port = parseInt(m[1], 10);
  } catch {}

  return new Promise(resolve => {
    const names = new Set();
    let buf = '', asked = false, quiet = null;
    const sock = net.connect({ host, port });
    const done = () => { sock.destroy(); resolve(asked ? names : null); };
    const fail = () => { sock.destroy(); resolve(asked ? names : null); };
    sock.setTimeout(1500, fail);
    sock.on('error', fail);
    sock.on('close', fail);   // server-side close (e.g. rejected login) must still resolve
    sock.on('data', chunk => {
      buf += chunk.toString();
      let i;
      while ((i = buf.indexOf('\n')) !== -1) {
        const line = buf.slice(0, i).trim();
        buf = buf.slice(i + 1);
        if (!asked) {
          if (line.startsWith('protocol')) sock.write('e1 standings\n');
          else if (line === 'username')    sock.write('admin\n');
          else if (line === 'password')    sock.write(password + '\n');
          else if (line === 'ok')          { asked = true; sock.write('who\n'); }
        } else {
          // "who" replies one line per connected player: name state gid rating k.
          const f = line.split(/\s+/);
          if (f.length === 5) names.add(f[0]);
        }
      }
      // The who reply has no terminator: finish after a quiet period.
      if (asked) { clearTimeout(quiet); quiet = setTimeout(done, 250); }
    });
  });
}

const db = new DatabaseSync(path.join(dataDir, 'cgos.state'), { readOnly: true });
db.exec('PRAGMA busy_timeout = 2000');   // wait out server writes instead of SQLITE_BUSY
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
// The seeded admin login row is credentials, not an engine.
players = players.filter(p => !p.name.startsWith('admin'));
const house = new Set(db.prepare('SELECT name FROM house').all().map(r => r.name));
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

queryConnected().then(connected => {
  // --live: only engines with a current role — house refs or connected guests.
  if (opts.live) {
    players = players.filter(p => house.has(p.name) || (connected && connected.has(p.name)));
  }
  const rows = players.map(p => ({
    name: p.name,
    games: gamesOf.get(p.name) || 0,
    elo: String(Math.round(p.rating)),
    prov: p.K > 16 ? '?' : ' ',   // provisional marker in its own column, digits stay aligned
    msMove: p.total_moves > 0 ? Util.fmt4(p.total_move_ms / p.total_moves) : '-',
    conn: house.has(p.name) ? '🏠' : (connected && connected.has(p.name) ? '🔗' : '  '),
    anchor: anchors.has(p.name) ? '⚓' : '',
  }));

  const W = Math.max(6, ...rows.map(r => r.name.length));
  console.log(`${'engine'.padEnd(W)}  ${'elo'.padStart(5)}   ${'games'.padStart(5)}  ${'ms/mv'.padStart(5)}`);
  for (const r of rows) {
    console.log(`${r.name.padEnd(W)}  ${r.elo.padStart(5)}${r.prov}  ${String(r.games).padStart(5)}  ${r.msMove.padStart(5)}  ${r.conn} ${r.anchor}`);
  }
  const legend = connected === null
    ? '⚓ = rating anchor.  (server offline — no connection info)'
    : '⚓ = rating anchor.  🏠 = house.  🔗 = connected guest.';
  console.log(`\n${games.length} games in database.  ${legend}`);
});
