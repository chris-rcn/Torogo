'use strict';

// join.js — connect one engine (usually the candidate you want rated) to a
// running toroidal CGOS server.
//
// Usage:
//   node cgos/join.js --p <policy> [options]
//
// Options:
//   --p       <policy>  AI policy: filename without .js inside ai/  (required)
//   --name    <user>    CGOS username        (default: <policy>[-<budget>])
//   --budget  <ms>      Per-move budget      (default 1)
//   --games   <n>       Stop after n more completed games (default: unlimited)
//   --house             Join as a HOUSE engine: idle until a trial engine
//                       needs an opponent (default: trial — the server
//                       keeps this engine playing continuously)
//   --size    <n>       Board size — selects the ladder to join (default 13)
//   --refs    <file>    Fleet file for host/password     (default cgos/refs.json)
//   --ini     <file>    Server config, read for the port (default cgos/torogo<size>.ini)
//   --data    <dir>     Data directory of the server     (default cgos/data/<size>x<size>)
//
// Runs in the foreground and prints the engine's rating after each game.
// The budget is part of the engine's identity: rate the same agent at two
// budgets by joining twice with two names (e.g. rave-100, rave-1000).

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const { DatabaseSync } = require('node:sqlite');
const Util = require('../util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help', 'house']);
if (opts.help || !opts.p) {
  console.log('Usage: node cgos/join.js --p <policy> [--name <user>] [--budget <ms>] [--games <n>] [--house]');
  process.exit(opts.help ? 0 : 1);
}

const agent    = opts.p;
const budget   = opts.getInt('budget', 1);
const name     = opts.get('name', budget > 1 ? `${agent}-${budget}` : agent);
const maxGames = opts.getInt('games', 0);
const size     = opts.getInt('size', 13);
const refsFile = path.resolve(opts.get('refs', path.join(__dirname, 'refs.json')));
const iniFile  = path.resolve(opts.get('ini',  path.join(__dirname, `torogo${size}.ini`)));
const dataDir  = path.resolve(opts.get('data', path.join(__dirname, 'data', `${size}x${size}`)));
const fleet    = JSON.parse(fs.readFileSync(refsFile, 'utf8'));

const portMatch = fs.readFileSync(iniFile, 'utf8').match(/^\s*portNumber\s*=\s*(\d+)/m);
const port = portMatch ? parseInt(portMatch[1], 10) : (fleet.port || 1919);

if (name.startsWith('admin')) {
  console.error('names starting with "admin" are reserved by the server');
  process.exit(1);
}

// Register the house/trial role.  The server re-reads the house table
// every scheduler tick, so this takes effect immediately.
{
  const db = new DatabaseSync(path.join(dataDir, 'cgos.state'));
  try {
    db.exec('CREATE TABLE IF NOT EXISTS house(name, primary key(name))');
    if (opts.house) db.prepare('INSERT OR REPLACE INTO house VALUES(?)').run(name);
    else            db.prepare('DELETE FROM house WHERE name = ?').run(name);
  } finally { db.close(); }
}

const killFile = path.join(dataDir, `kill-${name}.txt`);
try { fs.unlinkSync(killFile); } catch {}

const cfgPath = path.join(dataDir, 'clients', name + '.cfg');
fs.mkdirSync(path.dirname(cfgPath), { recursive: true });
fs.writeFileSync(cfgPath, `Common:
  KillFile = ${killFile}

GTPEngine:
  Name = ${name}
  CommandLine = node ${path.join(__dirname, 'gtp.js')} --p ${agent} --budget ${budget}
  ServerHost = ${fleet.host || '127.0.0.1'}
  ServerPort = ${port}
  ServerUser = ${name}
  ServerPassword = ${fleet.password || 'torogo'}
  NumberOfGames = 1
  LogFile = ${path.join(dataDir, 'logs', 'engine-' + name + '.log')}
`);

const client = spawn('python3',
  [path.join(__dirname, 'client', 'cgosclient.py'), cfgPath],
  { cwd: dataDir, stdio: ['ignore', 'inherit', 'inherit'] });

client.on('exit', code => process.exit(code ?? 0));
process.on('SIGINT', () => { client.kill('SIGTERM'); process.exit(0); });

// ── Progress: poll the server DB, report rating after each finished game ─────

const dbPath = path.join(dataDir, 'cgos.state');
let lastCount = -1;
let startCount = null;

function poll() {
  let db;
  try { db = new DatabaseSync(dbPath, { readOnly: true }); }
  catch { return; }
  try {
    const g = db.prepare(
      'SELECT count(*) AS n FROM games WHERE (w = ? OR b = ?) AND final = ?'
    ).get(name, name, 'y');
    const p = db.prepare(
      'SELECT rating, K, games FROM password WHERE name = ?'
    ).get(name);
    if (startCount === null && p !== undefined) startCount = g.n;
    if (p !== undefined && g.n !== lastCount) {
      lastCount = g.n;
      const uncertain = p.K > 16 ? '?' : '';
      console.log(`[${name}] games: ${g.n}  rating: ${Math.round(p.rating)}${uncertain} (K=${p.K.toFixed(1)})`);
      if (maxGames > 0 && g.n - startCount >= maxGames) {
        console.log(`[${name}] reached ${maxGames} games — stopping after current round`);
        fs.writeFileSync(killFile, '');
      }
    }
  } finally { db.close(); }
}

setInterval(poll, 5000).unref();
