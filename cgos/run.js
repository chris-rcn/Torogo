'use strict';

// run.js — launch the toroidal CGOS server plus one client per reference
// engine, all as child processes.  Games run concurrently (the server pairs
// every waiting player each round), so a fleet of N engines plays N/2 games
// at a time.
//
// Usage:
//   node cgos/run.js [options]
//
// Options:
//   --size <n>      Board size — selects config and data dir     (default 9)
//   --refs <file>   Reference fleet definition   (default cgos/refs.json)
//   --ini  <file>   Server config                (default cgos/torogo<size>.ini)
//   --data <dir>    Runtime data directory       (default cgos/data/<size>x<size>)
//   --server-only   Start only the server (attach engines with cgos/join.js)
//
// Each size is its own ladder: its own server config, port, database and
// ratings.  Run several sizes side by side (e.g. --size 9 and --size 13).
//
// The fleet file lists the reference engines (name, ai/ agent, per-move
// budget in ms) and the rating anchors.  Anchors are written into the
// server database on every start, so editing refs.json is enough.
//
// Everything runs until Ctrl-C, which shuts down all children.  State
// (ratings, game archive, SGF) persists in the data directory across runs.

const { spawn, execFileSync } = require('child_process');
const fs = require('fs');
const net = require('net');
const path = require('path');
const Util = require('../util.js');

const ROOT = path.join(__dirname, '..');

const opts = Util.parseArgs(process.argv.slice(2), ['help', 'server-only']);
if (opts.help) {
  console.log('Usage: node cgos/run.js [--size <n>] [--refs <file>] [--ini <file>] [--data <dir>] [--server-only]');
  process.exit(0);
}

const size     = opts.getInt('size', 9);
const refsFile = path.resolve(opts.get('refs', path.join(__dirname, 'refs.json')));
const iniFile  = path.resolve(opts.get('ini',  path.join(__dirname, `torogo${size}.ini`)));
const dataDir  = path.resolve(opts.get('data', path.join(__dirname, 'data', `${size}x${size}`)));

const fleet = JSON.parse(fs.readFileSync(refsFile, 'utf8'));

// The server config is the source of truth for the port, so one refs.json
// can serve every board size.
const portMatch = fs.readFileSync(iniFile, 'utf8').match(/^\s*portNumber\s*=\s*(\d+)/m);
const port = portMatch ? parseInt(portMatch[1], 10) : (fleet.port || 1919);

// ── Data directory layout ─────────────────────────────────────────────────────

fs.mkdirSync(path.join(dataDir, 'logs'), { recursive: true });
fs.mkdirSync(path.join(dataDir, 'clients'), { recursive: true });
fs.copyFileSync(path.join(__dirname, 'server', 'log.ini'), path.join(dataDir, 'log.ini'));
// A leftover kill file would make the server exit after one round.
try { fs.unlinkSync(path.join(dataDir, 'killCgos')); } catch {}

const children = [];
function launch(name, cmd, args, cwd) {
  const log = fs.openSync(path.join(dataDir, 'logs', name + '.out'), 'a');
  const child = spawn(cmd, args, { cwd, stdio: ['ignore', log, log] });
  child.on('exit', code => console.log(`[${name}] exited (${code})`));
  children.push(child);
  return child;
}

function shutdown() {
  console.log('\nshutting down...');
  for (const c of children) c.kill('SIGTERM');
  process.exit(0);
}
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

// ── Server ────────────────────────────────────────────────────────────────────

console.log(`server: ${iniFile}`);
console.log(`data:   ${dataDir}`);
launch('server', 'python3', [path.join(__dirname, 'server', 'cgos', 'server.py'), iniFile], dataDir);

function waitForPort(port, host, cb, tries = 50) {
  const s = net.connect(port, host);
  s.on('connect', () => { s.destroy(); cb(); });
  s.on('error', () => {
    s.destroy();
    if (tries <= 0) { console.error('server did not come up'); process.exit(1); }
    setTimeout(() => waitForPort(port, host, cb, tries - 1), 200);
  });
}

// ── Anchors and house list ────────────────────────────────────────────────────

// The reference fleet are HOUSE engines: they idle until a trial engine
// (cgos/join.js) needs an opponent.  Anchor ratings are pinned.
function writeFleetTables() {
  const anchors = fleet.anchors || {};
  const stmts = [
    'CREATE TABLE IF NOT EXISTS house(name, primary key(name));',
    'DELETE FROM house;',
    ...fleet.refs.map(r => `INSERT INTO house VALUES(${JSON.stringify(r.name)});`),
    ...Object.entries(anchors)
      .map(([n, r]) => `INSERT OR REPLACE INTO anchors VALUES(${JSON.stringify(n)}, ${r});`),
  ].join(' ');
  const py = `
import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
db.executescript(sys.argv[2])
db.commit()
`;
  execFileSync('python3', ['-c', py, path.join(dataDir, 'cgos.state'), stmts]);
  console.log(`house:   ${fleet.refs.map(r => r.name).join(', ')}`);
  console.log(`anchors: ${Object.entries(anchors).map(([n, r]) => `${n}=${r}`).join(', ') || '(none)'}`);
}

// ── Engine clients ────────────────────────────────────────────────────────────

function clientCfg(name, agent, budget) {
  const gtp = path.join(__dirname, 'gtp.js');
  return `Common:
  KillFile = ${path.join(dataDir, 'kill-' + name + '.txt')}

GTPEngine:
  Name = ${name}
  CommandLine = node ${gtp} --p ${agent} --budget ${budget}
  ServerHost = ${fleet.host || '127.0.0.1'}
  ServerPort = ${port}
  ServerUser = ${name}
  ServerPassword = ${fleet.password || 'torogo'}
  NumberOfGames = 1
  LogFile = ${path.join(dataDir, 'logs', 'engine-' + name + '.log')}
`;
}

waitForPort(port, fleet.host || '127.0.0.1', () => {
  writeFleetTables();
  if (opts['server-only']) {
    console.log('server up (server-only mode); attach engines with cgos/join.js');
    return;
  }
  for (const ref of fleet.refs) {
    const cfgPath = path.join(dataDir, 'clients', ref.name + '.cfg');
    fs.writeFileSync(cfgPath, clientCfg(ref.name, ref.agent, ref.budget));
    try { fs.unlinkSync(path.join(dataDir, `kill-${ref.name}.txt`)); } catch {}
    launch('client-' + ref.name, 'python3', [path.join(__dirname, 'client', 'cgosclient.py'), cfgPath], dataDir);
    console.log(`engine:  ${ref.name} (ai/${ref.agent}.js, ${ref.budget}ms/move)`);
  }
  console.log('\nhouse fleet idle and ready — games start when a trial engine');
  console.log('connects (node cgos/join.js --p <agent>).  Ctrl-C to stop.');
});
