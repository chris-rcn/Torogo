'use strict';

// ladder-timing.js — what ladder2 costs when single-stone chains are skipped.
//
// Replays games from a corpus (gen-games.js lines "<size> <move1,move2,...>",
// played from Game2(size, true)) and at each position runs
// getAllLadderStatuses twice on the same Game3: over all chains, and over
// chains of 2+ stones (minChainSize 2).  Reports wall time, chains read and
// read nodes for each, overall and by phase band.  The two passes alternate
// order by position so warm-up and cache effects cancel; node counts do not
// depend on machine load.
//
// Usage: node ladder-timing.js --corpus FILE [--games 100] [--stride 1]

const fs = require('fs');
const readline = require('readline');
const { Game2, parseMove } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');
const { getAllLadderStatuses } = require('./ladder2.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['corpus', 'games', 'stride']);
if (opts.help || !opts.corpus) {
  console.log(`Usage: node ladder-timing.js --corpus FILE [options]

Times ladder2's getAllLadderStatuses over every chain vs chains of 2+ stones,
at each position of the corpus's games (lines "<size> <moves>", '#' comments).

  --corpus FILE   gen-games.js corpus (required)
  --games N       games to replay, from the top of the file   (default 100)
  --stride K      time every Kth position of a game           (default 1)
  --help          show this message`);
  process.exit(opts.help ? 0 : 1);
}
const GAMES  = parseInt(opts.games || '100', 10);
const STRIDE = parseInt(opts.stride || '1', 10);
if (!(GAMES >= 1) || !(STRIDE >= 1)) { console.error('--games and --stride must be positive integers'); process.exit(1); }

const BANDS = 10;
const kinds = ['all', 'multi'];
const tot = () => ({ ns: 0n, reads: 0, nodes: 0 });
const total = { all: tot(), multi: tot() };
const byBand = Array.from({ length: BANDS }, () => ({ n: 0, all: tot(), multi: tot() }));
let positions = 0, games = 0;

function measure(g3, minSize, acc, band) {
  const t0 = process.hrtime.bigint();
  const res = getAllLadderStatuses(g3, minSize);
  const dt = process.hrtime.bigint() - t0;
  let nodes = 0;
  for (const { status } of res) if (status) nodes += status.readNodes;
  for (const a of [acc, band]) { a.ns += dt; a.reads += res.length; a.nodes += nodes; }
}

function timeGame(size, moves) {
  const g = new Game2(size, true);
  for (let i = 0; i < moves.length && !g.gameOver; i++) {
    if (i % STRIDE === 0) {
      const g3 = game3FromGame2(g);
      const b = byBand[Math.min(BANDS - 1, Math.floor(g.phase() * BANDS))];
      b.n++; positions++;
      const order = positions % 2 ? kinds : [...kinds].reverse();
      for (const k of order) measure(g3, k === 'all' ? 1 : 2, total[k], b[k]);
    }
    if (!g.play(moves[i])) { console.error(`illegal move ${i + 1} in game ${games + 1}`); process.exit(1); }
  }
}

function report() {
  const ms = ns => Number(ns) / 1e6;
  const pct = (a, b) => b > 0 ? `${(100 * (1 - a / b)).toFixed(0)}%` : '-';
  const A = total.all, M = total.multi;
  console.log(`games: ${games}  positions timed: ${positions}  (stride ${STRIDE})`);
  console.log(`all chains:    ${ms(A.ns).toFixed(0)} ms  (${(1000 * ms(A.ns) / positions).toFixed(1)} us/position)  reads ${A.reads}  nodes ${A.nodes}`);
  console.log(`2+ stone only: ${ms(M.ns).toFixed(0)} ms  (${(1000 * ms(M.ns) / positions).toFixed(1)} us/position)  reads ${M.reads}  nodes ${M.nodes}`);
  console.log(`saved by skipping single stones: time ${pct(ms(M.ns), ms(A.ns))}, reads ${pct(M.reads, A.reads)}, nodes ${pct(M.nodes, A.nodes)}`);
  console.log('');
  console.log('phase      positions   all us/pos  2+ us/pos  time saved  nodes saved');
  byBand.forEach((b, i) => {
    if (!b.n) return;
    console.log(`${(i / BANDS).toFixed(1)}-${((i + 1) / BANDS).toFixed(1)}  ${String(b.n).padStart(9)}  ${(1000 * ms(b.all.ns) / b.n).toFixed(1).padStart(10)}  ${(1000 * ms(b.multi.ns) / b.n).toFixed(1).padStart(9)}  ${pct(ms(b.multi.ns), ms(b.all.ns)).padStart(10)}  ${pct(b.multi.nodes, b.all.nodes).padStart(11)}`);
  });
}

const rl = readline.createInterface({ input: fs.createReadStream(opts.corpus), crlfDelay: Infinity });
rl.on('line', line => {
  if (games >= GAMES || !line || line[0] === '#') return;
  const [sizeS, movesS] = line.trim().split(/\s+/);
  const size = parseInt(sizeS, 10);
  if (!movesS || !(size >= 2)) return;
  timeGame(size, movesS.split(',').map(t => parseMove(t, size)));
  games++;
  if (games >= GAMES) rl.close();
});
rl.on('close', report);
