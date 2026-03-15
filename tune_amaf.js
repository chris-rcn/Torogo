'use strict';

/**
 * Tune the AMAF discount factor against the mc policy.
 *
 * AMAF_OPP_WEIGHT is fixed at 0.  This script sweeps AMAF_DISCOUNT values.
 *
 * Runs indefinitely, cycling through candidate values.  Each round
 * allocates batches via Wilson CI upper bounds — promising values get more
 * games while all values retain a minimum allocation for continuous
 * exploration.
 *
 * Usage:
 *   node tune_amaf.js [--size <n>] [--discounts <d,d,...>] [--budget <ms>]
 *
 *   --size      Board size (default: 9)
 *   --discounts Comma-separated AMAF_DISCOUNT values to test
 *               (default: 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)
 *   --budget    Time budget per move in ms (default: 5000)
 */

const { spawnSync } = require('child_process');
const path = require('path');

// ─── Argument parsing ─────────────────────────────────────────────────────────

const argv = process.argv.slice(2);
function argVal(name, def) {
  const i = argv.indexOf('--' + name);
  return i !== -1 ? argv[i + 1] : def;
}

const boardSize       = parseInt(argVal('size', '9'), 10);
const GAMES_PER_ROUND = 2; // 1 game as each colour per round
const FIXED_OPP_WEIGHT = 0;
const budgetMs        = argVal('budget', '5000');
const discounts       = (argVal('discounts', '0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9'))
  .split(',').map(Number).sort((a, b) => a - b);

const selfplayScript = path.join(__dirname, 'selfplay.js');

// ─── Per-weight accumulators ─────────────────────────────────────────────────

const stats = new Map(); // discount → { wins, games }
for (const d of discounts) stats.set(d, { wins: 0, games: 0 });

// ─── Helpers ──────────────────────────────────────────────────────────────────

function ts() {
  return new Date().toTimeString().slice(0, 8);
}

// Run selfplay.js with AMAF_DISCOUNT and AMAF_OPP_WEIGHT set, once as each
// colour against mc.  Returns amaf win count out of GAMES_PER_ROUND total.
function runBatch(discount) {
  let amafWins = 0;

  for (const [p1, p2, amafPlayer] of [
    ['amaf', 'mc', 'P1'],
    ['mc',   'amaf', 'P2'],
  ]) {
    const g = 1;

    const result = spawnSync(
      process.execPath,
      [selfplayScript, '--p1', p1, '--p2', p2, '--games', String(g), '--size', String(boardSize), '--budget', budgetMs],
      {
        env: {
          ...process.env,
          AMAF_DISCOUNT: String(discount),
          AMAF_OPP_WEIGHT: String(FIXED_OPP_WEIGHT),
        },
        encoding: 'utf8',
        timeout: 7200000, // 2-hour safety limit per batch
        maxBuffer: 50 * 1024 * 1024, // 50 MB
      }
    );

    if (result.error) throw result.error;

    // Parse "  P1:   14    70.0%  ..."  or "  P2:   14    70.0%  ..."
    const line = result.stdout.split('\n').find(l => l.trim().startsWith(amafPlayer + ':'));
    if (!line) {
      process.stderr.write(`[${ts()}] Could not parse output for discount=${discount}\n`);
      process.stderr.write(result.stdout + '\n');
      continue;
    }
    const m = line.match(/(\d+)\s+[\d.]+%/);
    if (m) amafWins += parseInt(m[1], 10);
  }

  return amafWins;
}

// Wilson CI upper bound for a discount's win rate.
function wilsonUpper(s) {
  if (s.games === 0) return Infinity; // untested → explore first
  const z = 1.96, n = s.games, p = s.wins / s.games;
  const centre = (p + z * z / (2 * n)) / (1 + z * z / n);
  const margin  = z / (1 + z * z / n) * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n));
  return centre + margin;
}

// Allocate batches per discount using Wilson CI upper bounds.
// Every value gets at least 1 batch (continuous exploration);
// top third by upper bound get 3 batches for faster convergence.
function batchesForRound() {
  const scored = discounts
    .map(d => ({ d, upper: wilsonUpper(stats.get(d)) }))
    .sort((a, b) => b.upper - a.upper);

  const top = Math.max(1, Math.ceil(discounts.length / 3));
  const alloc = new Map();
  for (let i = 0; i < scored.length; i++) {
    alloc.set(scored[i].d, i < top ? 3 : 1);
  }
  return alloc;
}

function printLeaderboard(round) {
  const rows = [];
  for (const [d, s] of stats) {
    const rate = s.games > 0 ? s.wins / s.games : null;
    rows.push({ d, rate, games: s.games, wins: s.wins });
  }
  rows.sort((a, b) => (b.rate ?? -1) - (a.rate ?? -1));

  console.log(`\n${'─'.repeat(62)}`);
  console.log(`Round ${round} complete  [${ts()}]  size=${boardSize}  opp_weight=${FIXED_OPP_WEIGHT}`);
  console.log(`${'─'.repeat(62)}`);
  console.log(`  discount      win%    games`);
  console.log(`${'─'.repeat(62)}`);
  for (const r of rows) {
    if (r.rate === null) {
      console.log(`  ${String(r.d).padEnd(10)}  —`);
    } else {
      const pct  = (r.rate  * 100).toFixed(1).padStart(5);
      const mark = r.rate > 0.5 ? ' ←' : '';
      console.log(`  ${String(r.d).padEnd(10)}  ${pct}%   ${String(r.games).padStart(5)}${mark}`);
    }
  }
  console.log(`${'─'.repeat(62)}\n`);

}

// ─── Main loop ────────────────────────────────────────────────────────────────

console.log(`AMAF discount tuning  size=${boardSize}  opp_weight=${FIXED_OPP_WEIGHT}  budget=${budgetMs}ms  ${GAMES_PER_ROUND} games/value/round  control=mc`);
console.log(`Discounts: ${discounts.join(', ')}`);
console.log(`Adaptive: top third by Wilson CI upper bound get 3x batches, rest get 1x`);

let round = 0;
// eslint-disable-next-line no-constant-condition
while (true) {
  round++;
  const alloc = batchesForRound();
  const totalBatches = [...alloc.values()].reduce((a, b) => a + b, 0);
  console.log(`[${ts()}] Starting round ${round}  (${alloc.size} values, ${totalBatches} batches)...`);

  for (const [d, batches] of alloc) {
    process.stdout.write(`  [${ts()}] discount=${d} x${batches} ... `);
    let wins = 0;
    for (let b = 0; b < batches; b++) wins += runBatch(d);
    const s = stats.get(d);
    s.wins  += wins;
    s.games += GAMES_PER_ROUND * batches;
    const pct = (100 * s.wins / s.games).toFixed(1);
    console.log(`${wins}/${GAMES_PER_ROUND * batches}  (cumulative: ${s.wins}/${s.games} = ${pct}%)`);
  }

  printLeaderboard(round);
}
