'use strict';

/**
 * Tune the AMAF discount factor against the mc policy.
 *
 * Runs indefinitely, cycling through candidate discount values and playing
 * 2 games per value per round (1 as each colour).  After each complete round
 * the cumulative leaderboard is printed and appended to results.jsonl so you
 * can inspect progress at any time.
 *
 * Usage:
 *   node tune_amaf.js [--size <n>] [--discounts <d,d,...>]
 *
 *   --size      Board size (default: 9)
 *   --discounts Comma-separated discount values to test
 *               (default: 0.0,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95,1.0)
 */

const { spawnSync } = require('child_process');
const path = require('path');

// ─── Argument parsing ─────────────────────────────────────────────────────────

const argv = process.argv.slice(2);
function argVal(name, def) {
  const i = argv.indexOf('--' + name);
  return i !== -1 ? argv[i + 1] : def;
}

const boardSize     = parseInt(argVal('size', '9'), 10);
const GAMES_PER_ROUND = 2; // 1 game as each colour per round
const discounts     = (argVal('discounts', '0.0,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95,1.0'))
  .split(',').map(Number).sort((a, b) => a - b);

const selfplayScript = path.join(__dirname, 'selfplay.js');

// ─── Per-discount accumulators ────────────────────────────────────────────────

const stats = new Map(); // discount → { wins, games }
for (const d of discounts) stats.set(d, { wins: 0, games: 0 });

// ─── Helpers ──────────────────────────────────────────────────────────────────

function ts() {
  return new Date().toTimeString().slice(0, 8);
}

// Run selfplay.js with AMAF_DISCOUNT set, once as each colour against mc.
// Returns amaf win count out of GAMES_PER_ROUND total.
function runBatch(discount) {
  let amafWins = 0;

  for (const [p1, p2, amafPlayer] of [
    ['amaf', 'mc', 'P1'],
    ['mc',   'amaf', 'P2'],
  ]) {
    const g = 1;

    const result = spawnSync(
      process.execPath,
      [selfplayScript, '--p1', p1, '--p2', p2, '--games', String(g), '--size', String(boardSize)],
      {
        env: { ...process.env, AMAF_DISCOUNT: String(discount) },
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

function printLeaderboard(round) {
  const rows = [];
  for (const [d, s] of stats) {
    const rate = s.games > 0 ? s.wins / s.games : null;
    // 95% Wilson confidence interval half-width
    let lo = 0, hi = 1;
    if (s.games > 0 && rate !== null) {
      const z = 1.96, n = s.games, p = rate;
      const centre = (p + z * z / (2 * n)) / (1 + z * z / n);
      const margin  = z / (1 + z * z / n) * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n));
      lo = centre - margin;
      hi = centre + margin;
    }
    rows.push({ d, rate, lo, hi, games: s.games, wins: s.wins });
  }
  rows.sort((a, b) => (b.rate ?? -1) - (a.rate ?? -1));

  console.log(`\n${'─'.repeat(58)}`);
  console.log(`Round ${round} complete  [${ts()}]  size=${boardSize}`);
  console.log(`${'─'.repeat(58)}`);
  console.log(`  discount    win%   95% CI              games`);
  console.log(`${'─'.repeat(58)}`);
  for (const r of rows) {
    if (r.rate === null) {
      console.log(`  ${String(r.d).padEnd(8)}  —`);
    } else {
      const pct  = (r.rate  * 100).toFixed(1).padStart(5);
      const plo  = (r.lo    * 100).toFixed(1).padStart(5);
      const phi  = (r.hi    * 100).toFixed(1).padStart(5);
      const mark = r.rate > 0.5 ? ' ←' : '';
      console.log(`  ${String(r.d).padEnd(8)}  ${pct}%   [${plo}%, ${phi}%]   ${String(r.games).padStart(5)}${mark}`);
    }
  }
  console.log(`${'─'.repeat(58)}\n`);

}

// ─── Main loop ────────────────────────────────────────────────────────────────

console.log(`AMAF discount tuning  size=${boardSize}  ${GAMES_PER_ROUND} games/value/round  control=mc`);
console.log(`Discounts: ${discounts.join(', ')}`);

let round = 0;
// eslint-disable-next-line no-constant-condition
while (true) {
  round++;
  console.log(`[${ts()}] Starting round ${round}...`);

  for (const d of discounts) {
    process.stdout.write(`  [${ts()}] discount=${d} ... `);
    const wins = runBatch(d);
    const s = stats.get(d);
    s.wins  += wins;
    s.games += GAMES_PER_ROUND;
    const pct = (100 * s.wins / s.games).toFixed(1);
    console.log(`${wins}/${GAMES_PER_ROUND}  (cumulative: ${s.wins}/${s.games} = ${pct}%)`);
  }

  printLeaderboard(round);
}
