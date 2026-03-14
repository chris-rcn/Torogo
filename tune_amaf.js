'use strict';

/**
 * Tune the AMAF discount factor against the influence policy.
 *
 * Runs indefinitely, cycling through candidate discount values and playing
 * games_per_round games per value per round.  After each complete round the
 * cumulative leaderboard is printed and appended to results.jsonl so you can
 * inspect progress at any time.
 *
 * Usage:
 *   node tune_amaf.js [--size <n>] [--games <n>] [--discounts <d,d,...>]
 *
 *   --size      Board size (default: 9)
 *   --games     Games per discount value per round (default: 20)
 *   --discounts Comma-separated discount values to test
 *               (default: 0.0,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95,1.0)
 */

const { spawnSync } = require('child_process');
const fs   = require('fs');
const path = require('path');

// ─── Argument parsing ─────────────────────────────────────────────────────────

const argv = process.argv.slice(2);
function argVal(name, def) {
  const i = argv.indexOf('--' + name);
  return i !== -1 ? argv[i + 1] : def;
}

const boardSize     = parseInt(argVal('size',  '9'), 10);
const gamesPerRound = parseInt(argVal('games', '20'), 10);
const discounts     = (argVal('discounts', '0.0,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95,1.0'))
  .split(',').map(Number).sort((a, b) => a - b);

const RESULTS_FILE = path.join(__dirname, 'amaf_tune_results.jsonl');
const selfplayScript = path.join(__dirname, 'selfplay.js');

// ─── Per-discount accumulators ────────────────────────────────────────────────

const stats = new Map(); // discount → { wins, games }
for (const d of discounts) stats.set(d, { wins: 0, games: 0 });

// ─── Helpers ──────────────────────────────────────────────────────────────────

function ts() {
  return new Date().toTimeString().slice(0, 8);
}

// Run selfplay.js with AMAF_DISCOUNT set, alternating amaf as p1 and p2
// to cancel colour bias.  Returns amaf win count out of `games` total.
function runBatch(discount, games) {
  const half = Math.ceil(games / 2);
  let amafWins = 0;

  for (const [p1, p2, amafPlayer] of [
    ['amaf', 'influence', 'P1'],
    ['influence', 'amaf',  'P2'],
  ]) {
    const g = (amafPlayer === 'P1') ? half : games - half;
    if (g === 0) continue;

    const result = spawnSync(
      process.execPath,
      [selfplayScript, '--p1', p1, '--p2', p2, '--games', String(g), '--size', String(boardSize)],
      {
        env: { ...process.env, AMAF_DISCOUNT: String(discount) },
        encoding: 'utf8',
        timeout: 7200000, // 2-hour safety limit per batch
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

  // Append a machine-readable snapshot
  const snapshot = { round, ts: new Date().toISOString(), boardSize, rows };
  fs.appendFileSync(RESULTS_FILE, JSON.stringify(snapshot) + '\n');
}

// ─── Main loop ────────────────────────────────────────────────────────────────

console.log(`AMAF discount tuning  size=${boardSize}  ${gamesPerRound} games/value/round`);
console.log(`Discounts: ${discounts.join(', ')}`);
console.log(`Results appended to: ${RESULTS_FILE}\n`);

let round = 0;
// eslint-disable-next-line no-constant-condition
while (true) {
  round++;
  console.log(`[${ts()}] Starting round ${round}...`);

  for (const d of discounts) {
    process.stdout.write(`  [${ts()}] discount=${d} ... `);
    const wins = runBatch(d, gamesPerRound);
    const s = stats.get(d);
    s.wins  += wins;
    s.games += gamesPerRound;
    const pct = (100 * s.wins / s.games).toFixed(1);
    console.log(`${wins}/${gamesPerRound}  (cumulative: ${s.wins}/${s.games} = ${pct}%)`);
  }

  printLeaderboard(round);
}
