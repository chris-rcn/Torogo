'use strict';

/**
 * Tune the AMAF opponent-move weight against the mc policy.
 *
 * AMAF_DISCOUNT is fixed at 0.5.  This script sweeps OPP_MOVE_WEIGHT values.
 *
 * Runs indefinitely, cycling through candidate values.  Each round
 * allocates batches via Wilson CI upper bounds — promising values get more
 * games while all values retain a minimum allocation for continuous
 * exploration.
 *
 * Usage:
 *   node tune_amaf.js [--size <n>] [--weights <w,w,...>]
 *
 *   --size    Board size (default: 9)
 *   --weights Comma-separated OPP_MOVE_WEIGHT values to test
 *             (default: 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)
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
const FIXED_DISCOUNT  = 0.5;
const weights         = (argVal('weights', '0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0'))
  .split(',').map(Number).sort((a, b) => a - b);

const selfplayScript = path.join(__dirname, 'selfplay.js');

// ─── Per-weight accumulators ─────────────────────────────────────────────────

const stats = new Map(); // weight → { wins, games }
for (const w of weights) stats.set(w, { wins: 0, games: 0 });

// ─── Helpers ──────────────────────────────────────────────────────────────────

function ts() {
  return new Date().toTimeString().slice(0, 8);
}

// Run selfplay.js with AMAF_DISCOUNT and AMAF_OPP_WEIGHT set, once as each
// colour against mc.  Returns amaf win count out of GAMES_PER_ROUND total.
function runBatch(oppWeight) {
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
        env: {
          ...process.env,
          AMAF_DISCOUNT: String(FIXED_DISCOUNT),
          AMAF_OPP_WEIGHT: String(oppWeight),
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
      process.stderr.write(`[${ts()}] Could not parse output for oppWeight=${oppWeight}\n`);
      process.stderr.write(result.stdout + '\n');
      continue;
    }
    const m = line.match(/(\d+)\s+[\d.]+%/);
    if (m) amafWins += parseInt(m[1], 10);
  }

  return amafWins;
}

// Wilson CI upper bound for a weight's win rate.
function wilsonUpper(s) {
  if (s.games === 0) return Infinity; // untested → explore first
  const z = 1.96, n = s.games, p = s.wins / s.games;
  const centre = (p + z * z / (2 * n)) / (1 + z * z / n);
  const margin  = z / (1 + z * z / n) * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n));
  return centre + margin;
}

// Allocate batches per weight using Wilson CI upper bounds.
// Every value gets at least 1 batch (continuous exploration);
// top third by upper bound get 3 batches for faster convergence.
function batchesForRound() {
  const scored = weights
    .map(w => ({ w, upper: wilsonUpper(stats.get(w)) }))
    .sort((a, b) => b.upper - a.upper);

  const top = Math.max(1, Math.ceil(weights.length / 3));
  const alloc = new Map();
  for (let i = 0; i < scored.length; i++) {
    alloc.set(scored[i].w, i < top ? 3 : 1);
  }
  return alloc;
}

function printLeaderboard(round) {
  const rows = [];
  for (const [w, s] of stats) {
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
    rows.push({ w, rate, lo, hi, games: s.games, wins: s.wins });
  }
  rows.sort((a, b) => (b.rate ?? -1) - (a.rate ?? -1));

  console.log(`\n${'─'.repeat(62)}`);
  console.log(`Round ${round} complete  [${ts()}]  size=${boardSize}  discount=${FIXED_DISCOUNT}`);
  console.log(`${'─'.repeat(62)}`);
  console.log(`  opp_weight    win%   95% CI              games`);
  console.log(`${'─'.repeat(62)}`);
  for (const r of rows) {
    if (r.rate === null) {
      console.log(`  ${String(r.w).padEnd(10)}  —`);
    } else {
      const pct  = (r.rate  * 100).toFixed(1).padStart(5);
      const plo  = (r.lo    * 100).toFixed(1).padStart(5);
      const phi  = (r.hi    * 100).toFixed(1).padStart(5);
      const mark = r.rate > 0.5 ? ' ←' : '';
      console.log(`  ${String(r.w).padEnd(10)}  ${pct}%   [${plo}%, ${phi}%]   ${String(r.games).padStart(5)}${mark}`);
    }
  }
  console.log(`${'─'.repeat(62)}\n`);

}

// ─── Main loop ────────────────────────────────────────────────────────────────

console.log(`AMAF opp-weight tuning  size=${boardSize}  discount=${FIXED_DISCOUNT}  ${GAMES_PER_ROUND} games/value/round  control=mc`);
console.log(`Weights: ${weights.join(', ')}`);
console.log(`Adaptive: top third by Wilson CI upper bound get 3x batches, rest get 1x`);

let round = 0;
// eslint-disable-next-line no-constant-condition
while (true) {
  round++;
  const alloc = batchesForRound();
  const totalBatches = [...alloc.values()].reduce((a, b) => a + b, 0);
  console.log(`[${ts()}] Starting round ${round}  (${alloc.size} values, ${totalBatches} batches)...`);

  for (const [w, batches] of alloc) {
    process.stdout.write(`  [${ts()}] opp_weight=${w} x${batches} ... `);
    let wins = 0;
    for (let b = 0; b < batches; b++) wins += runBatch(w);
    const s = stats.get(w);
    s.wins  += wins;
    s.games += GAMES_PER_ROUND * batches;
    const pct = (100 * s.wins / s.games).toFixed(1);
    console.log(`${wins}/${GAMES_PER_ROUND * batches}  (cumulative: ${s.wins}/${s.games} = ${pct}%)`);
  }

  printLeaderboard(round);
}
