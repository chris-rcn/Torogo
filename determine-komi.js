'use strict';

/**
 * determine-komi.js
 *
 * Plays self-play games under various komi values to find the "fair komi"
 * where black wins ~50% of the time.
 *
 * Usage:
 *   node determine-komi.js [--agent <name>] [--size <n>] [--budget <ms>]
 *
 * Defaults: agent=mcts  size=9  budget=100
 *
 * Komi is always a half-integer (0.5, 1.5, 2.5, …; never 0, 1, 2, …).
 * Output is printed at exponentially increasing intervals: 1s, 1.5s, 2.25s, …
 * The script runs indefinitely.
 */

const { Game2, PASS, BLACK, WHITE, setKomi } = require('./game2.js');

// ── CLI arguments ─────────────────────────────────────────────────────────────
const argv = process.argv.slice(2);
function arg(name, def) {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : def;
}

const agentName = arg('agent', 'mcts');
const size      = parseInt(arg('size',   '9'),   10);
const budget    = parseInt(arg('budget', '100'), 10);

const { getMove } = require(`./ai/${agentName}.js`);

// ── Per-komi statistics ───────────────────────────────────────────────────────
// Map<komi, { games: number, blackWins: number }>
const stats = new Map();

function getOrCreate(komi) {
  if (!stats.has(komi)) stats.set(komi, { games: 0, blackWins: 0 });
  return stats.get(komi);
}

// Black win-rate at a komi (0..1), or null if untested.
function winRate(komi) {
  const s = stats.get(komi);
  return s && s.games > 0 ? s.blackWins / s.games : null;
}

// ── Komi utilities ────────────────────────────────────────────────────────────
// Project x onto the nearest half-integer ≥ 0.5.
// halfInt(4.0) → 4.5   halfInt(4.5) → 4.5   halfInt(4.9) → 4.5
// halfInt(5.0) → 5.5   halfInt(3.7) → 3.5
function halfInt(x) {
  return Math.max(0.5, Math.floor(x) + 0.5);
}

// ── Adaptive bisection komi selector ─────────────────────────────────────────
//
// Maintains a bracket [loKomi, hiKomi] where:
//   loKomi = largest komi tested where black wins ≥ 50%
//   hiKomi = smallest komi tested where black wins  < 50%
//
// Strategy:
//   • No bracket yet  → extend the range in the appropriate direction.
//   • Bracket found   → bisect; when adjacent half-integers, refine
//                       whichever endpoint has fewer samples.
//
function selectKomi() {
  if (stats.size === 0) {
    return halfInt(size / 2);   // start near the "natural" komi
  }

  let loKomi = null;  // black-favoured: highest komi with wr >= 0.5
  let hiKomi = null;  // white-favoured: lowest  komi with wr <  0.5

  for (const [k, s] of stats) {
    if (s.games === 0) continue;
    const w = s.blackWins / s.games;
    if (w >= 0.5 && (loKomi === null || k > loKomi)) loKomi = k;
    if (w  < 0.5 && (hiKomi === null || k < hiKomi)) hiKomi = k;
  }

  // Extend search range until both sides are bracketed.
  if (loKomi === null && hiKomi === null) return halfInt(size / 2);
  if (loKomi === null) return halfInt(hiKomi - 2);   // white wins everywhere → go lower
  if (hiKomi === null) return halfInt(loKomi + 2);   // black wins everywhere → go higher

  // Adjacent half-integers: refine whichever side has fewer data.
  if (hiKomi - loKomi <= 1) {
    const sLo = stats.get(loKomi), sHi = stats.get(hiKomi);
    return sLo.games <= sHi.games ? loKomi : hiKomi;
  }

  // Bisect the bracket.
  return halfInt((loKomi + hiKomi) / 2);
}

// ── Play one full game at the given komi ──────────────────────────────────────
function playGame(komi) {
  setKomi(size, komi);
  const g = new Game2(size);
  while (!g.gameOver) {
    const move = getMove(g, budget);
    const idx  = move.type === 'place' ? move.y * size + move.x : PASS;
    g.play(idx);
  }
  const sc = g.calcScore();
  return sc.black > sc.white;   // true → black wins
}

// ── Find the komi whose win-rate is closest to 50% ───────────────────────────
function bestKomi() {
  let best = null, bestDist = Infinity;
  for (const [k, s] of stats) {
    if (s.games === 0) continue;
    const dist = Math.abs(s.blackWins / s.games - 0.5);
    if (dist < bestDist) { bestDist = dist; best = k; }
  }
  return best;
}

// ── Display ───────────────────────────────────────────────────────────────────
function fmtElapsed(sec) {
  if (sec < 60)   return `${sec.toFixed(0)}s`;
  if (sec < 3600) return `${(sec / 60).toFixed(1)}m`;
  return `${(sec / 3600).toFixed(2)}h`;
}

function printStatus(totalGames, elapsedSec) {
  const best = bestKomi();

  console.log(`\n[+${fmtElapsed(elapsedSec)} | ${totalGames} games]`);

  if (best !== null) {
    const bs   = stats.get(best);
    const rate = (bs.blackWins / bs.games * 100).toFixed(1);
    console.log(`  ★  Best komi: ${best.toFixed(1)}  |  black win rate: ${rate}%  (n=${bs.games})`);
  } else {
    console.log(`  (no data yet)`);
  }

  const sorted = [...stats.entries()]
    .filter(([, s]) => s.games > 0)
    .sort(([a], [b]) => a - b);

  for (const [k, s] of sorted) {
    const rate  = s.blackWins / s.games;
    const pct   = (rate * 100).toFixed(1).padStart(5);
    const fill  = Math.round(rate * 20);
    const bar   = '█'.repeat(fill) + '░'.repeat(20 - fill);
    const mark  = k === best ? ' ◄' : '  ';
    console.log(`    komi ${String(k.toFixed(1)).padStart(5)}: [${bar}] ${pct}% black  n=${String(s.games).padStart(4)}${mark}`);
  }
}

// ── Main loop (runs forever) ──────────────────────────────────────────────────
const startTime  = Date.now();
let   totalGames = 0;
let   interval   = 1000;                    // first output after 1 s
let   nextPrint  = startTime + interval;

console.log(`determine-komi  agent=${agentName}  size=${size}×${size}  budget=${budget}ms`);
console.log(`Output schedule: 1s × 1.5^n  (Ctrl-C to stop)\n`);

while (true) {
  const komi     = selectKomi();
  const blackWon = playGame(komi);
  const s        = getOrCreate(komi);
  s.games++;
  if (blackWon) s.blackWins++;
  totalGames++;

  const now = Date.now();
  if (now >= nextPrint) {
    printStatus(totalGames, (now - startTime) / 1000);
    interval  = Math.round(interval * 1.5);
    nextPrint = now + interval;
  }
}
