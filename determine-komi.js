'use strict';

/**
 * determine-komi.js
 *
 * Finds the "fair komi" (black win rate ≈ 50%) via Bayesian inference.
 *
 * Model: P(black wins | tested komi=k, fair komi=k*) = sigmoid(α × (k* − k))
 *   – monotonically decreasing in k
 *   – equals 0.5 exactly when k = k*
 *
 * A discrete prior over all half-integer komi values is updated after every
 * game using the likelihood above.  No hypothesis is ever eliminated — it
 * merely becomes less probable.  Thompson sampling (draw k* from the
 * posterior, test at that value) drives exploration toward the uncertain
 * boundary while still revisiting other regions.
 *
 * Usage:
 *   node determine-komi.js [--agent <name>] [--size <n>] [--budget <ms>]
 *                          [--sharpness <α>]
 *
 * Defaults: agent=mcts  size=9  budget=100  sharpness=0.4
 * Output: printed at exponentially increasing intervals (1s × 1.5^n).
 * Runs indefinitely.
 */

const { Game2, PASS, setKomi } = require('./game2.js');

// ── CLI arguments ─────────────────────────────────────────────────────────────
const argv = process.argv.slice(2);
function arg(name, def) {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : def;
}

const agentName = arg('agent',     'prod');
const size      = parseInt(arg('size',      '5'),   10);
const budget    = parseInt(arg('budget',    '1000'), 10);
// α: steepness of the sigmoid win-rate curve.
// Higher → strong agents with sharp transitions; lower → weak/random agents.
const ALPHA     = parseFloat(arg('sharpness', '0.4'));

const { getMove } = require(`./ai/${agentName}.js`);

// ── Komi grid ─────────────────────────────────────────────────────────────────
// All half-integers from 0.5 up to size*2+0.5.
const grid = [];
for (let k = 0.5; k <= size * 2 + 0.5; k += 1) grid.push(k);
const G = grid.length;

// ── Bayesian posterior (log-space for numerical stability) ────────────────────
// logPost[i]  ∝  log P(fair komi = grid[i] | observations so far)
// Initialized to 0 (uniform prior before normalization).
const logPost = new Float64Array(G); // all zeros = uniform

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

// After observing one game at `komi`, multiply each hypothesis's probability
// by the likelihood of that outcome under the logistic model, then renormalize.
function updatePosterior(komi, blackWon) {
  for (let i = 0; i < G; i++) {
    const p = sigmoid(ALPHA * (grid[i] - komi));
    logPost[i] += blackWon ? Math.log(p) : Math.log(1 - p);
  }
  // Subtract max before exponentiating to prevent underflow.
  let maxLP = -Infinity;
  for (let i = 0; i < G; i++) if (logPost[i] > maxLP) maxLP = logPost[i];
  let sum = 0;
  for (let i = 0; i < G; i++) { logPost[i] -= maxLP; sum += Math.exp(logPost[i]); }
  const logSum = Math.log(sum);
  for (let i = 0; i < G; i++) logPost[i] -= logSum;
}

// Return posterior as a normalized Float64Array of probabilities.
function posteriorProbs() {
  const p = new Float64Array(G);
  for (let i = 0; i < G; i++) p[i] = Math.exp(logPost[i]);
  return p;
}

// ── Thompson sampling ─────────────────────────────────────────────────────────
// Draw a fair-komi hypothesis from the posterior; test at that komi.
function thompsonSample() {
  const p = posteriorProbs();
  const r = Math.random();
  let cum = 0;
  for (let i = 0; i < G; i++) {
    cum += p[i];
    if (r <= cum) return grid[i];
  }
  return grid[G - 1];
}

// ── Posterior summaries ───────────────────────────────────────────────────────
function mapKomi(p) {
  let best = 0;
  for (let i = 1; i < G; i++) if (p[i] > p[best]) best = i;
  return grid[best];
}

function meanKomi(p) {
  let m = 0;
  for (let i = 0; i < G; i++) m += grid[i] * p[i];
  return m;
}

// ── Empirical per-komi stats (for display only) ───────────────────────────────
const empirical = new Map(); // komi → { games, blackWins }
function getEmp(komi) {
  if (!empirical.has(komi)) empirical.set(komi, { games: 0, blackWins: 0 });
  return empirical.get(komi);
}

// ── Play one full game at `komi` ──────────────────────────────────────────────
function playGame(komi) {
  setKomi(size, komi);
  const g = new Game2(size);
  while (!g.gameOver) {
    const move = getMove(g, budget);
    const idx  = move.type === 'place' ? move.y * size + move.x : -1; // PASS = -1
    g.play(idx);
  }
  const sc = g.calcScore();
  return sc.black > sc.white;
}

// ── Display ───────────────────────────────────────────────────────────────────
function fmtElapsed(sec) {
  if (sec < 60)   return `${sec.toFixed(0)}s`;
  if (sec < 3600) return `${(sec / 60).toFixed(1)}m`;
  return `${(sec / 3600).toFixed(2)}h`;
}

function printStatus(totalGames, elapsedSec) {
  const p    = posteriorProbs();
  const map  = mapKomi(p);
  const mean = meanKomi(p);
  const maxP = Math.max(...p);

  console.log(`\n[+${fmtElapsed(elapsedSec)} | ${totalGames} games]`);
  console.log(`  ★  MAP komi: ${map.toFixed(1)}  |  posterior mean: ${mean.toFixed(2)}`);
  console.log(`  Posterior over fair komi (bar scaled to mode):\n`);

  for (let i = 0; i < G; i++) {
    const k   = grid[i];
    const emp = empirical.get(k);

    // Skip lines with negligible probability and no empirical data.
    if (p[i] < 0.001 && !emp) continue;

    const fill   = Math.round(p[i] / maxP * 24);
    const bar    = '█'.repeat(fill) + '░'.repeat(24 - fill);
    const pct    = (p[i] * 100).toFixed(1).padStart(5);
    const marker = k === map ? ' ◄' : '  ';

    let empStr = '';
    if (emp && emp.games > 0) {
      const wr = (emp.blackWins / emp.games * 100).toFixed(0);
      empStr = `  ${wr.padStart(3)}% B  n=${emp.games}`;
    }

    console.log(`    komi ${String(k.toFixed(1)).padStart(5)}: [${bar}] ${pct}%${marker}${empStr}`);
  }
}

// ── Main loop (runs forever) ──────────────────────────────────────────────────
const startTime  = Date.now();
let   totalGames = 0;
let   interval   = 1000;
let   nextPrint  = startTime + interval;

console.log(`determine-komi  agent=${agentName}  size=${size}×${size}  budget=${budget}ms  sharpness=${ALPHA}`);
console.log(`Method: Bayesian logistic model + Thompson sampling`);
console.log(`Output: 1s × 1.5ⁿ  (Ctrl-C to stop)\n`);

while (true) {
  const komi     = thompsonSample();
  const blackWon = playGame(komi);

  updatePosterior(komi, blackWon);

  const emp = getEmp(komi);
  emp.games++;
  if (blackWon) emp.blackWins++;
  totalGames++;

  const now = Date.now();
  if (now >= nextPrint) {
    printStatus(totalGames, (now - startTime) / 1000);
    interval  = Math.round(interval * 1.5);
    nextPrint = now + interval;
  }
}
