'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.
// Wrapped in an IIFE to avoid polluting the global namespace.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

/**
 * Pattern-value policy.
 *
 * Full-width single-ply search using a learned linear value function:
 *   V(s) = σ( Σ  polarity_i · w[key_i] )  =  P(BLACK wins)
 *
 * BLACK maximises V(s'), WHITE minimises V(s').
 *
 * Interface: getMove(game, timeBudgetMs) → { type: 'pass' } | { type: 'place', x, y }
 *   game         - a live Game2 instance (read-only; do not mutate)
 *   timeBudgetMs - ignored (search is full-width, not time-bounded)
 *
 * Weights are read from the path in the PATTERN_WEIGHTS environment variable,
 * or from pattern-weights.json in the project root if the variable is unset.
 * In a browser context, supply weights via window.patternWeights (plain object).
 */

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
const { BLACK, WHITE, PASS } = _isNode ? require('../game2.js') : window.Game2;
const { pattern1, pattern2, pattern3 } = _isNode ? require('../patterns.js') : window.Patterns;

const MAX_LIBS = 1;

// ── Load weights ──────────────────────────────────────────────────────────────

const weights = new Map();  // pattern key (int32) → weight (float)

(function loadWeights() {
  let raw = null;
  if (_isNode) {
    const fs   = require('fs');
    const path = require('path');
    const file = process.env.PATTERN_WEIGHTS ||
                 path.join(__dirname, '..', 'pattern-weights.json');
    if (fs.existsSync(file)) raw = JSON.parse(fs.readFileSync(file, 'utf8'));
  } else if (typeof window.patternWeights !== 'undefined') {
    raw = window.patternWeights;
  }
  if (raw) {
    for (const [k, v] of Object.entries(raw)) weights.set(Number(k), v);
  }
})();

// ── Value function ────────────────────────────────────────────────────────────

function extractFeatures(game) {
  const cap = game.N * game.N;
  const out = [];
  for (let i = 0; i < cap; i++) {
    let f;
    f = pattern1(game, MAX_LIBS, i); if (f !== null) out.push(f);
    f = pattern2(game, MAX_LIBS, i); if (f !== null) out.push(f);
    f = pattern3(game, MAX_LIBS, i); if (f !== null) out.push(f);
  }
  return out;
}

function valueOf(features) {
  let z = 0;
  for (const f of features) {
    const w = weights.get(f.key);
    if (w !== undefined) z += f.polarity * w;
  }
  return 1 / (1 + Math.exp(-z));
}

function absoluteOutcome(game) {
  const w = game.calcWinner();
  return w === BLACK ? 1 : w === WHITE ? 0 : 0.5;
}

// ── Move selection ────────────────────────────────────────────────────────────

function getMove(game, _timeBudgetMs) {
  if (game.gameOver) return { type: 'pass', move: PASS };

  const N       = game.N;
  const cap     = N * N;
  const isBlack = game.current === BLACK;

  function scoreMove(idx) {
    const g = game.clone();
    g.play(idx);
    return g.gameOver ? absoluteOutcome(g) : valueOf(extractFeatures(g));
  }

  let bestIdx   = PASS;
  let bestScore = isBlack ? -Infinity : Infinity;

  function isBetter(s) { return isBlack ? s > bestScore : s < bestScore; }

  for (let i = 0; i < cap; i++) {
    if (game.cells[i] !== 0) continue;
    if (game.isTrueEye(i))   continue;
    if (!game.isLegal(i))    continue;
    const s = scoreMove(i);
    if (isBetter(s)) { bestScore = s; bestIdx = i; }
  }

  if (isBetter(scoreMove(PASS))) bestIdx = PASS;

  if (bestIdx === PASS) return { type: 'pass', move: PASS };
  return { type: 'place', move: bestIdx, x: bestIdx % N, y: (bestIdx / N) | 0 };
}

if (typeof module !== 'undefined') module.exports = { getMove };
else window.getMove = getMove;

})();
