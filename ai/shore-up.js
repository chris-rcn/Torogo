'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs (require, process, etc.).
// Wrapped in an IIFE to avoid polluting the global namespace.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

/**
 * Shore-up policy.
 *
 * Extends greedy-capture with three improvements:
 *   1. Capture / Escape prefer the *largest* group (most stones = most points).
 *   2. Shore up — NEW tier between Escape and Threaten: extend own groups that
 *      have exactly 2 liberties to ≥ 3, pre-empting greedy-capture's threaten
 *      cycle before it can put us in atari.
 *   3. Threaten targets the *largest* opponent group.
 *   4. Fallback skips self-atari moves (moves that immediately leave an own
 *      group with 1 liberty).
 *
 * Priority order each turn:
 *   1. Capture   — capture the largest opponent group in atari
 *   2. Escape    — save the largest own group in atari
 *   3. Shore up  — extend own 2-liberty groups to ≥ 3 liberties (largest first)
 *   4. Threaten  — put the largest opponent group into atari
 *   5. Fallback  — random legal non-eye move that avoids self-atari
 *
 * Interface: getMove(game, timeBudgetMs) → { type: 'pass' } | { type: 'place', x, y }
 *   game         - a live Game2 instance (read-only; do not mutate)
 *   timeBudgetMs - ignored (always fast)
 */

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

const { PASS, BLACK, WHITE } = _isNode ? require('../game2.js') : window.Game2;
const Util = _isNode ? require('../util.js') : window.Util;

// ── helpers ────────────────────────────────────────────────────────────────

// Return all groups of `color`, each as { firstIdx, size, libertyCount },
// sorted by size descending (largest first).
function groupsByColor(game2, color) {
  const cells   = game2.cells;
  const cap     = game2.N * game2.N;
  const seen    = new Set();
  const results = [];

  for (let i = 0; i < cap; i++) {
    if (cells[i] !== color) continue;
    const gid = game2._gid[i];
    if (seen.has(gid)) continue;
    seen.add(gid);
    results.push({ firstIdx: i, size: game2._ss[gid], libertyCount: game2._ls[gid] });
  }
  results.sort((a, b) => b.size - a.size);
  return results;
}

// True if any own group in game2 has exactly 1 liberty (we'd be in atari).
function leavesOwnGroupAtari(game2, color) {
  const cells = game2.cells;
  const cap   = game2.N * game2.N;
  const seen  = new Set();

  for (let i = 0; i < cap; i++) {
    if (cells[i] !== color) continue;
    const gid = game2._gid[i];
    if (seen.has(gid)) continue;
    seen.add(gid);
    if (game2._ls[gid] === 1) return true;
  }
  return false;
}

// ── tiers ──────────────────────────────────────────────────────────────────

// Tier 1: capture an opponent group in atari — largest first.
function findCapture(game2) {
  const N   = game2.N;
  const opp = game2.current === BLACK ? WHITE : BLACK;
  for (const g of groupsByColor(game2, opp)) {
    if (g.libertyCount !== 1) continue;
    const libs = game2.groupLibs(g.firstIdx);
    const idx  = libs[0];
    const clone = game2.clone();
    if (clone.play(idx)) return { type: 'place', x: idx % N, y: (idx / N) | 0 };
  }
  return null;
}

// Tier 2: save own group in atari — largest first.
function findEscape(game2) {
  const N     = game2.N;
  const color = game2.current;
  for (const g of groupsByColor(game2, color)) {
    if (g.libertyCount !== 1) continue;
    const libs = game2.groupLibs(g.firstIdx);
    const idx  = libs[0];
    const clone = game2.clone();
    if (clone.play(idx)) return { type: 'place', x: idx % N, y: (idx / N) | 0 };
  }
  return null;
}

// Tier 3: extend own groups at exactly 2 liberties to ≥ 3 — largest first.
function findShoreUp(game2) {
  const N     = game2.N;
  const color = game2.current;
  for (const g of groupsByColor(game2, color)) {
    if (g.libertyCount !== 2) continue;
    const libs = game2.groupLibs(g.firstIdx);
    for (let k = 0; k < libs.length; k++) {
      const idx   = libs[k];
      const clone = game2.clone();
      if (!clone.play(idx)) continue;
      // Only worthwhile if the extended group genuinely gains liberties.
      const newGid = clone._gid[idx];
      if (newGid !== -1 && clone._ls[newGid] >= 3)
        return { type: 'place', x: idx % N, y: (idx / N) | 0 };
    }
  }
  return null;
}

// Tier 4: put the largest opponent group into atari.
function findThreat(game2, candidates) {
  const N   = game2.N;
  const opp = game2.current === BLACK ? WHITE : BLACK;
  let bestMove = null;
  let bestSize = -1;

  for (const idx of candidates) {
    const clone = game2.clone();
    if (!clone.play(idx)) continue;

    const cells = clone.cells;
    const cap   = N * N;
    const seen  = new Set();
    for (let i = 0; i < cap; i++) {
      if (cells[i] !== opp) continue;
      const gid = clone._gid[i];
      if (seen.has(gid)) continue;
      seen.add(gid);
      if (clone._ls[gid] === 1 && clone._ss[gid] > bestSize) {
        bestSize = clone._ss[gid];
        bestMove = { type: 'place', x: idx % N, y: (idx / N) | 0 };
      }
    }
  }
  return bestMove;
}

// Tier 5: random legal non-eye move, skipping self-atari.
function findRandom(game2, candidates) {
  const N     = game2.N;
  const color = game2.current;
  for (const idx of candidates) {
    const clone = game2.clone();
    if (!clone.play(idx)) continue;
    if (leavesOwnGroupAtari(clone, color)) continue;
    return { type: 'place', x: idx % N, y: (idx / N) | 0 };
  }
  // If every legal move leaves us in atari, accept the least-bad option.
  for (const idx of candidates) {
    const clone = game2.clone();
    if (clone.play(idx)) return { type: 'place', x: idx % N, y: (idx / N) | 0 };
  }
  return null;
}

// ── main ───────────────────────────────────────────────────────────────────

function getMove(game, _timeBudgetMs) {
  if (game.gameOver) return { type: 'pass' };

  const game2 = game.cells ? game.clone() : game.toGame2();
  const N     = game2.N;
  const color = game2.current;

  const capture = findCapture(game2);
  if (capture) return capture;

  const escape = findEscape(game2);
  if (escape) return escape;

  const shoreUp = findShoreUp(game2);
  if (shoreUp) return shoreUp;

  // Build shuffled candidate list for tiers 4 & 5.
  const candidates = [];
  const cells = game2.cells;
  for (let i = 0; i < N * N; i++) {
    if (cells[i] !== 0) continue;
    if (game2.isTrueEye(i)) continue;
    candidates.push(i);
  }
  Util.shuffle(candidates);

  return findThreat(game2, candidates)
      || findRandom(game2, candidates)
      || { type: 'pass' };
}

if (typeof module !== 'undefined') module.exports = { getMove };
else window.getMove = getMove;

})();
