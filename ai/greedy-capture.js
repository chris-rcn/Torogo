'use strict';

/**
 * Greedy-capture policy.
 *
 * Priority order each turn:
 *   1. Capture  — play on the single liberty of any opponent group in atari.
 *                 Always legal (capturing relieves suicide), no clone needed.
 *   2. Escape   — play on the single liberty of any own group in atari.
 *                 One clone per candidate to verify legality.
 *   3. Threaten — play the first shuffled candidate that puts an opponent
 *                 group into atari (reduces it to 1 liberty).
 *   4. Fallback — random legal non-eye move (same as random).
 *
 * Interface: getMove(game) → { type: 'pass' } | { type: 'place', x, y }
 *   game  - a live Game instance (read-only; do not mutate)
 */

// ── helpers ────────────────────────────────────────────────────────────────

function isTrueEye(board, x, y, color) {
  const N = board.size;
  const ortho = board.getNeighbors(x, y);
  if (!ortho.every(([nx, ny]) => board.get(nx, ny) === color)) return false;
  const diags = [
    [(x + 1) % N,     (y + 1) % N],
    [(x - 1 + N) % N, (y + 1) % N],
    [(x + 1) % N,     (y - 1 + N) % N],
    [(x - 1 + N) % N, (y - 1 + N) % N],
  ];
  return diags.filter(([dx, dy]) => board.get(dx, dy) === color).length >= 3;
}

function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

// Scan all groups of `color` on board; return a map from liberty-key to group.
// Only includes groups with exactly `targetLibs` liberties.
function groupsWithLibCount(board, color, targetLibs) {
  const visited = new Set();
  const results = []; // [{ liberty: [x,y], group }]
  const N = board.size;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const key = `${x},${y}`;
      if (board.get(x, y) !== color || visited.has(key)) continue;
      const group = board.getGroup(x, y);
      group.forEach(([gx, gy]) => visited.add(`${gx},${gy}`));
      const libs = board.getLiberties(group);
      if (libs.size === targetLibs) {
        const [lx, ly] = [...libs][0].split(',').map(Number);
        results.push([lx, ly]);
      }
    }
  }
  return results;
}

// ── tiers ──────────────────────────────────────────────────────────────────

// Tier 1: capture an opponent group in atari.
function findCapture(game) {
  const opp = game.current === 'black' ? 'white' : 'black';
  const moves = groupsWithLibCount(game.board, opp, 1);
  // Shuffle so we don't always take the top-left capture.
  shuffle(moves);
  for (const [x, y] of moves) {
    // Capturing an atari group is always legal, but double-check for Ko.
    const clone = game.clone();
    if (clone.placeStone(x, y)) return { type: 'place', x, y };
  }
  return null;
}

// Tier 2: save own group in atari.
function findEscape(game) {
  const color = game.current;
  const moves = groupsWithLibCount(game.board, color, 1);
  shuffle(moves);
  for (const [x, y] of moves) {
    const clone = game.clone();
    if (clone.placeStone(x, y)) return { type: 'place', x, y };
  }
  return null;
}

// Tier 3: threaten — first candidate that puts an opponent group into atari.
function findThreat(game, candidates) {
  const opp = game.current === 'black' ? 'white' : 'black';
  for (const [x, y] of candidates) {
    const clone = game.clone();
    if (!clone.placeStone(x, y)) continue;
    // Check whether any opponent group now has exactly 1 liberty.
    const visited = new Set();
    let createsAtari = false;
    const N = clone.board.size;
    outer:
    for (let gy = 0; gy < N; gy++) {
      for (let gx = 0; gx < N; gx++) {
        const key = `${gx},${gy}`;
        if (clone.board.get(gx, gy) !== opp || visited.has(key)) continue;
        const group = clone.board.getGroup(gx, gy);
        group.forEach(([ax, ay]) => visited.add(`${ax},${ay}`));
        if (clone.board.getLiberties(group).size === 1) {
          createsAtari = true;
          break outer;
        }
      }
    }
    if (createsAtari) return { type: 'place', x, y };
  }
  return null;
}

// Tier 4: random legal non-eye move.
function findRandom(game, candidates) {
  for (const [x, y] of candidates) {
    const clone = game.clone();
    if (clone.placeStone(x, y)) return { type: 'place', x, y };
  }
  return null;
}

// ── main ───────────────────────────────────────────────────────────────────

module.exports = function getMove(game) {
  if (game.gameOver) return { type: 'pass' };

  const N = game.boardSize;
  const color = game.current;

  // Tier 1 & 2 don't need the candidate list.
  const capture = findCapture(game);
  if (capture) return capture;

  const escape = findEscape(game);
  if (escape) return escape;

  // Build shuffled candidate list for tiers 3 & 4.
  const candidates = [];
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      if (game.board.get(x, y) !== null) continue;
      if (isTrueEye(game.board, x, y, color)) continue;
      candidates.push([x, y]);
    }
  }
  shuffle(candidates);

  return findThreat(game, candidates)
      || findRandom(game, candidates)
      || { type: 'pass' };
};
