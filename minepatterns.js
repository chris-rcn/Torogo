#!/usr/bin/env node
'use strict';

// minepatterns.js — mine 3×3 pattern statistics from recorded games.
//
// Usage: node minepatterns.js --file <path> [--passes <n>]
//   --file    path to game records produced by recordgames.js (required)
//   --passes  number of ELO update passes (default 20)
//
// For every move in each game (excluding passes) the script:
//   1. Enumerates all legal non-true-eye placements as candidates.
//   2. Records a "seen" event for the pattern hash at each candidate.
//   3. Records a "selected" event for the pattern hash of the actual move.
//
// ELO is computed iteratively: the selected pattern "beats" each non-selected
// candidate.  K is divided by the number of candidates so one selection event
// contributes the same total rating change as one standard ELO game.
// Multiple passes are made until ratings settle.
//
// Output: one line per observed pattern hash:
//   <hash>,<selection_ratio>,<seen_count>,<elo>

const fs = require('fs');
const { Game, DEFAULT_KOMI } = require('./game.js');
const { patternHash, MAX_LIBS } = require('./patterns.js');
const { isLadderCaptured } = require('./ladder.js');

const args   = process.argv.slice(2);
const get    = (flag, def) => { const i = args.indexOf(flag); return i !== -1 ? args[i + 1] : def; };

const file   = get('--file', null);
const passes = parseInt(get('--passes', '20'), 10);
const LADDER_SAVE = process.env.LADDER_SAVE === 'true';
const LADDER_KILL = process.env.LADDER_KILL === 'true';

if (!file) {
  console.error('Usage: node minepatterns.js --file <path> [--passes <n>]');
  process.exit(1);
}

const lines = fs.readFileSync(file, 'utf8').trim().split('\n').filter(l => l.trim());

// ── Ladder-aware hash ─────────────────────────────────────────────────────────
//
// LADDER_SAVE and LADDER_KILL are independent binary dimensions appended to
// the base pattern hash, enabling their effectiveness to be measured separately.
//
// LADDER_SAVE: placing at (x,y) is the escape point of a friendly group that
//   is currently in a losing ladder (isLadderCaptured returns true).
//
// LADDER_KILL: placing at (x,y) puts an adjacent enemy group in atari and
//   that group cannot escape the resulting ladder.
//
// Each enabled flag adds one bit to the hash beyond the base hash space.
// The two flags are packed as independent binary digits:
//   offset = saveBit * HASH_SPACE  +  killBit * HASH_SPACE * saveDim
// where saveDim = 2 when LADDER_SAVE is on (1 otherwise), so the encodings
// for the two flags never collide regardless of which combination is active.
//
// Both properties are invariant under D4 symmetry (they depend only on the
// centre point), so the minimum-hash canonicalisation remains consistent.

// Total number of distinct base-hash values = 3^9 × (MAX_LIBS+1)^4
const HASH_SPACE = 19683 * Math.pow(MAX_LIBS + 1, 4);

function isLadderSave(game, x, y, color) {
  const board = game.board;
  for (const [nx, ny] of board.getNeighbors(x, y)) {
    if (board.get(nx, ny) !== color) continue;
    const grp  = board.getGroup(nx, ny);
    const libs = board.getLiberties(grp);
    if (libs.size === 1 && libs.has(`${x},${y}`)) {
      if (isLadderCaptured(game, nx, ny).captured) return true;
    }
  }
  return false;
}

function isLadderKill(game, x, y, color) {
  const g2 = game.clone();
  g2.current = color;
  if (g2.placeStone(x, y) === false) return false;
  for (const [nx, ny] of game.board.getNeighbors(x, y)) {
    const cell = g2.board.get(nx, ny);
    if (cell === null || cell === color) continue;
    const grp = g2.board.getGroup(nx, ny);
    if (grp.length === 0) continue;
    if (g2.board.getLiberties(grp).size === 1) {
      if (isLadderCaptured(g2, nx, ny).captured) return true;
    }
  }
  return false;
}

function computeHash(game, x, y, color) {
  const base = patternHash(game, x, y, color);
  if (!LADDER_SAVE && !LADDER_KILL) return base;
  const saveDim  = LADDER_SAVE ? 2 : 1;
  const saveBit  = LADDER_SAVE && isLadderSave(game, x, y, color) ? 1 : 0;
  const killBit  = LADDER_KILL && isLadderKill(game, x, y, color) ? 1 : 0;
  return base + HASH_SPACE * (saveBit + saveDim * killBit);
}

// Map from patternHash → { seen: number, selected: number }
const stats = new Map();

function bump(hash, selected) {
  let s = stats.get(hash);
  if (!s) { s = { seen: 0, selected: 0 }; stats.set(hash, s); }
  s.seen++;
  if (selected) s.selected++;
}

// Each entry: the selected pattern hash and the array of non-selected hashes.
const decisions = [];

for (let gi = 0; gi < lines.length; gi++) {
  const line   = lines[gi];
  const fields = line.split(',');
  const size   = parseInt(fields[0], 10);
  const moves  = fields.slice(1);

  const g = new Game(size, DEFAULT_KOMI);

  // Collect per-move data; we defer bumping until the winner is known.
  const gameMoves = [];

  // moves[0] is already placed by the constructor; process from moves[1] onward.
  for (let mi = 1; mi < moves.length; mi++) {
    const token = moves[mi];
    if (token === '..') { g.pass(); continue; }

    const color = g.current;
    const board = g.board;
    const N     = size;

    // Decode the actual move.
    const mx = token.charCodeAt(0) - 97;
    const my = token.charCodeAt(1) - 97;

    // Enumerate all legal non-true-eye candidates (excluding the selected move).
    const others = [];
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        if (x === mx && y === my) continue;
        if (board.get(x, y) !== null) continue;
        if (board.isTrueEye(x, y, color)) continue;
        if (board.isSuicide(x, y, color)) continue;
        if (board.isKo(x, y, color, g.koFlag)) continue;
        others.push(computeHash(g, x, y, color));
      }
    }

    // Warn if the selected move is questionable.
    if (board.get(mx, my) !== null)
      process.stderr.write(`WARNING: game ${gi + 1} move ${mi + 1}: selected move ${token} is occupied\n`);
    else if (board.isTrueEye(mx, my, color))
      process.stderr.write(`WARNING: game ${gi + 1} move ${mi + 1}: selected move ${token} is a true eye\n`);
    else if (board.isSuicide(mx, my, color))
      process.stderr.write(`WARNING: game ${gi + 1} move ${mi + 1}: selected move ${token} is suicide\n`);
    else if (board.isKo(mx, my, color, g.koFlag))
      process.stderr.write(`WARNING: game ${gi + 1} move ${mi + 1}: selected move ${token} is ko-illegal\n`);

    const selHash = computeHash(g, mx, my, color);
    gameMoves.push({ color, selHash, others });

    g.placeStone(mx, my);
  }

  // Determine the winner (force scoring if the game did not end naturally).
  if (!g.gameOver) g.endGame();
  const winner = g.scores.black.total > g.scores.white.total ? 'black' : 'white';

  // Only record patterns for moves made by the winning player.
  for (const { color, selHash, others } of gameMoves) {
    if (color !== winner) continue;
    for (const h of others) bump(h, false);
    bump(selHash, true);
    if (others.length > 0) decisions.push({ selected: selHash, others });
  }
}

// ── ELO computation ──────────────────────────────────────────────────────────
//
// For each decision: selected beat every candidate in others.
// K is divided by the number of candidates (N) so that one selection event
// contributes the same total ELO change as one standard Elo game (K=32).
// We snapshot both ratings before updating within a decision (Jacobi step)
// so that the order of opponents doesn't bias the result.

const K = 32;

const elos = new Map();
for (const [hash] of stats) elos.set(hash, 1500);

for (let pass = 0; pass < passes; pass++) {
  for (const { selected, others } of decisions) {
    const Rs = elos.get(selected);
    const N  = others.length;
    const kN = K / N;

    // Accumulate the total delta for the selected pattern, and per-hash deltas
    // for losers, using rating snapshots from the start of this decision.
    let dSelected = 0;
    const dOthers = new Map();

    for (const h of others) {
      const Rh = elos.get(h);
      // Expected score for selected against this opponent.
      const Es    = 1 / (1 + Math.pow(10, (Rh - Rs) / 400));
      const delta = kN * (1 - Es); // selected wins → positive delta
      dSelected  += delta;
      dOthers.set(h, (dOthers.get(h) ?? 0) - delta);
    }

    elos.set(selected, Rs + dSelected);
    for (const [h, d] of dOthers) elos.set(h, elos.get(h) + d);
  }
}

// Output: hash,ratio,seen,elo
for (const [hash, { seen, selected }] of stats) {
  const elo = elos.get(hash) ?? 1500;
  console.log(`${hash},${(selected / seen).toFixed(6)},${seen},${elo.toFixed(1)}`);
}
