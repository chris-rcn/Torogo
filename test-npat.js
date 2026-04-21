'use strict';

// Tests for npat-lib.js — the nine-pattern policy features.

const NPat = require('./npat-lib.js');
const { Game2, BLACK, WHITE, PASS } = require('./game2.js');

let passed = 0, failed = 0;
function check(label, ok) {
  if (ok) passed++;
  else    { failed++; console.error('FAIL:', label); }
}

// ── 1. Pattern-space accounting ──────────────────────────────────────────────
//
// Raw pattern space = 9 positions × 3^9 = 177,147 encoded entries, but only
// those with cells[relPos] === EMPTY are valid (9 × 3^8 = 59,049 valid raws).
// NUM_PATTERNS must be ≤ 59,049 and ≥ 59,049 / 8  ≈  7,381.

check(
  'NUM_PATTERNS within plausible bounds',
  NPat.NUM_PATTERNS > 7000 && NPat.NUM_PATTERNS < 59049
);

// ── 2. D4 equivalence via the canonical table ────────────────────────────────
//
// For each of the 8 D4 symmetries, applying σ to a (relPos, cells) pair must
// yield the same canonical ID.

function packRaw(relPos, cells) {
  let enc = 0;
  for (let i = 8; i >= 0; i--) enc = enc * 3 + cells[i];
  return relPos * NPat._CELLS_BASE + enc;
}

function applyD4(relPos, cells, perm) {
  const nc = new Array(9);
  for (let i = 0; i < 9; i++) nc[perm[i]] = cells[i];
  const nRelPos = perm[relPos];
  return { relPos: nRelPos, cells: nc };
}

// A few non-trivial patterns to sanity-check.
const cases = [
  { relPos: 4, cells: [0,1,0, 0,0,0, 0,2,0] },  // friend N, foe S, candidate center
  { relPos: 0, cells: [0,1,2, 2,1,0, 1,0,2] },  // candidate top-left
  { relPos: 8, cells: [1,0,0, 0,0,0, 0,0,0] },  // candidate bottom-right, friend top-left
  { relPos: 2, cells: [0,0,0, 1,2,1, 0,0,2] },  // candidate top-right
];

for (const c of cases) {
  const baseRaw = packRaw(c.relPos, c.cells);
  const baseId  = NPat._CANON_ID[baseRaw];
  check(`candidate cell is empty: relPos=${c.relPos}`, c.cells[c.relPos] === 0);
  check(`base canonical id ≥ 0 for relPos=${c.relPos}`, baseId >= 0);

  for (let di = 0; di < 8; di++) {
    const perm = NPat._D4[di];
    const t = applyD4(c.relPos, c.cells, perm);
    const tRaw = packRaw(t.relPos, t.cells);
    const tId  = NPat._CANON_ID[tRaw];
    check(`D4 symmetry ${di} preserves canonical ID (case ${c.relPos}/${c.cells.join('')})`,
      tId === baseId);
  }
}

// ── 3. Invalid raws (candidate cell non-empty) are marked ───────────────────
//
// We mark impossible patterns with CANON_ID = -1.  Verify by construction.
{
  let anyInvalid = false;
  for (let raw = 0; raw < NPat._CANON_ID.length; raw += 1000) {
    const relPos = (raw / NPat._CELLS_BASE) | 0;
    let packed = raw - relPos * NPat._CELLS_BASE;
    // Decode cells[relPos].
    let p = packed;
    let candCell = 0;
    for (let i = 0; i <= relPos; i++) {
      candCell = p % 3;
      p = (p / 3) | 0;
    }
    if (candCell !== 0) {
      anyInvalid = true;
      check(`invalid raw ${raw} gets CANON_ID = -1`, NPat._CANON_ID[raw] === -1);
    } else {
      check(`valid raw ${raw} gets CANON_ID ≥ 0`, NPat._CANON_ID[raw] >= 0);
    }
  }
  check('scanned some invalid raws', anyInvalid);
}

// ── 4. extractFeatures on an empty board ────────────────────────────────────
//
// Empty 5×5 board.  Every move should have 9 pattern IDs, and all of them
// should be the same canonical id (the all-EMPTY pattern), because all 9
// windows around every candidate are entirely empty.
{
  const N  = 5;
  const g  = new Game2(N, false);
  const st = NPat.createState(N);
  NPat.extractFeatures(g, st);

  // All 5×5 empty cells are legal non-eye moves.  (isTrueEye requires 4
  // friendly neighbours; there are no stones → emptyCount stays 25.)
  check('empty 5x5: count === 25', st.count === 25);

  // For the all-empty cell case, cells[9] are all zero; relPos varies 0..8.
  // The canonical ID should collapse all nine relPos values via D4 to just
  // two orbits (center, edge, corner).  Specifically, relPos ∈ {0,2,6,8} are
  // D4-equivalent; {1,3,5,7} are D4-equivalent; {4} is alone.  So each move's
  // 9 windows have exactly 3 distinct canonical IDs.
  const m0 = st.patIds.subarray(0, 9);
  const distinct = new Set();
  for (const id of m0) distinct.add(id);
  check('empty board, per-move window ids: exactly 3 distinct', distinct.size === 3);

  // Count occurrences per id: corner orbit (4), edge orbit (4), center (1).
  const counts = new Map();
  for (const id of m0) counts.set(id, (counts.get(id) ?? 0) + 1);
  const vals = [...counts.values()].sort((a, b) => a - b);
  check('empty board, occurrence counts are [1,4,4]', JSON.stringify(vals) === '[1,4,4]');

  // Every move produces the same multiset of 9 pattern IDs.
  let allSame = true;
  for (let i = 1; i < st.count; i++) {
    const a = st.patIds.subarray(i * 9, (i + 1) * 9);
    const sa = [...a].sort();
    const sm = [...m0].sort();
    if (sa.join(',') !== sm.join(',')) { allSame = false; break; }
  }
  check('empty board, every move has the same pattern multiset', allSame);
}

// ── 5. extractFeatures reacts to stones ─────────────────────────────────────
//
// Place a black stone and confirm that candidate moves immediately next to it
// have different pattern IDs than moves far away.
{
  const N  = 7;
  const g  = new Game2(N, false);
  const st = NPat.createState(N);
  // Drop a black stone at (3,3).
  g._place(3 * N + 3, BLACK);
  g.current = WHITE;                // White to move → black stone appears as FOE
  NPat.extractFeatures(g, st);

  // Find candidate moves: (3,4) is adjacent to the stone, (0,0) is far away.
  let adjIdx = -1, farIdx = -1;
  for (let i = 0; i < st.count; i++) {
    if (st.moves[i] === 3 * N + 4) adjIdx = i;
    if (st.moves[i] === 0)         farIdx = i;
  }
  check('stone test: adjacent candidate found', adjIdx >= 0);
  check('stone test: far candidate found',      farIdx >= 0);

  const adjIds = [...st.patIds.subarray(adjIdx * 9, (adjIdx + 1) * 9)].sort();
  const farIds = [...st.patIds.subarray(farIdx * 9, (farIdx + 1) * 9)].sort();
  check('stone test: adjacent move differs from far move (toroidal: stone only touches its 3×3 neighbourhood, and (0,0) is not in it)',
    adjIds.join(',') !== farIds.join(','));
}

// ── 6. Board symmetry: D4 of the board yields D4 of the features ────────────
//
// Empty board with one black stone at (1,0) on a 5×5 torus.  Now pick
// candidate (0,0) and its rot180 (4,0)   — wait, that's not general.  Easier:
// two boards related by a D4 rotation should give identical feature multisets
// at the two D4-related candidate positions.
{
  const N = 5;
  // Board A: black stone at (1, 2).   Candidate (0, 2) sits directly above.
  // Board B: 180°-rotated board — stone at (3, 2).  Candidate (4, 2).
  const gA = new Game2(N, false);
  gA._place(1 * N + 2, BLACK);
  gA.current = WHITE;

  const gB = new Game2(N, false);
  gB._place(3 * N + 2, BLACK);
  gB.current = WHITE;

  const stA = NPat.createState(N);
  const stB = NPat.createState(N);
  NPat.extractFeatures(gA, stA);
  NPat.extractFeatures(gB, stB);

  // Find the two candidates.
  function ids(state, idx) {
    for (let i = 0; i < state.count; i++) if (state.moves[i] === idx)
      return [...state.patIds.subarray(i * 9, (i + 1) * 9)].sort();
    return null;
  }
  const a = ids(stA, 0 * N + 2);
  const b = ids(stB, 4 * N + 2);
  check('D4-symmetric positions have D4-symmetric features: both candidates found',
    a !== null && b !== null);
  check('D4-symmetric positions have identical canonical-id multisets',
    JSON.stringify(a) === JSON.stringify(b));
}

// ── 7. policyMove returns a legal move on a fresh board ──────────────────────
{
  const N = 5;
  const g = new Game2(N, false);
  const st = NPat.createState(N);
  const w = new Map();   // empty weights: uniform policy
  const { move, index, prob } = NPat.policyMove(g, st, w);
  check('policyMove: returns a legal move', move !== PASS && g.isLegal(move));
  check('policyMove: returns a valid index', index >= 0 && index < st.count);
  check('policyMove: uniform prob ≈ 1/count', Math.abs(prob - 1 / st.count) < 1e-9);
}

// ── 8. REINFORCE update: positive advantage boosts chosen move's logit ──────
//
// On a featureless board (empty 5×5 torus) every move has the same pattern
// multiset, so the policy gradient is exactly zero by symmetry.  Break the
// symmetry by placing a stone; then updating any move must increase its logit
// relative to the others.
{
  const N = 5;
  const g = new Game2(N, false);
  g._place(1 * N + 2, BLACK);
  g.current = WHITE;
  const st = NPat.createState(N);
  const w = new Map();

  // populate features + probs
  NPat.policyMove(g, st, w);
  // Pick a move whose features differ from at least one other move (so the
  // gradient is non-trivial).  Index 0 is fine unless it's in a degenerate
  // equivalence class; stepping through the list guarantees we find one.
  let chosen = -1;
  for (let i = 0; i < st.count; i++) {
    const ai = [...st.patIds.subarray(i * 9, (i + 1) * 9)].sort().join(',');
    for (let j = 0; j < st.count; j++) {
      if (j === i) continue;
      const aj = [...st.patIds.subarray(j * 9, (j + 1) * 9)].sort().join(',');
      if (ai !== aj) { chosen = i; break; }
    }
    if (chosen >= 0) break;
  }
  check('REINFORCE test setup: found a non-symmetric move', chosen >= 0);
  const beforeScore = (() => {
    const off = chosen * 9;
    let s = 0;
    for (let k = 0; k < 9; k++) s += w.get(st.patIds[off + k]) || 0;
    return s;
  })();

  NPat.reinforceUpdate(st, chosen, +1, w, 0.1);

  const afterScore = (() => {
    const off = chosen * 9;
    let s = 0;
    for (let k = 0; k < 9; k++) s += w.get(st.patIds[off + k]) || 0;
    return s;
  })();

  check('REINFORCE(+1): chosen move logit increases', afterScore > beforeScore);

  // Also verify probability of chosen move now exceeds 1/n after the update.
  NPat.extractFeatures(g, st);
  // recompute softmax manually using updated weights
  const logits = new Float64Array(st.count);
  for (let i = 0; i < st.count; i++) {
    const off = i * 9;
    let s = 0;
    for (let k = 0; k < 9; k++) s += w.get(st.patIds[off + k]) || 0;
    logits[i] = s;
  }
  let maxL = -Infinity;
  for (let i = 0; i < st.count; i++) if (logits[i] > maxL) maxL = logits[i];
  let sum = 0;
  const probs = new Float64Array(st.count);
  for (let i = 0; i < st.count; i++) { probs[i] = Math.exp(logits[i] - maxL); sum += probs[i]; }
  for (let i = 0; i < st.count; i++) probs[i] /= sum;

  check('REINFORCE(+1): chosen move probability increases', probs[chosen] > 1 / st.count);
}

// ── Report ──────────────────────────────────────────────────────────────────

console.log(`npat: ${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
