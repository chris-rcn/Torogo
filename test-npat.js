'use strict';

// Tests for npat-lib.js — nine-pattern (3-state shape) policy features plus
// four orthogonal stacking tactical features.

const NPat = require('./npat-lib.js');
const { Game2, BLACK, WHITE, PASS, parseBoard } = require('./game2.js');

let passed = 0, failed = 0;
function check(label, ok) {
  if (ok) passed++;
  else    { failed++; console.error('FAIL:', label); }
}

const {
  CELL_EMPTY, CELL_FRIEND, CELL_FOE,
  CELL_BASE,  CELLS_BASE,
  N_TACT, TACT_STONE_LIMIT, N_TACT_SLOTS,
  TACT_URGENT_KILL, TACT_URGENT_SAVE,
  TACT_WASTED_EXTEND, TACT_WASTED_ATTACK,
  TACT_RAW_BASE,
  SHAPE33C_RAW_BASE,
  TYPE_A_CELLS, TYPE_A_BASE, TYPE_A_RAW_BASE,
  TYPE_B_CELLS, TYPE_B_BASE, TYPE_B_RAW_BASE,
  canonKey, canonKeyA, canonKeyB, _D4,
} = NPat;

check('CELL_BASE is 3', CELL_BASE === 3);
check('CELLS_BASE is 3^9', CELLS_BASE === 19683);
check('N_TACT is 4', N_TACT === 4);
check('TACT_RAW_BASE is 9 * CELLS_BASE', TACT_RAW_BASE === 9 * 19683);
check('TACT_STONE_LIMIT is 8', TACT_STONE_LIMIT === 8);
check('N_TACT_SLOTS is N_TACT * TACT_STONE_LIMIT',
  N_TACT_SLOTS === N_TACT * TACT_STONE_LIMIT);
check('SHAPE33C_RAW_BASE above tactical block',
  SHAPE33C_RAW_BASE === TACT_RAW_BASE + N_TACT_SLOTS);
check('TYPE_A_CELLS is 10', TYPE_A_CELLS === 10);
check('TYPE_A_BASE is 3^10', TYPE_A_BASE === 59049);
check('TYPE_B_CELLS is 11', TYPE_B_CELLS === 11);
check('TYPE_B_BASE is 3^11', TYPE_B_BASE === 177147);

// Helper: sum the stone-index sub-counts for a (cell, feature-type) pair.
// Equal to min(chain_size, LIMIT) per qualifying chain, so it matches the
// old "count of qualifying chains" semantics when every chain is size 1.
function tactSum(buf, idx, k) {
  const off = idx * N_TACT_SLOTS + k * TACT_STONE_LIMIT;
  let s = 0;
  for (let j = 0; j < TACT_STONE_LIMIT; j++) s += buf[off + j];
  return s;
}

// ── 1. canonKey respects D4 symmetry ────────────────────────────────────────
//
// Applying any D4 permutation σ to (relPos, cells) must yield the same
// canonical key: σ moves both the candidate's relative position and the cell
// values, so the canonical-min is invariant.

function applyD4(relPos, cells, perm) {
  const nc = new Array(9);
  for (let i = 0; i < 9; i++) nc[perm[i]] = cells[i];
  return { relPos: perm[relPos], cells: nc };
}

const cases = [
  { relPos: 4, cells: [0,1,0, 0,0,0, 0,2,0] },  // candidate center; friend N, foe S
  { relPos: 0, cells: [0,1,2, 2,1,0, 1,0,2] },  // candidate top-left, full stones
  { relPos: 8, cells: [1,0,0, 0,0,0, 0,0,0] },  // candidate bottom-right; friend at top-left
  { relPos: 2, cells: [0,0,0, 1,2,1, 0,0,2] },  // candidate top-right
  { relPos: 1, cells: [0,0,0, 2,1,2, 0,1,0] },  // mixed
];

for (const c of cases) {
  check(`case sanity: candidate cell is empty (relPos=${c.relPos})`,
    c.cells[c.relPos] === CELL_EMPTY);

  const baseKey = canonKey(c.relPos, c.cells);
  for (let di = 0; di < 8; di++) {
    const t = applyD4(c.relPos, c.cells, _D4[di]);
    const tKey = canonKey(t.relPos, t.cells);
    check(`D4 symmetry ${di} preserves canonical key (case relPos=${c.relPos})`,
      tKey === baseKey);
  }
}

// ── 2. Distinct patterns get distinct canonical keys (barring symmetry) ─────
{
  const k1 = canonKey(4, [0,0,0, 0,0,0, 0,0,0]);    // all empty
  const k2 = canonKey(4, [0,0,0, 0,0,0, 1,0,0]);    // one friend at corner
  const k3 = canonKey(4, [0,0,0, 0,0,0, 2,0,0]);    // one foe at corner
  check('all-empty pattern differs from single-friend pattern', k1 !== k2);
  check('single-friend pattern differs from single-foe pattern', k2 !== k3);

  // Four symmetric corner-friend patterns all collapse to the same key.
  check('corner NW friend ≡ corner NE friend', canonKey(4, [1,0,0, 0,0,0, 0,0,0]) === canonKey(4, [0,0,1, 0,0,0, 0,0,0]));
  check('corner NW friend ≡ corner SW friend', canonKey(4, [1,0,0, 0,0,0, 0,0,0]) === canonKey(4, [0,0,0, 0,0,0, 1,0,0]));
  check('corner NW friend ≡ corner SE friend', canonKey(4, [1,0,0, 0,0,0, 0,0,0]) === canonKey(4, [0,0,0, 0,0,0, 0,0,1]));
}

// ── 3. Empty 5×5 board — extractFeatures count and zero tactical ───────────
{
  const N  = 5;
  const g  = new Game2(N, false);
  const st = NPat.createState(N);
  NPat.extractFeatures(g, st);

  check('empty 5x5: count === 25', st.count === 25);
  let anyTact = false;
  for (let i = 0; i < st.count * N_TACT_SLOTS; i++) if (st.tact[i]) { anyTact = true; break; }
  check('empty board: all tactical counts are zero', !anyTact);
}

// ── 4. Ladder annotation: chain in atari sets URGENT_KILL at its liberty ────
//
// Set up a black chain in atari with exactly one liberty.  White's move.
// The liberty is urgent for white (playing there captures a foe chain).
{
  const N = 5;
  const g = new Game2(N, false);
  g._place(2 * N + 2, BLACK); // black at (2,2)
  g._place(1 * N + 2, WHITE); // white at (1,2)
  g._place(3 * N + 2, WHITE); // white at (3,2)
  g._place(2 * N + 3, WHITE); // white at (2,3)
  g.current = WHITE;
  const st = NPat.createState(N);
  NPat.extractFeatures(g, st);

  const libIdx = 2 * N + 1; // (2,1), the remaining liberty

  // For a size-1 chain, URGENT_KILL fires STONE_0 once at the liberty.
  // tactSum returns min(chain_size, LIMIT) per qualifying chain so it matches
  // the old "count of qualifying chains" semantics for size-1 chains.
  const tc = st.ladder.tactCount;
  check('ladder annotation: liberty cell marked URGENT_KILL',
    tactSum(tc, libIdx, TACT_URGENT_KILL) === 1);
  check('ladder annotation: liberty cell not marked URGENT_SAVE',
    tactSum(tc, libIdx, TACT_URGENT_SAVE) === 0);
  check('ladder annotation: liberty cell not marked WASTED_EXTEND',
    tactSum(tc, libIdx, TACT_WASTED_EXTEND) === 0);
  check('ladder annotation: liberty cell not marked WASTED_ATTACK',
    tactSum(tc, libIdx, TACT_WASTED_ATTACK) === 0);

  // The per-candidate tact buffer should reflect the same values.
  let urgIdx = -1;
  for (let i = 0; i < st.count; i++) if (st.moves[i] === libIdx) { urgIdx = i; break; }
  check('urgent liberty candidate present', urgIdx >= 0);
  check('candidate tact[URGENT_KILL] is 1',
    tactSum(st.tact, urgIdx, TACT_URGENT_KILL) === 1);

  // A far candidate has all-zero tactical counts.
  let farIdx = -1;
  for (let i = 0; i < st.count; i++) if (st.moves[i] === 0 * N + 0) { farIdx = i; break; }
  check('far candidate present', farIdx >= 0);
  let farZero = true;
  for (let k = 0; k < N_TACT_SLOTS; k++) if (st.tact[farIdx * N_TACT_SLOTS + k]) { farZero = false; break; }
  check('far candidate: all tactical counts zero', farZero);
}

// ── 5. Friend-in-atari (from mover perspective) sets URGENT_SAVE ────────────
{
  const N = 5;
  const g = new Game2(N, false);
  g._place(2 * N + 2, BLACK);
  g._place(1 * N + 2, WHITE);
  g._place(3 * N + 2, WHITE);
  g._place(2 * N + 3, WHITE);
  g.current = BLACK; // now mover is the chain owner

  const st = NPat.createState(N);
  NPat.extractFeatures(g, st);
  const libIdx = 2 * N + 1;

  // moverSucceeds may be false here (1 lib with neighbours hemming it in),
  // in which case the liberty is marked WASTED_EXTEND instead of URGENT_SAVE.
  // We just assert: exactly one of {URGENT_SAVE, WASTED_EXTEND} is set, and
  // the KILL / ATTACK slots are zero.
  const tc = st.ladder.tactCount;
  const save   = tactSum(tc, libIdx, TACT_URGENT_SAVE);
  const wExt   = tactSum(tc, libIdx, TACT_WASTED_EXTEND);
  const kill   = tactSum(tc, libIdx, TACT_URGENT_KILL);
  const wAtk   = tactSum(tc, libIdx, TACT_WASTED_ATTACK);
  check('friend-in-atari: either URGENT_SAVE or WASTED_EXTEND fires',
    (save === 1 && wExt === 0) || (save === 0 && wExt === 1));
  check('friend-in-atari: URGENT_KILL is 0', kill === 0);
  check('friend-in-atari: WASTED_ATTACK is 0', wAtk === 0);
}

// ── 6. D4 board symmetry → 3×3c feature symmetry ────────────────────────────
{
  const N = 5;
  const gA = new Game2(N, false);
  gA._place(1 * N + 2, BLACK);
  gA.current = WHITE;
  const gB = new Game2(N, false);
  gB._place(3 * N + 2, BLACK); // 180°-rotated
  gB.current = WHITE;

  const stA = NPat.createState(N);
  const stB = NPat.createState(N);
  NPat.extractFeatures(gA, stA);
  NPat.extractFeatures(gB, stB);

  function pid(state, idx) {
    for (let i = 0; i < state.count; i++) if (state.moves[i] === idx)
      return state.patIds33c[i];
    return -1;
  }
  const a = pid(stA, 0 * N + 2);
  const b = pid(stB, 4 * N + 2);
  check('D4-symmetric positions: both candidates found', a !== -1 && b !== -1);
  check('D4-symmetric positions: 3×3c canonical key matches', a === b);
}

// ── 7. policyMove returns a legal move with uniform prob on empty weights ───
{
  const N = 5;
  const g = new Game2(N, false);
  const st = NPat.createState(N);
  const w = NPat.createWeights({ use33c: true, useA: true, useB: true });
  const { move, index, prob } = NPat.policyMove(g, st, w);
  check('policyMove: returns a legal move', move !== PASS && g.isLegal(move));
  check('policyMove: returns a valid index', index >= 0 && index < st.count);
  check('policyMove: uniform prob ≈ 1/count', Math.abs(prob - 1 / st.count) < 1e-9);
}

// ── 8. REINFORCE update boosts chosen move once symmetry is broken ──────────
{
  const N = 5;
  const g = new Game2(N, false);
  g._place(1 * N + 2, BLACK);
  g.current = WHITE;
  const st = NPat.createState(N);
  const w = NPat.createWeights({ use33c: true, useA: true, useB: true });

  NPat.policyMove(g, st, w);
  // Pick any move whose 3×3c canonical key differs from move 0's (they will,
  // because the board is broken-symmetric).
  let chosen = -1;
  for (let i = 1; i < st.count; i++) {
    if (st.patIds33c[i] !== st.patIds33c[0]) { chosen = i; break; }
  }
  check('REINFORCE setup: found a non-symmetric move', chosen >= 0);

  function scoreShape(i) {
    return w.vals[st.patIds33c[i]] + w.vals[st.patIdsA[i]] + w.vals[st.patIdsB[i]];
  }
  const before = scoreShape(chosen);
  NPat.reinforceUpdate(st, chosen, +1, w, 0.1);
  const after = scoreShape(chosen);
  check('REINFORCE(+1): chosen move shape logit increases', after > before);
}

// ── 9. No urgent chains on a quiet board → all tactical counts zero ────────
{
  const N = 7;
  const g = new Game2(N, false);
  g._place(3 * N + 3, BLACK); // black stone with 4 liberties, not in a ladder
  g.current = WHITE;

  const ladder = NPat.annotateLadders(g);
  let any = false;
  for (let i = 0; i < N * N * N_TACT_SLOTS; i++) if (ladder.tactCount[i]) { any = true; break; }
  check('quiet board (chain with 4 libs): no tactical bits set', !any);
}

// ── 10. Rotated atari → matching shape multisets AND matching tact counts ──
{
  const N = 7;
  // Configuration A: black in atari at (2,2), surrounded N, E, S by white.
  const gA = new Game2(N, false);
  gA._place(2 * N + 2, BLACK);
  gA._place(1 * N + 2, WHITE);
  gA._place(3 * N + 2, WHITE);
  gA._place(2 * N + 3, WHITE);
  gA.current = WHITE;

  // Configuration B: the 180°-rotation of A.  (2,2) → (4,4).
  const gB = new Game2(N, false);
  gB._place(4 * N + 4, BLACK);
  gB._place(5 * N + 4, WHITE);
  gB._place(3 * N + 4, WHITE);
  gB._place(4 * N + 3, WHITE);
  gB.current = WHITE;

  const stA = NPat.createState(N);
  const stB = NPat.createState(N);
  NPat.extractFeatures(gA, stA);
  NPat.extractFeatures(gB, stB);

  const libA = 2 * N + 1;
  const libB = 4 * N + 5;

  function candIdx(state, idx) {
    for (let i = 0; i < state.count; i++) if (state.moves[i] === idx) return i;
    return -1;
  }
  const iA = candIdx(stA, libA), iB = candIdx(stB, libB);
  check('rotated atari: both urgent liberties present', iA >= 0 && iB >= 0);

  // Under a D4 rotation, the canonical keys for 3×3c, A, and B at the
  // rotated candidate must match.
  check('rotated atari: 3×3c canonical keys match',
    stA.patIds33c[iA] === stB.patIds33c[iB]);
  check('rotated atari: Type A canonical keys match',
    stA.patIdsA[iA] === stB.patIdsA[iB]);
  check('rotated atari: Type B canonical keys match',
    stA.patIdsB[iA] === stB.patIdsB[iB]);

  const tA = [...stA.tact.subarray(iA * N_TACT_SLOTS, (iA + 1) * N_TACT_SLOTS)];
  const tB = [...stB.tact.subarray(iB * N_TACT_SLOTS, (iB + 1) * N_TACT_SLOTS)];
  check('rotated atari: tactical counts match',
    JSON.stringify(tA) === JSON.stringify(tB));
  check('rotated atari: URGENT_KILL STONE_0 fires exactly once',
    tactSum(stA.tact, iA, TACT_URGENT_KILL) === 1 &&
    tactSum(stB.tact, iB, TACT_URGENT_KILL) === 1);
}

// ── 11. Stacking: WASTED_ATTACK fires twice at a lib shared by two chains ───
//
// Two separate black chains with 2 liberties each, placed with lots of
// surrounding open space so the mover cannot force a kill (moverSucceeds
// stays false for both).  They share one liberty — the shared cell's
// WASTED_ATTACK count must be 2, one per chain.
{
  const N = 11;
  const g = new Game2(N, false);
  // Chain 1: black at (3,5), walls N & W → libs E and S.
  g._place(3 * N + 5, BLACK);
  g._place(2 * N + 5, WHITE);
  g._place(3 * N + 4, WHITE);
  // Chain 2: black at (5,5), walls S & W → libs E and N.  Shares lib (4,5).
  g._place(5 * N + 5, BLACK);
  g._place(6 * N + 5, WHITE);
  g._place(5 * N + 4, WHITE);
  g.current = WHITE;

  const st = NPat.createState(N);
  NPat.extractFeatures(g, st);
  const libIdx = 4 * N + 5;

  let cand = -1;
  for (let i = 0; i < st.count; i++) if (st.moves[i] === libIdx) { cand = i; break; }
  check('stacking: shared-lib candidate found', cand >= 0);
  if (cand >= 0) {
    check('stacking: WASTED_ATTACK count is 2 (one per chain)',
      tactSum(st.tact, cand, TACT_WASTED_ATTACK) === 2);
    check('stacking: URGENT_KILL is 0',
      tactSum(st.tact, cand, TACT_URGENT_KILL) === 0);
  }
}

// ── 12. Coexistence: one move is URGENT_KILL and URGENT_SAVE simultaneously ─
//
// A shared liberty where playing captures a foe chain AND saves a friend
// chain.  Row 3 with white (mover) in atari at (3,1), black in atari at (3,3),
// shared liberty at (3,2) — playing there extends white's chain (saving it,
// if the resulting chain has enough libs) and captures the black chain
// neighbour.  Exact urgency depends on ladder outcome; we assert at least
// both URGENT_KILL and URGENT_SAVE fire on that cell.  If ladder2 reports
// WASTED_EXTEND instead of URGENT_SAVE (rare on 7×7 with this setup), we
// accept that outcome as well — the coexistence we really test is
// KILL + (SAVE or WASTED_EXTEND).
{
  const N = 7;
  const g = new Game2(N, false);
  // White chain in atari at (3,1): surround N, S, W with black.
  g._place(3 * N + 1, WHITE);
  g._place(2 * N + 1, BLACK);
  g._place(4 * N + 1, BLACK);
  g._place(3 * N + 0, BLACK);
  // Black chain in atari at (3,3): surround N, S, E with white.
  g._place(3 * N + 3, BLACK);
  g._place(2 * N + 3, WHITE);
  g._place(4 * N + 3, WHITE);
  g._place(3 * N + 4, WHITE);
  g.current = WHITE;

  const st = NPat.createState(N);
  NPat.extractFeatures(g, st);
  const libIdx = 3 * N + 2;

  let cand = -1;
  for (let i = 0; i < st.count; i++) if (st.moves[i] === libIdx) { cand = i; break; }
  check('coexist: candidate found', cand >= 0);
  if (cand >= 0) {
    const kill = tactSum(st.tact, cand, TACT_URGENT_KILL);
    const save = tactSum(st.tact, cand, TACT_URGENT_SAVE);
    const wExt = tactSum(st.tact, cand, TACT_WASTED_EXTEND);
    check('coexist: URGENT_KILL fires (captures foe chain)', kill >= 1);
    check('coexist: either URGENT_SAVE or WASTED_EXTEND fires for friend chain',
      save >= 1 || wExt >= 1);
  }
}

// ── 13. REINFORCE updates tactical weights ──────────────────────────────────
//
// With a board containing a single urgent-kill candidate and an unrelated far
// candidate, pushing +1 advantage on the urgent-kill move must raise its
// URGENT_KILL tactical weight (the only move that has a nonzero count).
{
  const N = 7;
  const g = new Game2(N, false);
  g._place(3 * N + 3, BLACK);
  g._place(2 * N + 3, WHITE);
  g._place(4 * N + 3, WHITE);
  g._place(3 * N + 4, WHITE);
  g.current = WHITE;

  const st = NPat.createState(N);
  const w  = NPat.createWeights({ use33: true, use34: true, useL: true });

  NPat.policyMove(g, st, w);
  const libIdx = 3 * N + 2;
  let cand = -1;
  for (let i = 0; i < st.count; i++) if (st.moves[i] === libIdx) { cand = i; break; }
  check('REINFORCE-tact: urgent-kill candidate found', cand >= 0);

  const killDense = w.tactIds[TACT_URGENT_KILL];
  const before = w.vals[killDense];
  NPat.reinforceUpdate(st, cand, +1, w, 0.1);
  const after = w.vals[killDense];
  check('REINFORCE(+1) on urgent-kill move raises URGENT_KILL weight',
    after > before);

  // And after that update, greedyMove must prefer libIdx (it's the only move
  // with a nonzero tactical count and positive weight).
  NPat.extractFeatures(g, st, undefined, undefined, w);
  const best = NPat.greedyMove(g, st, w);
  check('REINFORCE-tact: greedyMove picks the urgent-kill move', best === libIdx);
}

// ── Report ──────────────────────────────────────────────────────────────────

console.log(`npat: ${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
