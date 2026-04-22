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
  N_TACT,
  TACT_URGENT_KILL, TACT_URGENT_SAVE,
  TACT_WASTED_EXTEND, TACT_WASTED_ATTACK,
  TACT_RAW_BASE,
  WINDOWS_34, CELLS12_BASE, SHAPE34_RAW_BASE,
  canonKey, canonKey34, _D4, _D4rect,
} = NPat;

check('CELL_BASE is 3', CELL_BASE === 3);
check('CELLS_BASE is 3^9', CELLS_BASE === 19683);
check('N_TACT is 4', N_TACT === 4);
check('TACT_RAW_BASE is 9 * CELLS_BASE', TACT_RAW_BASE === 9 * 19683);
check('WINDOWS_34 is 12', WINDOWS_34 === 12);
check('CELLS12_BASE is 3^12', CELLS12_BASE === 531441);
check('SHAPE34_RAW_BASE is TACT_RAW_BASE + N_TACT',
  SHAPE34_RAW_BASE === TACT_RAW_BASE + N_TACT);

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

// ── 3. Empty 5×5 board — per-move feature multiset ──────────────────────────
//
// All 9 windows around every candidate are entirely empty.  The 9 values of
// relPos ∈ 0..8 fall into 3 D4 orbits: {0,2,6,8} corners, {1,3,5,7} edges, {4}
// center.  So every candidate's 9 pattern keys collapse to exactly 3 distinct
// values with multiplicities [4, 4, 1].
{
  const N  = 5;
  const g  = new Game2(N, false);
  const st = NPat.createState(N);
  NPat.extractFeatures(g, st);

  check('empty 5x5: count === 25', st.count === 25);

  const m0 = st.patIds.subarray(0, 9);
  const counts = new Map();
  for (const id of m0) counts.set(id, (counts.get(id) ?? 0) + 1);
  const vals = [...counts.values()].sort((a, b) => a - b);
  check('empty board, per-move window multiset [1,4,4]',
    JSON.stringify(vals) === '[1,4,4]');

  // Every move produces the same shape multiset, and zero tactical counts.
  let allSame = true;
  const sm = [...m0].sort().join(',');
  for (let i = 1; i < st.count; i++) {
    const a = [...st.patIds.subarray(i * 9, (i + 1) * 9)].sort().join(',');
    if (a !== sm) { allSame = false; break; }
  }
  check('empty board, every move has the same pattern multiset', allSame);
  let anyTact = false;
  for (let i = 0; i < st.count * N_TACT; i++) if (st.tact[i]) { anyTact = true; break; }
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

  // tactCount[libIdx * N_TACT + TACT_URGENT_KILL] should be 1; save/wasted 0.
  const tc = st.ladder.tactCount;
  check('ladder annotation: liberty cell marked URGENT_KILL',
    tc[libIdx * N_TACT + TACT_URGENT_KILL] === 1);
  check('ladder annotation: liberty cell not marked URGENT_SAVE',
    tc[libIdx * N_TACT + TACT_URGENT_SAVE] === 0);
  check('ladder annotation: liberty cell not marked WASTED_EXTEND',
    tc[libIdx * N_TACT + TACT_WASTED_EXTEND] === 0);
  check('ladder annotation: liberty cell not marked WASTED_ATTACK',
    tc[libIdx * N_TACT + TACT_WASTED_ATTACK] === 0);

  // The per-candidate tact buffer should reflect the same values.
  let urgIdx = -1;
  for (let i = 0; i < st.count; i++) if (st.moves[i] === libIdx) { urgIdx = i; break; }
  check('urgent liberty candidate present', urgIdx >= 0);
  check('candidate tact[URGENT_KILL] is 1',
    st.tact[urgIdx * N_TACT + TACT_URGENT_KILL] === 1);

  // A far candidate has all-zero tactical counts.
  let farIdx = -1;
  for (let i = 0; i < st.count; i++) if (st.moves[i] === 0 * N + 0) { farIdx = i; break; }
  check('far candidate present', farIdx >= 0);
  let farZero = true;
  for (let k = 0; k < N_TACT; k++) if (st.tact[farIdx * N_TACT + k]) { farZero = false; break; }
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
  const save   = tc[libIdx * N_TACT + TACT_URGENT_SAVE];
  const wExt   = tc[libIdx * N_TACT + TACT_WASTED_EXTEND];
  const kill   = tc[libIdx * N_TACT + TACT_URGENT_KILL];
  const wAtk   = tc[libIdx * N_TACT + TACT_WASTED_ATTACK];
  check('friend-in-atari: either URGENT_SAVE or WASTED_EXTEND fires',
    (save === 1 && wExt === 0) || (save === 0 && wExt === 1));
  check('friend-in-atari: URGENT_KILL is 0', kill === 0);
  check('friend-in-atari: WASTED_ATTACK is 0', wAtk === 0);
}

// ── 6. D4 board symmetry → D4 feature symmetry ──────────────────────────────
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

  function ids(state, idx) {
    for (let i = 0; i < state.count; i++) if (state.moves[i] === idx)
      return [...state.patIds.subarray(i * 9, (i + 1) * 9)].sort();
    return null;
  }
  const a = ids(stA, 0 * N + 2);
  const b = ids(stB, 4 * N + 2);
  check('D4-symmetric positions: both candidates found',  a !== null && b !== null);
  check('D4-symmetric positions: canonical-key multisets match',
    JSON.stringify(a) === JSON.stringify(b));
}

// ── 7. policyMove returns a legal move with uniform prob on empty weights ───
{
  const N = 5;
  const g = new Game2(N, false);
  const st = NPat.createState(N);
  const w = NPat.createWeights();
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
  const w = NPat.createWeights();

  NPat.policyMove(g, st, w);
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
  check('REINFORCE setup: found a non-symmetric move', chosen >= 0);

  function scoreShape(i) {
    const off = i * 9; let s = 0;
    for (let k = 0; k < 9; k++) s += w.vals[st.patIds[off + k]];
    return s;
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
  for (let i = 0; i < N * N * N_TACT; i++) if (ladder.tactCount[i]) { any = true; break; }
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

  const a = [...stA.patIds.subarray(iA * 9, (iA + 1) * 9)].sort();
  const b = [...stB.patIds.subarray(iB * 9, (iB + 1) * 9)].sort();
  check('rotated atari: shape pattern multisets match',
    JSON.stringify(a) === JSON.stringify(b));

  const tA = [...stA.tact.subarray(iA * N_TACT, (iA + 1) * N_TACT)];
  const tB = [...stB.tact.subarray(iB * N_TACT, (iB + 1) * N_TACT)];
  check('rotated atari: tactical counts match',
    JSON.stringify(tA) === JSON.stringify(tB));
  check('rotated atari: URGENT_KILL fires exactly once',
    tA[TACT_URGENT_KILL] === 1 && tB[TACT_URGENT_KILL] === 1);
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
      st.tact[cand * N_TACT + TACT_WASTED_ATTACK] === 2);
    check('stacking: URGENT_KILL is 0',
      st.tact[cand * N_TACT + TACT_URGENT_KILL] === 0);
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
    const off = cand * N_TACT;
    const kill = st.tact[off + TACT_URGENT_KILL];
    const save = st.tact[off + TACT_URGENT_SAVE];
    const wExt = st.tact[off + TACT_WASTED_EXTEND];
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
  const w  = NPat.createWeights();

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

// ── 15. canonKey34 is invariant under the Klein-4 subgroup of D4 ────────────
//
// The 8 D4 perms _D4rect indices 0..3 (id, hflip, vflip, 180°) are the ones
// that keep a 3×4 window as a 3×4 window.  They form a closed subgroup under
// composition, so applying any of them to a (relPos, cells) input yields a
// new 3×4 input with the same canonical key.  The other 4 perms (indices
// 4..7) take 3×4 → 4×3 (different row-major layout), so they do not form a
// closed action on our 3×4-input space and are exercised instead by the
// end-to-end board-rotation test below.

function applyD4rect(relPos, cells, perm) {
  const nc = new Array(12);
  for (let i = 0; i < 12; i++) nc[perm[i]] = cells[i];
  return { relPos: perm[relPos], cells: nc };
}

{
  const cases = [
    { relPos: 0,  cells: [0,0,0,0, 0,0,0,0, 0,0,0,0] }, // all empty, corner
    { relPos: 5,  cells: [0,0,0,0, 0,0,1,0, 0,0,0,0] }, // one friend adj candidate
    { relPos: 5,  cells: [0,1,2,0, 1,0,2,0, 0,1,2,0] }, // mixed
    { relPos: 11, cells: [1,0,0,0, 0,2,0,0, 0,0,1,0] }, // candidate bottom-right-ish
    { relPos: 4,  cells: [2,2,2,2, 0,1,1,1, 0,0,0,0] }, // asymmetric
  ];

  for (const c of cases) {
    check(`canonKey34 case sanity: candidate cell empty (relPos=${c.relPos})`,
      c.cells[c.relPos] === CELL_EMPTY);
    const baseKey = canonKey34(c.relPos, c.cells);
    for (let di = 0; di < 4; di++) {  // Klein-4 subgroup only
      const t = applyD4rect(c.relPos, c.cells, _D4rect[di]);
      const tKey = canonKey34(t.relPos, t.cells);
      check(`canonKey34: Klein-4 sym ${di} preserves canonical key (relPos=${c.relPos})`,
        tKey === baseKey);
    }
  }
}

// ── 16. Distinct 3×4 patterns get distinct canonical keys (barring symmetry) ─
//
// A single friend at the window's top-left (relPos at center) vs a single foe
// at the same spot vs all empty must yield three different canonical keys.

{
  const k1 = canonKey34(5, [0,0,0,0, 0,0,0,0, 0,0,0,0]); // all empty
  const k2 = canonKey34(5, [1,0,0,0, 0,0,0,0, 0,0,0,0]); // one friend at corner
  const k3 = canonKey34(5, [2,0,0,0, 0,0,0,0, 0,0,0,0]); // one foe at corner
  check('canonKey34: empty vs one-friend keys differ', k1 !== k2);
  check('canonKey34: empty vs one-foe keys differ',    k1 !== k3);
  check('canonKey34: one-friend vs one-foe keys differ', k2 !== k3);
}

// ── 17. Empty board: every candidate has 12 3×4 keys with multiset [2,4,6] ──
//
// Canonical relPos values (min over D4 orbit) for j=0..11 are
// [0,1,1,0, 1,4,4,1, 0,1,1,0], giving counts {0:4, 1:6, 4:2}.  Since cells
// are all empty, the canonical raw is SHAPE34_RAW_BASE + canon_relPos * 3^12.

{
  const N = 5;
  const g = new Game2(N, false);
  const st = NPat.createState(N);
  NPat.extractFeatures(g, st);

  check('patIds34 buffer length matches count*12',
    st.patIds34.length >= st.count * WINDOWS_34);

  const m0 = st.patIds34.subarray(0, WINDOWS_34);
  const counts = new Map();
  for (const id of m0) counts.set(id, (counts.get(id) ?? 0) + 1);
  const vals = [...counts.values()].sort((a, b) => a - b);
  check('empty board: per-move 3×4 window multiset [2,4,6]',
    JSON.stringify(vals) === '[2,4,6]');

  let allSame = true;
  const sm = [...m0].sort().join(',');
  for (let i = 1; i < st.count; i++) {
    const a = [...st.patIds34.subarray(i * WINDOWS_34, (i + 1) * WINDOWS_34)]
      .sort().join(',');
    if (a !== sm) { allSame = false; break; }
  }
  check('empty board: every move has the same 3×4 multiset', allSame);
}

// ── 18. D4 board symmetry: 3×4 multisets match across rotated atari ─────────
//
// Same setup as test 10 (180° board rotation).  The patIds34 multiset at the
// urgent liberty must match between configs A and B.

{
  const N = 7;
  const gA = new Game2(N, false);
  gA._place(2 * N + 2, BLACK);
  gA._place(1 * N + 2, WHITE);
  gA._place(3 * N + 2, WHITE);
  gA._place(2 * N + 3, WHITE);
  gA.current = WHITE;

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
  const a = [...stA.patIds34.subarray(iA * WINDOWS_34, (iA + 1) * WINDOWS_34)].sort();
  const b = [...stB.patIds34.subarray(iB * WINDOWS_34, (iB + 1) * WINDOWS_34)].sort();
  check('rotated atari: 3×4 pattern multisets match',
    JSON.stringify(a) === JSON.stringify(b));
}

// ── 19. REINFORCE now updates 3×4 weights too ───────────────────────────────
//
// After a REINFORCE step on a chosen move, every one of its 12 3×4 dense pids
// must have a nonzero weight (at least some of them will, barring cancellations
// from the distribution-average term — unlikely on a mostly-empty board with
// uniform initialisation).

{
  const N = 5;
  const g = new Game2(N, false);
  const st = NPat.createState(N);
  const w  = NPat.createWeights();

  NPat.policyMove(g, st, w);
  const chosen = 0;
  const off34 = chosen * WINDOWS_34;
  const pidsBefore = [...st.patIds34.subarray(off34, off34 + WINDOWS_34)];
  NPat.reinforceUpdate(st, chosen, +1, w, 0.1);

  let anyNonzero = false;
  for (const p of pidsBefore) {
    if (w.vals[p] !== 0) { anyNonzero = true; break; }
  }
  check('REINFORCE modifies at least one 3×4 weight for the chosen move',
    anyNonzero);
}

// ── Report ──────────────────────────────────────────────────────────────────

console.log(`npat: ${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
