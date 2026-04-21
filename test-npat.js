'use strict';

// Tests for npat-lib.js — the nine-pattern policy features with ladder-aware
// cell encoding.

const NPat = require('./npat-lib.js');
const { Game2, BLACK, WHITE, PASS, parseBoard } = require('./game2.js');

let passed = 0, failed = 0;
function check(label, ok) {
  if (ok) passed++;
  else    { failed++; console.error('FAIL:', label); }
}

const {
  CELL_EMPTY, CELL_EMPTY_URGENT,
  CELL_FRIEND, CELL_FRIEND_URGENT,
  CELL_FOE,    CELL_FOE_URGENT,
  CELL_BASE,   CELLS_BASE,
  canonKey, _D4,
} = NPat;

check('CELL_BASE is 6', CELL_BASE === 6);
check('CELLS_BASE is 6^9', CELLS_BASE === 10077696);

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
  { relPos: 4, cells: [0,2,0, 0,0,0, 0,4,0] },  // candidate center; friend N, foe S
  { relPos: 0, cells: [0,2,4, 4,2,0, 2,0,4] },  // candidate top-left, full stones
  { relPos: 8, cells: [2,0,0, 0,0,0, 0,0,0] },  // candidate bottom-right; friend at top-left
  { relPos: 2, cells: [0,0,0, 2,4,2, 0,0,4] },  // candidate top-right
  { relPos: 4, cells: [1,3,5, 5,0,1, 3,1,5] },  // with urgency markers everywhere
  { relPos: 1, cells: [0,1,0, 3,2,4, 0,5,0] },  // mixed
];

for (const c of cases) {
  check(`case sanity: candidate cell is empty (relPos=${c.relPos})`,
    c.cells[c.relPos] === CELL_EMPTY || c.cells[c.relPos] === CELL_EMPTY_URGENT);

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
  // (relPos=4, center friend) vs (relPos=4, center foe) are not D4-equivalent.
  const k1 = canonKey(4, [0,0,0, 0,0,0, 0,0,0]);    // all empty
  const k2 = canonKey(4, [0,0,0, 0,0,0, 2,0,0]);    // one friend at corner
  const k3 = canonKey(4, [0,0,0, 0,0,0, 4,0,0]);    // one foe at corner
  check('all-empty pattern differs from single-friend pattern', k1 !== k2);
  check('single-friend pattern differs from single-foe pattern', k2 !== k3);

  // Four symmetric corner-friend patterns all collapse to the same key.
  check('corner NW friend ≡ corner NE friend', canonKey(4, [2,0,0, 0,0,0, 0,0,0]) === canonKey(4, [0,0,2, 0,0,0, 0,0,0]));
  check('corner NW friend ≡ corner SW friend', canonKey(4, [2,0,0, 0,0,0, 0,0,0]) === canonKey(4, [0,0,0, 0,0,0, 2,0,0]));
  check('corner NW friend ≡ corner SE friend', canonKey(4, [2,0,0, 0,0,0, 0,0,0]) === canonKey(4, [0,0,0, 0,0,0, 0,0,2]));
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

  // Every move produces the same multiset.
  let allSame = true;
  const sm = [...m0].sort().join(',');
  for (let i = 1; i < st.count; i++) {
    const a = [...st.patIds.subarray(i * 9, (i + 1) * 9)].sort().join(',');
    if (a !== sm) { allSame = false; break; }
  }
  check('empty board, every move has the same pattern multiset', allSame);
}

// ── 4. Ladder annotation: chain in atari is marked urgent ───────────────────
//
// Set up a black chain in atari with exactly one liberty.  White's move.
// The liberty is urgent for white (playing there captures).  Per ladder2,
// urgentLibs is non-empty → stoneUrgent on the chain's stones and libUrgent
// on the liberty point.
{
  const N = 5;
  const g = new Game2(N, false);
  // White surrounds a black stone at (2,2): white at N,E,S; black has 1 lib (W).
  g._place(2 * N + 2, BLACK); // black at (2,2)
  g._place(1 * N + 2, WHITE); // white at (1,2)
  g._place(3 * N + 2, WHITE); // white at (3,2)
  g._place(2 * N + 3, WHITE); // white at (2,3)
  g.current = WHITE;
  const st = NPat.createState(N);
  NPat.extractFeatures(g, st);

  const blackStoneIdx = 2 * N + 2;
  const libIdx        = 2 * N + 1; // (2,1), the remaining liberty

  check('ladder annotation: black stone marked stoneUrgent',
    st.ladder.stoneUrgent[blackStoneIdx] === 1);
  check('ladder annotation: liberty cell marked libUrgent',
    st.ladder.libUrgent[libIdx] === 1);

  // The urgent liberty move, when extracted as a candidate, has its candidate
  // cell encoded as EMPTY_URGENT — even after canonicalisation this must
  // distinguish it from a plain EMPTY-centered candidate in an otherwise
  // identical neighbourhood.  Here we just check the pattern features differ
  // between the urgent-liberty candidate and an unrelated far candidate.
  let urgIdx = -1, farIdx = -1;
  for (let i = 0; i < st.count; i++) {
    if (st.moves[i] === libIdx)       urgIdx = i;
    if (st.moves[i] === 0 * N + 0)    farIdx = i;
  }
  check('urgent liberty candidate present', urgIdx >= 0);
  check('far candidate present',            farIdx >= 0);

  const urg = [...st.patIds.subarray(urgIdx * 9, (urgIdx + 1) * 9)].sort().join(',');
  const far = [...st.patIds.subarray(farIdx * 9, (farIdx + 1) * 9)].sort().join(',');
  check('urgent-liberty move has different features from empty-area move',
    urg !== far);
}

// ── 5. Ladder-off vs ladder-on produce different pattern keys ───────────────
//
// Same board, but flip one of the white stones away so the black chain now
// has 2 liberties (still analyzable by ladder2, but urgentLibs may be empty
// if neither side can force a result).  Confirm the keys change when the
// tactical landscape changes.
{
  const N = 5;
  const g1 = new Game2(N, false);
  g1._place(2 * N + 2, BLACK);
  g1._place(1 * N + 2, WHITE);
  g1._place(3 * N + 2, WHITE);
  g1._place(2 * N + 3, WHITE);
  g1.current = WHITE;

  const g2 = new Game2(N, false);
  g2._place(2 * N + 2, BLACK);
  g2._place(1 * N + 2, WHITE);
  // no south white — chain has 2 liberties (W and S) instead of 1.
  g2._place(2 * N + 3, WHITE);
  g2.current = WHITE;

  const st1 = NPat.createState(N);
  const st2 = NPat.createState(N);
  NPat.extractFeatures(g1, st1);
  NPat.extractFeatures(g2, st2);

  const libIdx = 2 * N + 1;
  function idsFor(st, idx) {
    for (let i = 0; i < st.count; i++) if (st.moves[i] === idx)
      return [...st.patIds.subarray(i * 9, (i + 1) * 9)].sort().join(',');
    return null;
  }
  const a = idsFor(st1, libIdx);
  const b = idsFor(st2, libIdx);
  check('atari position vs 2-lib position produce different pattern keys',
    a !== null && b !== null && a !== b);
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

  function score(i) {
    const off = i * 9; let s = 0;
    for (let k = 0; k < 9; k++) s += w.vals[st.patIds[off + k]];
    return s;
  }
  const before = score(chosen);
  NPat.reinforceUpdate(st, chosen, +1, w, 0.1);
  const after = score(chosen);
  check('REINFORCE(+1): chosen move logit increases', after > before);
}

// ── 9. No urgent chains on a quiet board → no urgent cell-encoding bits ────
//
// A chain with ≥3 liberties is not tactically urgent.  All nine-pattern keys
// around the chain should use only CELL_EMPTY / CELL_FRIEND / CELL_FOE — i.e.
// the annotateLadders flags must be zero.
{
  const N = 7;
  const g = new Game2(N, false);
  g._place(3 * N + 3, BLACK); // black stone with 4 liberties, not in a ladder
  g.current = WHITE;

  const ladder = NPat.annotateLadders(g);
  let anyUrgent = false;
  for (let i = 0; i < N * N; i++) {
    if (ladder.stoneUrgent[i] || ladder.libUrgent[i]) { anyUrgent = true; break; }
  }
  check('quiet board (chain with 4 libs): no urgent cells', !anyUrgent);
}

// ── 10. Urgent liberty and its D4-symmetric counterpart get the same key ────
//
// On a toroidal 7×7 board, set up TWO identical atari situations at positions
// related by a D4 rotation.  The urgent liberties of both ataris must produce
// the same multiset of 9 canonical pattern keys.
{
  const N = 7;
  // Configuration A: black in atari at (2,2), surrounded N, E, S by white.
  const gA = new Game2(N, false);
  gA._place(2 * N + 2, BLACK);
  gA._place(1 * N + 2, WHITE);
  gA._place(3 * N + 2, WHITE);
  gA._place(2 * N + 3, WHITE);
  gA.current = WHITE;

  // Configuration B: the 180°-rotation of A.  (2,2) → (4,4).  N→S, E→W, S→N.
  const gB = new Game2(N, false);
  gB._place(4 * N + 4, BLACK);
  gB._place(5 * N + 4, WHITE);   // rotated N (1,2) → (5,4)
  gB._place(3 * N + 4, WHITE);   // rotated S (3,2) → (3,4)
  gB._place(4 * N + 3, WHITE);   // rotated E (2,3) → (4,3)
  gB.current = WHITE;

  const stA = NPat.createState(N);
  const stB = NPat.createState(N);
  NPat.extractFeatures(gA, stA);
  NPat.extractFeatures(gB, stB);

  const libA = 2 * N + 1;   // remaining liberty of A
  const libB = 4 * N + 5;   // remaining liberty of B (rotated from (2,1))

  function ids(state, idx) {
    for (let i = 0; i < state.count; i++) if (state.moves[i] === idx)
      return [...state.patIds.subarray(i * 9, (i + 1) * 9)].sort();
    return null;
  }
  const a = ids(stA, libA);
  const b = ids(stB, libB);
  check('rotated atari: both urgent liberties present', a !== null && b !== null);
  check('rotated atari: pattern multisets match',
    JSON.stringify(a) === JSON.stringify(b));
}

// ── Report ──────────────────────────────────────────────────────────────────

console.log(`npat: ${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
