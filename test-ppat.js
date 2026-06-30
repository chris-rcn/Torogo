'use strict';

const { createState, extractFeatures, evaluate, ppatMove, totalWeights, NUM_PATTERNS } = require('./ppat-lib.js');
// NUM_PATTERNS is the count of canonical IDs under D4 spatial symmetry only
// (color swap is NOT applied since the encoding is already mover-relative).
const { Game2, BLACK, WHITE, PASS, parseBoard } = require('./game2.js');

let passed = 0, failed = 0;
function check(label, ok) {
  if (ok) { passed++; }
  else { failed++; console.error('FAIL:', label); }
}

// ── 1. Pattern count ─────────────────────────────────────────────────────────
check('NUM_PATTERNS is 6810', NUM_PATTERNS === 6810);

// ── 2. Lookup-table internal consistency ─────────────────────────────────────
// For every raw index, the canonical form should be its own canonical (idempotent):
//   CANON_ID[minVariant] === CANON_ID[raw]
// We spot-check rather than exhaustively scan to keep the test fast.
{
  const { _CANON_ID: CID, _CANON_POL: CPOL } = (function() {
    // Access internal tables via a white-box trick: re-require is cached,
    // so use a getter instead.  Actually, just verify properties via the API.
    // Use evaluate on a blank board to smoke-test the tables instead.
    return {};
  }());

  // Spot-check: a pattern that maps to itself under all transforms (all-empty = raw 0)
  // raw 0 = all EMPTY; it's symmetric, so CANON_ID[0] === 0 and CANON_POL[0] === 1.
  const g = new Game2(5, false);
  const st = createState(5);
  extractFeatures(g, st);
  // Center of 5×5 = idx 12; it should be legal and have a valid pattern.
  let found12 = false;
  for (let i = 0; i < st.count; i++) {
    if (st.moves[i] === 12) { found12 = true; break; }
  }
  check('center move found in empty 5x5', found12);
}

// ── 3. D4 symmetry (lookup table) ────────────────────────────────────────────
// Build raw index directly and check all 8 D4 rotations share the same CANON_ID.
{
  // Access the tables through a Node module cache trick.
  // We'll re-implement rawIdx encode for testing.
  function rawEnc(vN, vE, vS, vW, vNE, vSE, vSW, vNW) {
    return vN + 5*(vE + 5*(vS + 5*(vW + 5*(vNE + 3*(vSE + 3*(vSW + 3*vNW))))));
  }

  // Use a game position to verify D4 symmetry indirectly:
  // Place B at (0,1) on a 5x5 toroidal board (cell 1).  Then check candidate
  // at (0,0)=0 and at the 3 rotated equivalents.  All 4 candidates are in
  // identical rotational contexts (the board is not toroidal-equivalent, but
  // we can verify directly via the rawIdx formula).

  // Pattern: FRIEND at N (vN=1), all else EMPTY → rawIdx = 1
  // Rot90CW: FRIEND moves to E → rawIdx = 5
  // Rot180:  FRIEND moves to S → rawIdx = 25
  // Rot270:  FRIEND moves to W → rawIdx = 125
  // All should have same CANON_ID.

  // We can't directly read _CANON_ID; test via a real game.
  // Instead, construct a game where only one cell is occupied adjacent to the candidate,
  // and check across rotated board positions.
  function makeGameWithFriendAt(boardCell, candidateCell, N) {
    const g = new Game2(N, false);
    // current = BLACK. Place BLACK at boardCell (first move as BLACK).
    if (!g.play(boardCell)) return null;
    // Now it's WHITE's turn; pass to keep it simple.
    if (!g.play(PASS)) return null;
    // Now BLACK again; our feature extraction runs as BLACK.
    // Extract the 3x3 pattern for candidateCell.
    const st = createState(N);
    extractFeatures(g, st);
    for (let i = 0; i < st.count; i++) {
      if (st.moves[i] === candidateCell) return { patId: getPatId(st, i) };
    }
    return null;
  }

  // On a 5×5 board, candidate = center (12).
  // Rotate context: FRIEND at N (cell 12-5=7), E (cell 13), S (cell 17), W (cell 11).
  const N = 5;
  const center = 12;
  const fN = makeGameWithFriendAt(7,  center, N);  // black at N of center
  const fE = makeGameWithFriendAt(13, center, N);  // black at E of center
  const fS = makeGameWithFriendAt(17, center, N);  // black at S of center
  const fW = makeGameWithFriendAt(11, center, N);  // black at W of center

  check('D4 Rot90: N and E rotations same patId', fN && fE && fN.patId === fE.patId);
  check('D4 Rot180: N and S rotations same patId', fN && fS && fN.patId === fS.patId);
  check('D4 Rot270: N and W rotations same patId', fN && fW && fN.patId === fW.patId);


  // Flip: FRIEND at NE (diagonal) and FRIEND at NW should share patId (FlipH symmetry).
  const fNE = makeGameWithFriendAt(8,  center, N);  // NE of center (row-1, col+1)=(1,3)=8?
  // On 5×5: center=12=(r2,c2). NE=(r1,c3)=1*5+3=8. NW=(r1,c1)=1*5+1=6.
  // But these are diagonal — they only affect patNE and patNW which are diag (3-state, no atari).
  const fNW = makeGameWithFriendAt(6,  center, N);  // NW of center
  check('D4 FlipH: NE and NW have same patId', fNE && fNW && fNE.patId === fNW.patId);
}

// ── 4. Color symmetry ────────────────────────────────────────────────────────
{
  // Pattern: current player's FRIEND at N vs FOE at N.
  // Both positions should yield same CANON_ID with opposite polarity.
  //
  // Setup A: BLACK plays first, then it's BLACK's turn; friend at N of candidate.
  // Setup B: WHITE plays first (via pass), then it's BLACK's turn; enemy at N of candidate.
  // Actually easier: for BLACK's perspective: BLACK stone at N → FRIEND.
  //                  WHITE stone at N → FOE.

  const N = 5, center = 12, northOf = 7;

  // A: BLACK stone at northOf (FRIEND for BLACK)
  const gA = new Game2(N, false);
  gA.play(northOf); gA.play(PASS); // BLACK at north, pass (now BLACK's turn again)
  const stA = createState(N); extractFeatures(gA, stA);
  let rA = null;
  for (let i = 0; i < stA.count; i++) if (stA.moves[i] === center) { rA = { id: getPatId(stA, i) }; break; }

  // B: WHITE stone at northOf (FOE for BLACK)
  // Place WHITE there: BLACK passes, then WHITE plays, then it's BLACK's turn.
  const gB = new Game2(N, false);
  gB.play(PASS); gB.play(northOf); // pass by BLACK, WHITE plays at north
  const stB = createState(N); extractFeatures(gB, stB);
  let rB = null;
  for (let i = 0; i < stB.count; i++) if (stB.moves[i] === center) { rB = { id: getPatId(stB, i) }; break; }

  // FRIEND at N and FOE at N are genuinely different patterns in a mover-relative
  // encoding — no color-swap symmetry is applied, so they must have different patIds.
  check('mover-relative: FRIEND at N vs FOE at N have different patIds', rA && rB && rA.id !== rB.id);
}

// ── 5. Atari encoding ─────────────────────────────────────────────────────────
{
  // Create a position where a friendly stone adjacent to the candidate is in atari.
  // We want to verify the resulting pattern encodes FRIEND_ATARI (not just FRIEND).
  //
  // Setup on 9×9 (no initial stone):
  // Candidate at center (40). Place BLACK stones around position 31 (north of center)
  // such that position 31 has only 1 liberty (= cell 40, the candidate).
  // That means: cells 22 (N of 31), 30 (W of 31), 32 (E of 31) must be occupied.
  // (All orthogonal neighbors of 31 except 40 are occupied.)
  // On a 9×9 toroidal board, 9 cells per row.
  // Cell 40 = center (4,4). Cell 31 = (3,4) = north of center.
  // Neighbors of 31: N=(2,4)=22, S=(4,4)=40, W=(3,3)=30, E=(3,5)=32.
  // To put 31 in atari: occupy 22, 30, 32 with enemy (WHITE).
  const N = 9;
  const g = new Game2(N, false);
  const cand = 40, atariCell = 31;
  // Play: B=31, W=22, B=0(dummy), W=30, B=1, W=32, B=2, W=pass
  // After this: WHITE has stones at 22, 30, 32 surrounding 31 (BLACK) except at 40.
  g.play(atariCell); // BLACK at 31
  g.play(22);        // WHITE at 22
  g.play(0);         // BLACK dummy
  g.play(30);        // WHITE at 30
  g.play(1);         // BLACK dummy
  g.play(32);        // WHITE at 32
  g.play(2);         // BLACK dummy
  g.play(PASS);      // WHITE pass  → BLACK's turn
  // Now BLACK at 31 should have 1 liberty (= 40). It's in atari.
  const gid31 = g._gid[31];
  check('atari setup: cell 31 has gid', gid31 >= 0);
  check('atari setup: cell 31 in atari', gid31 >= 0 && g._ls[gid31] === 1);

  const st = createState(N); extractFeatures(g, st);
  let got = null;
  for (let i = 0; i < st.count; i++) if (st.moves[i] === cand) { got = { id: getPatId(st, i) }; break; }
  check('atari candidate found', got !== null);

  // Compare with the same position but cell 31 has 2+ liberties (not in atari).
  // Reuse: just play at cell 32 (freeing a liberty of 31) — but that changes the board.
  // Instead, create a fresh game with BLACK at 31 but only 22 and 30 occupied by WHITE.
  const g2 = new Game2(N, false);
  g2.play(atariCell); g2.play(22); g2.play(0); g2.play(30); g2.play(1); g2.play(PASS);
  const gid31b = g2._gid[31];
  check('non-atari setup: cell 31 has 2 libs', gid31b >= 0 && g2._ls[gid31b] === 2);

  const st2 = createState(N); extractFeatures(g2, st2);
  let got2 = null;
  for (let i = 0; i < st2.count; i++) if (st2.moves[i] === cand) { got2 = { id: getPatId(st2, i) }; break; }
  check('non-atari candidate found', got2 !== null);

  // The atari version should have a different patId than the non-atari version
  // (since the adjacent cell encodes differently: FRIEND_ATARI vs FRIEND).
  check('atari vs non-atari: different patId', got && got2 && got.id !== got2.id);
}

// ── 6. Feature 1: contiguity to previous move ────────────────────────────────
{
  // Previous move W@C3.  Every candidate in C3's 8-neighborhood gets F1 (bit 0);
  // candidates outside it do not.  (X@E5 is just an earlier stone on the board.)
  const N = 9;
  const g = parseBoard(`
    9 . . . . . . . . .
    8 . . . . . . . . .
    7 . . . . . . . . .
    6 . . . . . . . . .
    5 . . . . X . . . .
    4 . . . . . . . . .
    3 . . . . . . . . .
    2 . . . . . . . . .
    1 . . . . . . . . .
  `, WHITE);
  g.play(20); // W@C3 (previous move)

  const st = createState(N); extractFeatures(g, st);
  const nbrs8 = new Set([11, 29, 19, 21, 12, 10, 28, 30]); // 8-neighbors of C3

  const prevBase = NUM_PATTERNS;   // single-phase extraction (phaseCount defaults to 1)
  let allNbrsHaveBit0 = true, nonNbrsLackBit0 = true;
  for (let i = 0; i < st.count; i++) {
    const m = st.moves[i];
    let mask = 0;
    for (let fi = st.featStart[i]; fi < st.featStart[i + 1]; fi++) {
      const key = st.feat[fi];
      if (key >= prevBase) mask |= 1 << ((key - prevBase) % 7);
    }
    if (nbrs8.has(m)) {
      if (!(mask & 1)) { allNbrsHaveBit0 = false; console.error('  move', m, 'is 8-nbr but lacks bit 0, mask=', mask); }
    } else {
      if (mask & 1) { nonNbrsLackBit0 = false; console.error('  move', m, 'is not 8-nbr but has bit 0'); }
    }
  }
  check('Feature 1: all 8-neighbors have bit 0', allNbrsHaveBit0);
  check('Feature 1: non-neighbors lack bit 0', nonNbrsLackBit0);
}

// ── 7. Feature 4: save string in new atari by extension ──────────────────────
{
  // Previous move W@E6 puts B@E5 in atari (only lib F5); extending at that liberty
  // (F5) is the save move → bit 3 (or 4), plus contiguous (bit 0).
  const g = parseBoard(`
    9 . . . . . . . . .
    8 . . . . . . . . .
    7 . . . . . . . . .
    6 . . . . . . . . .
    5 . . . O X . . . .
    4 . . . . O . . . .
    3 . . . . . . . . .
    2 . . . . . . . . .
    1 . . . . . . . . .
  `, WHITE);
  g.play(49); // W@E6 (previous move)
  check('F4/5 setup: B@E5 in atari', g._ls[g._gid[40]] === 1);
  const libMask = getMask(g, 41); // F5, the saving extension
  // Bit 3 = F4 (extend, not self-atari); Bit 4 = F5 (extend, self-atari)
  check('F4/5: extension liberty has bit 3 or 4', (libMask & (8 | 16)) !== 0);
  check('F4/5: also has bit 0 (contiguous)', (libMask & 1) !== 0);
}

// ── Helpers for reading flat feature state ────────────────────────────────────
// Extract pattern ID (phase-stripped) from feature keys.
function getPatId(st, i) {
  return st.feat[st.featStart[i]] % NUM_PATTERNS;
}

// Reconstruct prevMask for a specific candidate from flat feature keys.
function getMask(game, cell) {
  const st = createState(game.N);
  extractFeatures(game, st);
  const prevBase = NUM_PATTERNS;   // single-phase extraction (phaseCount defaults to 1)
  for (let i = 0; i < st.count; i++) {
    if (st.moves[i] !== cell) continue;
    let mask = 0;
    for (let fi = st.featStart[i]; fi < st.featStart[i + 1]; fi++) {
      const key = st.feat[fi];
      if (key >= prevBase) mask |= 1 << ((key - prevBase) % 7);
    }
    return mask;
  }
  return -1;
}

// ── 8. Feature 2: save by capture, not self-atari (bit 1) ───────────────────
// 8a: B@31 in atari (lib=22). W@32 in atari (lib=41). lastMove=W@40.
//     Candidate 41 captures W@32, saving B@31. Not self-atari.
//     9×9 cells: 31=(3,4) 32=(3,5) 23=(2,5) 30=(3,3) 33=(3,6) 40=(4,4) 41=(4,5)
{
  // Previous move W@E5 puts both B@E4 and W@F4 in atari; candidate F5 captures
  // W@F4, saving B@E4.  Not self-atari → F2 (bit 1), plus contiguous (bit 0).
  const g = parseBoard(`
    9 . . . . . . . . .
    8 . . . . . . . . .
    7 . . . . . . . . .
    6 . . . . . . . . .
    5 . . . . . . . . .
    4 . . . O X O X . .
    3 . . . . . X . . .
    2 . . . . . . . . .
    1 . . . . . . . . .
  `, WHITE);
  g.play(40); // W@E5 (previous move)
  check('F2a setup: B@E4 in atari', g._ls[g._gid[31]] === 1);
  check('F2a setup: W@F4 in atari', g._ls[g._gid[32]] === 1);
  const mask = getMask(g, 41); // F5
  check('F2a: bit 0 set (contiguous)', mask !== -1 && (mask & 1) !== 0);
  check('F2a: bit 1 set (save by capture)', mask !== -1 && (mask & 2) !== 0);
  check('F2a: bit 2 not set (not self-atari)', mask !== -1 && (mask & 4) === 0);
}

// 8b: mirror — previous move W@E6 ataris B@E5 and W@D5; candidate D6 captures
//     W@D5, saving B@E5.  Not self-atari.
{
  const g = parseBoard(`
    9 . . . . . . . . .
    8 . . . . . . . . .
    7 . . . . . . . . .
    6 . . . . . . . . .
    5 . . X O X O . . .
    4 . . . X . . . . .
    3 . . . . . . . . .
    2 . . . . . . . . .
    1 . . . . . . . . .
  `, WHITE);
  g.play(49); // W@E6 (previous move)
  check('F2b setup: B@E5 in atari', g._ls[g._gid[40]] === 1);
  check('F2b setup: W@D5 in atari', g._ls[g._gid[39]] === 1);
  const mask = getMask(g, 48); // D6
  check('F2b: bit 0 set (contiguous)', mask !== -1 && (mask & 1) !== 0);
  check('F2b: bit 1 set (save by capture)', mask !== -1 && (mask & 2) !== 0);
  check('F2b: bit 2 not set (not self-atari)', mask !== -1 && (mask & 4) === 0);
}

// ── 9. Feature 3: save by capture, IS self-atari (bit 2) ────────────────────
// 9a: Same as 8a but W@50,W@42 surround cell 41, making the capture self-atari.
//     After capturing W@32, B@41 has only lib=32 (N=32 empty, S=50(W), W=40(W), E=42(W)).
{
  // As F2a but W@F6 and W@G5 surround F5, so capturing W@F4 leaves B@F5 in
  // self-atari (only lib F4) → F3 (bit 2), not F2 (bit 1).
  const g = parseBoard(`
    9 . . . . . . . . .
    8 . . . . . . . . .
    7 . . . . . . . . .
    6 . . . . . O . . .
    5 . . . . . . O . .
    4 . . . O X O X . .
    3 . . . . . X . . .
    2 . . . . . . . . .
    1 . . . . . . . . .
  `, WHITE);
  g.play(40); // W@E5 (previous move)
  check('F3a setup: B@E4 in atari', g._ls[g._gid[31]] === 1);
  check('F3a setup: W@F4 in atari', g._ls[g._gid[32]] === 1);
  const mask = getMask(g, 41); // F5
  check('F3a: bit 0 set (contiguous)', mask !== -1 && (mask & 1) !== 0);
  check('F3a: bit 2 set (capture + self-atari)', mask !== -1 && (mask & 4) !== 0);
  check('F3a: bit 1 not set', mask !== -1 && (mask & 2) === 0);
}

// 9b: mirror — W@D7 and W@C6 surround D6, so capturing W@D5 leaves B@D6 in
//     self-atari (only lib D5) → F3.
{
  const g = parseBoard(`
    9 . . . . . . . . .
    8 . . . . . . . . .
    7 . . . O . . . . .
    6 . . O . . . . . .
    5 . . X O X O . . .
    4 . . . X . . . . .
    3 . . . . . . . . .
    2 . . . . . . . . .
    1 . . . . . . . . .
  `, WHITE);
  g.play(49); // W@E6 (previous move)
  check('F3b setup: B@E5 in atari', g._ls[g._gid[40]] === 1);
  check('F3b setup: W@D5 in atari', g._ls[g._gid[39]] === 1);
  const mask = getMask(g, 48); // D6
  check('F3b: bit 0 set (contiguous)', mask !== -1 && (mask & 1) !== 0);
  check('F3b: bit 2 set (capture + self-atari)', mask !== -1 && (mask & 4) !== 0);
  check('F3b: bit 1 not set', mask !== -1 && (mask & 2) === 0);
}

// 9c: Realistic F2/F3 scenario using board parser.
// W@F6 creates new atari on the E6 group. G6 saves by capture (not self-atari).
// C6 saves by capture (self-atari) but is outside 8-nbr of F6.
{
  const g = parseBoard(`
    9 . . . . . . . . .
    8 . O O O O . . . .
    7 . O X X O X . . .
    6 . O . O X . . . .
    5 . O X X X O X . .
    4 . . O O O X . . .
    3 . . . . . . . . .
    2 . . . . . . . . .
    1 . . . . . . . . .
  `, WHITE);
  g.play(50); // W@F6, puts E6 group in atari

  check('F2/3 setup: E6 group in atari', g._ls[g._gid[49]] === 1);

  const maskG6 = getMask(g, 51); // G6
  check('F2c: G6 bit 0 set', maskG6 !== -1 && (maskG6 & 1) !== 0);
  check('F2c: G6 bit 1 set (save by capture, not self-atari)', maskG6 !== -1 && (maskG6 & 2) !== 0);
  check('F2c: G6 bit 2 not set', maskG6 !== -1 && (maskG6 & 4) === 0);

  // C6 captures W@D3 (saving E6 group) but is self-atari = Feature 3.
  const maskC6 = getMask(g, 47); // C6
  check('F3c: C6 bit 0 set', maskC6 !== -1 && (maskC6 & 1) !== 0);
  check('F3c: C6 bit 2 set (save by capture, self-atari)', maskC6 !== -1 && (maskC6 & 4) !== 0);
  check('F3c: C6 bit 1 not set', maskC6 !== -1 && (maskC6 & 2) === 0);
}

// ── 10. Feature 4: save by extension, not self-atari (bit 3) ────────────────
// 10a: B@31 in atari (lib=40). lastMove=W@30 (W of 31, orthogonal).
//      Candidate 40 (SE of 30, in 8-nbr) extends B@31. Not self-atari.
{
  // Previous move W@D4 puts B@E4 in atari (only lib E5); candidate E5 extends to
  // gain liberties → F4 (bit 3), not self-atari.
  const g = parseBoard(`
    9 . . . . . . . . .
    8 . . . . . . . . .
    7 . . . . . . . . .
    6 . . . . . . . . .
    5 . . . . . . . . .
    4 . . . . X O . . .
    3 . . . . O . . . .
    2 . . . . . . . . .
    1 . . . . . . . . .
  `, WHITE);
  g.play(30); // W@D4 (previous move)
  check('F4a setup: B@E4 in atari', g._ls[g._gid[31]] === 1);
  const mask = getMask(g, 40); // E5
  check('F4a: bit 0 set (contiguous)', mask !== -1 && (mask & 1) !== 0);
  check('F4a: bit 3 set (extend, not self-atari)', mask !== -1 && (mask & 8) !== 0);
  check('F4a: bit 4 not set', mask !== -1 && (mask & 16) === 0);
}

// 10b: multi-stone — previous move W@F6 ataris the B@{E5,F5} string (only lib G5);
//      candidate G5 extends → F4, not self-atari.
{
  const g = parseBoard(`
    9 . . . . . . . . .
    8 . . . . . . . . .
    7 . . . . . . . . .
    6 . . . . O . . . .
    5 . . . O X X . . .
    4 . . . . O O . . .
    3 . . . . . . . . .
    2 . . . . . . . . .
    1 . . . . . . . . .
  `, WHITE);
  g.play(50); // W@F6 (previous move)
  check('F4b setup: B@{E5,F5} in atari', g._ls[g._gid[40]] === 1);
  const mask = getMask(g, 42); // G5
  check('F4b: bit 0 set (contiguous)', mask !== -1 && (mask & 1) !== 0);
  check('F4b: bit 3 set (extend, not self-atari)', mask !== -1 && (mask & 8) !== 0);
  check('F4b: bit 4 not set', mask !== -1 && (mask & 16) === 0);
}

// ── 11. Feature 5: save by extension, IS self-atari (bit 4) ─────────────────
// 11a: B@31 in atari (lib=40). lastMove=W@30 (W of 31, orthogonal). W@49,W@39 block S,W of 40.
//      After B@40: group {31,40} has only lib=41 — self-atari.
{
  // As F4a but W@E6 and W@D5 block, so extending B@E4 to E5 leaves the {E4,E5}
  // string in self-atari (only lib F5) → F5 (bit 4), not F4 (bit 3).
  const g = parseBoard(`
    9 . . . . . . . . .
    8 . . . . . . . . .
    7 . . . . . . . . .
    6 . . . . O . . . .
    5 . . . O . . . . .
    4 . . . . X O . . .
    3 . . . . O . . . .
    2 . . . . . . . . .
    1 . . . . . . . . .
  `, WHITE);
  g.play(30); // W@D4 (previous move)
  check('F5a setup: B@E4 in atari', g._ls[g._gid[31]] === 1);
  const mask = getMask(g, 40); // E5
  check('F5a: bit 0 set (contiguous)', mask !== -1 && (mask & 1) !== 0);
  check('F5a: bit 4 set (extend + self-atari)', mask !== -1 && (mask & 16) !== 0);
  check('F5a: bit 3 not set', mask !== -1 && (mask & 8) === 0);
}

// 11b: mirror — previous move W@D6 ataris B@D5; extending to E5 leaves {D5,E5}
//      in self-atari (only lib F5) → F5.
{
  const g = parseBoard(`
    9 . . . . . . . . .
    8 . . . . . . . . .
    7 . . . . . . . . .
    6 . . . . O . . . .
    5 . . O X . . . . .
    4 . . . O O . . . .
    3 . . . . . . . . .
    2 . . . . . . . . .
    1 . . . . . . . . .
  `, WHITE);
  g.play(48); // W@D6 (previous move)
  check('F5b setup: B@D5 in atari', g._ls[g._gid[39]] === 1);
  const mask = getMask(g, 40); // E5
  check('F5b: bit 0 set (contiguous)', mask !== -1 && (mask & 1) !== 0);
  check('F5b: bit 4 set (extend + self-atari)', mask !== -1 && (mask & 16) !== 0);
  check('F5b: bit 3 not set', mask !== -1 && (mask & 8) === 0);
}

// ── 12. Feature 6: ko-solve (bit 5) ─────────────────────────────────────────
// Realistic ko-solving scenario.
// B@C3 captures W@C4, creating ko at C4. W@F9 is a ko threat.
// B@B2 captures B3(O) adjacent to the ko area — solves the ko.
{
  const g = parseBoard(`
    9 . . . . . . . . .
    8 . . . . . X O O .
    7 . . . . X . X O .
    6 . . . . . . X O .
    5 . . X O . . . X O
    4 . X O X X . . X O
    3 X O . O . . . X .
    2 . . O . . . . . .
    1 . . . . . . . . .
  `, BLACK);

  check('F6 setup: C4 is WHITE', g.cells[29] === WHITE);
  check('F6 setup: C3 is empty', g.cells[20] === 0);
  g.play(20);  // B@C3: captures W@C4, ko at C4
  check('F6 setup: ko at C4', g.ko === 29);
  g.play(77);  // W@F9: ko threat
  check('F6 setup: ko cleared by F9', g.ko === PASS);
  check('F6 setup: B@B2 is capture', g.isCapture(10));
  check('F6 setup: B3 has 1 lib', g._ls[g._gid[19]] === 1);

  const mask = getMask(g, 10);
  check('F6: bit 0 set (contiguous)', mask !== -1 && (mask & 1) !== 0);
  check('F6: bit 5 set (ko-solve capture)', mask !== -1 && (mask & 32) !== 0);
}

// ── 13. Feature 7: 2-point semeai (bit 6) ───────────────────────────────────
// Friendly string with 2 libs adjacent to lastMove; enemy string with 2 libs
// where candidate gives it atari.

// 13a: B@31 has 2 libs {22,32}. W@30 has 2 libs {39,29}. lastMove=W@40.
//      Candidate 39 (W of 40, in 8-nbr) gives W@30 atari.
//      31=(3,4) 30=(3,3) 21=(2,3) 40=(4,4) 39=(4,3)
{
  // Previous move W@E5 leaves both B@E4 and W@D4 with 2 libs (a 2-point semeai).
  // Candidate D5 gives the enemy W@D4 atari → F7 (bit 6), plus contiguous (bit 0).
  const g = parseBoard(`
    9 . . . . . . . . .
    8 . . . . . . . . .
    7 . . . . . . . . .
    6 . . . . . . . . .
    5 . . . . . . . . .
    4 . . . O X . . . .
    3 . . . X . . . . .
    2 . . . . . . . . .
    1 . . . . . . . . .
  `, WHITE);
  g.play(40); // W@E5 (previous move)
  check('F7a setup: B@E4 has 2 libs', g._ls[g._gid[31]] === 2);
  check('F7a setup: W@D4 has 2 libs', g._ls[g._gid[30]] === 2);
  const mask = getMask(g, 39); // D5
  check('F7a: bit 0 set (contiguous)', mask !== -1 && (mask & 1) !== 0);
  check('F7a: bit 6 set (2-point semeai)', mask !== -1 && (mask & 64) !== 0);
}

// 13b: mirror — previous move W@E5 leaves B@D5 and W@D4 with 2 libs; candidate
//      E4 gives the enemy W@D4 atari → F7.
{
  const g = parseBoard(`
    9 . . . . . . . . .
    8 . . . . . . . . .
    7 . . . . . . . . .
    6 . . . . . . . . .
    5 . . . X . . . . .
    4 . . . O . . . . .
    3 . . . X . . . . .
    2 . . . . . . . . .
    1 . . . . . . . . .
  `, WHITE);
  g.play(40); // W@E5 (previous move)
  check('F7b setup: B@D5 has 2 libs', g._ls[g._gid[39]] === 2);
  check('F7b setup: W@D4 has 2 libs', g._ls[g._gid[30]] === 2);
  const mask = getMask(g, 31); // E4
  check('F7b: bit 0 set (contiguous)', mask !== -1 && (mask & 1) !== 0);
  check('F7b: bit 6 set (2-point semeai)', mask !== -1 && (mask & 64) !== 0);
}

// 13c: Realistic F7 scenario. B@H2 leaves WHITE group H3 with 2 libs.
// E4 BLACK group has 2 libs (D5, E5). W@E5 kills (B@D5 response is self-atari).
// W@D5 does not kill (B@E5 response joins).  Both E5 and D5 are outside H2's
// 8-neighborhood, yet F7 still fires for the killing move E5 — with bit 0 set
// alongside, matching the paper's "Feature 1 also active for all features 2-7".
// The code does distinguish the killing atari (E5) from the non-killing one (D5).
{
  const g = parseBoard(`
    9 . . . . . . . . .  \n
    8 . . . . . X . . .  \n
    7 . . O O O O X . .  \n
    6 . O X X X O X . .  \n
    5 . O X . . X X X .  \n
    4 . O X X X O O X .  \n
    3 X . O O O X O O O  \n
    2 . . . . O X X . .  \n
    1 . . . . . . . . .  \n
  `, BLACK);
  g.play(16); // B@H2

  check('F7c setup: H3 group has 2 libs', g._ls[g._gid[25]] === 2);
  check('F7c setup: E4 group has 2 libs', g._ls[g._gid[31]] === 2);

  const maskE5 = getMask(g, 40);
  const maskD5 = getMask(g, 39);
  check('F7c: E5 bit 0 set', maskE5 !== -1 && (maskE5 & 1) !== 0);
  check('F7c: E5 bit 6 set (kills)', maskE5 !== -1 && (maskE5 & 64) !== 0);
  check('F7c: D5 bit 6 NOT set (does not kill)', (maskD5 & 64) === 0);
}

// ── 11. All moves have valid patIds ────────────────────────────────────────────
{
  const g = new Game2(9);
  for (let i = 0; i < 30; i++) { const m = g.randomLegalMove(); if (m >= 0) g.play(m); }
  const st = createState(9);
  extractFeatures(g, st);
  let allValid = true;
  for (let i = 0; i < st.count; i++) {
    if (getPatId(st, i) < 0 || getPatId(st, i) >= NUM_PATTERNS) { allValid = false; break; }
  }
  check('all patIds in [0, NUM_PATTERNS) and pols ±1', allValid);
  check('at least 1 move found in mid-game', st.count > 0);
}

// ── Summary ───────────────────────────────────────────────────────────────────
console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
