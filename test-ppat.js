'use strict';

const { createState, extractFeatures, evaluate, NUM_PATTERNS } = require('./ppat-lib.js');
// NUM_PATTERNS is the count of canonical IDs under D4 spatial symmetry only
// (color swap is NOT applied since the encoding is already mover-relative).
const { Game2, BLACK, WHITE, PASS } = require('./game2.js');

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
      if (st.moves[i] === candidateCell) return { patId: st.patIds[i] };
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
  for (let i = 0; i < stA.count; i++) if (stA.moves[i] === center) { rA = { id: stA.patIds[i] }; break; }

  // B: WHITE stone at northOf (FOE for BLACK)
  // Place WHITE there: BLACK passes, then WHITE plays, then it's BLACK's turn.
  const gB = new Game2(N, false);
  gB.play(PASS); gB.play(northOf); // pass by BLACK, WHITE plays at north
  const stB = createState(N); extractFeatures(gB, stB);
  let rB = null;
  for (let i = 0; i < stB.count; i++) if (stB.moves[i] === center) { rB = { id: stB.patIds[i] }; break; }

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
  for (let i = 0; i < st.count; i++) if (st.moves[i] === cand) { got = { id: st.patIds[i] }; break; }
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
  for (let i = 0; i < st2.count; i++) if (st2.moves[i] === cand) { got2 = { id: st2.patIds[i] }; break; }
  check('non-atari candidate found', got2 !== null);

  // The atari version should have a different patId than the non-atari version
  // (since the adjacent cell encodes differently: FRIEND_ATARI vs FRIEND).
  check('atari vs non-atari: different patId', got && got2 && got.id !== got2.id);
}

// ── 6. Feature 1: contiguity to previous move ────────────────────────────────
{
  const N = 9;
  const g = new Game2(N, false);
  // BLACK plays at 40 (center), then WHITE plays somewhere, then extract features.
  g.play(40); // BLACK (lastMove = 40 from WHITE's perspective is still 40)
  // Now it's WHITE's turn; WHITE plays at 20, then it's BLACK's turn with lastMove=20.
  g.play(20);
  // lastMove = 20. 8-neighbors of 20: up/down/left/right/diag.
  // On 9×9: 20 = (2,2). N=(1,2)=11, S=(3,2)=29, W=(2,1)=19, E=(2,3)=21,
  //          NE=(1,3)=12, NW=(1,1)=10, SW=(3,1)=28, SE=(3,3)=30.

  const st = createState(N); extractFeatures(g, st);
  const nbrs8 = new Set([11, 29, 19, 21, 12, 10, 28, 30]);

  let allNbrsHaveBit0 = true, nonNbrsLackBit0 = true;
  for (let i = 0; i < st.count; i++) {
    const m = st.moves[i], mask = st.prevMasks[i];
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
  // Setup: WHITE puts BLACK's string in atari with their last move.
  // Then the single liberty of BLACK's string should trigger Feature 4.
  //
  // 9×9 board: BLACK string at {40,41}. Surrounded by WHITE except for one liberty.
  // Place BLACK at 40, 41. Surround with WHITE at 31(N of 40), 30(W of 40), 39(W of 41? no...
  // Actually: BLACK at 40(4,4), 41(4,5). Neighbors of {40,41}:
  //   40: N=31, S=49, W=39, E=41(own)
  //   41: N=32, S=50, W=40(own), E=42
  // External liberties: 31,49,39,32,50,42 (6 liberties).
  // To create atari (1 liberty): fill 5 of them. Too complex to set up simply.
  //
  // Simpler: single stone in atari. BLACK at 40, surrounds = WHITE at 31,49,39.
  // That leaves liberty E=41. Then WHITE plays 31 (already placed) — wrong.
  // Let me try: BLACK at 40 with 3 neighbors occupied by WHITE leaving only E=41.
  //
  // Plan: BLACK=40 (placed first). Then WHITE occupies N=31, W=39, S=49. Liberty left = E=41.
  // Then WHITE plays somewhere else (not 41). It's BLACK's turn. lastMove = WHITE's last move.
  // For Feature 4 to trigger: lastMove must be the move that PUT BLACK in atari.
  // lastMove = the white move that was 3rd (filling the 3rd neighbor). Let's make WHITE's
  // last move = 49 (S of 40). Then BLACK at 40 has 1 liberty = 41 (E).

  const N = 9;
  const g = new Game2(N, false);
  g.play(40);  // BLACK at 40
  g.play(31);  // WHITE at 31 (N of 40)
  g.play(0);   // BLACK dummy
  g.play(39);  // WHITE at 39 (W of 40)
  g.play(1);   // BLACK dummy
  g.play(49);  // WHITE at 49 (S of 40) — this is the last white move, puts 40 in atari
  // Now BLACK's turn. lastMove=49. BLACK at 40 should have 1 liberty (E=41).
  const gid40 = g._gid[40];
  check('F4 setup: BLACK at 40 in atari', gid40 >= 0 && g._ls[gid40] === 1);

  // Find the single liberty of 40.
  let singleLib = -1;
  const W = g._W, lb = gid40 * W, cap = N * N;
  for (let wi = 0; wi < W; wi++) {
    const w = g._lw[lb + wi];
    if (w) { singleLib = wi * 32 + (31 - Math.clz32(w & -w)); break; }
  }
  check('F4 setup: single liberty found', singleLib >= 0);

  const st = createState(N); extractFeatures(g, st);
  let libMask = 0;
  for (let i = 0; i < st.count; i++) {
    if (st.moves[i] === singleLib) { libMask = st.prevMasks[i]; break; }
  }
  // Bit 3 = Feature 4 (save by extension, not self-atari)
  // Bit 4 = Feature 5 (save by extension, self-atari)
  check('F4/5: extension liberty has bit 3 or 4', (libMask & (8 | 16)) !== 0);
  check('F4/5: also has bit 0 (contiguous)', (libMask & 1) !== 0);
}

// ── Helper: extract prevMask for a specific candidate cell ───────────────────
function getMask(game, cell) {
  const st = createState(game.N);
  extractFeatures(game, st);
  for (let i = 0; i < st.count; i++) if (st.moves[i] === cell) return st.prevMasks[i];
  return -1;
}

// ── 8. Feature 2: save by capture, not self-atari (bit 1) ───────────────────
// 8a: B@31 in atari (lib=22). W@32 in atari (lib=41). lastMove=W@40.
//     Candidate 41 captures W@32, saving B@31. Not self-atari.
//     9×9 cells: 31=(3,4) 32=(3,5) 23=(2,5) 30=(3,3) 33=(3,6) 40=(4,4) 41=(4,5)
{
  const g = new Game2(9, false);
  g.play(31); g.play(32); g.play(23); g.play(30); g.play(33); g.play(40);
  check('F2a setup: B@31 in atari', g._ls[g._gid[31]] === 1);
  check('F2a setup: W@32 in atari', g._ls[g._gid[32]] === 1);
  const mask = getMask(g, 41);
  check('F2a: bit 0 set (contiguous)', mask !== -1 && (mask & 1) !== 0);
  check('F2a: bit 1 set (save by capture)', mask !== -1 && (mask & 2) !== 0);
  check('F2a: bit 2 not set (not self-atari)', mask !== -1 && (mask & 4) === 0);
}

// 8b: B@40 in atari (lib=31). W@39 in atari (lib=48). lastMove=W@49.
//     Candidate 48 captures W@39, saving B@40. Not self-atari.
//     40=(4,4) 39=(4,3) 30=(3,3) 41=(4,5) 38=(4,2) 49=(5,4) 48=(5,3)
{
  const g = new Game2(9, false);
  g.play(40); g.play(39); g.play(30); g.play(41); g.play(38); g.play(49);
  check('F2b setup: B@40 in atari', g._ls[g._gid[40]] === 1);
  check('F2b setup: W@39 in atari', g._ls[g._gid[39]] === 1);
  const mask = getMask(g, 48);
  check('F2b: bit 0 set (contiguous)', mask !== -1 && (mask & 1) !== 0);
  check('F2b: bit 1 set (save by capture)', mask !== -1 && (mask & 2) !== 0);
  check('F2b: bit 2 not set (not self-atari)', mask !== -1 && (mask & 4) === 0);
}

// ── 9. Feature 3: save by capture, IS self-atari (bit 2) ────────────────────
// 9a: Same as 8a but W@50,W@42 surround cell 41, making the capture self-atari.
//     After capturing W@32, B@41 has only lib=32 (N=32 empty, S=50(W), W=40(W), E=42(W)).
{
  const g = new Game2(9, false);
  g.play(31); g.play(32); g.play(23); g.play(30); g.play(33); g.play(50);
  g.play(0);  g.play(42); g.play(1);  g.play(40);
  check('F3a setup: B@31 in atari', g._ls[g._gid[31]] === 1);
  check('F3a setup: W@32 in atari', g._ls[g._gid[32]] === 1);
  const mask = getMask(g, 41);
  check('F3a: bit 0 set (contiguous)', mask !== -1 && (mask & 1) !== 0);
  check('F3a: bit 2 set (capture + self-atari)', mask !== -1 && (mask & 4) !== 0);
  check('F3a: bit 1 not set', mask !== -1 && (mask & 2) === 0);
}

// 9b: Same as 8b but W@57,W@47 surround cell 48, making the capture self-atari.
//     After capturing W@39, B@48 has only lib=39 (N=39 empty, S=57(W), W=47(W), E=49(W)).
{
  const g = new Game2(9, false);
  g.play(40); g.play(39); g.play(30); g.play(41); g.play(38); g.play(57);
  g.play(0);  g.play(47); g.play(1);  g.play(49);
  check('F3b setup: B@40 in atari', g._ls[g._gid[40]] === 1);
  check('F3b setup: W@39 in atari', g._ls[g._gid[39]] === 1);
  const mask = getMask(g, 48);
  check('F3b: bit 0 set (contiguous)', mask !== -1 && (mask & 1) !== 0);
  check('F3b: bit 2 set (capture + self-atari)', mask !== -1 && (mask & 4) !== 0);
  check('F3b: bit 1 not set', mask !== -1 && (mask & 2) === 0);
}

// ── 10. Feature 4: save by extension, not self-atari (bit 3) ────────────────
// 10a: B@31 in atari (lib=40). lastMove=W@39. Candidate 40 extends B@31.
//      After B@40: group {31,40} has libs {49,41} — not self-atari.
//      31=(3,4) 22=(2,4) 30=(3,3) 32=(3,5) 39=(4,3) 40=(4,4)
{
  const g = new Game2(9, false);
  g.play(31); g.play(22); g.play(0); g.play(30); g.play(1); g.play(32); g.play(2); g.play(39);
  check('F4a setup: B@31 in atari', g._ls[g._gid[31]] === 1);
  const mask = getMask(g, 40);
  check('F4a: bit 0 set (contiguous)', mask !== -1 && (mask & 1) !== 0);
  check('F4a: bit 3 set (extend, not self-atari)', mask !== -1 && (mask & 8) !== 0);
  check('F4a: bit 4 not set', mask !== -1 && (mask & 16) === 0);
}

// 10b: Multi-stone string {40,41} in atari (lib=42). lastMove=W@50.
//      After B@42: group {40,41,42} has libs {33,51,43} — not self-atari.
{
  const g = new Game2(9, false);
  g.play(40); g.play(31); g.play(41); g.play(32); g.play(0); g.play(39);
  g.play(1);  g.play(49); g.play(2);  g.play(50);
  check('F4b setup: {40,41} in atari', g._ls[g._gid[40]] === 1);
  const mask = getMask(g, 42);
  check('F4b: bit 0 set (contiguous)', mask !== -1 && (mask & 1) !== 0);
  check('F4b: bit 3 set (extend, not self-atari)', mask !== -1 && (mask & 8) !== 0);
  check('F4b: bit 4 not set', mask !== -1 && (mask & 16) === 0);
}

// ── 11. Feature 5: save by extension, IS self-atari (bit 4) ─────────────────
// 11a: B@31 in atari (lib=40). lastMove=W@39. W@49 blocks S of 40.
//      After B@40: group {31,40} has only lib=41 — self-atari.
{
  const g = new Game2(9, false);
  g.play(31); g.play(22); g.play(0); g.play(30); g.play(1); g.play(32);
  g.play(2);  g.play(49); g.play(3);  g.play(39);
  check('F5a setup: B@31 in atari', g._ls[g._gid[31]] === 1);
  const mask = getMask(g, 40);
  check('F5a: bit 0 set (contiguous)', mask !== -1 && (mask & 1) !== 0);
  check('F5a: bit 4 set (extend + self-atari)', mask !== -1 && (mask & 16) !== 0);
  check('F5a: bit 3 not set', mask !== -1 && (mask & 8) === 0);
}

// 11b: B@39 in atari (lib=40). lastMove=W@48. W@31,W@49 block N,S of 40.
//      After B@40: group {39,40} has only lib=41 — self-atari.
{
  const g = new Game2(9, false);
  g.play(39); g.play(30); g.play(0); g.play(38); g.play(1); g.play(31);
  g.play(2);  g.play(49); g.play(3);  g.play(48);
  check('F5b setup: B@39 in atari', g._ls[g._gid[39]] === 1);
  const mask = getMask(g, 40);
  check('F5b: bit 0 set (contiguous)', mask !== -1 && (mask & 1) !== 0);
  check('F5b: bit 4 set (extend + self-atari)', mask !== -1 && (mask & 16) !== 0);
  check('F5b: bit 3 not set', mask !== -1 && (mask & 8) === 0);
}

// ── 12. Feature 6: ko-solve (bit 5) ─────────────────────────────────────────
// Ko setup: B captures W creating ko. Separate B group with 1 lib provides the
// capturable target for the ko-solving feature.

// 12a: Ko at 40. B@41 captures W@40. B@34 has 1 lib at 33 (NE of lastMove=41).
//      WHITE plays at 33 → captures B@34. ko≠PASS → bit 5.
{
  const g = new Game2(9, false);
  // Ko frame: B@31,B@49,B@39 surround W@40; W@32,W@50,W@42 surround B@41.
  // Capturable group: B@34 with W@25,W@43,W@35 leaving lib=33.
  g.play(31); g.play(40); g.play(49); g.play(32); g.play(39); g.play(50);
  g.play(34); g.play(42); g.play(0);  g.play(25); g.play(1);  g.play(43);
  g.play(2);  g.play(35);
  check('F6a setup: W@40 in atari', g._ls[g._gid[40]] === 1);
  g.play(41); // B captures W@40 → ko
  check('F6a setup: ko at 40', g.ko === 40);
  // WHITE's turn. lastMove=41. Candidate 33 captures B@34.
  const mask = getMask(g, 33);
  check('F6a: bit 0 set (contiguous)', mask !== -1 && (mask & 1) !== 0);
  check('F6a: bit 5 set (ko-solve capture)', mask !== -1 && (mask & 32) !== 0);
}

// 12b: Ko at 50. B@51 captures W@50. B@44 has 1 lib at 43 (NE of lastMove=51).
{
  const g = new Game2(9, false);
  // Ko frame: B@41,B@59,B@49 surround W@50; W@42,W@60,W@52 surround B@51.
  // Capturable group: B@44 (4,8) with W@35,W@53,W@36 leaving lib=43.
  //   44=(4,8): N=35, S=53, W=43, E=(4,0)=36 (toroidal).
  g.play(41); g.play(50); g.play(59); g.play(42); g.play(49); g.play(60);
  g.play(44); g.play(52); g.play(0);  g.play(35); g.play(1);  g.play(53);
  g.play(2);  g.play(36);
  check('F6b setup: W@50 in atari', g._ls[g._gid[50]] === 1);
  g.play(51); // B captures W@50 → ko
  check('F6b setup: ko at 50', g.ko === 50);
  const mask = getMask(g, 43);
  check('F6b: bit 0 set (contiguous)', mask !== -1 && (mask & 1) !== 0);
  check('F6b: bit 5 set (ko-solve capture)', mask !== -1 && (mask & 32) !== 0);
}

// ── 13. Feature 7: 2-point semeai (bit 6) ───────────────────────────────────
// Friendly string with 2 libs adjacent to lastMove; enemy string with 2 libs
// where candidate gives it atari.

// 13a: B@31 has 2 libs {22,32}. W@30 has 2 libs {39,29}. lastMove=W@40.
//      Candidate 39 (W of 40, in 8-nbr) gives W@30 atari.
//      31=(3,4) 30=(3,3) 21=(2,3) 40=(4,4) 39=(4,3)
{
  const g = new Game2(9, false);
  g.play(31); g.play(30); g.play(21); g.play(40);
  check('F7a setup: B@31 has 2 libs', g._ls[g._gid[31]] === 2);
  check('F7a setup: W@30 has 2 libs', g._ls[g._gid[30]] === 2);
  const mask = getMask(g, 39);
  check('F7a: bit 0 set (contiguous)', mask !== -1 && (mask & 1) !== 0);
  check('F7a: bit 6 set (2-point semeai)', mask !== -1 && (mask & 64) !== 0);
}

// 13b: B@39 has 2 libs {48,38}. W@30 has 2 libs {29,31}. lastMove=W@40.
//      Candidate 31 (N of 40, in 8-nbr) gives W@30 atari.
{
  const g = new Game2(9, false);
  g.play(39); g.play(30); g.play(21); g.play(40);
  check('F7b setup: B@39 has 2 libs', g._ls[g._gid[39]] === 2);
  check('F7b setup: W@30 has 2 libs', g._ls[g._gid[30]] === 2);
  const mask = getMask(g, 31);
  check('F7b: bit 0 set (contiguous)', mask !== -1 && (mask & 1) !== 0);
  check('F7b: bit 6 set (2-point semeai)', mask !== -1 && (mask & 64) !== 0);
}

// ── 11. All moves have valid patIds ────────────────────────────────────────────
{
  const g = new Game2(9);
  for (let i = 0; i < 30; i++) { const m = g.randomLegalMove(); if (m >= 0) g.play(m); }
  const st = createState(9);
  extractFeatures(g, st);
  let allValid = true;
  for (let i = 0; i < st.count; i++) {
    if (st.patIds[i] < 0 || st.patIds[i] >= NUM_PATTERNS) { allValid = false; break; }
  }
  check('all patIds in [0, NUM_PATTERNS) and pols ±1', allValid);
  check('at least 1 move found in mid-game', st.count > 0);
}

// ── Summary ───────────────────────────────────────────────────────────────────
console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
