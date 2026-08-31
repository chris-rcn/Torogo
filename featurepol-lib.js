'use strict';

// featurepol — a hash-keyed, spec-driven linear softmax move policy.
//
// Like npat (npat-lib.js) it scores every legal move with a linear sum of
// sparse feature weights and samples from a softmax, learning by REINFORCE.
// Unlike npat, the feature set is chosen at runtime from a CLI --spec string
// and every feature key is a 32-bit hash (the "hashing trick"), so new feature
// combinations need no hand-laid key layout.
//
// SPEC GRAMMAR
//   spec   := space (',' space)*          comma-separated INDEPENDENT feature
//                                         spaces; each contributes one weight.
//   space  := term ('+' term)*            '+'-joined terms form ONE composite
//                                         (conjunction) hashed into a single key.
//   term   := name [number]               e.g. capture6, atari4, stones4, ladderStatus
//
//   Example:  --spec 'capture6,atari4+stones4'
//     space A = capture6                  (capture-size bucket 0..6)
//     space B = atari4+stones4            (atari-size bucket 0..4 conjoined with
//                                          the encoded nearest-4 cell pattern)
//
// FEATURE TERMS (extend by adding a case to _makeTerm):
//   stones<n>   D4-canonical encoded states (empty/friend/foe) of the nearest n
//               cells; n in {4,8,12,20} (the D4-closed rings).
//   adjLib<n>   D4-canonical 4 orthogonal neighbours, each a stone encoded with its
//               chain's liberty count capped at n (radix 2n+1).  A descriptor.
//   stone8AdjLib<n>  JOINT liberty-aware 3×3 pattern: the 8 nearest cells canonicalised
//               as ONE unit (orthogonals liberty-aware cap n, diagonals shape-only), so
//               shape+liberties stay in register — what stones8+adjLib4 cannot do (it
//               canonicalises each half separately).  n=2 ≈ ppat's pattern.  Descriptor.
//
// The size<n> terms below are CUMULATIVE (thermometer): each expands at parse
// time into n additive "≥k present" indicator spaces, so a size-s feature lights
// up levels 1..min(s,n) and the logit sums over size — like npat's tactical
// slots, which gives far better size generalisation than one-hot bucketing.
//   capture<n>  stones this move captures (cumulative over size, up to n)
//   atari<n>    total enemy stones this move puts in atari, summed over chains
//   selfAtari<n> stone count of the resulting self-atari'd group (own group → 1
//               liberty); 0 if the move does not self-atari
//   ko          binary: 1 iff the move creates a ko (captures one lone stone
//               into a ko shape), else 0
//   anyKo       binary BOARD-CONTEXT flag: 1 iff a ko is currently active on the
//               board (some point is ko-banned).  Same value for every candidate
//               move, so only meaningful in conjunction (e.g. stones8+anyKo).
//   flags       6-bit descriptor combining tactical event flags for the move:
//               self-atari(1) | capture(2) | atari(4) | ko(8) | join≥2(16) | local(32).
//               Always emits one key (mask 0 = none, its own category).
//   local       binary LOCALITY flag: 1 iff the move is in the 8-neighbourhood
//               (Moore) of the previous move; 0 (incl. no previous move) emits
//               nothing.  The one feature that conditions on the opponent's move.
//   koSolve     binary (ppat Feature 6): 1 iff the move captures an atari'd enemy
//               group adjacent to my own ko-stone (game.koStone[cur+1]) — resolving
//               a ko I just made by capturing the threat rather than fighting it.
//   dist<n>     cumulative blended-toroidal distance to the previous move, as
//               levels floor(Game2.distance*2-1) capped at n (orthogonal-adjacent
//               = 1, rising with distance); no previous move emits nothing.
//   ladderStatus 4-bit ladder presence mask at the move from ladder2
//               (urgent-kill/urgent-save/wasted-extend/wasted-attack)
//   urgentKill<n> / urgentSave<n> / wastedExtend<n> / wastedAttack<n>
//               one ladder flag's summed chain stone-count (cumulative, up to n)
//   hpat<n>     rank of the move under a fixed hierarchical-pattern (hpatterns)
//               model, mover-relative: 1 = that model's top choice, 2 = its
//               second, up to n.  CUMULATIVE in rank quality: rank r contributes
//               levels 1..n+1-r, so the top choice lights every level and rank n
//               only level 1; rank past n emits nothing and shares the implicit
//               zero baseline.  Still exactly n weights per space combination,
//               but level k is estimated from every move ranked <= n+1-k rather
//               than from one rank alone.  The model file comes from FP_HPAT_DATA
//               and is never trained here.
//
// BROWSER-COMPATIBLE: no Node-only APIs at top level.

(function () {

const Util = (typeof require === 'function') ? require('./util.js') : window.Util;
const { PASS, BLACK }          = Util.load('./game2.js', 'Game2');
const { game3FromGame2 }       = Util.load('./game3.js', 'Game3');
const { getAllLadderStatuses } = Util.load('./ladder2.js', 'Ladder2');
const HPatterns                = Util.load('./hpatterns.js', 'HPatterns');

// ── 32-bit hashing ────────────────────────────────────────────────────────────

function _mix32(x) {
  x = x >>> 0;
  x = Math.imul(x ^ (x >>> 16), 0x45d9f3b);
  x = Math.imul(x ^ (x >>> 16), 0x45d9f3b);
  return (x ^ (x >>> 16)) >>> 0;
}
function _hashCombine(h, v) { return (_mix32((h >>> 0) ^ _mix32(v))) >>> 0; }
function _hashStr(s) {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 0x01000193);
  return h >>> 0;
}

// ── Per-move feature helpers ──────────────────────────────────────────────────

// Number of enemy stones captured by playing empty cell idx (sum over distinct
// adjacent enemy chains whose last liberty is idx).
function _captureCount(game, idx) {
  const foe = -game.current;
  const nbr = game._nbr, cells = game.cells, gid = game._gid, ls = game._ls, ss = game._ss;
  const b = idx * 4;
  let total = 0, s0 = -1, s1 = -1, s2 = -1;
  for (let d = 0; d < 4; d++) {
    const ni = nbr[b + d];
    if (cells[ni] !== foe) continue;
    const g = gid[ni];
    if (ls[g] !== 1) continue;
    if (g === s0 || g === s1 || g === s2) continue;
    if (s0 < 0) s0 = g; else if (s1 < 0) s1 = g; else s2 = g;
    total += ss[g];
  }
  return total;
}

// Total enemy stones this move puts in atari: the summed sizes of the distinct
// adjacent enemy chains with exactly 2 liberties (playing idx reduces each to 1).
// Opponent stones only — never the mover's own (self-atari is excluded by
// construction, since we only sum foe chains).  0 if none.
function _atariStones(game, idx) {
  const foe = -game.current;
  const nbr = game._nbr, cells = game.cells, gid = game._gid, ls = game._ls, ss = game._ss;
  const b = idx * 4;
  let total = 0, s0 = -1, s1 = -1, s2 = -1;
  for (let d = 0; d < 4; d++) {
    const ni = nbr[b + d];
    if (cells[ni] !== foe) continue;
    const g = gid[ni];
    if (ls[g] !== 2) continue;
    if (g === s0 || g === s1 || g === s2) continue;
    if (s0 < 0) s0 = g; else if (s1 < 0) s1 = g; else s2 = g;
    total += ss[g];
  }
  return total;
}

// Number of DISTINCT friendly chains adjacent to empty cell idx (0..4) — i.e. how
// many of the mover's own groups this move would connect together.
function _adjFriendlyChains(game, idx) {
  const me = game.current;
  const nbr = game._nbr, cells = game.cells, gid = game._gid;
  const b = idx * 4;
  let n = 0, s0 = -1, s1 = -1, s2 = -1;
  for (let d = 0; d < 4; d++) {
    const ni = nbr[b + d];
    if (cells[ni] !== me) continue;
    const g = gid[ni];
    if (g === s0 || g === s1 || g === s2) continue;
    if (s0 < 0) s0 = g; else if (s1 < 0) s1 = g; else s2 = g;
    n++;
  }
  return n;
}

// Per-cell ladder sizes.  For each cell, out[cell*4 + flag] is the summed stone
// count of the chains for which that cell is a flagged liberty, where flag is:
//   0 urgent-kill, 1 urgent-save, 2 wasted-extend, 3 wasted-attack.
// (ladderStatus reads presence/0; urgentKill/urgentSave/wastedExtend/wastedAttack<n> read size.)
const URGENT_KILL = 0, URGENT_SAVE = 1, WASTED_EXTEND = 2, WASTED_ATTACK = 3;
function _buildLadderSizes(game, game3, out) {
  out.fill(0);
  const cap = game.N * game.N;
  if (game.emptyCount === cap) return out;
  const infos = getAllLadderStatuses(game3);
  const cur = game.current;
  for (const info of infos) {
    if (!info.status) continue;
    const { libs, moverSucceeds, urgentLibs } = info.status;
    const defending = info.color === cur;
    const size = game3.groupSize(info.gid);
    let flag, targets;
    if (urgentLibs.length > 0)  { flag = defending ? URGENT_SAVE : URGENT_KILL; targets = urgentLibs; }
    else if (!moverSucceeds)    { flag = defending ? WASTED_EXTEND : WASTED_ATTACK; targets = libs; }
    else continue;
    for (const lib of targets) out[lib * 4 + flag] += size;
  }
  return out;
}

// ── Spec parsing ──────────────────────────────────────────────────────────────

// Nearest-cell offsets (dy, dx) in increasing Euclidean distance; stones<n>
// reads the first n.  Rings: 4 orthogonal (d=1), 4 diagonal (√2), 4 distance-2
// orthogonal (2), 8 knight's-move (√5).  Order is nearest-first so the offset
// lists nest (stones4 ⊂ stones8 ⊂ stones12 ⊂ stones20).
const _NEAR_OFFSETS = [
  [-1, 0], [0, 1], [1, 0], [0, -1],            //  0..3   d = 1
  [-1, 1], [1, 1], [1, -1], [-1, -1],          //  4..7   d = √2
  [-2, 0], [0, 2], [2, 0], [0, -2],            //  8..11  d = 2
  [-2, -1], [-2, 1], [-1, 2], [1, 2],          // 12..19  d = √5
  [2, 1], [2, -1], [1, -2], [-1, -2],
];
const NEAR_MAX = _NEAR_OFFSETS.length;   // 20

// D4 index permutations of the nearest-cell offsets, for canonicalising the
// stones feature.  _NEAR_PERM[s*NEAR_MAX + i] = the offset index whose cell sits
// at logical position i under board symmetry s.  Each closed prefix (4/8/12/20)
// maps into itself, so stones<n> is canonicalised by taking the min base-3
// encoding over the 8 symmetries — making the resulting hash D4-invariant.
const _NEAR_PERM = new Int32Array(8 * NEAR_MAX);
const _STONES_N = new Set([4, 8, 12, 20]);   // D4-closed nearest-cell prefixes
const _cvScratch = new Int8Array(NEAR_MAX);

// Unrolled fast path for the n=4 case (stones4, adjLib): the 4 orthogonal cells
// under the 8 D4 symmetries.  Loads cv[0..3] once into locals and evaluates the 8
// base-radix encodings as straight-line arithmetic — no _NEAR_PERM gather, no inner
// loop, and the 8 independent encodings expose instruction-level parallelism.  The
// 8 index permutations are exactly the first-4 columns of _NEAR_PERM (4 rotations +
// 4 reflections), so the result is identical to _canonRadix(cv, 4, radix); the
// self-check below verifies this exhaustively at load.
function _canonRadix4(cv, radix) {
  const c0 = cv[0], c1 = cv[1], c2 = cv[2], c3 = cv[3], R = radix;
  let best = ((c0 * R + c1) * R + c2) * R + c3, v;       // [0,1,2,3]
  v = ((c1 * R + c2) * R + c3) * R + c0; if (v < best) best = v;   // [1,2,3,0]
  v = ((c2 * R + c3) * R + c0) * R + c1; if (v < best) best = v;   // [2,3,0,1]
  v = ((c3 * R + c0) * R + c1) * R + c2; if (v < best) best = v;   // [3,0,1,2]
  v = ((c0 * R + c3) * R + c2) * R + c1; if (v < best) best = v;   // [0,3,2,1]
  v = ((c2 * R + c1) * R + c0) * R + c3; if (v < best) best = v;   // [2,1,0,3]
  v = ((c3 * R + c2) * R + c1) * R + c0; if (v < best) best = v;   // [3,2,1,0]
  v = ((c1 * R + c0) * R + c3) * R + c2; if (v < best) best = v;   // [1,0,3,2]
  return best;
}

// Unrolled fast path for stone8AdjLib (the n=8 mixed-radix join): the 4 orthogonal
// cells (base R = 2n+1) followed by the 4 diagonals (base 3), under the 8 D4 symmetries,
// as ONE canonical encoding.  Loads cv[0..7] once and evaluates the 8 encodings straight-
// line — no _NEAR_PERM gather.  For each symmetry the orthogonals permute as the first-4
// columns of _NEAR_PERM (as in _canonRadix4) and the diagonals as the next-4 columns; the
// result is identical to the table-driven loop, verified exhaustively at load.
function _canon8AdjLib(cv, R) {
  const c0 = cv[0], c1 = cv[1], c2 = cv[2], c3 = cv[3], c4 = cv[4], c5 = cv[5], c6 = cv[6], c7 = cv[7];
  // each line: orthogonals [o0,o1,o2,o3] base R, then diagonals [d0,d1,d2,d3] base 3
  let best = ((((((c0 * R + c1) * R + c2) * R + c3) * 3 + c4) * 3 + c5) * 3 + c6) * 3 + c7, v;       // o[0,1,2,3] d[4,5,6,7]
  v = ((((((c1 * R + c2) * R + c3) * R + c0) * 3 + c5) * 3 + c6) * 3 + c7) * 3 + c4; if (v < best) best = v;   // o[1,2,3,0] d[5,6,7,4]
  v = ((((((c2 * R + c3) * R + c0) * R + c1) * 3 + c6) * 3 + c7) * 3 + c4) * 3 + c5; if (v < best) best = v;   // o[2,3,0,1] d[6,7,4,5]
  v = ((((((c3 * R + c0) * R + c1) * R + c2) * 3 + c7) * 3 + c4) * 3 + c5) * 3 + c6; if (v < best) best = v;   // o[3,0,1,2] d[7,4,5,6]
  v = ((((((c0 * R + c3) * R + c2) * R + c1) * 3 + c7) * 3 + c6) * 3 + c5) * 3 + c4; if (v < best) best = v;   // o[0,3,2,1] d[7,6,5,4]
  v = ((((((c2 * R + c1) * R + c0) * R + c3) * 3 + c5) * 3 + c4) * 3 + c7) * 3 + c6; if (v < best) best = v;   // o[2,1,0,3] d[5,4,7,6]
  v = ((((((c3 * R + c2) * R + c1) * R + c0) * 3 + c6) * 3 + c5) * 3 + c4) * 3 + c7; if (v < best) best = v;   // o[3,2,1,0] d[6,5,4,7]
  v = ((((((c1 * R + c0) * R + c3) * R + c2) * 3 + c4) * 3 + c7) * 3 + c6) * 3 + c5; if (v < best) best = v;   // o[1,0,3,2] d[4,7,6,5]
  return best;
}

// Unrolled fast path for the n=8 uniform-radix case (stones8): same 8 D4 permutations
// as _canon8AdjLib, but every cell uses the same radix (no orthogonal/diagonal split).
// Loads cv[0..7] once and evaluates the 8 encodings straight-line; identical to
// _canonRadix(cv, 8, radix), verified at load.
function _canon8(cv, R) {
  const c0 = cv[0], c1 = cv[1], c2 = cv[2], c3 = cv[3], c4 = cv[4], c5 = cv[5], c6 = cv[6], c7 = cv[7];
  let best = ((((((c0 * R + c1) * R + c2) * R + c3) * R + c4) * R + c5) * R + c6) * R + c7, v;       // [0,1,2,3,4,5,6,7]
  v = ((((((c1 * R + c2) * R + c3) * R + c0) * R + c5) * R + c6) * R + c7) * R + c4; if (v < best) best = v;   // [1,2,3,0,5,6,7,4]
  v = ((((((c2 * R + c3) * R + c0) * R + c1) * R + c6) * R + c7) * R + c4) * R + c5; if (v < best) best = v;   // [2,3,0,1,6,7,4,5]
  v = ((((((c3 * R + c0) * R + c1) * R + c2) * R + c7) * R + c4) * R + c5) * R + c6; if (v < best) best = v;   // [3,0,1,2,7,4,5,6]
  v = ((((((c0 * R + c3) * R + c2) * R + c1) * R + c7) * R + c6) * R + c5) * R + c4; if (v < best) best = v;   // [0,3,2,1,7,6,5,4]
  v = ((((((c2 * R + c1) * R + c0) * R + c3) * R + c5) * R + c4) * R + c7) * R + c6; if (v < best) best = v;   // [2,1,0,3,5,4,7,6]
  v = ((((((c3 * R + c2) * R + c1) * R + c0) * R + c6) * R + c5) * R + c4) * R + c7; if (v < best) best = v;   // [3,2,1,0,6,5,4,7]
  v = ((((((c1 * R + c0) * R + c3) * R + c2) * R + c4) * R + c7) * R + c6) * R + c5; if (v < best) best = v;   // [1,0,3,2,4,7,6,5]
  return best;
}

// Canonical encoding of the first n cell values in cv at the given radix: min
// value over the 8 D4 symmetries (so the result is D4-invariant).  n must be a
// closed prefix; every cv[i] must be in [0, radix).
function _canonRadix(cv, n, radix) {
  if (n === 4) return _canonRadix4(cv, radix);
  if (n === 8) return _canon8(cv, radix);
  let best = Infinity;
  for (let s = 0; s < 8; s++) {
    const po = s * NEAR_MAX;
    let raw = 0;
    for (let i = 0; i < n; i++) raw = raw * radix + cv[_NEAR_PERM[po + i]];
    if (raw < best) best = raw;
  }
  return best;
}
// stones<n>: ternary cell values (0 empty / 1 own / 2 enemy).
function _canonStones(cv, n) { return _canonRadix(cv, n, 3); }
(function () {
  const d4 = [
    (r, c) => [ r,  c], (r, c) => [ c, -r], (r, c) => [-r, -c], (r, c) => [-c,  r],
    (r, c) => [ r, -c], (r, c) => [-r,  c], (r, c) => [ c,  r], (r, c) => [-c, -r],
  ];
  const key = (r, c) => r * 100 + c;
  const index = new Map();
  for (let i = 0; i < NEAR_MAX; i++) index.set(key(_NEAR_OFFSETS[i][0], _NEAR_OFFSETS[i][1]), i);
  for (let s = 0; s < 8; s++) {
    for (let i = 0; i < NEAR_MAX; i++) {
      const [r, c] = d4[s](_NEAR_OFFSETS[i][0], _NEAR_OFFSETS[i][1]);
      const j = index.get(key(r, c));
      if (j === undefined) throw new Error('featurepol: nearest-cell offsets are not D4-closed');
      _NEAR_PERM[s * NEAR_MAX + i] = j;
    }
  }
})();

// Self-check: the unrolled n=4 fast path must agree with the generic table-driven
// canonicalisation exactly — any divergence would shift canonical keys and silently
// invalidate every trained model.  Exhaustive over radix 5: the canonical selection
// depends only on the relative order of the 4 cell values, and radix 5 (≥4 distinct
// symbols) exercises every weak ordering of 4 cells, so agreement here holds for all
// radices.  625 patterns — trivial cost.
(function () {
  const cv = new Int8Array(NEAR_MAX);
  const radix = 5;
  for (let a = 0; a < radix; a++) for (let b = 0; b < radix; b++)
    for (let c = 0; c < radix; c++) for (let d = 0; d < radix; d++) {
      cv[0] = a; cv[1] = b; cv[2] = c; cv[3] = d;
      let ref = Infinity;                                    // generic, straight from _NEAR_PERM
      for (let s = 0; s < 8; s++) {
        const po = s * NEAR_MAX;
        let raw = 0;
        for (let i = 0; i < 4; i++) raw = raw * radix + cv[_NEAR_PERM[po + i]];
        if (raw < ref) ref = raw;
      }
      if (_canonRadix4(cv, radix) !== ref) throw new Error('featurepol: _canonRadix4 disagrees with generic canonicalisation');
    }
})();

// Self-check for the unrolled stone8AdjLib (n=8 mixed radix), same rationale as above.
// Exhaustive over orthogonals in [0,4) (≥4 distinct → every weak ordering of the 4
// orthogonals, incl. ties that hand the decision to the diagonals) × diagonals in [0,3)
// (their real range).  Validates the permutation structure for all radices.  20736 patterns.
(function () {
  const cv = new Int8Array(NEAR_MAX);
  const Ro = 4;
  const total = 4 * 4 * 4 * 4 * 3 * 3 * 3 * 3;
  for (let code = 0; code < total; code++) {
    let x = code;
    cv[0] = x % 4; x = (x / 4) | 0; cv[1] = x % 4; x = (x / 4) | 0;
    cv[2] = x % 4; x = (x / 4) | 0; cv[3] = x % 4; x = (x / 4) | 0;
    cv[4] = x % 3; x = (x / 3) | 0; cv[5] = x % 3; x = (x / 3) | 0;
    cv[6] = x % 3; x = (x / 3) | 0; cv[7] = x % 3;
    let ref = Infinity;                                    // generic mixed-radix, from _NEAR_PERM
    for (let s = 0; s < 8; s++) {
      const po = s * NEAR_MAX;
      let raw = 0;
      for (let i = 0; i < 4; i++) raw = raw * Ro + cv[_NEAR_PERM[po + i]];
      for (let i = 4; i < 8; i++) raw = raw * 3 + cv[_NEAR_PERM[po + i]];
      if (raw < ref) ref = raw;
    }
    if (_canon8AdjLib(cv, Ro) !== ref) throw new Error('featurepol: _canon8AdjLib disagrees with generic canonicalisation');
  }
})();

// Self-check for the unrolled _canon8 (n=8 uniform radix, used by stones8).  Two passes:
// (a) EXHAUSTIVE over radix 3 — stones8's complete real pattern space (3^8 = 6561); and
// (b) a deterministic LCG sample at radix 9, which (unlike radix 3) can give 8 distinct
// values, so it exercises all-distinct cell orderings and would catch any permutation
// transcription error that ties at radix 3 hide.  Both compared to the generic routine.
(function () {
  const cv = new Int8Array(NEAR_MAX);
  const genRef = (radix) => {
    let best = Infinity;
    for (let s = 0; s < 8; s++) {
      const po = s * NEAR_MAX;
      let raw = 0;
      for (let i = 0; i < 8; i++) raw = raw * radix + cv[_NEAR_PERM[po + i]];
      if (raw < best) best = raw;
    }
    return best;
  };
  for (let code = 0; code < 6561; code++) {            // (a) exhaustive radix 3
    let x = code;
    for (let i = 0; i < 8; i++) { cv[i] = x % 3; x = (x / 3) | 0; }
    if (_canon8(cv, 3) !== genRef(3)) throw new Error('featurepol: _canon8 disagrees with generic canonicalisation (radix 3)');
  }
  let seed = 0x9e3779b1;                                // (b) sampled radix 9
  for (let t = 0; t < 30000; t++) {
    for (let i = 0; i < 8; i++) { seed = (Math.imul(seed, 1103515245) + 12345) & 0x7fffffff; cv[i] = seed % 9; }
    if (_canon8(cv, 9) !== genRef(9)) throw new Error('featurepol: _canon8 disagrees with generic canonicalisation (radix 9)');
  }
})();

// ── hpat: rank under a hierarchical-pattern model ────────────────────────────
//
// hpat<n> ranks every candidate of the CURRENT position by the hpatterns model's
// Delta-z (mover-relative: higher = better for the side to move) and feeds that
// rank in as a CUMULATIVE (thermometer) size, exactly like capture<n> and the
// other size terms: rank r maps to size n+1-r, so the model's top choice
// contributes levels 1..n, its second levels 1..n-1, and rank n only level 1.
// Rank > n maps to size 0, which gates the space off, so everything past n
// shares the implicit zero baseline.
//
// The term still costs exactly n weights per space combination, but they now
// partition by THRESHOLD rather than by exact rank: level k fires for every move
// ranked <= n+1-k, so level 1 ("in the top n") is estimated from n times more
// data than level n ("is the top choice").  The measured rank-quality curve is
// smooth and monotone through the first several ranks, which is precisely where
// sharing statistics across adjacent ranks pays and one-hot ranks waste data.
//
// The model is a FIXED external evaluator; featurepol never trains it.  It comes
// from the file named by FP_HPAT_DATA (a train-hpatterns save) -- or, in a
// browser, from window.hpatternsModel.
//
// Ranking is a whole-position computation, not a per-move one, so it runs once
// per position in a prepare hook rather than in an evalFn.

let _hpatModel = null;
let _hpatIncremental = false;     // saturated caps => deltaZ is usable
function _hpatLoad() {
  if (_hpatModel) return _hpatModel;
  let raw;
  const envPath = (typeof process !== 'undefined' && process.env) ? process.env.FP_HPAT_DATA : null;
  if (envPath) {
    raw = require(require('path').resolve(envPath));
  } else if (typeof window !== 'undefined' && window.hpatternsModel) {
    raw = window.hpatternsModel;
  } else {
    throw new Error('featurepol: the hpat<n> feature needs a hierarchical-pattern model — set FP_HPAT_DATA to a train-hpatterns save file');
  }
  const model = HPatterns.createModel(raw.maxStones, raw.maxSize);
  model.weights = HPatterns.weightsMap(raw);
  // A BINDING stone cap needs per-window stone counts that the incremental path
  // does not maintain, so deltaZ is only valid when every cap is saturated.
  _hpatIncremental = HPatterns.saturatedOnly(model.maxStones);
  _hpatModel = model;
  return model;
}

// Swap in shadow hash buffers so a speculative full extraction does not clobber
// the buffers deltaZ reads for the current position.
function _hpatFallback(model, fn) {
  if (!model._fbBufs) {
    model._fbBufs  = model._hBufs.map(b => new Int32Array(b.length));
    model._fbBufsI = model._hBufsInv.map(b => new Int32Array(b.length));
  }
  const sN = model._hBufs, sI = model._hBufsInv;
  model._hBufs = model._fbBufs; model._hBufsInv = model._fbBufsI;
  const r = fn();
  model._hBufs = sN; model._hBufsInv = sI;
  return r;
}

let _hpatRank = null;      // _hpatRank[boardIdx] = 1-based rank, 0 = not a candidate
let _hpatOrder = null;     // scratch: candidate board indices, sorted best-first
let _hpatScore = null;     // scratch: _hpatScore[boardIdx] = mover-relative score

// Fill _hpatRank for every legal non-true-eye move of ctx.game.  Runs once per
// position; each hpat<n> evalFn then just reads and clamps.
function _hpatPrepare(ctx) {
  const game = ctx.game, N = game.N, cap = N * N;
  const model = _hpatLoad(), w = model.weights;
  if (!_hpatRank || _hpatRank.length < cap) {
    _hpatRank  = new Int32Array(cap);
    _hpatOrder = new Int32Array(cap);
    _hpatScore = new Float64Array(cap);
  }
  _hpatRank.fill(0, 0, cap);
  const black = ctx.cur === BLACK;   // hpat z is the BLACK-wins logit

  let zBase = 0;
  if (_hpatIncremental) {
    HPatterns.extractFeatures(game, model);        // primes the buffers deltaZ reads
    const zCum = HPatterns.zFromBuffers(model, w, N);
    zBase = zCum[zCum.length - 1];
  }

  const emC = game._emptyCells, ec = game.emptyCount;
  let n = 0;
  for (let ei = 0; ei < ec; ei++) {
    const idx = emC[ei];
    if (!game.isLegal(idx) || game.isTrueEye(idx)) continue;
    let z;
    if (_hpatIncremental) {
      const d = HPatterns.deltaZ(game, model, w, idx);
      // NaN = the move captures; the incremental path cannot express that, and
      // capture moves are exactly where this model disagrees most usefully, so
      // pay for a full extraction rather than dropping them.
      z = Number.isNaN(d)
        ? _hpatFallback(model, () => _zOf(HPatterns.extractFeatures(game, model, undefined, idx), w))
        : zBase + d;
    } else {
      z = _zOf(HPatterns.extractFeatures(game, model, undefined, idx), w);
    }
    _hpatScore[idx] = black ? z : -z;             // mover-relative: higher = better
    _hpatOrder[n++] = idx;
  }
  const order = _hpatOrder.subarray(0, n);
  order.sort((a, b) => _hpatScore[b] - _hpatScore[a]);
  for (let r = 0; r < n; r++) _hpatRank[order[r]] = r + 1;
}

// Raw logit of an hpatterns feature set (HPatterns.evaluateFeatures returns the
// sigmoid; ranking only needs the monotone pre-image).
function _zOf(f, weights) {
  let z = 0;
  const keys = f.keys, pols = f.pols, count = f.count;
  for (let i = 0; i < count; i++) {
    const v = weights.get(keys[i]);
    if (v !== undefined) z += pols[i] * v;
  }
  return z;
}

// Build one feature term { str, kind, param, salt, evalFn, needsLadder } from a
// token like "capture6" / "stones4" / "ladderStatus".
function _makeTerm(str) {
  // kind = leading word (may contain interior digits, e.g. stone8AdjLib); param =
  // the OPTIONAL trailing run of digits.  Lazy kind + greedy trailing \d* split them.
  const m = /^([a-zA-Z][a-zA-Z0-9]*?)(\d*)$/.exec(str);
  if (!m) throw new Error(`featurepol: bad feature term "${str}"`);
  const kind = m[1];
  const param = m[2] ? parseInt(m[2], 10) : null;
  const salt = _hashStr(str);
  // A size<n> term is CUMULATIVE: it carries sizeFn (the raw size) and is expanded
  // by parseSpec into n additive "≥k present" indicator spaces (thermometer
  // encoding) so the logit sums over size, like npat's tactical slots.  Other
  // terms set evalFn directly (one value → one key).
  // maxNear: how many of the nearest cells this term reads from the nearNbr table
  // (0 if it reads none).  parseSpec takes the max across the spec to size the table.
  let evalFn = null, sizeFn = null, cumulative = false, needsLadder = false, binary = false, maxNear = 0;
  let prepare = null;
  switch (kind) {
    case 'hpat': {
      // Rank under the hpatterns model as a cumulative size: rank r -> n+1-r, so
      // the top choice lights levels 1..n and rank n only level 1.  Rank > n
      // gives 0, which gates the space off.  n weights per space.
      if (param === null || param < 1) throw new Error(`featurepol: hpat<n> needs a rank count n >= 1, got "${str}"`);
      const n = param;
      prepare = _hpatPrepare;
      cumulative = true;
      sizeFn = (ctx, idx) => { const r = _hpatRank[idx]; return (r >= 1 && r <= n) ? (n + 1 - r) : 0; };
      break;
    }
    case 'stones': {
      if (!_STONES_N.has(param)) throw new Error(`featurepol: stones<n> needs n in {4,8,12,20} (D4-closed), got "${str}"`);
      const n = param;
      maxNear = n;
      // D4-canonical ternary encoding (0 empty / 1 own / 2 enemy) of the nearest n cells.
      evalFn = (ctx, idx) => {
        const nn = ctx.nearNbr, base = idx * ctx.nearStride, cells = ctx.game.cells, cur = ctx.cur, cv = _cvScratch;
        for (let i = 0; i < n; i++) { const c = cells[nn[base + i]]; cv[i] = c === 0 ? 0 : c === cur ? 1 : 2; }
        return _canonStones(cv, n);
      };
      break;
    }
    case 'stoneExpand': {
      // Adaptive-radius shape.  Starts with the stones8 region (the 8 nearest cells)
      // and keeps adding the next D4-closed distance shell until at least N of the
      // discovered cells hold a stone (own OR enemy), capping at the 20-cell region.
      // Under Game2's mixed distance the shells are exactly stones8 ⊂ stones12 ⊂
      // stones20 (cumulative cell counts 8, 12, 20), so each region is D4-closed and
      // canonicalisable.  The ternary pattern (0 empty / 1 own / 2 enemy) is
      // D4-canonicalised over the region it reached, and the region SIZE is folded
      // into the key so the same canonical value at a different extent is a distinct
      // feature.  A descriptor — one key per move.  Larger N expands further (more,
      // rarer keys); N is a stone count, not a cell count.
      if (param === null || param < 1) throw new Error(`featurepol: stoneExpand<N> needs a stone target N >= 1, got "${str}"`);
      const target = param;
      const SHELLS = [8, 12, NEAR_MAX];   // floor, then the cumulative shell boundaries
      maxNear = NEAR_MAX;                  // may expand to the full 20-cell region
      evalFn = (ctx, idx) => {
        const nn = ctx.nearNbr, base = idx * ctx.nearStride, cells = ctx.game.cells, cur = ctx.cur, cv = _cvScratch;
        let n = 0, stones = 0, si = 0;
        for (; n < 8; n++) {                // fill the stones8 floor and count its stones
          const c = cells[nn[base + n]];
          cv[n] = c === 0 ? 0 : c === cur ? 1 : 2;
          if (c !== 0) stones++;
        }
        while (stones < target && n < NEAR_MAX) {   // expand shell by shell (→12, →20)
          const next = SHELLS[++si];
          for (; n < next; n++) {
            const c = cells[nn[base + n]];
            cv[n] = c === 0 ? 0 : c === cur ? 1 : 2;
            if (c !== 0) stones++;
          }
        }
        return _hashCombine(_canonStones(cv, n), n) >>> 0;
      };
      break;
    }
    case 'adjLib': {
      // Like stones4 (the 4 orthogonal neighbours) but each neighbouring stone is
      // encoded with its chain's CURRENT liberty count capped at n.  Per-cell
      // symbol (radix 2n+1): 0 = empty, 1..n = own stone with that many libs,
      // n+1..2n = enemy stone with that many libs.  The move-point itself is a
      // liberty of those chains, so an adjacent enemy at 1 lib = capturable here,
      // at 2 libs = atari-able, etc.  Canonicalised over D4.  A descriptor.
      if (!param) throw new Error(`featurepol: adjLib<n> needs a liberty cap, got "${str}"`);
      const n = param, radix = 2 * n + 1;
      maxNear = 4;
      evalFn = (ctx, idx) => {
        const nn = ctx.nearNbr, base = idx * ctx.nearStride, game = ctx.game;
        const cells = game.cells, cur = ctx.cur, ls = game._ls, gid = game._gid, cv = _cvScratch;
        for (let i = 0; i < 4; i++) {
          const ni = nn[base + i], c = cells[ni];
          if (c === 0) { cv[i] = 0; continue; }
          let lib = ls[gid[ni]]; if (lib > n) lib = n;
          cv[i] = (c === cur) ? lib : n + lib;
        }
        return _canonRadix(cv, 4, radix);
      };
      break;
    }
    case 'stone8AdjLib': {
      // JOINT liberty-aware 3x3 pattern (ppat-style), canonicalised as ONE unit so
      // shape and liberties stay IN REGISTER — unlike the stones8+adjLib4 conjunction,
      // which D4-canonicalises each half independently and so loses their relative
      // orientation.  The 8 nearest cells: the 4 ORTHOGONAL neighbours carry liberty
      // counts capped at n (like adjLib: radix 2n+1 — 0 empty, 1..n own-with-libs,
      // n+1..2n enemy-with-libs); the 4 DIAGONALS carry shape only (radix 3 — 0 empty,
      // 1 own, 2 enemy).  Key = min over the 8 D4 symmetries of the mixed-radix
      // encoding (D4 preserves the orthogonal/diagonal split, so the radices line up).
      // A descriptor.  n=2 reproduces ppat's pattern; larger n adds liberty resolution
      // (and keys) — cardinality grows ~ (2n+1)^4, so prefer small n.
      if (!param) throw new Error(`featurepol: stone8AdjLib<n> needs a liberty cap, got "${str}"`);
      const n = param, R = 2 * n + 1;
      maxNear = 8;
      evalFn = (ctx, idx) => {
        const nn = ctx.nearNbr, base = idx * ctx.nearStride, game = ctx.game;
        const cells = game.cells, cur = ctx.cur, ls = game._ls, gid = game._gid, cv = _cvScratch;
        for (let i = 0; i < 4; i++) {                 // orthogonal: liberty-aware
          const ni = nn[base + i], c = cells[ni];
          if (c === 0) { cv[i] = 0; continue; }
          let lib = ls[gid[ni]]; if (lib > n) lib = n;
          cv[i] = (c === cur) ? lib : n + lib;
        }
        for (let i = 4; i < 8; i++) {                 // diagonal: shape only
          const c = cells[nn[base + i]];
          cv[i] = c === 0 ? 0 : (c === cur ? 1 : 2);
        }
        return _canon8AdjLib(cv, R);
      };
      break;
    }
    case 'capture': {
      if (!param) throw new Error(`featurepol: capture<n> needs a size, got "${str}"`);
      cumulative = true; sizeFn = (ctx, idx) => _captureCount(ctx.game, idx);
      break;
    }
    case 'atari': {
      if (!param) throw new Error(`featurepol: atari<n> needs a size, got "${str}"`);
      cumulative = true; sizeFn = (ctx, idx) => _atariStones(ctx.game, idx);
      break;
    }
    case 'selfAtari': {
      if (!param) throw new Error(`featurepol: selfAtari<n> needs a size, got "${str}"`);
      cumulative = true; sizeFn = (ctx, idx) => ctx.game.selfAtariSize(idx);
      break;
    }
    case 'lib': {
      // Liberty count of the group that would RESULT from playing the move
      // (static: joined-chain liberties ∪ idx's empty neighbours, minus idx, plus
      // cells freed by captures).  Cumulative: levels 1..min(libs, n).  lib=0
      // (suicide-without-capture) is the reference state and emits nothing.
      if (!param) throw new Error(`featurepol: lib<n> needs a size, got "${str}"`);
      cumulative = true; sizeFn = (ctx, idx) => ctx.game.resultingLibertyCount(idx);
      break;
    }
    case 'joins': {
      // DESCRIPTOR (takes no parameter): the count of distinct friendly chains
      // adjacent to the move, 0..4.  Always emits exactly one key encoding that
      // count — including 0 (connects nothing), which is its own category, not a
      // suppressed reference state.  So it spans the full set {0,1,2,3,4}.
      if (param !== null) throw new Error(`featurepol: joins takes no parameter, got "${str}"`);
      evalFn = (ctx, idx) => _adjFriendlyChains(ctx.game, idx);
      break;
    }
    case 'flags': {
      // DESCRIPTOR (no parameter): a 6-bit mask combining tactical event flags for
      // the move, always emitted as one key (mask 0 = no flags, its own category):
      //   bit 0 (1)  self-atari : the resulting own group has exactly 1 liberty
      //   bit 1 (2)  capture    : the move captures >= 1 enemy stone
      //   bit 2 (4)  atari      : the move puts >= 1 enemy chain in atari
      //   bit 3 (8)  ko         : the move creates a ko
      //   bit 4 (16) join       : the move connects >= 2 distinct friendly chains
      //   bit 5 (32) local      : the move is in the 8-neighbourhood of the prev move
      if (param !== null) throw new Error(`featurepol: flags takes no parameter, got "${str}"`);
      maxNear = 8;   // the local bit reads the 8-neighbourhood from nearNbr
      evalFn = (ctx, idx) => {
        const g = ctx.game;
        let m = 0;
        if (g.selfAtariSize(idx) > 0)        m |= 1;
        if (_captureCount(g, idx) > 0)       m |= 2;
        if (_atariStones(g, idx) > 0)        m |= 4;
        if (g.createsKo(idx))                m |= 8;
        if (_adjFriendlyChains(g, idx) >= 2) m |= 16;
        const prev = g.lastMove;
        if (prev >= 0) { const nn = ctx.nearNbr, base = idx * ctx.nearStride; for (let i = 0; i < 8; i++) if (nn[base + i] === prev) { m |= 32; break; } }
        return m;
      };
      break;
    }
    case 'ko': {
      // Binary: 1 iff the move creates a ko (captures one lone stone into a ko shape).
      binary = true;
      evalFn = (ctx, idx) => (ctx.game.createsKo(idx) ? 1 : 0);
      break;
    }
    case 'anyKo': {
      // Binary board-context flag: 1 iff a ko is currently active on the board
      // (some point is ko-banned).  The same value for every candidate move, so
      // it is meaningful only in conjunction (e.g. stones8+anyKo) — it modulates
      // other terms by whether a ko fight is on, not by which move is played.
      binary = true;
      evalFn = (ctx) => (ctx.game.ko !== PASS ? 1 : 0);
      break;
    }
    case 'local': {
      // Binary LOCALITY flag: 1 iff the move lies in the 8-neighbourhood (Moore)
      // of the previous move.  The reference state 0 (not adjacent, or no previous
      // move / previous was a pass) emits nothing.  Membership is symmetric, so we
      // test whether the previous move is one of idx's 8 nearest cells.
      binary = true;
      maxNear = 8;   // reads the 8-neighbourhood from nearNbr
      evalFn = (ctx, idx) => {
        const prev = ctx.game.lastMove;
        if (prev < 0) return 0;
        const nn = ctx.nearNbr, base = idx * ctx.nearStride;
        for (let i = 0; i < 8; i++) if (nn[base + i] === prev) return 1;
        return 0;
      };
      break;
    }
    case 'localAlways': {
      // Two-valued sibling of `local` that fires in BOTH states: the non-adjacent
      // value 0 is a real category (its own key), not a suppressed reference.  Being
      // non-binary, the 0/1 locality value is folded into the key and the space never
      // gates, so a conjoined space (e.g. stones8+localAlways) splits each pattern
      // into a local AND a non-local variant — a symmetric split — rather than adding
      // a one-sided local-only correction the way stones8+local does.
      maxNear = 8;   // reads the 8-neighbourhood from nearNbr
      evalFn = (ctx, idx) => {
        const prev = ctx.game.lastMove;
        if (prev < 0) return 0;
        const nn = ctx.nearNbr, base = idx * ctx.nearStride;
        for (let i = 0; i < 8; i++) if (nn[base + i] === prev) return 1;
        return 0;
      };
      break;
    }
    case 'koSolve': {
      // ppat Feature 6, faithful: binary.  1 iff the move resolves a ko I just made
      // by capturing the threatening enemy — i.e. I have a live ko-stone
      // (game.koStone[cur+1], the lone stone I played that created the ko), an enemy
      // group adjacent to it is in atari, and this move is that group's single
      // liberty (the capturing move).  0 (no live ko-stone, or not such a capture)
      // emits nothing.
      binary = true;
      evalFn = (ctx, idx) => {
        const g = ctx.game, ks = g.koStone[ctx.cur + 1];
        if (ks === PASS) return 0;
        const foe = -ctx.cur, nbr = g._nbr, cells = g.cells, gid = g._gid, ls = g._ls, b = ks * 4;
        for (let d = 0; d < 4; d++) {
          const ni = nbr[b + d];
          if (cells[ni] !== foe) continue;
          const eg = gid[ni];
          if (ls[eg] !== 1) continue;                 // adjacent enemy must be in atari
          if (g.groupLibs2(ni).lib0 === idx) return 1; // idx is its capturing move
        }
        return 0;
      };
      break;
    }
    case 'dist': {
      // Cumulative thermometer of the (blended toroidal) distance from the move to
      // the previous move, mapped to integer levels via floor(Game2.distance*2-1):
      // an orthogonal-adjacent move is level 1 and the level rises with distance,
      // up to the cap n.  No previous move (or a pass) yields level 0 — the
      // reference state, which emits nothing.  Graded sibling of `local`.
      if (!param) throw new Error(`featurepol: dist<n> needs a cap, got "${str}"`);
      cumulative = true;
      sizeFn = (ctx, idx) => {
        const g = ctx.game, prev = g.lastMove;
        if (prev < 0) return 0;
        return Math.floor(g.distance(idx, prev) * 2 - 1);
      };
      break;
    }
    case 'ladderStatus': {
      // 4-bit presence mask over the four ladder flags (kill 1, save 2,
      // extend 4, attack 8); a cell may carry several at once.
      needsLadder = true;
      evalFn = (ctx, idx) => {
        const s = ctx.ladderSizes, b = idx * 4;
        return (s[b] ? 1 : 0) | (s[b + 1] ? 2 : 0) | (s[b + 2] ? 4 : 0) | (s[b + 3] ? 8 : 0);
      };
      break;
    }
    case 'urgentKill': case 'urgentSave': case 'wastedExtend': case 'wastedAttack': {
      // Size-bucketed variant of one ladder flag: the summed stone count of the
      // chains for which this move is that flag's liberty.
      if (!param) throw new Error(`featurepol: ${kind}<n> needs a size, got "${str}"`);
      needsLadder = true;
      const flag = { urgentKill: URGENT_KILL, urgentSave: URGENT_SAVE,
                     wastedExtend: WASTED_EXTEND, wastedAttack: WASTED_ATTACK }[kind];
      cumulative = true; sizeFn = (ctx, idx) => ctx.ladderSizes[idx * 4 + flag];
      break;
    }
    default:
      throw new Error(`featurepol: unknown feature kind "${kind}" in "${str}"`);
  }
  return { str, kind, param, salt, evalFn, sizeFn, cumulative, maxLevel: param, needsLadder, binary, prepare, maxNear };
}

// Parse a full spec string into a runtime spec.  Every feature space emits keys
// dynamically: a term contributes either a value (descriptor) or a set of present
// thermometer levels (gated event), and the space emits the CROSS-PRODUCT.  Term
// roles:
//   - DESCRIPTOR (stones / ladderStatus): always present; folds its value into
//     the key.  No "absent" state — every move has a shape, including the all-empty
//     one — so there is nothing to gate on.
//   - GATED EVENT: a count/indicator whose 0 value means "nothing happened" and so
//     needs no weight (it is the softmax reference).  Two flavours:
//       · CUMULATIVE size term (capture / atari / urgentKill / …): contributes the
//         present thermometer levels 1..min(size, maxLevel); contributes NOTHING
//         when size is 0.
//       · BINARY indicator (ko): contributes one level when on, nothing when off.
// A space emits the cross-product of its terms' contributions, so a conjunction
// fires ONLY when every gated term is present (size ≥ 1) — AND semantics.  This is
// O(present levels), and the absent state is never materialised as a key (which,
// combined with a descriptor's varying value, would otherwise alias the descriptor
// feature and distort learning).
function parseSpec(specStr) {
  if (specStr && typeof specStr === 'object' && specStr.spaces) return specStr;   // already parsed
  const str = String(specStr || '').trim();
  if (!str) throw new Error('featurepol: empty --spec');
  const spaces = [];
  let needsLadder = false;
  let nearMax = 0;            // widest nearNbr reach across all terms (sizes the table)
  const slotOf = new Map();   // term salt → memo slot
  const computers = [];       // computers[slot] = value fn (sizeFn for cumulative, else evalFn)
  const prepares = [];        // whole-position hooks, run once per position before the move loop
  function slotFor(t) {
    let slot = slotOf.get(t.salt);
    if (slot === undefined) { slot = computers.length; slotOf.set(t.salt, slot); computers.push(t.cumulative ? t.sizeFn : t.evalFn); }
    return slot;
  }
  for (const spaceStr of str.split(',').map(s => s.trim()).filter(Boolean)) {
    const terms = spaceStr.split('+').map(t => t.trim()).filter(Boolean).map(_makeTerm);
    if (terms.length === 0) throw new Error(`featurepol: empty feature space in "${spaceStr}"`);
    const gate = [];        // slots that must be ≥ 1 for the space to fire at all
    const baseTerms = [];   // descriptors (bin:false, fold value) + binary events (bin:true, fold level 1)
    const cumTerms = [];    // cumulative size terms (thermometer cross-product)
    for (const t of terms) {
      if (t.needsLadder) needsLadder = true;
      if (t.maxNear > nearMax) nearMax = t.maxNear;
      const slot = slotFor(t);
      if (t.prepare && !prepares.includes(t.prepare)) prepares.push(t.prepare);
      if (t.cumulative)       { gate.push(slot); cumTerms.push({ salt: t.salt, slot, maxLevel: t.maxLevel }); }
      else if (t.binary)      { gate.push(slot); baseTerms.push({ salt: t.salt, slot, bin: true }); }
      else                    { baseTerms.push({ salt: t.salt, slot, bin: false }); }
    }
    let maxKeys = 1;
    for (const c of cumTerms) maxKeys *= c.maxLevel;
    spaces.push({ str: spaceStr, salt: _hashStr('space:' + spaceStr), gate, baseTerms, cumTerms, maxKeys });
  }
  if (spaces.length === 0) throw new Error(`featurepol: no feature spaces in "${str}"`);
  let maxKeysPerMove = 0;
  for (const sp of spaces) maxKeysPerMove += sp.maxKeys;
  return { str, spaces, computers, numSlots: computers.length, prepares, maxKeysPerMove, needsLadder, nearMax };
}

// ── Weights store (hash → dense idx → Float32 weight) ─────────────────────────

function createWeights(opts = {}) {
  const spec = parseSpec(opts.spec);
  const initialCapacity = opts.initialCapacity || 1024;
  return {
    spec,
    nSpaces: spec.spaces.length,
    map:   new Map(),                          // 32-bit hash → dense idx
    vals:  new Float32Array(initialCapacity),  // weight[dense idx]
    delta: new Float32Array(initialCapacity),  // reusable scatter buffer
    count: new Int32Array(initialCapacity),    // per-key contributor count (for per-key gradient mean)
    size:  0,
  };
}

function _intern(w, key) {
  const map = w.map;
  const existing = map.get(key);
  if (existing !== undefined) return existing;
  const idx = w.size;
  if (idx >= w.vals.length) {
    const cap = w.vals.length * 2;
    const nv = new Float32Array(cap); nv.set(w.vals); w.vals = nv;
    const nd = new Float32Array(cap); nd.set(w.delta); w.delta = nd;
    const nc = new Int32Array(cap); nc.set(w.count); w.count = nc;
  }
  map.set(key, idx);
  w.size = idx + 1;
  return idx;
}

// ── State (per board size, reused across calls) ───────────────────────────────

function _wrap(x, N) { x %= N; return x < 0 ? x + N : x; }

function createState(N, spec) {
  spec = parseSpec(spec);
  const cap = N * N;
  const maxK = spec.maxKeysPerMove;   // upper bound on keys emitted per move
  // Toroidal nearest-cell table: nearNbr[idx*stride + k] = flat index of the k-th
  // nearest cell to idx.  The stride is sized to the spec's actual reach (the max
  // nearest-cells any term in this spec reads), not the global NEAR_MAX — a spec that
  // only needs the 4 orthogonals (adjLib, stones4) gets a 4-wide table, not 20-wide.
  // Measurably faster: the smaller table keeps the per-move neighbour reads in cache.
  const stride = spec.nearMax;
  const nearNbr = new Int32Array(cap * stride);
  for (let idx = 0; idx < cap; idx++) {
    const r = (idx / N) | 0, c = idx - r * N, base = idx * stride;
    for (let k = 0; k < stride; k++) {
      const dy = _NEAR_OFFSETS[k][0], dx = _NEAR_OFFSETS[k][1];
      nearNbr[base + k] = _wrap(r + dy, N) * N + _wrap(c + dx, N);
    }
  }
  return {
    N,
    nearNbr,
    nearStride: stride,
    moves:    new Int32Array(cap),
    keys:     new Int32Array(cap * maxK),       // variable: move i's keys are keys[keyOff[i]..keyOff[i+1])
    keyOff:   new Int32Array(cap + 1),
    memo:     new Float64Array(spec.numSlots),  // per-move scratch: one value per distinct fixed-space term
    ladderSizes: new Uint16Array(cap * 4),
    logits:   new Float64Array(cap),
    probs:    new Float64Array(cap),
    touched:  new Int32Array(maxK * (cap + 1)),
    accA:     new Int32Array(maxK),             // cross-product scratch (≥2 cumulative terms)
    accB:     new Int32Array(maxK),
    count:    0,
  };
}

// ── Feature extraction ────────────────────────────────────────────────────────

// Fill state.keys (interned dense indices) for every legal non-true-eye move.
function extractFeatures(game, state, weights, game3) {
  const spec = weights.spec;
  const spaces = spec.spaces, nSpaces = spaces.length;
  const memo = state.memo;
  const ctx = { game, cur: game.current, nearNbr: state.nearNbr, nearStride: state.nearStride, ladderSizes: null, memo };
  if (spec.needsLadder) {
    const g3 = game3 || game3FromGame2(game);
    ctx.ladderSizes = _buildLadderSizes(game, g3, state.ladderSizes);
  }
  // Whole-position precomputes (e.g. hpat ranking) -- once per position, before
  // any per-move term runs.
  const preps = spec.prepares;
  if (preps) for (let i = 0; i < preps.length; i++) preps[i](ctx);
  const computers = spec.computers, numSlots = spec.numSlots;
  const emC = game._emptyCells, ec = game.emptyCount;
  const moves = state.moves, keys = state.keys, keyOff = state.keyOff;
  let accA = state.accA, accB = state.accB;
  let count = 0, pos = 0;
  keyOff[0] = 0;
  for (let ei = 0; ei < ec; ei++) {
    const idx = emC[ei];
    if (!game.isLegal(idx) || game.isTrueEye(idx)) continue;
    // Compute each distinct term once for this move, then read by slot per space.
    for (let sl = 0; sl < numSlots; sl++) memo[sl] = computers[sl](ctx, idx);
    for (let s = 0; s < nSpaces; s++) {
      const sp = spaces[s];
      // Gate: every cumulative/binary term must be present (≥1) or the space fires nothing.
      const gate = sp.gate; let gated = false;
      for (let gi = 0; gi < gate.length; gi++) if (memo[gate[gi]] < 1) { gated = true; break; }
      if (gated) continue;
      // Base fold: space salt + descriptor values + present binary indicators.
      let base = sp.salt;
      const bt = sp.baseTerms;
      for (let bi = 0; bi < bt.length; bi++) {
        const t = bt[bi];
        base = _hashCombine(base, _hashCombine(t.salt, t.bin ? 1 : (memo[t.slot] >>> 0)));
      }
      const ct = sp.cumTerms;
      if (ct.length === 0) {
        keys[pos++] = _intern(weights, base >>> 0);
      } else if (ct.length === 1) {
        // Single thermometer: present levels 1..min(size, maxLevel) (size ≥ 1, gated above).
        const t = ct[0]; let sz = memo[t.slot]; if (sz > t.maxLevel) sz = t.maxLevel;
        for (let k = 1; k <= sz; k++) keys[pos++] = _intern(weights, _hashCombine(base, _hashCombine(t.salt, k)) >>> 0);
      } else {
        // ≥2 cumulative terms: cross-product of their present levels (rare).
        let acc = accA, nxt = accB, nAcc = 1; acc[0] = base;
        for (let ci = 0; ci < ct.length; ci++) {
          const t = ct[ci]; let sz = memo[t.slot]; if (sz > t.maxLevel) sz = t.maxLevel;
          let on = 0;
          for (let a = 0; a < nAcc; a++) { const ba = acc[a]; for (let k = 1; k <= sz; k++) nxt[on++] = _hashCombine(ba, _hashCombine(t.salt, k)); }
          const tmp = acc; acc = nxt; nxt = tmp; nAcc = on;
        }
        for (let a = 0; a < nAcc; a++) keys[pos++] = _intern(weights, acc[a] >>> 0);
      }
    }
    moves[count] = idx;
    count++;
    keyOff[count] = pos;
  }
  state.count = count;
}

// ── Scoring / softmax / sampling ──────────────────────────────────────────────

function _score(state, i, weights) {
  const vals = weights.vals, keys = state.keys, keyOff = state.keyOff;
  const end = keyOff[i + 1];
  let s = 0;
  for (let k = keyOff[i]; k < end; k++) s += vals[keys[k]];
  return s;
}

function computeSoftmax(state, weights, temperature = 1) {
  const n = state.count;
  if (n === 0) return 0;
  const lg = state.logits, pr = state.probs;
  let maxL = -Infinity, maxI = 0;
  for (let i = 0; i < n; i++) {
    const s = _score(state, i, weights);
    lg[i] = s;
    if (s > maxL) { maxL = s; maxI = i; }
  }
  if (temperature === 0) {
    for (let i = 0; i < n; i++) pr[i] = 0;
    pr[maxI] = 1;
    return n;
  }
  const invT = 1 / temperature;
  let sum = 0;
  for (let i = 0; i < n; i++) { pr[i] = Math.exp((lg[i] - maxL) * invT); sum += pr[i]; }
  const inv = 1 / sum;
  for (let i = 0; i < n; i++) pr[i] *= inv;
  return n;
}

function evaluate(game, state, weights) {
  extractFeatures(game, state, weights);
  const out = [];
  for (let i = 0; i < state.count; i++) out.push({ move: state.moves[i], score: _score(state, i, weights) });
  return out.sort((a, b) => b.score - a.score);
}

// Fill out[i] with the raw linear score (Σ weights of move i's keys) for every candidate
// move, in state.moves order, without softmax or sorting — the model's per-move prediction.
// For trainers/agents that read raw per-move values (e.g. point/territory prediction).
// Assumes extractFeatures has already populated state.  Returns the move count.
function scoreAll(state, weights, out) {
  const n = state.count;
  for (let i = 0; i < n; i++) out[i] = _score(state, i, weights);
  return n;
}

function policyMove(game, state, weights, rng, game3, temperature = 1) {
  extractFeatures(game, state, weights, game3);
  const n = state.count;
  if (n === 0) return { move: PASS, index: -1, prob: 1 };
  computeSoftmax(state, weights, temperature);
  const probs = state.probs;
  if (temperature === 0) {
    let best = 0;
    for (let i = 1; i < n; i++) if (probs[i] > probs[best]) best = i;
    return { move: state.moves[best], index: best, prob: 1 };
  }
  let r = (rng || Math).random(), chosen = n - 1;
  for (let i = 0; i < n; i++) { r -= probs[i]; if (r <= 0) { chosen = i; break; } }
  return { move: state.moves[chosen], index: chosen, prob: probs[chosen] };
}

function greedyMove(game, state, weights, game3) {
  return policyMove(game, state, weights, null, game3, 0).move;
}

// ── SGD step (generic) + REINFORCE ────────────────────────────────────────────
//
// The linear model is shared across trainers; only the per-move objective differs.
// applyScoreGradient is the generic backward — scatter a per-move score-gradient onto
// the sparse keys, dedup keys shared across moves, apply decoupled L2 decay — and each
// trainer (REINFORCE, sim-balancing, point/territory prediction) computes its own grad.
// reinforceUpdate is the REINFORCE specialisation, kept as a fused fast path.

// Apply each TOUCHED weight's net accumulated delta exactly once, with decoupled L2
// shrink (w ← w + Δw − lr·decay·w), then clear its delta.  The `d !== 0` guard (matching
// npat) skips duplicate visits — delta is zeroed on the first — so a feature shared across
// moves is updated and decayed once, not once per occurrence.  Optional `stats`
// (absSum/count) frequency-weights |weight| over genuine updates, for the avgW column.
//
// Optional `counts`: a per-key contributor count.  When supplied, each key's accumulated
// delta is divided by its count before being applied — i.e. the MEAN gradient per key, not
// the sum.  This is diagonal (Jacobi) preconditioning: a key shared by j candidates of a
// position would otherwise take a j× step (the branching blow-up), so the mean normalises
// every key's effective curvature to ~1 regardless of how many candidates touched it, while
// leaving rarely-shared keys (j≈1) at full step.  Counts are cleared alongside the deltas.
function _applyTouchedDelta(state, tc, weights, decayStep, stats, counts) {
  const vals = weights.vals, delta = weights.delta, touched = state.touched;
  for (let i = 0; i < tc; i++) {
    const idx = touched[i], d = delta[idx];
    if (d !== 0) {
      vals[idx] += (counts ? d / counts[idx] : d) - decayStep * vals[idx];
      delta[idx] = 0;
      if (stats) { stats.absSum += Math.abs(vals[idx]); stats.count++; }
    }
    if (counts) counts[idx] = 0;
  }
}

// Generic SGD step for sparse per-move features.  grad[i] = ∂objective/∂score_i for each
// candidate move (ASCENT convention: weights move +lr·grad, so for a minimisation loss
// pass the negative gradient).  Scatters each move's gradient onto its keys (summing where
// keys recur across moves), then applies _applyTouchedDelta.  This is the shared update
// the REINFORCE / sim-balancing / point-prediction trainers all build on.
//
// Two orthogonal normalisations make lr independent of spec/position geometry:
//   • per-MOVE, fan-in: each move's gradient is divided by its active-feature count K (keys
//     per move).  A move's score is the SUM of its keys, so an un-normalised step moves the
//     score by lr·grad·K; dividing by K makes the step independent of spec WIDTH.
//   • per-KEY, fan-out: each key's accumulated gradient is divided by the number of candidates
//     that touched it (the mean, via _applyTouchedDelta's `counts`).  A key shared by j
//     candidates would otherwise take a j× step — the branching-driven blow-up — so the mean
//     normalises every key's curvature to ~1 (diagonal/Jacobi preconditioning), removing the
//     instability ceiling without throttling rarely-shared (informative, move-local) keys.
function applyScoreGradient(state, grad, weights, lr, weightDecay = 0, stats) {
  const n = state.count;
  if (n === 0) return 0;
  const keys = state.keys, keyOff = state.keyOff, delta = weights.delta, count = weights.count, touched = state.touched;
  let tc = 0;
  for (let i = 0; i < n; i++) {
    const k0 = keyOff[i], e = keyOff[i + 1], K = e - k0;
    if (K === 0 || grad[i] === 0) continue;
    const gi = lr * grad[i] / K;
    for (let k = k0; k < e; k++) { const idx = keys[k]; touched[tc++] = idx; delta[idx] += gi; count[idx]++; }
  }
  _applyTouchedDelta(state, tc, weights, lr * weightDecay, stats, count);
  return tc;
}

// REINFORCE specialisation: per-move score-gradient is advantage·(1{i=chosen} − π_i), so
// the chosen move's keys get +lr·advantage and every move's keys get −lr·advantage·π_i.
// Each move's contribution is divided by its active-feature count (keys per move), matching
// applyScoreGradient's per-example normalisation so lr is comparable across spec widths.
function reinforceUpdate(state, chosenIndex, advantage, weights, lr, weightDecay = 0, stats) {
  const n = state.count;
  if (n === 0 || chosenIndex < 0) return 0;
  const step = lr * advantage;
  if (step === 0) return 0;
  const keys = state.keys, keyOff = state.keyOff, probs = state.probs;
  const delta = weights.delta, touched = state.touched;
  let tc = 0;
  {
    const k0 = keyOff[chosenIndex], e = keyOff[chosenIndex + 1], K = e - k0;
    if (K > 0) {
      const add = step / K;
      for (let k = k0; k < e; k++) { const idx = keys[k]; touched[tc++] = idx; delta[idx] += add; }
    }
  }
  for (let i = 0; i < n; i++) {
    const pi = probs[i];
    if (pi === 0) continue;
    const k0 = keyOff[i], e = keyOff[i + 1], K = e - k0;
    if (K === 0) continue;
    const sub = step * pi / K;
    for (let k = k0; k < e; k++) { const idx = keys[k]; touched[tc++] = idx; delta[idx] -= sub; }
  }
  _applyTouchedDelta(state, tc, weights, lr * weightDecay, stats);
  return tc;
}

// ── Model serialization ───────────────────────────────────────────────────────
//
// File: base64 of [Int32 keys][Int16 qvals], plus spec / scale / ema /
// totalUpdates.  weight ≈ qval / scale.  Mirrors train-npat's format.

function modelWeights(raw) {
  const keys = raw.keys, qvals = raw.qvals;
  const count = raw.count != null ? raw.count : keys.length;
  const inv = 1 / raw.scale;
  return { count, forEach(cb) { for (let i = 0; i < count; i++) cb(keys[i] >>> 0, qvals[i] * inv); } };
}

function serialize(weights, meta = {}) {
  const count = weights.map.size;
  let maxAbs = 0;
  for (const [, d] of weights.map) { const a = Math.abs(weights.vals[d]); if (a > maxAbs) maxAbs = a; }
  const scale = maxAbs > 0 ? 32767 / maxAbs : 1;
  const keys = new Int32Array(count), qvals = new Int16Array(count);
  let i = 0;
  for (const [key, d] of weights.map) {
    keys[i] = key | 0;
    let q = Math.round(weights.vals[d] * scale);
    if (q > 32767) q = 32767; else if (q < -32768) q = -32768;
    qvals[i] = q;
    i++;
  }
  let buf, b64;
  if (typeof Buffer !== 'undefined') {
    buf = Buffer.alloc(count * 6);
    Buffer.from(keys.buffer, keys.byteOffset, count * 4).copy(buf, 0);
    Buffer.from(qvals.buffer, qvals.byteOffset, count * 2).copy(buf, count * 4);
    b64 = buf.toString('base64');
  } else {
    const bytes = new Uint8Array(count * 6);
    bytes.set(new Uint8Array(keys.buffer, keys.byteOffset, count * 4), 0);
    bytes.set(new Uint8Array(qvals.buffer, qvals.byteOffset, count * 2), count * 4);
    let s = ''; for (let j = 0; j < bytes.length; j++) s += String.fromCharCode(bytes[j]);
    b64 = btoa(s);
  }
  return [
    "'use strict';",
    '// Auto-generated by a featurepol trainer — do not edit by hand.',
    'const featurepolModel = (() => {',
    `  const count = ${count};`,
    `  const scale = ${scale};`,
    `  const spec = ${JSON.stringify(meta.spec || weights.spec.str)};`,
    `  const ema = ${+(meta.ema || 0).toFixed(6)};`,
    `  const totalUpdates = ${Math.round(meta.totalUpdates || 0)};`,
    `  const b64 = '${b64}';`,
    "  const bytes = typeof Buffer !== 'undefined'",
    "    ? Buffer.from(b64, 'base64')",
    "    : Uint8Array.from(atob(b64), c => c.charCodeAt(0));",
    "  const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + count * 6);",
    "  const keys  = new Int32Array(buf, 0, count);",
    "  const qvals = new Int16Array(buf, count * 4, count);",
    "  return { spec, ema, totalUpdates, count, scale, keys, qvals };",
    "})();",
    "if (typeof module !== 'undefined') module.exports = featurepolModel;",
    "else window.featurepolModel = featurepolModel;",
  ].join('\n') + '\n';
}

// Build runtime weights from a saved model.
//   name — message prefix; path — model file path (Node only).
function loadModel({ name = 'featurepol', path: pathOverride } = {}) {
  let raw, modelName;
  if (typeof window !== 'undefined' && !pathOverride) {
    if (!window.featurepolModel) throw new Error(`${name}: window.featurepolModel is not set`);
    raw = window.featurepolModel; modelName = 'window.featurepolModel';
  } else {
    const path = require('path');
    raw = require(path.resolve(pathOverride));
    modelName = path.basename(pathOverride);
  }
  const mw = modelWeights(raw);
  const weights = createWeights({ spec: raw.spec, initialCapacity: Math.max(1024, mw.count | 0) });
  mw.forEach((key, val) => { weights.vals[_intern(weights, key)] = val; });
  return { weights, modelName, spec: weights.spec, ema: raw.ema || 0, totalUpdates: raw.totalUpdates || 0 };
}

const FeaturePol = {
  parseSpec,
  createWeights,
  createState,
  internKey: _intern,
  extractFeatures,
  computeSoftmax,
  evaluate,
  scoreAll,
  policyMove,
  greedyMove,
  applyScoreGradient,
  reinforceUpdate,
  modelWeights,
  serialize,
  loadModel,
  NEAR_MAX,
  // exposed for tests
  _hashStr, _captureCount, _atariStones,
  _hpatRanks: () => _hpatRank,
};

if (typeof module !== 'undefined') module.exports = FeaturePol;
else window.FeaturePol = FeaturePol;

})();
