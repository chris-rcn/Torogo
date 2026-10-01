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
//   adjHealth<n>  as adjLib<n>, but each neighbouring stone carries its chain's
//               HEALTH bucket (P(survives a standard playout) under a frozen
//               train-health.js model, split into n levels) instead of its
//               liberty count.  Liberty count is a crude proxy for the same
//               thing; in the vpatterns families the learned version beat it
//               outright.  Needs FP_HEALTH_DATA.  A descriptor.
//   adjLib<n>   D4-canonical 4 orthogonal neighbours, each a stone encoded with its
//               chain's liberty count capped at n (radix 2n+1).  A descriptor.
//   stoneExpand<N>  Adaptive-radius shape: starts at the stones8 region and adds
//               the next D4-closed shell (→12, →20) until at least N of the
//               cells seen hold a stone, own or enemy.  N is a STONE count, not
//               a cell count.  The region size is folded into the key, so the
//               same canonical pattern reached at a different extent is a
//               distinct feature.  Larger N reaches further and yields more,
//               rarer keys.  A descriptor.
//   stoneLimit<N>  Adaptive-diamond sparse shape: the LARGEST L1 diamond
//               around the move containing at most N stones (stoneLimit0 =
//               the largest empty diamond, a pure distance profile;
//               stoneLimit2 = the largest window still sparse enough to
//               read).  Hashed with the stones12b recursion at the reached
//               levels: the FULL stack — the base (level-1) pattern for
//               EVERY candidate plus each deeper within-limit level's
//               pattern, one key per level, with a reached-level thermometer
//               alongside.  OPENING ONLY: emits nothing at board fullness
//               >= 0.1, and costs nothing there either.  Radius runs to the
//               board maximum; N is semantic and stays in the key salt.
//   stones12b   The stones12 cells PLUS the centre (13), hashed as a recursive
//               plus-of-plusses instead of a min over 8 D4 permutations: invariant
//               by construction, so much cheaper, at 90.4% of the true D4 orbits.
//               Takes no size.  Descriptor.
//   stones24    25-cell shape (stones24 + centre): the stones12b trick
//               recursed once more — a plus of five 13-cell t-hashes,
//               covering the L1-radius-3 diamond.  Invariant by
//               construction.  Fidelity is HIGHER than stones12b's: the
//               five sub-diamonds overlap heavily, anchoring sub-shape
//               orientations — 96.20% exact on the binary alphabet (all
//               2^25 patterns; stones12b: 84.95% binary, 90.36% ternary),
//               ternary extrapolates to ~97.5-98.5%.
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
//   dist<n>     ONE-HOT blended-toroidal distance to the previous move: the
//               reached level floor(Game2.distance*2-1) capped at n
//               (orthogonal-adjacent = 1, rising with distance), one key per
//               move; no previous move emits nothing (level 0, gated).
//   ladderStatus 4-bit ladder presence mask at the move from ladder2
//               (urgent-kill/urgent-save/wasted-extend/wasted-attack)
//   urgentKill<n> / urgentSave<n> / wastedExtend<n> / wastedAttack<n>
//               one ladder flag's summed chain stone-count (cumulative, up to n)
//   t3Status    ladderStatus's 4-bit mask from tactics3 instead (chains of 1-3
//               liberties and at least t3MinChain stones, read to 4 liberties
//               within t3DepthLimit / t3NodeLimit);
//               an inconclusive read sets no flag
//   t3Kill<n> / t3Save<n> / t3WastedExtend<n> / t3WastedAttack<n>
//               one tactics3 flag's summed chain stone-count (cumulative, up to n)
//   vpat<n>     rank of the move under a fixed external vpatterns value model,
//               mover-relative: 1 = that model's top choice, 2 = its second,
//               up to n.  CUMULATIVE in rank quality: rank r contributes
//               levels 1..n+1-r, so the top choice lights every level and rank n
//               only level 1; rank past n emits nothing and shares the implicit
//               zero baseline.  Still exactly n weights per space combination,
//               but level k is estimated from every move ranked <= n+1-k rather
//               than from one rank alone.  The value model is embedded in the
//               featurepol file (attached to the weights) and is never trained
//               here.  Pattern-only models rank
//               incrementally (deltaZ); ladder-coded models (size:L) are
//               supported but rank NON-incrementally — a full extraction on a
//               clone per candidate, several times slower per position.
//   amaf<n>:<P>  rank of the move by an aggregate all-moves-as-first win ratio
//               over P uniform first-play-wins playouts from the position (own
//               moves full weight, opponent's at 0.3, first-play weight decaying
//               0.5/area per move), mover-relative and CUMULATIVE in rank quality
//               exactly like vpat<n>.  One cheap batch ranks the WHOLE board — no
//               external model, no top-N gate.  The batch rng is seeded from the
//               position (game.hash), so the feature is deterministic.
//
// BROWSER-COMPATIBLE: no Node-only APIs at top level.

(function () {

const Util = (typeof require === 'function') ? require('./util.js') : window.Util;
const { PASS, BLACK }          = Util.load('./game2.js', 'Game2');
const { game3FromGame2 }       = Util.load('./game3.js', 'Game3');
const { getAllLadderStatuses } = Util.load('./ladder2.js', 'Ladder2');
const Tactics3                 = Util.load('./tactics3.js', 'Tactics3');
const VPatterns                = Util.load('./vpatterns.js', 'VPatterns');
const HealthLib                = Util.load('./health-lib.js', 'HealthLib');
const { makeIntMap }           = Util.load('./int-map.js', 'IntMap');
const { makeRng }              = Util.load('./xorshift.js', 'XorShift');

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
// count of the chains for which that cell is a flagged liberty, over chains of
// at least `minChain` stones (the model's ladderMinChain), where flag is:
//   0 urgent-kill, 1 urgent-save, 2 wasted-extend, 3 wasted-attack.
// (ladderStatus reads presence/0; urgentKill/urgentSave/wastedExtend/wastedAttack<n> read size.)
const URGENT_KILL = 0, URGENT_SAVE = 1, WASTED_EXTEND = 2, WASTED_ATTACK = 3;
function _buildLadderSizes(game, game3, out, minChain) {
  out.fill(0);
  const cap = game.N * game.N;
  if (game.emptyCount === cap) return out;
  const infos = getAllLadderStatuses(game3, minChain);
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

// Per-cell tactics3 sizes: _buildLadderSizes's layout and flags, from
// tactics3.searchChains over chains of at least `minChain` stones, read
// within `depthLimit` / `nodeLimit` (the model's t3MinChain, t3DepthLimit,
// t3NodeLimit).  An urgent flag needs a move
// that definitely succeeds; a wasted flag needs a definite failure
// (moverSucceeds === false), so an inconclusive read (null) sets nothing.
function _buildT3Sizes(game, game3, out, minChain, depthLimit, nodeLimit) {
  out.fill(0);
  const cap = game.N * game.N;
  if (game.emptyCount === cap) return out;
  const infos = Tactics3.searchChains(game3, nodeLimit, depthLimit, minChain);
  const cur = game.current;
  for (const info of infos) {
    if (!info.status) continue;
    const { libs, moverSucceeds, urgentLibs } = info.status;
    const defending = info.color === cur;
    const size = game3.groupSize(info.gid);
    let flag, targets;
    if (urgentLibs.length > 0)        { flag = defending ? URGENT_SAVE : URGENT_KILL; targets = urgentLibs; }
    else if (moverSucceeds === false) { flag = defending ? WASTED_EXTEND : WASTED_ATTACK; targets = libs; }
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

// ── stones12b: recursive plus-of-plusses ("t"-hash) ──────────────────────────
// D4-invariant BY CONSTRUCTION rather than by enumerating 8 permutations and
// taking a min.  Two levels:
//
//   th5(p)  = mix( uh(uh(N,S), uh(E,W)), centre )      one 5-point plus at p
//   key(p)  = mix( uh(uh(th5(N),th5(S)), uh(th5(E),th5(W))), th5(p) )
//
// uh is unordered, so pairing opposite cells across each axis is invariant to
// exactly the group preserving the partition {{N,S},{E,W}} — which is D4.  The
// centre is fixed by every D4 element, so it folds in with an ORDERED mix; a
// second uh there would confuse "centre X, ring Y" with "centre Y, ring X".
//
// The 13 cells covered are stones12's twelve plus the centre.  Fidelity is
// 90.4% of true D4 orbits (186270 of 206145): what merges is a sub-plus's own
// internal orientation, since each sub-hash has already discarded it.  Same
// trade hpatterns makes above 2x2, and the reason this is a SEPARATE feature
// from stones12 rather than a replacement for it.
//
// Leaves enter as value+1 (1..3): uh(a,b) = C + (1+a)(1+b) - 1, so a leaf of
// exactly -1 would absorb its partner.
function _uh(a, b) { return (1234567 + a + b + Math.imul(a, b)) | 0; }

// th5 for every board cell, filled once per position by a prepare hook.  Every
// value is read five times — as one candidate's centre and as four neighbours'
// arms — so computing it per candidate would do the work five times over.
let _t5Val = null;
function _t5Prepare(ctx) {
  const game = ctx.game, N = game.N, cap = N * N, cur = ctx.cur;
  const cells = game.cells, nn = ctx.nearNbr, stride = ctx.nearStride;
  if (!_t5Val || _t5Val.length < cap) _t5Val = new Int32Array(cap);
  for (let idx = 0; idx < cap; idx++) {
    const base = idx * stride;
    // Symbols inlined rather than via a helper: this is the hot loop, and
    // 1 empty / 2 own / 3 enemy keeps every leaf clear of uh's absorbing -1.
    const cN = cells[nn[base]],     sN = cN === 0 ? 1 : cN === cur ? 2 : 3;
    const cE = cells[nn[base + 1]], sE = cE === 0 ? 1 : cE === cur ? 2 : 3;
    const cS = cells[nn[base + 2]], sS = cS === 0 ? 1 : cS === cur ? 2 : 3;
    const cW = cells[nn[base + 3]], sW = cW === 0 ? 1 : cW === cur ? 2 : 3;
    const cC = cells[idx],          sC = cC === 0 ? 1 : cC === cur ? 2 : 3;
    _t5Val[idx] = _hashCombine(_uh(_uh(sN, sS), _uh(sE, sW)), sC);
  }
}

// t13 (the stones12b value) for every board cell: the same plus-of-plusses
// composition one level up.  Runs its own t5 pass — when stones12b is also
// in the spec its prepare fills _t5Val a second time (harmless, ~cheap).
let _t13Val = null;
function _t13Prepare(ctx) {
  _t5Prepare(ctx);
  const cap = ctx.game.N * ctx.game.N, nn = ctx.nearNbr, stride = ctx.nearStride;
  if (!_t13Val || _t13Val.length < cap) _t13Val = new Int32Array(cap);
  const t5 = _t5Val;
  for (let idx = 0; idx < cap; idx++) {
    const base = idx * stride;
    _t13Val[idx] = _hashCombine(_uh(_uh(t5[nn[base]], t5[nn[base + 2]]),
                                    _uh(t5[nn[base + 1]], t5[nn[base + 3]])), t5[idx]);
  }
}

// ── stoneLimit<N>: the largest diamond holding at most N stones ──────────────
// Exact counts come from a per-stone scatter into per-distance planes
// (stones x area), each cell's level from a prefix walk, and a t-hash
// pyramid run only to the deepest level any cell reached.
let _slCnt = null;        // (Rmax+1) x area stone counts by exact distance
const _slRowD = new Int32Array(64), _slColD = new Int32Array(64);   // per-stone wrapped deltas
const _listBuf = new Int32Array(64);        // list-term emission scratch
let _slDist = null;       // torus L1 distance lookup, indexed by (dr*N + dc)
let _slDistN = 0;

// stoneLimit is an OPENING feature: it emits only while board fullness is
// under SL_MAX_PHASE, and outside that window the prepare exits before any
// work, so mid- and endgame positions pay nothing.
const SL_MAX_PHASE = 0.1;
// Training-time dropout: within the phase<SL_MAX_PHASE window, keep stoneLimit
// active for only this fraction of positions (1 = always, the inference/eval
// default).  The featurepol trainer lowers it (setStoneLimitTrainKeep) so the
// shared features can't co-adapt to a memorising opening feature and its noisy
// opening keys steer fewer self-play games.  The window is unchanged — this is
// a coin flip inside it, not a wider/narrower gate.
let _slTrainKeep = 1;

function _slPrepare(ctx, limit, st) {
  const game = ctx.game, N = game.N, area = N * N, cur = ctx.cur;
  st.active = (1 - game.emptyCount / area) < SL_MAX_PHASE
              && (_slTrainKeep >= 1 || Math.random() < _slTrainKeep);
  if (!st.active) return;
  const cells = game.cells, nn = ctx.nearNbr, stride = ctx.nearStride;
  const Rmax = (N >> 1) * 2;
  if (!st.lkeys || st.lkeys.length < (Rmax + 1) * area) {
    st.lkeys = new Int32Array((Rmax + 1) * area); st.depth = new Int32Array(area);
    st.a = new Int32Array(area); st.b = new Int32Array(area);
  }
  if (!_slCnt || _slCnt.length < (Rmax + 1) * area) _slCnt = new Int16Array((Rmax + 1) * area);
  if (_slDistN !== N) {
    // Pre-multiplied by area: the scatter's plane index needs no multiply.
    _slDist = new Int32Array(area);
    const half = N >> 1;
    for (let dr = 0; dr < N; dr++) for (let dc = 0; dc < N; dc++) {
      const wr = dr > half ? N - dr : dr, wc = dc > half ? N - dc : dc;
      _slDist[dr * N + dc] = (wr + wc) * area;
    }
    _slDistN = N;
  }
  const cnt = _slCnt, dist = _slDist;
  cnt.fill(0, 0, (Rmax + 1) * area);
  // Scatter: every stone bumps its exact-distance plane at every cell.  The
  // inner loop is two lookups and an add: per stone, the wrapped row/col
  // deltas are precomputed into dist-table strides, so no division and no
  // branch survives in the area loop.
  const rowD = _slRowD, colD = _slColD;
  for (let s = 0; s < area; s++) {
    if (cells[s] === 0) continue;
    const sr = (s / N) | 0, sc = s - sr * N;
    for (let r = 0; r < N; r++) rowD[r] = ((r - sr + N) % N) * N;
    for (let c = 0; c < N; c++) colD[c] = (c - sc + N) % N;
    let i = 0;
    for (let r = 0; r < N; r++) {
      const dRow = rowD[r];
      for (let c = 0; c < N; c++, i++) cnt[dist[dRow + colD[c]] + i]++;
    }
  }
  // Per cell: the largest level with cumulative stones <= limit (level 0 =
  // just the cell itself, which never emits — depth 0 gates the host space).
  const depth = st.depth;
  let maxL = 0;
  for (let i = 0; i < area; i++) {
    let cum = cnt[i];               // distance 0
    let L = 0;
    for (let o = area + i; o <= Rmax * area + i; o += area) {
      cum += cnt[o];
      if (cum > limit) break;
      L++;
    }
    depth[i] = L;
    if (L > maxL) maxL = L;
  }
  // t-hash pyramid: the BASE (level-1) key is stored for every cell, and each
  // deeper level's key for the cells whose limit reaches it — the full stack.
  const lkeys = st.lkeys;
  let a = st.a, b = st.b;
  {
    for (let idx = 0; idx < area; idx++) {
      const base = idx * stride;
      const cN = cells[nn[base]],     sN = cN === 0 ? 1 : cN === cur ? 2 : 3;
      const cE = cells[nn[base + 1]], sE = cE === 0 ? 1 : cE === cur ? 2 : 3;
      const cS = cells[nn[base + 2]], sS = cS === 0 ? 1 : cS === cur ? 2 : 3;
      const cW = cells[nn[base + 3]], sW = cW === 0 ? 1 : cW === cur ? 2 : 3;
      const cC = cells[idx],          sC = cC === 0 ? 1 : cC === cur ? 2 : 3;
      const v = _hashCombine(_uh(_uh(sN, sS), _uh(sE, sW)), sC) | 0;
      a[idx] = v;
      lkeys[area + idx] = _hashCombine(v, 1) | 0;
    }
  }
  for (let k = 2; k <= maxL; k++) {
    const ko = k * area;
    for (let idx = 0; idx < area; idx++) {
      const base = idx * stride;
      const v = _hashCombine(_uh(_uh(a[nn[base]], a[nn[base + 2]]),
                                 _uh(a[nn[base + 1]], a[nn[base + 3]])), a[idx]) | 0;
      b[idx] = v;
      if (depth[idx] >= k) lkeys[ko + idx] = _hashCombine(v, k) | 0;
    }
    const t = a; a = b; b = t;
  }
}

const _slByN = new Map();
function _slShared(limit) {
  let sh = _slByN.get(limit);
  if (!sh) {
    const st = { lkeys: null, depth: null, a: null, b: null };
    sh = { st, prepare: ctx => _slPrepare(ctx, limit, st) };
    _slByN.set(limit, sh);
  }
  return sh;
}

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

// Optional candidate restriction: when set, only these board indices are ranked
// (everything else gets rank 0, i.e. the space stays dark for them).  Used to
// evaluate "rank only the top-N moves by the cheap features" -- ranks then run
// 1..N within that shortlist rather than over the whole board.  null = rank
// every legal non-true-eye move, which is the normal behaviour.
let _rankAllow = null;     // Set of board indices, or null -- external tooling
let _rankMask  = null;     // Uint8Array board mask, or null -- the top-N path
// The rank-topn width is NOT a process global: it is a per-weights field,
// weights.rankTopN, set by each caller and read by extractFeatures, so two
// models in one process can rank at different widths.  Its value:
//   0  (the default) — the rank feature is OFF: no ranking pass, the rank
//                      spaces emit nothing.  A model whose caller never opts in
//                      pays nothing for a vpat<n> term.
//   N > 0            — rank only the best N moves by the other spaces (a
//                      shortlist), the deployment setting.
//   N < 0            — rank the whole board (every candidate).
// Fraction of positions in which the rank feature is computed at all.  1 = every
// position (the default).  Below 1, the ranking is skipped on the rest and the
// rank spaces emit nothing there, so the feature keeps its whole-board meaning
// -- rank 1 is still "best on the board" -- and only its FREQUENCY drops.  That
// is the difference from weights.rankTopN, which narrows what rank means.
let _rankPosRatio = 1;
let _rank = null;      // _rank[boardIdx] = 1-based rank, 0 = not a candidate
let _rankOrder = null;     // scratch: candidate board indices, sorted best-first
let _rankScore = null;     // scratch: _rankScore[boardIdx] = mover-relative score

// Zero the ranking so the rank spaces stay dark for this position.
function _rankBlank(cap) {
  if (!_rank || _rank.length < cap) {
    _rank  = new Int32Array(cap);
    _rankOrder = new Int32Array(cap);
    _rankScore = new Float64Array(cap);
  }
  _rank.fill(0, 0, cap);
}

// ── vpat: the rank feature — candidates scored by a vpatterns model ────────
//
// Candidates are scored by the vpatterns model's deltaZ (V = sigma(zBase+dz)),
// mirroring vpatsearch's incremental depth-1 loop.  Captures fall back to a
// full extraction on a SEPARATE prepared spec so the base planes deltaZ reads
// stay valid for the remaining candidates.  Ladder-coded models (size:L) have
// no incremental contract at all, so they rank via that fallback for EVERY
// candidate — supported, but several times slower per position.

// The vpat<n> value model is NOT a process-global: it is embedded in the
// featurepol file (loadModel reconstructs it onto weights.vpatModel) or attached
// to the weights by a trainer, and reaches the ranking through ctx.vpatModel.
// This keeps the model bound to the policy it was trained against — two policies
// in one process can hold different vpat models, and the ranking model can never
// drift from the one training used.

// Fill the rank arrays for every legal non-true-eye move of ctx.game.  Runs
// once per position; each vpat<n> evalFn then just reads and clamps.
// ── adjHealth: per-position chain health ──────────────────────────────────────
// P(chain survives) is a per-POSITION quantity, so it is computed once here and
// reused by every candidate move — ~200 ninecell hashes amortised over ~100
// candidates.  Stored per CELL as p*255 so the prepare hook is independent of
// any one term's bucket count.
let _healthModel = null;
function _healthLoad() {
  if (_healthModel) return _healthModel;
  const envPath = (typeof process !== 'undefined' && process.env) ? process.env.FP_HEALTH_DATA : null;
  if (!envPath && typeof window === 'undefined') {
    throw new Error('featurepol: the adjHealth<n> feature needs a health model — set FP_HEALTH_DATA to a train-health.js save file');
  }
  _healthModel = HealthLib.resolveHealthModel(envPath || '');
  return _healthModel;
}
let _hpP = new Uint8Array(0);
function _healthPrepare(ctx) {
  const game = ctx.game, cap = game.N * game.N;
  const model = _healthLoad();
  if (_hpP.length < cap) _hpP = new Uint8Array(cap);
  _hpP.fill(0, 0, cap);
  const chains = HealthLib.chainHealthAll(model, game, ctx.game3);
  for (let i = 0; i < chains.length; i++) {
    const r = chains[i];
    let v = (r.p * 255) | 0;
    if (v > 255) v = 255; else if (v < 0) v = 0;
    for (let k = 0; k < r.stones.length; k++) _hpP[r.stones[k]] = v;
  }
}

function _vpatPrepare(ctx) {
  const game = ctx.game, N = game.N, cap = N * N;
  const model = ctx.vpatModel;
  if (!model) throw new Error('featurepol: the vpat<n> feature needs a vpat model, but none is ' +
    'attached to the weights — the featurepol file has no embedded vpat model.');
  _rankBlank(cap);                   // (re)size + zero the shared rank arrays
  const black = ctx.cur === BLACK;   // vpat z is the BLACK-wins logit

  const prep = model.preparedSpecs;
  const incremental = !prep.hasLadder;
  let zBase = 0;
  if (incremental) {
    const f = VPatterns.extractFeatures(game, prep);   // primes the planes deltaZ reads
    VPatterns.evaluateFeatures(f, model.weights);
    zBase = f.z;
  }

  const emC = game._emptyCells, ec = game.emptyCount;
  let n = 0;
  for (let ei = 0; ei < ec; ei++) {
    const idx = emC[ei];
    if (_rankMask  !== null && _rankMask[idx] === 0) continue;
    if (_rankAllow !== null && !_rankAllow.has(idx)) continue;
    if (!game.isLegal(idx) || game.isTrueEye(idx)) continue;
    let z;
    if (incremental) {
      z = VPatterns.deltaZ(game, prep, model.weights, idx);
      if (z !== z) {
        // Capture: full extraction on a separate prepared spec (vpatsearch's rule).
        const fb = model._fbPrep || (model._fbPrep = VPatterns.prepareSpecs(model.specs,
          { health: model.preparedSpecs && model.preparedSpecs.healthModel,
            ladderMinChain: model.preparedSpecs && model.preparedSpecs.ladderMinChain }));
        const g = game.clone();
        g.play(idx);
        const ff = VPatterns.extractFeatures(g, fb);
        VPatterns.evaluateFeatures(ff, model.weights);
        z = ff.z - zBase;
      }
    } else {
      // Ladder-coded model: full extraction on a clone.  Absolute logits —
      // the position's own zBase is a shared constant, so it cancels in the
      // ranking and never needs computing.  Reuse ctx.game3 (synced to `game`),
      // advancing it per candidate with play/undo instead of rebuilding one.
      const g3 = ctx.game3;
      const g = game.clone();
      g.play(idx);
      g3.play(idx);
      const ff = VPatterns.extractFeatures(g, prep, false, undefined, false, g3);
      VPatterns.evaluateFeatures(ff, model.weights);
      z = ff.z;
      g3.undo();
    }
    _rankScore[idx] = black ? z : -z;             // mover-relative: higher = better
    _rankOrder[n++] = idx;
  }
  const order = _rankOrder.subarray(0, n);
  order.sort((a, b) => _rankScore[b] - _rankScore[a]);
  for (let r = 0; r < n; r++) _rank[order[r]] = r + 1;
}

// The rank-prepare tag: everything that special-cases "the rank feature" (the
// top-N split, the pos-ratio gate) keys on this, not on function identity.
_vpatPrepare._isRank = true;

// ── amaf: rank the whole board by an aggregate all-moves-as-first win ratio ──
//
// One batch of P uniform, first-play-wins playouts from the position scores
// EVERY cell at once (the all-moves-as-first trick): each cell's aggregate win
// ratio folds its own moves at full weight and the opponent's at
// _AMAF_OPP_WEIGHT, the first-play weight decaying _AMAF_DECAY/area per move.
// Candidates are ranked by that ratio and then encoded exactly like vpat<n>
// (cumulative rank thermometer) -- but with NO top-N gate: one cheap batch
// ranks the whole board, so amaf spaces are plain (non-_isRank) and always run.
//
// Determinism without plumbing: the batch rng is seeded from a hash of the
// board (cells + side to move + P), so the feature is a pure function of the
// position -- identical in training, inference and inspection -- and never
// touches or perturbs the caller's rng.  The rollout constants are the amaf
// agent's fielded defaults.
const _AMAF_DECAY = 0.5, _AMAF_OPP_WEIGHT = 0.3;
const _amafByP = new Map();     // P -> { prepare, rank, order, ratio, wins, plays, played }

function _amafSeed(game, P) {
  // game.hash() already folds in cells + side to move; combine with P so two
  // playout budgets on the same position draw independent batches.
  return (_hashCombine(game.hash(), P >>> 0) >>> 0) || 1;
}

function _amafEntry(P) {
  let e = _amafByP.get(P);
  if (e) return e;
  e = { prepare: null, rank: null, order: null, ratio: null, wins: null, plays: null, played: null };
  e.prepare = (ctx) => _amafRun(ctx, P, e);
  _amafByP.set(P, e);
  return e;
}

// Fill e.rank[boardIdx] (1-based rank over legal non-true-eye moves, 0 = not a
// candidate) from a P-playout AMAF batch.  Runs once per position; each amaf<n>
// evalFn then just reads and cumulatively encodes the rank.
function _amafRun(ctx, P, e) {
  const game = ctx.game, cap = game.N * game.N;
  if (!e.rank || e.rank.length < cap) {
    e.rank   = new Int32Array(cap);
    e.order  = new Int32Array(cap);
    e.ratio  = new Float64Array(cap);
    e.wins   = new Float64Array(cap);
    e.plays  = new Float64Array(cap);
    e.played = new Float32Array(cap);
  }
  const rank = e.rank, order = e.order, ratio = e.ratio;
  const wins = e.wins, plays = e.plays, played = e.played;
  rank.fill(0, 0, cap); wins.fill(0, 0, cap); plays.fill(0, 0, cap);

  const player = ctx.cur, playerSign = player === BLACK ? 1 : -1;
  const weightStep = _AMAF_DECAY / cap, moveLimit = cap + 20;
  const rng = makeRng(_amafSeed(game, P));

  for (let p = 0; p < P; p++) {
    const clone = game.clone();
    played.fill(0, 0, cap);
    let moves = 0, weight = 1.0;
    while (!clone.gameOver && moves < moveLimit) {
      const current = clone.current;
      const idx = clone.randomLegalMove(rng);
      if (idx === PASS) { clone.play(PASS); moves++; continue; }
      if (played[idx] === 0) played[idx] = current === BLACK ? weight : -weight;
      clone.play(idx);
      moves++;
      weight -= weightStep; if (weight < 0) weight = 0;
    }
    const won = clone.estimateWinner() === player ? 1 : 0;
    for (let k = 0; k < cap; k++) {
      const w = played[k];
      if (w === 0) continue;
      const aw = w < 0 ? -w : w;                       // magnitude (decayed first-play weight)
      if (w * playerSign > 0) { plays[k] += aw;                     wins[k] += won * aw; }
      else                    { const wt = aw * _AMAF_OPP_WEIGHT;   plays[k] += wt; wins[k] += (1 - won) * wt; }
    }
  }

  // Rank every legal non-true-eye move by its aggregate win ratio (best first).
  const emC = game._emptyCells, ec = game.emptyCount;
  let n = 0;
  for (let ei = 0; ei < ec; ei++) {
    const idx = emC[ei];
    if (!game.isLegal(idx) || game.isTrueEye(idx)) continue;
    ratio[idx] = plays[idx] > 0 ? wins[idx] / plays[idx] : -1;   // unplayed candidate sorts last
    order[n++] = idx;
  }
  const ord = order.subarray(0, n);
  ord.sort((a, b) => ratio[b] - ratio[a]);
  for (let r = 0; r < n; r++) rank[ord[r]] = r + 1;
}

// Build one feature term { str, kind, param, salt, evalFn, needsLadder } from a
// token like "capture6" / "stones4" / "ladderStatus".
function _makeTerm(str) {
  // A term may carry a ':'-suffixed SECOND parameter (currently only amaf<n>:<P>
  // -- rank levels : playout count).  Split it off before the name+digits parse;
  // the full str still salts the key, so the second parameter stays in it.
  let colonParam = null, core = str;
  const ci = str.indexOf(':');
  if (ci >= 0) { core = str.slice(0, ci); colonParam = parseInt(str.slice(ci + 1), 10); }
  // kind = leading word (may contain interior digits, e.g. stone8AdjLib); param =
  // the OPTIONAL trailing run of digits.  Lazy kind + greedy trailing \d* split them.
  const m = /^([a-zA-Z][a-zA-Z0-9]*?)(\d*)$/.exec(core);
  if (!m) throw new Error(`featurepol: bad feature term "${str}"`);
  // stones24 is a fixed-shape NAME (the stones12b recursion), not stones<n>;
  // the trailing-digit split cannot tell, so match the exact token.
  const kind = str === 'stones24' ? 'stones24' : m[1];
  const param = kind === 'stones24' ? null : (m[2] ? parseInt(m[2], 10) : null);
  let salt = _hashStr(str);
  // A size<n> term is CUMULATIVE: it carries sizeFn (the raw size) and is expanded
  // by parseSpec into n additive "≥k present" indicator spaces (thermometer
  // encoding) so the logit sums over size, like npat's tactical slots.  Other
  // terms set evalFn directly (one value → one key).
  // maxNear: how many of the nearest cells this term reads from the nearNbr table
  // (0 if it reads none).  parseSpec takes the max across the spec to size the table.
  let evalFn = null, sizeFn = null, cumulative = false, oneHot = false, needsLadder = false, needsT3 = false, binary = false, maxNear = 0;
  let prepare = null, stacked = null;   // stacked: an additional weight space this keyword emits into (see parseSpec)
  let listFn = null;                    // list term: emits one key per value it writes into the shared buffer
  switch (kind) {
    case 'vpat': {
      // Rank under the external value model as a cumulative size: rank r ->
      // n+1-r, so the top choice lights levels 1..n and rank n only level 1.
      // Rank > n gives 0, which gates the space off.  n weights per space.
      if (param === null || param < 1) throw new Error(`featurepol: vpat<n> needs a rank count n >= 1, got "${str}"`);
      const n = param;
      prepare = _vpatPrepare;
      cumulative = true;
      sizeFn = (ctx, idx) => { const r = _rank[idx]; return (r >= 1 && r <= n) ? (n + 1 - r) : 0; };
      break;
    }
    case 'amaf': {
      // Rank the whole board by the aggregate AMAF win ratio over P playouts,
      // then encode the mover-relative rank as a cumulative size exactly like
      // vpat<n> (rank r -> n+1-r).  No external model and no top-N gate: one
      // batch ranks everyone, so this is a plain (always-run) prepare.
      if (param === null || param < 1) throw new Error(`featurepol: amaf<n>:<P> needs rank levels n >= 1, got "${str}"`);
      if (!(colonParam >= 1)) throw new Error(`featurepol: amaf<n>:<P> needs a playout count P >= 1, got "${str}"`);
      const n = param, e = _amafEntry(colonParam);
      prepare = e.prepare;
      cumulative = true;
      sizeFn = (ctx, idx) => { const r = e.rank[idx]; return (r >= 1 && r <= n) ? (n + 1 - r) : 0; };
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
    case 'stones12b': {
      // 13-cell shape (stones12 + centre) hashed as a plus of five 5-point
      // plusses.  Invariant by construction, so no permutation enumeration.
      if (param !== null) throw new Error(`featurepol: stones12b takes no size, got "${str}"`);
      maxNear = 4;   // reads only the 4 orthogonals — of the move AND of its neighbours
      prepare = _t5Prepare;
      evalFn = (ctx, idx) => {
        const nn = ctx.nearNbr, base = idx * ctx.nearStride, t5 = _t5Val;
        return _hashCombine(_uh(_uh(t5[nn[base]], t5[nn[base + 2]]),
                                _uh(t5[nn[base + 1]], t5[nn[base + 3]])), t5[idx]);
      };
      break;
    }
    case 'stones24': {
      // 25-cell shape: stones12b recursed — plus of five 13-cell t-hashes.
      if (param !== null) throw new Error(`featurepol: stones24 takes no size, got "${str}"`);
      maxNear = 4;
      prepare = _t13Prepare;
      evalFn = (ctx, idx) => {
        const nn = ctx.nearNbr, base = idx * ctx.nearStride, t13 = _t13Val;
        return _hashCombine(_uh(_uh(t13[nn[base]], t13[nn[base + 2]]),
                                _uh(t13[nn[base + 1]], t13[nn[base + 3]])), t13[idx]);
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
    case 'stoneLimit': {
      // The largest diamond holding at most N stones, hashed at its reached
      // level; a reached-level thermometer stacks alongside it.  N is semantic
      // (patterns mean "<= N stones"), so it stays in the salts.
      if (param === null || param < 0 || param > 8) throw new Error(`featurepol: stoneLimit<N> needs a stone limit N in 0..8, got "${str}"`);
      {
        maxNear = 4;
        const sh = _slShared(param), st = sh.st;
        prepare = sh.prepare;
        // FULL STACK, base always: the level-1 pattern for every candidate,
        // plus every deeper within-limit level's pattern (a list term — one
        // key per level).  The reached-level thermometer stacks alongside.
        listFn = (ctx, idx, out) => {
          if (!st.active) return 0;
          const area = ctx.game.N * ctx.game.N, lkeys = st.lkeys;
          const L = st.depth[idx];
          out[0] = lkeys[area + idx];
          for (let k = 2; k <= L; k++) out[k - 1] = lkeys[k * area + idx];
          return L > 1 ? L : 1;
        };
        stacked = { str: `_stoneLimitThermometer${param}`, saltStr: `_stoneLimitThermometer${param}`,
                    salt: _hashStr(`_stoneLimitThermometer${param}`),
                    cumulative: true, maxLevel: 0, maxNear: 4, needsLadder: false,
                    prepare: sh.prepare, sizeFn: (ctx, idx) => st.active ? st.depth[idx] : 0 };
      }
      break;
    }
    case 'adjHealth': {
      // adjLib's shape with health buckets in place of liberty counts: per-cell
      // symbol (radix 2n+1) 0 = empty, 1..n = own stone in that health bucket,
      // n+1..2n = enemy stone likewise.  Canonicalised over D4.  A descriptor.
      if (!param || param < 2) throw new Error(`featurepol: adjHealth<n> needs a bucket count n >= 2, got "${str}"`);
      {
        const n = param, radix = 2 * n + 1;
        maxNear = 4;
        prepare = _healthPrepare;
        evalFn = (ctx, idx) => {
          const nn = ctx.nearNbr, base = idx * ctx.nearStride, game = ctx.game;
          const cells = game.cells, cur = ctx.cur, cv = _cvScratch;
          for (let i = 0; i < 4; i++) {
            const ni = nn[base + i], c = cells[ni];
            if (c === 0) { cv[i] = 0; continue; }
            let b = ((_hpP[ni] * n) >> 8) + 1;
            if (b > n) b = n;
            cv[i] = (c === cur) ? b : n + b;
          }
          return _canonRadix(cv, 4, radix);
        };
      }
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
      // ONE-HOT (blended toroidal) distance from the move to the previous move,
      // mapped to an integer level via floor(Game2.distance*2-1): an
      // orthogonal-adjacent move is level 1, rising with distance, capped at n.
      // Exactly ONE key per move — the reached level — not a 1..k thermometer
      // (distance bands are not a monotone accumulation, and a thermometer
      // emitted up to n keys per candidate for marginal signal).  No previous
      // move (or a pass) yields level 0, the reference state, which emits
      // nothing (gated).  Graded sibling of `local`.
      if (!param) throw new Error(`featurepol: dist<n> needs a cap, got "${str}"`);
      cumulative = true;
      oneHot = true;
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
    case 't3Status': {
      // ladderStatus's mask over the tactics3 flags.
      needsT3 = true;
      evalFn = (ctx, idx) => {
        const s = ctx.t3Sizes, b = idx * 4;
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
    case 't3Kill': case 't3Save': case 't3WastedExtend': case 't3WastedAttack': {
      // The same size variant over the tactics3 flags.
      if (!param) throw new Error(`featurepol: ${kind}<n> needs a size, got "${str}"`);
      needsT3 = true;
      const flag = { t3Kill: URGENT_KILL, t3Save: URGENT_SAVE,
                     t3WastedExtend: WASTED_EXTEND, t3WastedAttack: WASTED_ATTACK }[kind];
      cumulative = true; sizeFn = (ctx, idx) => ctx.t3Sizes[idx * 4 + flag];
      break;
    }
    default:
      throw new Error(`featurepol: unknown feature kind "${kind}" in "${str}"`);
  }
  return { str, kind, param, salt, evalFn, sizeFn, cumulative, oneHot, maxLevel: param, needsLadder, needsT3, binary, prepare, maxNear, stacked, listFn };
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
  let needsLadder = false, needsT3 = false;
  let nearMax = 0;            // widest nearNbr reach across all terms (sizes the table)
  const slotOf = new Map();   // term salt → memo slot
  const computers = [];       // computers[slot] = value fn (sizeFn for cumulative, else evalFn)
  const prepares = [];        // whole-position hooks, run once per position before the move loop
  function slotFor(t) {
    let slot = slotOf.get(t.salt);
    if (slot === undefined) { slot = computers.length; slotOf.set(t.salt, slot); computers.push(t.cumulative ? t.sizeFn : t.evalFn); }
    return slot;
  }
  const stackedTerms = [];    // synthetic companion terms (a feature's depth thermometer), deduped by salt
  let boardMaxSpaces = 0;     // spaces whose key count resolves at createState (board maximum)
  for (const spaceStr of str.split(',').map(s => s.trim()).filter(Boolean)) {
    const terms = spaceStr.split('+').map(t => t.trim()).filter(Boolean).map(_makeTerm);
    if (terms.length === 0) throw new Error(`featurepol: empty feature space in "${spaceStr}"`);
    const gate = [];        // slots that must be ≥ 1 for the space to fire at all
    const baseTerms = [];   // descriptors (bin:false, fold value) + binary events (bin:true, fold level 1)
    const cumTerms = [];    // cumulative size terms (thermometer cross-product)
    // A LIST term emits one key per value (per-level pattern stacks); it
    // owns its space — '+' composition has no defined cross-product with a
    // variable-length key list.
    if (terms.some(t => t.listFn)) {
      if (terms.length !== 1) throw new Error(`featurepol: "${spaceStr}" — a list term cannot be composed with '+'`);
      const t = terms[0];
      if (t.needsLadder) needsLadder = true;
      if (t.needsT3) needsT3 = true;
      if (t.maxNear > nearMax) nearMax = t.maxNear;
      if (t.prepare && !prepares.includes(t.prepare)) prepares.push(t.prepare);
      if (t.stacked && !stackedTerms.some(x => x.salt === t.stacked.salt)) stackedTerms.push(t.stacked);
      spaces.push({ str: spaceStr, salt: _hashStr('space:' + spaceStr), gate: [], baseTerms: [],
                    cumTerms: [], maxKeys: 0, usesRank: false,
                    listFn: t.listFn, listSalt: t.salt });
      boardMaxSpaces++;          // keys per move bounded by the board maximum
      continue;
    }
    for (const t of terms) {
      if (t.needsLadder) needsLadder = true;
      if (t.needsT3) needsT3 = true;
      if (t.maxNear > nearMax) nearMax = t.maxNear;
      const slot = slotFor(t);
      if (t.prepare && !prepares.includes(t.prepare)) prepares.push(t.prepare);
      if (t.cumulative)       { gate.push(slot); cumTerms.push({ salt: t.salt, slot, maxLevel: t.maxLevel, oneHot: t.oneHot }); }
      else if (t.binary)      { gate.push(slot); baseTerms.push({ salt: t.salt, slot, bin: true }); }
      else                    { baseTerms.push({ salt: t.salt, slot, bin: false }); }
      if (t.stacked && !stackedTerms.some(s => s.salt === t.stacked.salt)) stackedTerms.push(t.stacked);
    }
    let maxKeys = 1;
    for (const c of cumTerms) maxKeys *= (c.oneHot ? 1 : c.maxLevel);
    const usesRank = terms.some(t => t.prepare && t.prepare._isRank);
    spaces.push({ str: spaceStr, salt: _hashStr('space:' + spaceStr),
                  gate, baseTerms, cumTerms, maxKeys, usesRank });
  }
  // A keyword is not limited to one key family: `stacked` is an additional
  // weight space the keyword emits into (a per-level thermometer, e.g.
  // stoneLimit's, 0..N keys per move) alongside whatever its host space emits.
  // Its prepare is the parent term's object, so the includes() dedupe above
  // already covered it.  maxLevel 0 = board maximum: the size values the
  // prepare produces are already capped at the board's largest L1 distance, so
  // the emission clamp becomes Infinity (never binds) and the space's key
  // budget is deferred to createState, where N is known (boardMaxSpaces).
  for (const t of stackedTerms) {
    const slot = slotFor(t);
    const unbounded = t.maxLevel === 0;
    if (unbounded) boardMaxSpaces++;
    // synthetic: internal plumbing, not part of the user's spec — interfaces
    // listing spaces (the trainer's resume diff) skip these; they track their
    // parent term exactly.
    spaces.push({ str: t.str, salt: _hashStr('space:' + (t.saltStr || t.str)), gate: [slot], baseTerms: [],
                  cumTerms: [{ salt: t.salt, slot, maxLevel: unbounded ? Infinity : t.maxLevel }],
                  maxKeys: unbounded ? 0 : t.maxLevel, usesRank: false, synthetic: true });
  }
  if (spaces.length === 0) throw new Error(`featurepol: no feature spaces in "${str}"`);
  let maxKeysPerMove = 0;
  for (const sp of spaces) maxKeysPerMove += sp.maxKeys;
  // Split for the top-N path: which spaces need the rank feature’s ranking, and which memo
  // slots the remaining ("plain") spaces read.  Plain spaces can be scored
  // before the ranking exists, which is what makes the shortlist possible.
  const rankSpaces  = spaces.filter(sp => sp.usesRank);
  const plainSpaces = spaces.filter(sp => !sp.usesRank);
  const plainSlots = [];
  for (const sp of plainSpaces)
    for (const t of [...sp.baseTerms, ...sp.cumTerms]) if (!plainSlots.includes(t.slot)) plainSlots.push(t.slot);
  const rankMaxKeys = rankSpaces.reduce((a, sp) => a + sp.maxKeys, 0);
  return { str, spaces, plainSpaces, rankSpaces, plainSlots, rankMaxKeys,
           computers, numSlots: computers.length, prepares, maxKeysPerMove, boardMaxSpaces, needsLadder, needsT3, nearMax };
}

// ── Weights store (hash → dense idx → Float32 weight) ─────────────────────────

function createWeights(opts = {}) {
  const spec = parseSpec(opts.spec);
  const initialCapacity = opts.initialCapacity || 1024;
  return {
    spec,
    nSpaces: spec.spaces.length,
    map:   makeIntMap(Math.max(64, initialCapacity * 2)),   // 32-bit hash → dense idx (int-map: key 0 reserved, get miss = -1)
    vals:  new Float32Array(initialCapacity),  // weight[dense idx]
    delta: new Float32Array(initialCapacity),  // reusable scatter buffer
    count: new Int32Array(initialCapacity),    // per-key contributor count (for per-key gradient mean)
    size:  0,
    rankTopN: 0,   // vpat<n> rank width (per-instance): 0 = OFF (default), N>0 = top-N, N<0 = whole board
    // Smallest chain the ladder terms read (ladder2's minChainSize).  A
    // property of the trained model, saved with it; files without the field
    // are 1, every chain.  2 skips single stones, ~57% of the ladder time.
    ladderMinChain: opts.ladderMinChain || 1,
    // Smallest chain the tactics3 terms read, and their search bounds; saved
    // like ladderMinChain.  Files without the bounds get tactics3's defaults.
    t3MinChain: opts.t3MinChain || 1,
    t3DepthLimit: opts.t3DepthLimit ?? Tactics3.DEFAULT_DEPTH_LIMIT,
    t3NodeLimit:  opts.t3NodeLimit  ?? Tactics3.DEFAULT_NODE_LIMIT,
  };
}

function _intern(w, key) {
  key |= 0;                       // int-map stores int32; hashes arrive as uint32
  const map = w.map;
  const existing = map.get(key);
  if (existing >= 0) return existing;
  // no-add mode (weights.noAdd, the trainers' --no-add): unknown keys are
  // not interned — the caller skips them, so they contribute zero to the
  // logit and receive no gradient, and the key set stays exactly as loaded.
  // maxWeights (the trainers' --max-weights) is the size-triggered form:
  // existing keys keep training, new ones stop interning at the cap.
  if (w.noAdd || (w.maxWeights > 0 && w.size >= w.maxWeights)) return -1;
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

// opts.components: also record, per move and space, where that space's keys
// start in `keys`, so componentValues can give a per-space breakdown.
function createState(N, spec, opts) {
  spec = parseSpec(spec);
  const cap = N * N;
  // Upper bound on keys emitted per move; a board-maximum thermometer space
  // (a list term's) contributes its resolved depth here, N finally being known.
  const maxK = spec.maxKeysPerMove + spec.boardMaxSpaces * ((N >> 1) * 2);
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
    keys2:    new Int32Array(cap * maxK),       // top-N path: rebuild target
    rKeys:    new Int32Array(cap * Math.max(1, spec.rankMaxKeys)),   // shortlisted moves' rank-space keys
    rCount:   new Int32Array(cap),
    rMask:    new Uint8Array(cap),              // top-N path: shortlist membership
    pScore:   new Float64Array(cap),            // plain-space score, drives the shortlist
    pOrder:   new Int32Array(cap),
    memo:     new Float64Array(spec.numSlots),  // per-move scratch: one value per distinct fixed-space term
    spaceOff: (opts && opts.components) ? new Int32Array(cap * spec.spaces.length) : null,   // see componentValues
    ladderSizes: new Uint16Array(cap * 4),
    t3Sizes:  new Uint16Array(cap * 4),
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
// Emit one space's keys for the move whose term values are in `memo`.  Same
// logic as the main loop below; factored out because the top-N path emits the
// plain spaces and the rank spaces in two separate phases.
function _emitSpace(sp, memo, weights, out, pos, accA, accB, ctx, idx) {
  if (sp.listFn) {
    const n = sp.listFn(ctx, idx, _listBuf);
    for (let j = 0; j < n; j++) {
      const ix = _intern(weights, _hashCombine(sp.salt, _hashCombine(sp.listSalt, _listBuf[j] >>> 0)) >>> 0);
      if (ix >= 0) out[pos++] = ix;
    }
    return pos;
  }
  const gate = sp.gate;
  for (let gi = 0; gi < gate.length; gi++) if (memo[gate[gi]] < 1) return pos;   // space stays dark
  let base = sp.salt;
  const bt = sp.baseTerms;
  for (let bi = 0; bi < bt.length; bi++) {
    const t = bt[bi];
    base = _hashCombine(base, _hashCombine(t.salt, t.bin ? 1 : (memo[t.slot] >>> 0)));
  }
  const ct = sp.cumTerms;
  if (ct.length === 0) {
    { const ix = _intern(weights, base >>> 0); if (ix >= 0) out[pos++] = ix; }
  } else if (ct.length === 1) {
    const t = ct[0]; let sz = memo[t.slot]; if (sz > t.maxLevel) sz = t.maxLevel;
    for (let k = t.oneHot ? sz : 1; k <= sz; k++) { const ix = _intern(weights, _hashCombine(base, _hashCombine(t.salt, k)) >>> 0); if (ix >= 0) out[pos++] = ix; }
  } else {
    let acc = accA, nxt = accB, nAcc = 1; acc[0] = base;
    for (let ci = 0; ci < ct.length; ci++) {
      const t = ct[ci]; let sz = memo[t.slot]; if (sz > t.maxLevel) sz = t.maxLevel;
      let on = 0;
      for (let a = 0; a < nAcc; a++) { const ba = acc[a]; for (let k = t.oneHot ? sz : 1; k <= sz; k++) nxt[on++] = _hashCombine(ba, _hashCombine(t.salt, k)); }
      const tmp = acc; acc = nxt; nxt = tmp; nAcc = on;
    }
    for (let a = 0; a < nAcc; a++) { const ix = _intern(weights, acc[a] >>> 0); if (ix >= 0) out[pos++] = ix; }
  }
  return pos;
}

// Top-N extraction: score every candidate on the spaces that do NOT need the
// ranking, then rank within the best `weights.rankTopN` of them only.  The
// whole point is that the ranking -- the expensive part -- runs over a handful
// of candidates instead of the whole board.  Ranks are then 1..N within that
// shortlist, so this is an APPROXIMATION of the whole-board feature: a move the
// plain spaces dislike can never be ranked, however good the value model thinks it is.
function _extractTopN(game, state, weights, ctx, spec, useRank) {
  // Non-rank whole-position precomputes still have to run here: this path
  // returns before extractFeatures' prepare block ever executes.  (The rank
  // ranking is this function's own Phase B, hence the exclusion.)
  const prePreps = spec.prepares;
  if (prePreps) for (let i = 0; i < prePreps.length; i++)
    if (!prePreps[i]._isRank) prePreps[i](ctx);
  const computers = spec.computers, memo = state.memo, vals = weights.vals;
  const plainSlots = spec.plainSlots, plainSpaces = spec.plainSpaces, rankSpaces = spec.rankSpaces;
  const emC = game._emptyCells, ec = game.emptyCount;
  const moves = state.moves, keyOff = state.keyOff, pScore = state.pScore;
  const accA = state.accA, accB = state.accB;
  let keys = state.keys;
  let count = 0, pos = 0;
  keyOff[0] = 0;
  // Phase A — plain spaces, and the score that picks the shortlist.
  for (let ei = 0; ei < ec; ei++) {
    const idx = emC[ei];
    if (!game.isLegal(idx) || game.isTrueEye(idx)) continue;
    for (let si = 0; si < plainSlots.length; si++) { const sl = plainSlots[si]; memo[sl] = computers[sl](ctx, idx); }
    const start = pos;
    for (let s = 0; s < plainSpaces.length; s++) pos = _emitSpace(plainSpaces[s], memo, weights, keys, pos, accA, accB, ctx, idx);
    let sc = 0;
    for (let k = start; k < pos; k++) sc += vals[keys[k]];
    pScore[count] = sc;
    moves[count] = idx; count++; keyOff[count] = pos;
  }
  state.count = count;
  if (count === 0) return;
  if (useRank === false) { _rankBlank(game.N * game.N); return; }   // the rank feature sits out this position
  // Phase B — run the ranking over the top-N by plain score, then emit their keys.
  // Partial selection, not a sort: n is 2-6 against ~30 candidates, so a few
  // linear max passes beat a comparator sort and allocate nothing.
  const n = Math.min(weights.rankTopN, count), rank = state.pOrder, mask = state.rMask;
  for (let i = 0; i < n; i++) {
    let best = -1, bestS = -Infinity;
    for (let j = 0; j < count; j++) {
      if (mask[moves[j]] !== 0) continue;                 // already taken
      if (best < 0 || pScore[j] > bestS) { best = j; bestS = pScore[j]; }
    }
    rank[i] = best; mask[moves[best]] = 1;
  }
  _rankMask = mask;
  // Only the rank prepare belongs here: it is the ranking, and it must see the
  // shortlist mask.  The others already ran at the top of this function.
  const preps = spec.prepares;
  for (let i = 0; i < preps.length; i++) if (preps[i]._isRank) preps[i](ctx);
  _rankMask = null;
  for (let i = 0; i < n; i++) mask[moves[rank[i]]] = 0;   // O(n) clear, not O(board)
  const rKeys = state.rKeys, rCount = state.rCount, stride = spec.rankMaxKeys, numSlots = spec.numSlots;
  rCount.fill(0, 0, count);
  for (let ii = 0; ii < n; ii++) {
    const ci = rank[ii], idx = moves[ci];
    // The rank spaces may read terms the plain spaces never used (stones4 in
    // stones4+vpat6, say), so recompute every slot for these few moves.
    for (let sl = 0; sl < numSlots; sl++) memo[sl] = computers[sl](ctx, idx);
    const base = ci * stride;
    let hp = base;
    for (let s = 0; s < rankSpaces.length; s++) hp = _emitSpace(rankSpaces[s], memo, weights, rKeys, hp, accA, accB, ctx, idx);
    rCount[ci] = hp - base;
  }
  // Splice the rank-space keys into each move's range, keeping the flat layout every
  // downstream consumer (_score, the gradient paths) expects.
  const keys2 = state.keys2;
  let p2 = 0;
  for (let i = 0; i < count; i++) {
    const a = keyOff[i], b = keyOff[i + 1], startNew = p2;
    for (let k = a; k < b; k++) keys2[p2++] = keys[k];
    const hc = rCount[i], hb = i * stride;
    for (let k = 0; k < hc; k++) keys2[p2++] = rKeys[hb + k];
    keyOff[i] = startNew;
  }
  keyOff[count] = p2;
  state.keys = keys2; state.keys2 = keys;
}

function extractFeatures(game, state, weights, game3) {
  const spec = weights.spec;
  const spaces = spec.spaces, nSpaces = spaces.length;
  const memo = state.memo;
  const ctx = { game, cur: game.current, nearNbr: state.nearNbr, nearStride: state.nearStride, ladderSizes: null, t3Sizes: null, memo, vpatModel: weights.vpatModel || null, game3: null };
  // Build one Game3 (synced to `game`) if featurepol's own ladder or tactics3
  // terms or the embedded vpat<n> ladder model needs one, and stash it on ctx
  // so the vpat rank prepare can reuse it per candidate instead of rebuilding.
  if (spec.needsLadder || spec.needsT3 || (ctx.vpatModel && VPatterns.needsGame3(ctx.vpatModel.preparedSpecs))) {
    ctx.game3 = game3 || game3FromGame2(game);
    if (spec.needsLadder) ctx.ladderSizes = _buildLadderSizes(game, ctx.game3, state.ladderSizes, weights.ladderMinChain);
    if (spec.needsT3) ctx.t3Sizes = _buildT3Sizes(game, ctx.game3, state.t3Sizes, weights.t3MinChain, weights.t3DepthLimit, weights.t3NodeLimit);
  }
  // Is the rank feature live for this position at all?
  const useRank = _rankPosRatio >= 1 || Math.random() < _rankPosRatio;
  // rankTopN > 0: rank a top-N shortlist.  (rankTopN == 0 is OFF; < 0 is
  // whole-board — both handled in the whole-position branch below.)
  if (weights.rankTopN > 0 && spec.rankSpaces && spec.rankSpaces.length > 0) return _extractTopN(game, state, weights, ctx, spec, useRank);
  // Whole-position precomputes (e.g. the vpat ranking) -- once per position, before
  // any per-move term runs.
  const preps = spec.prepares;
  if (preps) for (let i = 0; i < preps.length; i++) {
    // Non-rank precomputes always run.  The rank prepare runs only in whole-board
    // mode (rankTopN < 0); at rankTopN == 0 the feature is off, so the rank stays
    // blank and the rank spaces emit nothing.  (useRank gates it per position.)
    if (!preps[i]._isRank) preps[i](ctx);
    else if (weights.rankTopN < 0 && useRank) preps[i](ctx);
    else _rankBlank(game.N * game.N);
  }
  const computers = spec.computers, numSlots = spec.numSlots;
  const emC = game._emptyCells, ec = game.emptyCount;
  const moves = state.moves, keys = state.keys, keyOff = state.keyOff, spaceOff = state.spaceOff;
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
      if (spaceOff) spaceOff[count * nSpaces + s] = pos;
      if (sp.listFn) {
        const n = sp.listFn(ctx, idx, _listBuf);
        for (let j = 0; j < n; j++) {
          const ix = _intern(weights, _hashCombine(sp.salt, _hashCombine(sp.listSalt, _listBuf[j] >>> 0)) >>> 0);
          if (ix >= 0) keys[pos++] = ix;
        }
        continue;
      }
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
        { const ix = _intern(weights, base >>> 0); if (ix >= 0) keys[pos++] = ix; }
      } else if (ct.length === 1) {
        // One key: the reached level for a one-hot term (dist), else the
        // present thermometer levels 1..min(size, maxLevel) (size >= 1, gated).
        const t = ct[0]; let sz = memo[t.slot]; if (sz > t.maxLevel) sz = t.maxLevel;
        for (let k = t.oneHot ? sz : 1; k <= sz; k++) { const ix = _intern(weights, _hashCombine(base, _hashCombine(t.salt, k)) >>> 0); if (ix >= 0) keys[pos++] = ix; }
      } else {
        // ≥2 cumulative terms: cross-product of their present levels (rare).
        let acc = accA, nxt = accB, nAcc = 1; acc[0] = base;
        for (let ci = 0; ci < ct.length; ci++) {
          const t = ct[ci]; let sz = memo[t.slot]; if (sz > t.maxLevel) sz = t.maxLevel;
          let on = 0;
          for (let a = 0; a < nAcc; a++) { const ba = acc[a]; for (let k = t.oneHot ? sz : 1; k <= sz; k++) nxt[on++] = _hashCombine(ba, _hashCombine(t.salt, k)); }
          const tmp = acc; acc = nxt; nxt = tmp; nAcc = on;
        }
        for (let a = 0; a < nAcc; a++) { const ix = _intern(weights, acc[a] >>> 0); if (ix >= 0) keys[pos++] = ix; }
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

// Per-space breakdown of every candidate move's score: values[i * nSpaces + s]
// is the sum of move i's weights from space s (the spaces in spec order, named
// in `spaces`).  The state must have been created with { components: true }.
// Whole-position extraction only (no top-N rank shortlist).  For tools that
// convert one space of a model into another representation.
function componentValues(game, state, weights, game3) {
  if (!state.spaceOff) throw new Error('featurepol componentValues: create the state with { components: true }');
  const spec = weights.spec, nSpaces = spec.spaces.length;
  if (weights.rankTopN > 0 && spec.rankSpaces && spec.rankSpaces.length > 0) throw new Error('featurepol componentValues: not available on the top-N rank path');
  extractFeatures(game, state, weights, game3);
  const n = state.count, vals = weights.vals, keys = state.keys, keyOff = state.keyOff, so = state.spaceOff;
  const values = new Float64Array(n * nSpaces);
  for (let i = 0; i < n; i++) {
    for (let s = 0; s < nSpaces; s++) {
      const start = so[i * nSpaces + s], end = s + 1 < nSpaces ? so[i * nSpaces + s + 1] : keyOff[i + 1];
      let v = 0; for (let k = start; k < end; k++) v += vals[keys[k]];
      values[i * nSpaces + s] = v;
    }
  }
  return { count: n, moves: state.moves, spaces: spec.spaces.map(sp => sp.str), values };
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
    // Greedy with reservoir sampling over the argmax ties: scan once, and on
    // meeting the j-th tied max replace the pick with probability 1/j —
    // uniform over the tied class with no second pass or buffer.  Ties are
    // EXACT float equality, which symmetry twins satisfy bitwise (identical
    // key families summed in the same per-space order); without this, board
    // order always picked the same member of the best class — on a torus,
    // the same opening every game.  Scans logits: probs is one-hot here.
    const lg = state.logits, R = rng || Math;
    let best = 0, ties = 1;
    for (let i = 1; i < n; i++) {
      const s = lg[i];
      if (s > lg[best]) { best = i; ties = 1; }
      else if (s === lg[best] && R.random() * ++ties < 1) best = i;
    }
    return { move: state.moves[best], index: best, prob: 1 / ties };
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

function serialize(weights, meta = {}, saveZeros = false) {
  let maxAbs = 0;
  weights.map.forEach((k, d) => { const a = Math.abs(weights.vals[d]); if (a > maxAbs) maxAbs = a; });
  const scale = maxAbs > 0 ? 32767 / maxAbs : 1;
  // Keys whose weight quantizes to 0 are NOT written: they contribute
  // nothing on load and would bloat both the file and the reloaded table
  // (at stones20 scale the sub-floor tail was ~70% of all keys).
  // saveZeros keeps them — for models whose KEY SET is the payload, like the
  // orphans file featurepol-subtract.js consumes: an enumeration pass interns
  // keys at exact zero, and the default path would write an empty model.
  let count = 0;
  weights.map.forEach((k, d) => { if (saveZeros || Math.round(weights.vals[d] * scale) !== 0) count++; });
  const keys = new Int32Array(count), qvals = new Int16Array(count);
  let i = 0;
  weights.map.forEach((key, d) => {
    let q = Math.round(weights.vals[d] * scale);
    if (q === 0 && !saveZeros) return;
    if (q > 32767) q = 32767; else if (q < -32768) q = -32768;
    keys[i] = key | 0;
    qvals[i] = q;
    i++;
  });
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
  // A vpat<n> spec's value model is embedded so it travels with the policy it
  // was trained against — the sole source at inference.
  const vpatModel = meta.vpat || weights.vpatModel || null;
  const vpatLine = vpatModel ? `  const vpat = ${VPatterns.modelLiteral(vpatModel)};` : null;
  const returnLine = vpatModel
    ? "  return { spec, ema, totalUpdates, komi, ladderMinChain, t3MinChain, t3DepthLimit, t3NodeLimit, count, scale, keys, qvals, vpat };"
    : "  return { spec, ema, totalUpdates, komi, ladderMinChain, t3MinChain, t3DepthLimit, t3NodeLimit, count, scale, keys, qvals };";
  return [
    "'use strict';",
    '// Auto-generated by a featurepol trainer — do not edit by hand.',
    'const featurepolModel = (() => {',
    `  const count = ${count};`,
    `  const scale = ${scale};`,
    `  const spec = ${JSON.stringify(meta.spec || weights.spec.str)};`,
    `  const ema = ${+(meta.ema || 0).toFixed(6)};`,
    `  const totalUpdates = ${Math.round(meta.totalUpdates || 0)};`,
    `  const komi = ${meta.komi === undefined ? 'null' : +meta.komi};`,
    `  const ladderMinChain = ${meta.ladderMinChain || weights.ladderMinChain || 1};`,
    `  const t3MinChain = ${weights.t3MinChain || 1};`,
    `  const t3DepthLimit = ${weights.t3DepthLimit};`,
    `  const t3NodeLimit = ${weights.t3NodeLimit};`,
    `  const b64 = '${b64}';`,
    "  const bytes = typeof Buffer !== 'undefined'",
    "    ? Buffer.from(b64, 'base64')",
    "    : Uint8Array.from(atob(b64), c => c.charCodeAt(0));",
    "  const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + count * 6);",
    "  const keys  = new Int32Array(buf, 0, count);",
    "  const qvals = new Int16Array(buf, count * 4, count);",
    ...(vpatLine ? [vpatLine] : []),
    returnLine,
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
  const weights = createWeights({ spec: raw.spec, initialCapacity: Math.max(1024, mw.count | 0),
                                  ladderMinChain: raw.ladderMinChain || 1,      // files without the field: every chain
                                  t3MinChain: raw.t3MinChain || 1,
                                  t3DepthLimit: raw.t3DepthLimit, t3NodeLimit: raw.t3NodeLimit });
  mw.forEach((key, val) => { weights.vals[_intern(weights, key)] = val; });
  // Reconstruct the embedded vpat<n> value model (if any) onto the weights, so
  // the ranking reads it through ctx without any process-global or env var.  A
  // spec with vpat<n> but no embedded model errors at first ranking (in
  // _vpatPrepare), not silently.
  if (raw.vpat) weights.vpatModel = VPatterns.modelFromRaw(raw.vpat, undefined);
  return { weights, modelName, spec: weights.spec, ema: raw.ema || 0, totalUpdates: raw.totalUpdates || 0,
           komi: raw.komi === undefined ? null : raw.komi };
}

const FeaturePol = {
  parseSpec,
  createWeights,
  createState,
  internKey: _intern,
  extractFeatures,
  computeSoftmax,
  componentValues,
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
  _ranks: () => _rank,
  _setRankAllow: (set) => { _rankAllow = set; },
  setRankPositionRatio: (p) => { _rankPosRatio = p; },
  setStoneLimitTrainKeep: (p) => { _slTrainKeep = p; },
  getRankPositionRatio: () => _rankPosRatio,
};

if (typeof module !== 'undefined') module.exports = FeaturePol;
else window.FeaturePol = FeaturePol;

})();
