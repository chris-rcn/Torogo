'use strict';

// health-lib.js — CHAIN HEALTH: P(this chain is still on the board when the
// playout ends).  Lifted out of vpatterns.js, which is where it grew up, so
// that the one implementation serves both consumers: vpatterns' health-coded
// specs score with it, and train-health.js fits it.  The
// propagation used to exist twice, once in each, which is why INIT_HEALTH and
// the live pin each had to be changed in two places.
//
// It also carries the NINECELL HASH and its x-hash primitives.  Those are not
// health-specific — vpatterns' pattern families use them too — but chainSurvKeys
// is built on them, and a chain-health file that imported them back from
// vpatterns would close a cycle.  So this is the lower layer and vpatterns
// imports them from here.
//
// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.
// Wrapped in an IIFE to avoid polluting the global namespace.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

const { isTrueEye } = _isNode ? require('./game2.js') : window.Game2;
const { makeIntFloat64Map } = _isNode ? require('./int-map.js') : window.IntMap;
const { game3FromGame2 } = _isNode ? require('./game3.js') : window.Game3;
const { getAllLadderStatuses } = _isNode ? require('./ladder2.js') : window.Ladder2;

// Per-group chain ladder state for the optional --ladder2 feature: a Map from a
// GAME3 group id to {1 alive, 2 dead, 3 unsettled}, from one getAllLadderStatuses
// pass.  Groups the read skips (>3 liberties) are absent and default to alive at
// the lookup.  The state derivation matches ladder2's own alive/dead/unsettled
// convention (unsettled when either side can flip it, else relative to whose
// move it is).  Keyed by group id, looked up per chain via its stone0 cell.
function ladderStateByGid(game3) {
  const cur = game3.current, m = new Map();
  const infos = getAllLadderStatuses(game3);
  for (const info of infos) {
    const st = info.status;
    if (!st) continue;
    let s;
    if (st.urgentLibs.length > 0 || st.moverSucceeds === null) s = 3;   // unsettled
    else if (st.moverSucceeds) s = info.color === cur ? 1 : 2;          // alive : dead
    else                       s = info.color === cur ? 2 : 1;
    m.set(info.gid, s);
  }
  return m;
}

// Weight tables are open-addressing int32→float64 maps (int-map.js): far
// cheaper get/set than a V8 Map at large sizes, same get/set/size surface,
// plus forEach(k, v) and clone().  Key 0 is the empty-slot sentinel; a
// canonical key of exactly 0 (a ~2^-31 hash coincidence) is silently
// dropped — the same class of accepted risk as any collision.
function makeWeights(minCap) {
  const m = makeIntFloat64Map(minCap || 1024);
  m.suppressZeroWarning();
  return m;
}

// ── X-hash (hpatterns' scheme) ────────────────────────────────────────────────
// Symmetry-invariant hierarchical "X" hash, replacing the min-over-16-
// transforms canonicalisation: uh(a,b) is an unordered combiner; a 2×2
// window hashes its two diagonals unordered, which is invariant to exactly
// D4; 3×3 recurses on its four corner 2×2 sub-windows in the same X
// arrangement, and 4×4 on its four corner 3×3 sub-windows (deliberately
// lossy above 2×2 — distinct shapes may share weights, the trade hpatterns
// measured at ~83% key fidelity for 3-state 3×3).
// Color canonicalisation compares the hash of the board against the hash of
// the color-inverted board: equal → color-twin (zero value by symmetry,
// dropped — all-empty and self-inverse-under-D4 patterns); else key = min,
// polarity says which coloring won.  Leaves enter as raw + maxLibs + 1 (≥ 1: uh has an
// absorbing element at -1, and 0 is unsafe as a map key downstream).
function uh(a, b) {
  return (1234567 + a + b + Math.imul(a, b)) | 0;
}
// Chain-relative hash of the NINECELL centred on l — the 3x3 region, of which
// only the eight surrounding cells are coded, the centre being the anchor.
// The same function therefore serves a stone (its own chain's surround) and an
// empty point such as a liberty.  Two xh4 combines, orthogonals and diagonals,
// each pairing opposite cells, giving D4 invariance.  Shared with
// train-health.js.
//
// Three-state alphabet by default (empty / owner's color / enemy).  Pass
// gid and chainGid for the FOUR-state alphabet, which splits the owner's
// color into "this chain" and "friendly but a different chain" — the
// distinction the strict eye rule needs, since four friendly stones around a
// point make an eye only when they are all the SAME chain.
//
// Pass ls (liberty count by gid) and maxLibs to split the OTHER-chain stone
// states by capped liberty count, giving 2 + 2*maxLibs states: at maxLibs 2
// that is six — empty / this chain / friendly-other in atari /
// friendly-other 2+ / enemy in atari / enemy 2+.  The subject chain is one
// state: splitting it too was measured worse, since its liberty count is
// already implicit in how many liberty ninecells the chain emits.
// Layout: 1 empty, then this-chain, friendly-other, enemy, each 1..maxLibs.
// Exact index + canonicalisation table, the scheme ppat-lib has always used
// (see its _buildTables): encode the eight cells in a fixed radix, then look the
// raw index up in a table that maps it to its D4 orbit, built once per alphabet.
//
// This REPLACES a symmetric-combine hash used from 2026-09-09 to 2026-09-11.
// That scheme paired opposite orthogonals and opposite diagonals through uh(),
// which is symmetric, so it discarded the order within each pair
// unconditionally — four order-discards on top of D4.  Measured against true D4
// orbits: 46.2% fidelity at 3 states, 34.6% at 4, and 20.6% at the 6 states
// every deployed health model used.  A table is both exact and faster.
//
// Positions in encoding order: N, E, S, W, NE, SE, SW, NW.  game2's nbr is
// N,S,W,E and dnbr is NW,NE,SW,SE, so the reads below are permuted to match.
const _D4 = (() => {
  // index order: 0 N, 1 E, 2 S, 3 W, 4 NE, 5 SE, 6 SW, 7 NW
  const ROT = [1, 2, 3, 0, 5, 6, 7, 4];   // 90 degrees: N->E, NE->SE, ...
  const REF = [0, 3, 2, 1, 7, 6, 5, 4];   // mirror: E<->W, NE<->NW, SE<->SW
  const ap = (p, q) => q.map(i => p[i]);
  const out = [];
  let cur = [0, 1, 2, 3, 4, 5, 6, 7];
  for (let r = 0; r < 4; r++) { out.push(cur.slice()); out.push(ap(cur, REF)); cur = ap(cur, ROT); }
  return out;
})();

// raw index -> D4 orbit id.  3^8 = 6561 entries canonicalising to 954 orbits,
// built once at load: one pass, eight transforms each.
const NC_STATES = 3, NC_RAW = 6561;
const _CANON = (() => {
  const S = NC_STATES, rawSize = NC_RAW;
  const canonId = new Int32Array(rawSize);
  const v = new Int32Array(8), tv = new Int32Array(8);
  const idMap = new Map();
  let next = 0;
  for (let raw = 0; raw < rawSize; raw++) {
    let r = raw;
    for (let i = 0; i < 8; i++) { v[i] = r % S; r = (r / S) | 0; }
    let minV = raw;
    for (let d = 0; d < 8; d++) {
      const p = _D4[d];
      for (let i = 0; i < 8; i++) tv[p[i]] = v[i];
      let enc = 0;
      for (let i = 7; i >= 0; i--) enc = enc * S + tv[i];
      if (enc < minV) minV = enc;
    }
    let id = idMap.get(minV);
    if (id === undefined) { id = next++; idMap.set(minV, id); }
    canonId[raw] = id;
  }
  return canonId;
})();


// Canonical ninecell id for the point l, coded relative to `owner`: 3 states per
// cell — empty, owner's colour, the other colour.  The chain-identity and
// liberty-split alphabets were dropped 2026-09-11: color matched them on loss
// with 2.6x fewer keys, and it is the one that keeps the canonicalisation table
// at 3^8 = 6561 raw entries (954 orbits) instead of 1.68M.
function ninecellId(cells, nbr, dnbr, l, owner) {
  const raw = ninecellRaw(cells, nbr, dnbr, l);
  return owner === 1 ? _CANON[raw] : _CANON_NEG[raw];
}

// The same eight cells coded in ABSOLUTE colours — empty 0, black 1, white 2 —
// so ONE encode serves both owners: black reads its orbit out of _CANON, white
// out of _CANON_NEG.  This is what lets the board scan encode a liberty shared
// by two chains once instead of once per chain, and it is exact rather than an
// approximation: for owner black the absolute trits already equal the relative
// ones, and for owner white they are the relative ones with 1 and 2 exchanged,
// which is precisely what _CANON_NEG undoes.
function ninecellRaw(cells, nbr, dnbr, l) {
  const b4 = l * 4;
  const n0 = nbr[b4], n1 = nbr[b4 + 1], n2 = nbr[b4 + 2], n3 = nbr[b4 + 3];      // N,S,W,E
  const m0 = dnbr[b4], m1 = dnbr[b4 + 1], m2 = dnbr[b4 + 2], m3 = dnbr[b4 + 3];  // NW,NE,SW,SE
  // Encoding order N, E, S, W, NE, SE, SW, NW, to match _D4 above.
  const c0 = _cA(cells, n0), c1 = _cA(cells, n3);
  const c2 = _cA(cells, n1), c3 = _cA(cells, n2);
  const c4 = _cA(cells, m1), c5 = _cA(cells, m3);
  const c6 = _cA(cells, m2), c7 = _cA(cells, m0);
  return c0 + 3*(c1 + 3*(c2 + 3*(c3 + 3*(c4 + 3*(c5 + 3*(c6 + 3*c7))))));
}
function _cA(cells, i) { const c = cells[i]; return c === 0 ? 0 : c === 1 ? 1 : 2; }

// _CANON read through the 1<->2 trit swap: the orbit of the same ninecell seen
// by a WHITE owner.  Built once at load, 6561 entries.
const _CANON_NEG = (() => {
  const out = new Int32Array(NC_RAW);
  for (let raw = 0; raw < NC_RAW; raw++) {
    let r = raw, sw = 0, pw = 1;
    for (let i = 0; i < 8; i++) {
      const t = r % 3; r = (r / 3) | 0;
      sw += (t === 1 ? 2 : t === 2 ? 1 : 0) * pw; pw *= 3;
    }
    out[raw] = _CANON[sw];
  }
  return out;
})();

function xh4(tl, tr, bl, br) {
  return uh(uh(tl, br), uh(tr, bl));
}

// Key for the chain-level LIBERTY-COUNT one-hot: exactly one weight per
// example, indexed by the chain's liberty count capped at subjectMaxLibs.  The
// ninecell sum can only move linearly with liberty count (one term per
// liberty), but survival is sharply non-linear in it — the step from one
// liberty to two is worth ~1.5 in logit, later steps ~0.5 — so this lets the
// model fit that curve directly.  Its own salt keeps it clear of the ninecell
// key space.
// Key for the BEST-SINGLE-JOIN liberty one-hot: the liberty count this chain
// would have after the most favourable connecting move available to it.
// Playing at a join point p merges this chain with EVERY friendly chain
// adjacent to p at once, so the result is
//     (my liberties U those friends' liberties U p's empty neighbours) \ {p}
// maximised over p.  Unlike every other chain aggregate, this is a
// COUNTERFACTUAL: no sum over current neighbourhoods can compute what the
// liberty count becomes after a move that has not been played, at any window
// size.  Chains with no join available fall back to their own liberty count.
const CHAIN_JOIN_SALT = 0x3b9aca07 | 0;
function chainJoinLibsKey(nLibs, cap) {
  const b = nLibs > cap ? cap : nLibs;
  const k = (Math.imul(b + 1, 0x45d9f3b3) ^ CHAIN_JOIN_SALT) | 0;
  return k === 0 ? 1 : k;
}

// Generation-stamped scratch for the union counts: no allocation, no clearing.
let _joinMark = null, _joinGen = 0;

// Key for the chain-level STONE-COUNT one-hot, the sibling of the liberty one.
// Chain SIZE is not derivable from the ninecell sum at any window: two chains
// with identical local shapes emit the same keys per point and differ only in
// HOW MANY they emit, which a sum cannot separate from a smaller chain whose
// points happen to score higher.  One weight per capped count, its own salt to
// stay clear of the liberty one-hot's key space.
// Dense canonical id -> weight-map key.  Its own salt keeps the ninecells clear
// of the one-hots' key space.
const NINECELL_SALT = 0x1b873593 | 0;
function _ncKey(id) {
  const k = (Math.imul(id + 1, 0x27220A95) ^ NINECELL_SALT) | 0;
  return k === 0 ? 1 : k;
}

// Liberty-PAIR key: a symmetric hash of two liberties' ninecell ids, in its own
// key space.  uh is commutative, so the key is order-invariant for the pair.
const CHAIN_PAIR_SALT = 0x3e9a1773 | 0;
function chainPairKey(ida, idb) {
  const k = (Math.imul(uh(ida, idb) + 1, 2654435761) ^ CHAIN_PAIR_SALT) | 0;
  return k === 0 ? 1 : k;
}
let _pairIds = new Int32Array(64);

const CHAIN_STONE_SALT = 0x71c3a5d9 | 0;
function chainStoneCountKey(nStones, cap) {
  const b = nStones > cap ? cap : nStones;
  const k = (Math.imul(b + 1, 0x85EBCA6B) ^ CHAIN_STONE_SALT) | 0;
  return k === 0 ? 1 : k;
}

// Key for the JOINT liberty-count x stone-count one-hot: one weight per
// (capped libs, capped stones) cell.  The model is additive over its one-hots,
// so separate liberty and stone features cannot express the interaction — and
// the interaction is the whole signal.  Two liberties on a 2-stone chain and
// two liberties on a 20-stone chain are different situations, and their
// difference is not a sum of "two liberties" and "20 stones".  Liberties per
// stone is the shape ratio: low means a blob, high means a string.
const CHAIN_LIBSTONE_SALT = 0x5a2f8b31 | 0;
function chainLibStoneKey(nLibs, nStones, libCap, stoneCap) {
  const l = nLibs > libCap ? libCap : nLibs;
  const s = nStones > stoneCap ? stoneCap : nStones;
  const k = (Math.imul(l * (stoneCap + 1) + s + 1, 0x9E3779B1) ^ CHAIN_LIBSTONE_SALT) | 0;
  return k === 0 ? 1 : k;
}


const CHAIN_LIB_SALT = 0x2f1d3b77 | 0;
function chainLibCountKey(nLibs, cap) {
  const b = nLibs > cap ? cap : nLibs;
  const k = (Math.imul(b + 1, 0x9E3779B1) ^ CHAIN_LIB_SALT) | 0;
  return k === 0 ? 1 : k;
}

const CHAIN_LADDER_SALT = 0x6b3f27d1 | 0;
// Chain ladder-status one-hot (the --ladder2 feature): state 1 alive / 2 dead /
// 3 unsettled.  Its own salt keeps it clear of the other one-hots.
function chainLadderKey(state) {
  const k = (Math.imul(state + 1, 0x2545F491) ^ CHAIN_LADDER_SALT) | 0;
  return k === 0 ? 1 : k;
}

// Health of the healthiest FRIENDLY chain the subject could join with (the
// friend relation runs through a SHARED LIBERTY — two friendly chains in
// contact would be one chain), bucketed into `buckets` uniform bins in p.
// Uniform in p, matching the H family's bucketing: the boundaries then do not
// depend on where the logit is clipped, and at 8+ buckets the healthy tail
// still gets bins of its own.  One weight per bucket, so the resolution is
// free — it is a one-hot over a scalar, not another dimension of the ninecell
// key space.  A chain with no joinable friend emits no key at all; absence is
// distinguishable from any bucket by the weight simply not being there.
const FRIEND_HEALTH_SALT = 0x6d5a19c3 | 0;
function chainFriendHealthKey(p, buckets) {
  let b = (p * buckets) | 0;
  if (b >= buckets) b = buckets - 1;
  const k = (Math.imul(b + 1, 0x27d4eb2f) ^ FRIEND_HEALTH_SALT) | 0;
  return k === 0 ? 1 : k;
}

// Health of the WEAKEST enemy chain in contact, bucketed in its own key space.
// What it says is whether there is something nearby worth attacking —
// liberties to take, eyespace to steal, tempo.  A minimum rather than an
// average over the enemies in contact: the two disagree exactly where it
// matters, e.g. a doomed enemy stone sitting inside the subject's own area
// gives min 0.033 against a contact-weighted mean of 0.838.  Both the mean and
// the maximum were implemented and measured on the full corpus, and neither
// moved the loss at any bucket count, so the enemy field only speaks through
// its weakest member.  Contact length does not weight an order statistic:
// every enemy in contact is a candidate however short the shared border.
const FOE_MIN_HEALTH_SALT = 0x4c72b915 | 0;
function chainFoeMinHealthKey(p, buckets) {
  let b = (p * buckets) | 0;
  if (b >= buckets) b = buckets - 1;
  const k = (Math.imul(b + 1, 0xc2b2ae35) ^ FOE_MIN_HEALTH_SALT) | 0;
  return k === 0 ? 1 : k;
}

// Ninecell keys for ONE chain, in the order train-health.js emits them: one
// per liberty, then one per stone (stone ninecells XORed into their own key
// space), then the chain-level one-hots: liberty count when subjectMaxLibs > 0,
// then best-single-join liberties when joinCap > 0, which additionally needs
// chainsByGid — this position's chainsOf() byGid map, so a friend's liberties
// are looked up rather than rescanned.  Shared by the trainer and by the C
// family's survival attribute so the two can never drift apart.  libs/stones
// may be any iterable of cells.
// stoneNinecells / libertyNinecells === false drop the per-STONE and
// per-LIBERTY ninecells respectively.  Together they are the bulk of the cost —
// a chain emits one hash per stone and one per liberty — so these are the main
// levers on this model's price, and they are separate because the two halves
// are partly redundant: dropping stones while keeping liberties was measured
// near-free, and the mirror is the open question.  Anything other than false
// keeps a half, so a model saved before these options existed behaves as it
// always did.  Dropping BOTH is allowed and is the bottom of the ablation
// ladder: the model is then the one-hots plus whatever neighbour propagation is
// configured, which is the baseline the ninecells have to beat.
// The ninecells are NOT here — scanBoard emits those, because they are a
// property of a point on the board rather than of a chain, and a point can
// belong to several chains.  What is left is the per-chain half: the count
// one-hots, and the best-single-join count, which is the only feature that has
// to read other chains' liberty lists and so cannot be settled during the scan.
//
// joinLibGate gates the join feature on the subject's own liberty count (0 =
// no gate), for the same reason the ninecell caps exist: its walk covers the
// chain's liberties and then every adjacent friend's liberty list, so it costs
// the most on chains whose survival was never in question.
function chainCountKeys(cells, nbr, gid, owner, chainGid, libs, nStones,
                        out, subjectMaxLibs, joinCap, chainsByGid,
                        subjectMaxStones, libStoneLibCap, libStoneStoneCap,
                        joinLibGate, ladderState) {
  if (ladderState > 0) out.push(chainLadderKey(ladderState));
  const nl = libs.length;
  if (subjectMaxStones > 0) {
    out.push(chainStoneCountKey(nStones, subjectMaxStones));
  }
  if (subjectMaxLibs > 0) {
    out.push(chainLibCountKey(nl, subjectMaxLibs));
  }
  if (libStoneLibCap > 0 && libStoneStoneCap > 0) {
    out.push(chainLibStoneKey(nl, nStones, libStoneLibCap, libStoneStoneCap));
  }
  if (joinCap > 0 && (joinLibGate <= 0 || nl <= joinLibGate)) {
    if (_joinMark === null || _joinMark.length < cells.length) _joinMark = new Int32Array(cells.length);
    const mark = _joinMark;
    // Stamp my own liberties once; each candidate join point then counts the
    // union incrementally against a fresh stamp.
    const gMine = ++_joinGen;
    let nMine = 0;
    for (const l of libs) { mark[l] = gMine; nMine++; }
    let best = nMine;
    for (const p of libs) {
      const b4p = p * 4;
      // friends adjacent to p — connecting here merges with all of them
      let hasFriend = false;
      const gU = ++_joinGen;
      let n = nMine - 1;                       // p itself stops being a liberty
      for (let d = 0; d < 4; d++) {
        const j = nbr[b4p + d];
        const c = cells[j];
        if (c === 0) {
          // an empty neighbour of p is a liberty of the connecting stone;
          // new only if it is not already one of mine
          if (mark[j] !== gMine && mark[j] !== gU) { mark[j] = gU; n++; }
          continue;
        }
        if (c !== owner) continue;
        const gj = gid[j];
        if (gj === chainGid) continue;
        hasFriend = true;
        // that friend's liberties join mine — looked up, never rescanned
        const fl = chainsByGid[gj].libs;
        for (let i = 0; i < fl.length; i++) {
          const q = fl[i];
          if (q === p) continue;
          if (mark[q] !== gMine && mark[q] !== gU) { mark[q] = gU; n++; }
        }
      }
      if (hasFriend && n > best) best = n;
    }
    out.push(chainJoinLibsKey(best, joinCap));
  }
  return out;
}

// Every chain on the board, as { chains, byGid }: chains[i] = { gid, c, idx,
// stones, libs, p } with libs DEDUPED, and byGid mapping group id to that
// record.  Every health consumer needs exactly this pass — the H family, the C
// family's health bucket, featurepol's adjHealth, the error tool and the
// trainer each used to open-code it — so it lives here once.
// byGid is a gid -> chain-record array, not a Map: game2 allocates gids densely
// from 0 with reuse (MAX_G = area + 4), so a plain array indexed by gid replaces
// a Map.get in the innermost loops — one per stone and up to four per empty
// point.  Measured 2026-09-11: chainsOf 6.5us -> 4.4us per evaluation.
//
// The array is MODULE-LEVEL and reused across calls, and it is never cleared.
// Freshness comes from stamping each record with this call's `chains` array:
// a record whose stamp is a different array is left over from a previous
// position and is rebuilt.  Clearing MAX_G slots per call would cost more than
// the Map did, and a stale entry can never be read, because the only gids read
// are ones this call has already written.
//
// Two consequences the old Map did not have:
//   - chain records now OUTLIVE the call that made them.  Every consumer today
//     uses them within one scoring pass; holding a record across a later
//     chainsOf on the same board size would see it rebuilt underneath.
//   - byGid is no longer a Map, so it has no .get.  Only health-lib dereferences
//     it (chainSurvKeys, friendsOfChain, foesOfChain, markLiveChains); every
//     other caller passes it through opaquely.
// The position's liberty points, filled by chainsOf and consumed by scanBoard.
// _lpPt[i] is the point and _lpEye[i] the enclosing colour when all four of its
// neighbours are stones of one colour (0 otherwise); the chains it is a liberty
// of are _lpFlat[_lpStart[i] ... +_lpCnt[i]].  CSR rather than a fixed four
// slots because the great majority of points have one or two chains, and the
// padding was read on every one of them.  Module-level and overwritten by the
// next chainsOf, exactly like _byGid.
let _byGid = [];
let _lpPt = new Int32Array(0), _lpEye = new Int32Array(0);
let _lpStart = new Int32Array(0), _lpCnt = new Int32Array(0), _lpFlat = new Int32Array(0);
let _lpN = 0, _lpFlatN = 0;
// collectStones (default true): with it false the per-chain stone LISTS are not
// built -- only nStones and stone0.  The lists' sole consumers are the stone
// ninecells and the foe relation, so a caller whose model uses neither skips
// ~80 pushes per position; scanBoard hard-fails if the model disagrees.
function chainsOf(cells, nbr, gid, collectStones) {
  const noStones = collectStones === false;
  _lpN = 0; _lpFlatN = 0;
  const areaLp = cells.length;
  if (_lpPt.length < areaLp) {
    // Typed, and holding chain INDICES rather than records: an array of object
    // references costs a GC write barrier on every store, and this one takes
    // ~190 stores per position.
    _lpPt = new Int32Array(areaLp); _lpEye = new Int32Array(areaLp);
    _lpStart = new Int32Array(areaLp); _lpCnt = new Int32Array(areaLp);
    _lpFlat = new Int32Array(areaLp * 4);
  }
  const area = cells.length;
  const chains = [];
  const byGid = _byGid;
  for (let i = 0; i < area; i++) {
    const c = cells[i];
    if (c === 0) continue;
    const g = gid[i];
    let r = byGid[g];
    if (r === undefined || r.stamp !== chains) {
      // A FRESH record and fresh arrays every position, which looks like churn
      // and measures as the opposite: reusing them in place cost 3.2us/position
      // here (5.5 -> 8.7), because a long-lived array takes a GC write barrier
      // on every store while a nursery one does not.  keys/friends/foes are
      // left null for scanBoard to fill in only when a feature needs them.
      r = { gid: g, c, idx: chains.length, stones: noStones ? null : [], libs: [],
            nStones: 0, stone0: i,
            keys: null, friends: null, foes: null, p: 0, live: false, stamp: chains,
            wantLib: false, wantStone: false, wantFriend: false, wantFoe: false };
      byGid[g] = r; chains.push(r);
    }
    r.nStones++;
    if (!noStones) r.stones.push(i);
  }
  for (let l = 0; l < area; l++) {
    if (cells[l] !== 0) continue;
    const b4 = l * 4;
    // At most four distinct chains touch an empty point, and once three are
    // held the fourth cannot repeat one of them, so three slots dedupe exactly.
    let s0 = -1, s1 = -1, s2 = -1;
    let occ = 0, col = 0, mono = true;
    const flatStart = _lpFlatN;
    for (let d = 0; d < 4; d++) {
      const j = nbr[b4 + d], cj = cells[j];
      if (cj === 0) { mono = false; continue; }
      occ++;
      if (col === 0) col = cj; else if (cj !== col) mono = false;
      const gj = gid[j];
      if (gj === s0 || gj === s1 || gj === s2) continue;
      if (s0 < 0) s0 = gj; else if (s1 < 0) s1 = gj; else s2 = gj;
      const rj = byGid[gj];
      rj.libs.push(l);
      _lpFlat[_lpFlatN++] = rj.idx;
    }
    if (_lpFlatN === flatStart) continue;
    // The LIBERTY POINTS of the position, each with the chains it belongs to,
    // so the board scan never has to rediscover them: it would otherwise repeat
    // this loop's neighbour reads and its dedup over the whole board.  _lpEye
    // carries the colour of a point enclosed by four stones of one colour (0
    // otherwise), which is the only kind of point the life proof can use.
    _lpPt[_lpN] = l;
    _lpStart[_lpN] = flatStart; _lpCnt[_lpN] = _lpFlatN - flatStart;
    _lpEye[_lpN] = (mono && occ === 4) ? col : 0;
    _lpN++;
  }
  return { chains, byGid };
}

// The friendly chains a chain could join with, as INDICES into `chains`,
// appended to `out`.  Two friendly chains are never in contact — that would
// make them one chain — so the relation runs through a SHARED LIBERTY, the
// same neighbour set the best-single-join one-hot considers.  Shared with
// train-health.js so the trained and the scored friend set cannot drift.
// The final weight key of the ninecell at `l` as seen by `owner`: salt 0 for a
// liberty, the model's stoneSalt for a stone.  Only the trainer's diagnostics
// call this — the scan computes its keys inline — but it has to compute them
// the same way, so it lives next to the code that does.
function ninecellKey(cells, nbr, dnbr, l, owner, salt) {
  const k = (_ncKey(ninecellId(cells, nbr, dnbr, l, owner)) ^ salt) | 0;
  return k === 0 ? 1 : k;
}

function _pushUniq(list, k) {
  for (let q = 0; q < list.length; q++) if (list[q] === k) return;
  list.push(k);
}
// Two chains sharing the empty point under inspection.  Same colour makes them
// joinable friends; different colours means nothing here, since enemy contact
// runs through adjacent STONES and is collected at the stone instead.
function _friendPair(a, b) {
  if (a === null || b === null || a.c !== b.c) return;
  if (a.wantFriend) _pushUniq(a.friends, b.idx);
  if (b.wantFriend) _pushUniq(b.friends, a.idx);
}

// ── The board scan ────────────────────────────────────────────────────────────
// ONE pass over the board, producing everything the per-chain walks used to
// produce separately: the ninecell key of every stone and of every liberty, the
// friendly chains each chain can join, the enemy chains it is in contact with,
// and the eye points that prove life.  It replaces five traversals — a liberty
// walk, a stone walk, a friend walk and a foe walk per chain, plus a pass over
// the empties for the eye proof — with one, and the reads at a point serve all
// of its consumers at once: the eight cells of the ninecell window are the same
// eight isTrueEye wants, and the four orthogonals that tell a stone whether it
// touches a liberty are the four that find its enemies.
//
// chainsOf still runs first.  It is the cheap pass — four reads per empty point,
// no window, no encode, no table lookup — and the liberty and stone counts it
// produces are exactly what the --max-lib-ninecells and --max-stone-ninecells
// gates test, so they must be known before this pass can decide what to emit.
// The gates are resolved once per chain here, into wantLib / wantStone, and a
// gated-out chain then costs this pass nothing.
//
// A liberty shared by several chains is encoded ONCE; each chain reads its own
// orbit from the table for its colour, and two chains of the SAME colour share
// the key and its weight lookup as well.  Under the old per-chain walk every
// one of them re-read the window and re-encoded it.
//
// `weights` non-null accumulates the logit into r.z (what the scorer wants);
// `keysOn` collects the keys into r.keys (what the trainer needs for its
// gradient).  Both may be on; neither may be, which is how a caller asks only
// for the relations.
const _eg4 = new Int32Array(4);
// Pooled per-chain key arrays for the keysOn (trainer) path, reused across
// positions by chain INDEX; a fresh array per chain per position measured as
// allocation churn in the trainer's tPos.
const _keyPool = [];
function scanBoard(model, cells, nbr, dnbr, gid, chains, byGid, weights, keysOn, zOut) {
  const nChains = chains.length;
  const stoneNC = model.stoneNinecells !== false;
  const libNC = model.libertyNinecells !== false;
  const mln = model.maxLibNinecells, msn = model.maxStoneNinecells;
  const fhb = model.friendHealthMaxBuckets, fnb = model.foeHealthMinBuckets;
  const fGate = model.friendLibGate, oGate = model.foeLibGate;
  const salt = model.stoneSalt;
  if ((stoneNC || fnb > 0) && nChains > 0 && chains[0].stones === null) {
    throw new Error('health-lib.scanBoard: the model wants stone ninecells or foe-health ' +
                    'but chainsOf was called with collectStones false');
  }
  for (let i = 0; i < nChains; i++) {
    const r = chains[i];
    const nl = r.libs.length;
    // The liberty cap gates BOTH ninecell halves: a chain with that many
    // liberties is not in question, so nothing about its shape is worth a key.
    const inQuestion = mln <= 0 || nl <= mln;
    r.wantLib = libNC && inQuestion;
    r.wantStone = stoneNC && inQuestion && (msn <= 0 || r.nStones <= msn);
    r.wantFriend = fhb > 0 && (fGate <= 0 || nl <= fGate);
    r.wantFoe = fnb > 0 && (oGate <= 0 || nl <= oGate);
    if (zOut !== null) zOut[i] = 0;
    if (keysOn) { let ka = _keyPool[i]; if (ka === undefined) ka = _keyPool[i] = []; ka.length = 0; r.keys = ka; }
    if (r.wantFriend) r.friends = [];
    if (r.wantFoe) r.foes = [];
  }
  // The stones, chain by chain.  A chain wanting neither its ninecells nor its
  // enemy contacts never has its stones walked at all.
  for (let i = 0; i < nChains; i++) {
    const r = chains[i];
    const wantStone = r.wantStone, wantFoe = r.wantFoe;
    if (!wantStone && !wantFoe) continue;
    const c = r.c, stones = r.stones;
    for (let a = 0; a < stones.length; a++) {
      const st = stones[a], b4 = st * 4;
      let touchesEmpty = false;
      for (let d = 0; d < 4; d++) {
        const j = nbr[b4 + d], cj = cells[j];
        if (cj === 0) { touchesEmpty = true; continue; }
        if (cj === c || !wantFoe) continue;
        _pushUniq(r.foes, byGid[gid[j]].idx);
      }
      // A stone with no empty orthogonal neighbour is interior: not where the
      // chain lives or dies, and its surround says nothing about the boundary.
      if (!touchesEmpty || !wantStone) continue;
      const raw = ninecellRaw(cells, nbr, dnbr, st);
      let k = (_ncKey(c === 1 ? _CANON[raw] : _CANON_NEG[raw]) ^ salt) | 0;
      if (k === 0) k = 1;
      if (keysOn) r.keys.push(k);
      if (weights !== null) zOut[r.idx] += weights.get(k) || 0;
    }
  }
  // The liberty points, each already carrying the chains it belongs to.
  _eyeG.length = 0; _eyeN.length = 0;
  for (let i = 0; i < _lpN; i++) {
    const s = _lpStart[i], n = _lpCnt[i];
    // The window is encoded on FIRST demand and reused by every chain here;
    // _ncKey never returns 0, so 0 is a safe "not computed yet" for each
    // colour's key.  Two friendly chains sharing this liberty therefore cost
    // one encode, one table lookup and one weight lookup between them.
    let raw = -1, kb = 0, kw = 0;
    for (let a = 0; a < n; a++) {
      const r = chains[_lpFlat[s + a]];
      if (!r.wantLib) continue;
      if (raw < 0) raw = ninecellRaw(cells, nbr, dnbr, _lpPt[i]);
      let k;
      if (r.c === 1) { if (kb === 0) kb = _ncKey(_CANON[raw]); k = kb; }
      else { if (kw === 0) kw = _ncKey(_CANON_NEG[raw]); k = kw; }
      if (keysOn) r.keys.push(k);
      if (weights !== null) zOut[r.idx] += weights.get(k) || 0;
    }
    if (n > 1) {
      for (let a = 0; a < n; a++) {
        for (let b = a + 1; b < n; b++) _friendPair(chains[_lpFlat[s + a]], chains[_lpFlat[s + b]]);
      }
    }
    const eyeCol = _lpEye[i];
    if (eyeCol !== 0) {
      const b4 = _lpPt[i] * 4;
      for (let d = 0; d < 4; d++) _eg4[d] = gid[nbr[b4 + d]];
      _eyeCandidate(cells, dnbr, b4, eyeCol);
    }
  }
  // Liberty-PAIR ninecells: for a chain with few enough liberties (life still in
  // question), a key per unordered pair of NON-ADJACENT liberties, combining
  // their two ninecell ids.  Orthogonally adjacent liberties are one eyespace,
  // not two eyes, so they are excluded; every other pair lights one key, letting
  // the model learn the two-eyes-suffice jump a sum of singles cannot.  Capped at
  // libertyPairs liberties to bound the O(libs^2) pairs.
  const pairGate = model.libertyPairs;
  if (pairGate > 0) {
    for (let i = 0; i < nChains; i++) {
      const r = chains[i];
      const libs = r.libs, nl = libs.length;
      if (nl < 2 || nl > pairGate) continue;
      if (_pairIds.length < nl) _pairIds = new Int32Array(nl * 2);
      for (let a = 0; a < nl; a++) _pairIds[a] = ninecellId(cells, nbr, dnbr, libs[a], r.c);
      for (let a = 0; a < nl; a++) {
        const b4 = libs[a] * 4, n0 = nbr[b4], n1 = nbr[b4 + 1], n2 = nbr[b4 + 2], n3 = nbr[b4 + 3];
        for (let b = a + 1; b < nl; b++) {
          const lb = libs[b];
          if (lb === n0 || lb === n1 || lb === n2 || lb === n3) continue;   // adjacent: one eyespace
          const k = chainPairKey(_pairIds[a], _pairIds[b]);
          if (keysOn) r.keys.push(k);
          if (weights !== null) zOut[r.idx] += weights.get(k) || 0;
        }
      }
    }
  }
  _markLiveFromEyes(byGid);
  return chains;
}

// The neighbour-health keys for chain i under the current health estimates,
// written into `out` (capacity 2); returns how many.  `cfg` carries the bucket
// counts under the same field names a health model uses, so a scorer can pass
// the model itself.  A chain with no
// joinable friend, or none in contact with an enemy, emits nothing for that
// family — absence is distinguishable from any bucket by the weight not being
// there.  The trainer and the scorer both go through this, so what is trained
// and what is scored cannot drift.
function neighbourHealthKeys(chains, i, health, cfg, out) {
  const friendBuckets = cfg.friendHealthMaxBuckets;
  const foeMinBuckets = cfg.foeHealthMinBuckets;
  const r = chains[i];
  let n = 0;
  if (friendBuckets > 0) {
    const fr = r.friends;
    let best = -1;
    if (fr !== null) for (let a = 0; a < fr.length; a++) { const v = health[fr[a]]; if (v > best) best = v; }
    if (best >= 0) out[n++] = chainFriendHealthKey(best, friendBuckets);
  }
  if (foeMinBuckets > 0) {
    const fo = r.foes;
    let worst = 2, nf = 0;
    if (fo !== null) { nf = fo.length;
      for (let a = 0; a < nf; a++) { const v = health[fo[a]]; if (v < worst) worst = v; } }
    if (nf > 0) out[n++] = chainFoeMinHealthKey(worst, foeMinBuckets);
  }
  return n;
}

// A chain flagged `live` is PROVEN uncapturable, so its health is pinned to 1
// rather than predicted — in the propagation too, where it feeds neighbours as
// a certainty.  The flag is only ever set, never cleared, so a caller may set
// it on a chainsOf() record before scoring and a new proof can be added here
// without touching chainHealthAll.
//
// The proof implemented is TWO EYES OF A GROUP.  An empty point is an eye of a
// group when all four of its orthogonal neighbours are stones of one color;
// the group is the set of chains those neighbours belong to (one chain when a
// single chain owns all four — the classic case — up to four).  Two eye points
// with the SAME group are a proof of life for every chain in it: the opponent
// playing on either has no liberty and captures nothing, because every chain in
// the group still touches the other eye, so the move is suicide.  Keying on the
// exact set is what makes that hold — it guarantees every member touches both
// points, which a merged-component version would not.
//
// The OWNER filling its own eye has to be excluded separately, and this is
// where a first version of the rule was wrong: 2.4% of the chains it marked
// died in playouts.  So the point must pass THE rule (game2.isTrueEye) verbatim,
// diagonals included — a multi-chain eye with a hostile diagonal is one the
// policy is still free to fill, merging the group and leaving it one eye, so it
// proves nothing.  Measured 2026-09-10: the
// model rated 5% of provably-alive chains below 0.9, one 19-stone group with
// two eyes at 0.504, because at 2 liberties the liberty one-hot dominates and
// nothing contradicts it.  One pass over the empty points, ~7us on 13x13
// against ~180us for the scoring it guards.  The collection now rides along
// with scanBoard, which is already at the point with all eight neighbours in
// hand; _eyeCandidate is the per-point half and _markLiveFromEyes the pairing.
const _eyeG = [], _eyeN = [];
// The point at b4/4 is empty with four occupied neighbours, all of colour
// `col` — scanBoard's caller has established that much.  _eg4 holds their gids.
function _eyeCandidate(cells, dnbr, b4, col) {
  const firstGid = _eg4[0];
  let sameGroup = 0, n = 0;
  let a0 = -1, a1 = -1, a2 = -1, a3 = -1;
  for (let d = 0; d < 4; d++) {
    const q = _eg4[d];
    if (q === firstGid) sameGroup++;
    if (q === a0 || q === a1 || q === a2 || q === a3) continue;
    // insertion sort into the four slots, so identical groups compare equal
    if (q < a0 || a0 < 0) { a3 = a2; a2 = a1; a1 = a0; a0 = q; }
    else if (q < a1 || a1 < 0) { a3 = a2; a2 = a1; a1 = q; }
    else if (q < a2 || a2 < 0) { a3 = a2; a2 = q; }
    else a3 = q;
    n++;
  }
  // The opponent can never fill it, but the OWNER can unless the playout
  // policy declines to — so the point must also be an eye by THE rule
  // (game2.isTrueEye), which ppat-lib's move filter now shares.  A first version
  // of this check omitted the test entirely and 2.4% of the chains it marked
  // died in playouts.
  let enemyDiag = 0;
  for (let d = 0; d < 4; d++) if (cells[dnbr[b4 + d]] === -col) enemyDiag++;
  if (!isTrueEye(4, 0, sameGroup, enemyDiag)) return;
  _eyeG.push(a0, a1, a2, a3); _eyeN.push(n);
}
function _markLiveFromEyes(byGid) {
  const m = _eyeN.length;
  for (let i = 0; i < m; i++) {
    const bi = i * 4;
    for (let j = i + 1; j < m; j++) {
      const bj = j * 4;
      if (_eyeN[j] !== _eyeN[i]) continue;
      if (_eyeG[bj] !== _eyeG[bi] || _eyeG[bj+1] !== _eyeG[bi+1] ||
          _eyeG[bj+2] !== _eyeG[bi+2] || _eyeG[bj+3] !== _eyeG[bi+3]) continue;
      for (let k = 0; k < 4; k++) {
        const q = _eyeG[bi + k];
        if (q < 0) break;
        byGid[q].live = true;
      }
      break;
    }
  }
}

// P(chain survives a standard playout) for EVERY chain, written to chains[i].p
// and returned.  A frozen train-health model gives sigma(bias + sum of the
// chain's ninecell and one-hot weights); with friendHealthMax on, one of those
// keys is a function of the NEIGHBOURS' health, which makes the POSITION — not
// the chain — the honest unit of work: every chain starts at health 0.5 and
// the model is applied model.iterations times, each pass reading the previous
// pass's healths.  Chains start at model.initHealth — the base rate of chain
// survival on the band the model was fitted on, not 0.5, so the first pass
// evaluates the neighbour terms at the population mean rather than telling
// every chain its neighbours are unusually weak.  That is exactly the forward
// iteration train-health.js trains through, so inference reproduces training.  The ninecell sum is
// computed once per chain and reused across passes; only the one neighbour
// weight moves.  The model carries its own alphabet parameters, so a model
// trained with different settings still scores correctly.
const _survScratch = [];
let _baseZ = new Float64Array(0), _live = new Uint8Array(0), _keys = new Int32Array(0);
// A ladder2 model's ladder read needs a Game3.  When a caller doesn't supply
// one, chainHealthAll builds it (game3FromGame2), but that is the slow path, so
// it is warned once per process — unless the caller passed game3RebuildOk to say
// it legitimately has no Game3 to reuse (a one-shot offline scorer).
let _warnedHealthGame3Rebuild = false;

// The health system's entry point: given a Game2 (and optionally a synced Game3,
// which it builds itself when a ladder2 model needs one and none is supplied),
// return every chain with its survival probability in `.p`.  All the position
// machinery it needs — the chain list, the ninecell/one-hot features, and the
// ladder read — is derived here; callers pass games, not arrays.  game3RebuildOk
// acknowledges building a Game3 here (no warning) for callers with none to pass.
function chainHealthAll(model, game2, game3, game3RebuildOk) {
  const cells = game2.cells, nbr = game2._nbr, dnbr = game2._dnbr, gid = game2._gid, ls = game2._ls;
  const { chains, byGid } = chainsOf(cells, nbr, gid);
  const n = chains.length, w = model.weights;
  if (_baseZ.length < n) {
    _baseZ = new Float64Array(n * 2);
  }
  // Ladder-status feature: one getAllLadderStatuses pass on a Game3, looked up
  // per chain via its stone0 cell's group (default alive for chains the read
  // skipped).  Off unless the model was trained with --ladder2.
  let ladderMap = null, g3gid = null;
  if (model.ladder2) {
    let g3 = game3;
    if (!g3) {
      if (!game3RebuildOk && !_warnedHealthGame3Rebuild) {
        _warnedHealthGame3Rebuild = true;
        console.error('health-lib.chainHealthAll: no game3 supplied for a ladder2 model — ' +
          'building one (slow path; pass a synced game3 to reuse it, or game3RebuildOk to ' +
          'acknowledge).  First occurrence:\n' +
          (new Error().stack || '').split('\n').slice(2, 7).join('\n'));
      }
      g3 = game3FromGame2(game2);
    }
    ladderMap = ladderStateByGid(g3);
    g3gid = g3._gid;
  }
  // The ninecell half of every chain's logit, plus the friend and foe relations
  // and the life proof, in one pass over the board.  The scorer takes the sum
  // rather than the keys, so nothing per-key is materialised.
  scanBoard(model, cells, nbr, dnbr, gid, chains, byGid, w, false, _baseZ);
  for (let i = 0; i < n; i++) {
    const r = chains[i];
    const ladderState = ladderMap ? (ladderMap.get(g3gid[r.stone0]) || 1) : 0;
    _survScratch.length = 0;
    chainCountKeys(cells, nbr, gid, r.c, r.gid, r.libs, r.nStones,
                   _survScratch, model.maxLibs, model.maxJoinLibs, byGid,
                   model.maxStones, model.libStoneLibs, model.libStoneStones,
                   model.joinLibGate, ladderState);
    let z = model.bias + _baseZ[i];
    for (let j = 0; j < _survScratch.length; j++) z += w.get(_survScratch[j]) || 0;
    _baseZ[i] = z;
  }
  const fhb = model.friendHealthMaxBuckets, fnb = model.foeHealthMinBuckets;
  if (fhb <= 0 && fnb <= 0) {
    for (let i = 0; i < n; i++) chains[i].p = chains[i].live ? 1 : 1 / (1 + Math.exp(-_baseZ[i]));
    return chains;
  }
  if (_live.length < n) { _live = new Uint8Array(n * 2); _keys = new Int32Array(n * 2 * NB_SLOTS); }
  for (let i = 0; i < n; i++) _live[i] = chains[i].live ? 1 : 0;
  propagateHealth(model, w, _baseZ, _live, n, chains, _keys);
  // The propagation leaves the final neighbour keys; the probability is one
  // more sigmoid over them, which is also exactly what the trainer scores.
  for (let i = 0; i < n; i++) {
    if (chains[i].live) { chains[i].p = 1; continue; }
    let z = _baseZ[i];
    const b = i * NB_SLOTS;
    for (let k = 0; k < NB_SLOTS; k++) { const key = _keys[b + k]; if (key !== 0) z += w.get(key) || 0; }
    chains[i].p = 1 / (1 + Math.exp(-z));
  }
  return chains;
}

// The saved model stores its weights in a plain Map; move them into an
// open-addressing int map, which is markedly cheaper for the ~900 int-keyed
// lookups this costs per position.
function _survIntern(raw) {
  if (!(raw.weights instanceof Map)) return raw;
  const w = makeWeights(raw.weights.size * 2);
  raw.weights.forEach((v, k) => w.set(k, v));
  return { bias: raw.bias, maxLibs: raw.maxLibs || 0, ladder2: !!raw.ladder2,
           maxJoinLibs: raw.maxJoinLibs || 0, friendHealthMaxBuckets: raw.friendHealthMaxBuckets || 0,
           foeHealthMinBuckets: raw.foeHealthMinBuckets || 0, iterations: raw.iterations || 1,
           initHealth: raw.initHealth !== undefined ? raw.initHealth : 0.5,
           stoneNinecells: raw.stoneNinecells !== false,
           libertyNinecells: raw.libertyNinecells !== false,
           maxStones: raw.maxStones || 0,
           maxLibNinecells: raw.maxLibNinecells || 0,
           maxStoneNinecells: raw.maxStoneNinecells || 0,
           libertyPairs: raw.libertyPairs || 0,
           joinLibGate: raw.joinLibGate || 0,
           friendLibGate: raw.friendLibGate || 0,
           foeLibGate: raw.foeLibGate || 0,
           libStoneLibs: raw.libStoneLibs || 0, libStoneStones: raw.libStoneStones || 0,
           stoneSalt: raw.stoneSalt, minPhase: raw.minPhase, maxPhase: raw.maxPhase,
           delta: raw.delta, weights: w };
}

// Resolve a health model from a path, a already-loaded model object, or the
// browser global.  The CALLER supplies it — a library reading the environment
// behind the caller's back cannot be given per-agent values, and a
// module-level singleton would force every agent in a process to share one.
// The result is attached to the prepared specs (prepareSpecs opts), so two
// agents can carry different health models.
const _healthCache = new Map();          // path -> interned model
function resolveHealthModel(pathOrModel) {
  if (pathOrModel && typeof pathOrModel === 'object') return _survIntern(pathOrModel);
  if (typeof pathOrModel === 'string' && pathOrModel !== '') {
    const abs = _isNode ? require('path').resolve(pathOrModel) : pathOrModel;
    let m = _healthCache.get(abs);
    if (!m) { m = _survIntern(require(abs)); _healthCache.set(abs, m); }
    return m;
  }
  if (typeof window !== 'undefined' && window.chainSurvModel) return _survIntern(window.chainSurvModel);
  throw new Error('health-lib: health-coded specs (size:H<N>) need a health model, and none ' +
                  'was supplied.  From the trainers, set ' +
                  'HEALTH_DATA to a train-health.js save file (e.g. HEALTH_DATA=out/health-xxxx.js ' +
                  'node train-vpatterns.js ...); featurepol reads FP_HEALTH_DATA.  In code, pass ' +
                  'prepareSpecs(specs, { health: <path or model> }) or loadWeights(file, health).');
}

// ── Neighbour-health propagation ─────────────────────────────────────────────
// The iterative core, shared by scoring and training so the two cannot drift.
//
// Health starts at 1 for chains PROVEN alive and at model.initHealth for the
// rest; each pass rescores every chain from the PREVIOUS pass's neighbour
// healths, with the weights fixed for the whole propagation.  The two callers
// want different things out of it — chainHealthAll wants the probability, the
// trainer wants the KEYS that produced it, since the example must carry them
// as fixed inputs so no gradient flows back through a neighbour's health — so
// this returns neither: it leaves the FINAL pass's keys in outKeys, NB_SLOTS
// per chain with 0 for an absent feature, and each caller finishes from there
// with one sigmoid.
//
// model supplies friendHealthMaxBuckets, foeHealthMinBuckets, iterations and
// initHealth; weights is anything with .get(key); live is indexable and
// truthy for a chain pinned alive.
const NB_SLOTS = 2;                     // friendHealthMax, foeHealthMin
const _propKeys = new Int32Array(NB_SLOTS);
let _pCur = new Float64Array(0), _pNext = new Float64Array(0);
function propagateHealth(model, weights, baseZ, live, n, chains, outKeys) {
  if (_pCur.length < n) { _pCur = new Float64Array(n * 2); _pNext = new Float64Array(n * 2); }
  for (let i = 0; i < n; i++) _pCur[i] = live[i] ? 1 : model.initHealth;
  for (let it = 1; ; it++) {
    for (let i = 0; i < n; i++) {
      const nk = neighbourHealthKeys(chains, i, _pCur, model, _propKeys);
      const b = i * NB_SLOTS;
      for (let k = 0; k < NB_SLOTS; k++) outKeys[b + k] = k < nk ? _propKeys[k] : 0;
    }
    if (it >= model.iterations) break;
    for (let i = 0; i < n; i++) {
      if (live[i]) { _pNext[i] = 1; continue; }
      let z = baseZ[i];
      const b = i * NB_SLOTS;
      for (let k = 0; k < NB_SLOTS; k++) { const key = outKeys[b + k]; if (key !== 0) z += weights.get(key) || 0; }
      _pNext[i] = 1 / (1 + Math.exp(-z));
    }
    for (let i = 0; i < n; i++) _pCur[i] = _pNext[i];
  }
}

// Serialize a runtime health model back to an object literal carrying exactly
// the fields _survIntern reads, so it can be EMBEDDED in another model file (a
// vpat model trained against it) and reconstructed with resolveHealthModel — no
// external HEALTH_DATA at load.  Zero-quantizing weights are skipped (they read
// back as 0), matching the vpat / featurepol serializers.
function modelLiteral(h) {
  const pairs = [];
  h.weights.forEach((k, v) => { const q = +v.toFixed(6); if (q !== 0) pairs.push(`[${k},${q}]`); });
  const scalars = [
    ['bias', h.bias], ['maxLibs', h.maxLibs], ['ladder2', !!h.ladder2],
    ['maxJoinLibs', h.maxJoinLibs], ['friendHealthMaxBuckets', h.friendHealthMaxBuckets],
    ['foeHealthMinBuckets', h.foeHealthMinBuckets], ['iterations', h.iterations],
    ['initHealth', h.initHealth], ['stoneNinecells', h.stoneNinecells],
    ['libertyNinecells', h.libertyNinecells], ['maxStones', h.maxStones],
    ['maxLibNinecells', h.maxLibNinecells], ['maxStoneNinecells', h.maxStoneNinecells],
    ['libertyPairs', h.libertyPairs],
    ['joinLibGate', h.joinLibGate], ['friendLibGate', h.friendLibGate],
    ['foeLibGate', h.foeLibGate], ['libStoneLibs', h.libStoneLibs],
    ['libStoneStones', h.libStoneStones], ['stoneSalt', h.stoneSalt],
    ['minPhase', h.minPhase], ['maxPhase', h.maxPhase], ['delta', h.delta],
  ];
  const fields = scalars.map(([k, v]) => `${k}: ${v}`).join(', ');
  return `{ ${fields}, weights: new Map([${pairs.join(',')}]) }`;
}

// ── Exports ───────────────────────────────────────────────────────────────────

const HealthLib = {
  makeWeights,
  modelLiteral,
  uh,
  xh4,
  ninecellId,
  ninecellRaw,
  ninecellKey,
  chainLibCountKey,
  chainStoneCountKey,
  chainLibStoneKey,
  chainJoinLibsKey,
  chainFriendHealthKey,
  chainFoeMinHealthKey,
  chainCountKeys,
  chainLadderKey,
  ladderStateByGid,
  scanBoard,
  chainsOf,
  neighbourHealthKeys,
  propagateHealth,
  chainHealthAll,
  resolveHealthModel,
  NB_SLOTS,
};

if (typeof module !== 'undefined') module.exports = HealthLib;
else window.HealthLib = HealthLib;

})();
