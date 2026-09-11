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
const { makeIntFloatMap } = _isNode ? require('./int-map.js') : window.IntMap;

// Weight tables are open-addressing int32→float64 maps (int-map.js): far
// cheaper get/set than a V8 Map at large sizes, same get/set/size surface,
// plus forEach(k, v) and clone().  Key 0 is the empty-slot sentinel; a
// canonical key of exactly 0 (a ~2^-31 hash coincidence) is silently
// dropped — the same class of accepted risk as any collision.
function makeWeights(minCap) {
  const m = makeIntFloatMap(minCap || 1024);
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
// each pairing opposite cells, giving D4 invariance.  Shared by the E
// (eye-pair) family and train-health.js.
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
  const b4 = l * 4;
  const n0 = nbr[b4], n1 = nbr[b4 + 1], n2 = nbr[b4 + 2], n3 = nbr[b4 + 3];      // N,S,W,E
  const m0 = dnbr[b4], m1 = dnbr[b4 + 1], m2 = dnbr[b4 + 2], m3 = dnbr[b4 + 3];  // NW,NE,SW,SE
  // Encoding order N, E, S, W, NE, SE, SW, NW, to match _D4 above.
  const c0 = _c3(cells, n0, owner) - 1, c1 = _c3(cells, n3, owner) - 1;
  const c2 = _c3(cells, n1, owner) - 1, c3 = _c3(cells, n2, owner) - 1;
  const c4 = _c3(cells, m1, owner) - 1, c5 = _c3(cells, m3, owner) - 1;
  const c6 = _c3(cells, m2, owner) - 1, c7 = _c3(cells, m0, owner) - 1;
  const raw = c0 + 3*(c1 + 3*(c2 + 3*(c3 + 3*(c4 + 3*(c5 + 3*(c6 + 3*c7))))));
  return _CANON[raw];
}
// Cell coders for the three alphabets: 3-state, 4-state (chain-relative),
// and the liberty-split alphabet.
function _c3(cells, i, owner) { const c = cells[i]; return c === 0 ? 1 : c === owner ? 2 : 3; }

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

const CHAIN_STONE_SALT = 0x71c3a5d9 | 0;
function chainStoneCountKey(nStones, cap) {
  const b = nStones > cap ? cap : nStones;
  const k = (Math.imul(b + 1, 0x85EBCA6B) ^ CHAIN_STONE_SALT) | 0;
  return k === 0 ? 1 : k;
}


const CHAIN_LIB_SALT = 0x2f1d3b77 | 0;
function chainLibCountKey(nLibs, cap) {
  const b = nLibs > cap ? cap : nLibs;
  const k = (Math.imul(b + 1, 0x9E3779B1) ^ CHAIN_LIB_SALT) | 0;
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
function chainSurvKeys(cells, nbr, dnbr, gid, owner, chainGid, libs, stones,
                       stoneSalt, out, subjectMaxLibs, joinCap, chainsByGid,
                       stoneNinecells, libertyNinecells, subjectMaxStones,
                       maxLibNinecells) {
  // maxLibNinecells caps the chain's LIBERTY COUNT for the singleton ninecells:
  // above it the chain emits none.  A chain with many liberties is not in
  // question, its liberty shapes are all saying the same thing, and the
  // liberty-count one-hot already says it — so the keys are spent where the
  // answer is in doubt.  0 means NO CAP (use --liberty-ninecells 0 to turn the
  // feature off); that differs from the other knobs on purpose, because 0-as-off
  // is already taken.
  const nl = libs.length;
  const wantSingles = libertyNinecells !== false &&
                      (maxLibNinecells <= 0 || nl <= maxLibNinecells);
  if (wantSingles) {
    for (let a = 0; a < nl; a++) {
      const h = _ncKey(ninecellId(cells, nbr, dnbr, libs[a], owner));
      out.push(h === 0 ? 1 : h);
    }
  }
  if (stoneNinecells !== false) {
    for (const st of stones) {
      const h = (_ncKey(ninecellId(cells, nbr, dnbr, st, owner)) ^ stoneSalt) | 0;
      out.push(h === 0 ? 1 : h);
    }
  }
  if (subjectMaxStones > 0) {
    let n = 0;
    for (const _ of stones) n++;
    out.push(chainStoneCountKey(n, subjectMaxStones));
  }
  if (subjectMaxLibs > 0) {
    let n = 0;
    for (const _ of libs) n++;
    out.push(chainLibCountKey(n, subjectMaxLibs));
  }
  if (joinCap > 0) {
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
let _byGid = [];
function chainsOf(cells, nbr, gid) {
  const area = cells.length;
  const chains = [];
  const byGid = _byGid;
  for (let i = 0; i < area; i++) {
    const c = cells[i];
    if (c === 0) continue;
    const g = gid[i];
    let r = byGid[g];
    if (r === undefined || r.stamp !== chains) {
      r = { gid: g, c, idx: chains.length, stones: [], libs: [], p: 0, live: false, stamp: chains };
      byGid[g] = r; chains.push(r);
    }
    r.stones.push(i);
  }
  for (let l = 0; l < area; l++) {
    if (cells[l] !== 0) continue;
    const b4 = l * 4;
    // At most four distinct chains touch an empty point, and once three are
    // held the fourth cannot repeat one of them, so three slots dedupe exactly.
    let s0 = -1, s1 = -1, s2 = -1;
    for (let d = 0; d < 4; d++) {
      const j = nbr[b4 + d];
      if (cells[j] === 0) continue;
      const gj = gid[j];
      if (gj === s0 || gj === s1 || gj === s2) continue;
      if (s0 < 0) s0 = gj; else if (s1 < 0) s1 = gj; else s2 = gj;
      byGid[gj].libs.push(l);
    }
  }
  return { chains, byGid };
}

// The friendly chains a chain could join with, as INDICES into `chains`,
// appended to `out`.  Two friendly chains are never in contact — that would
// make them one chain — so the relation runs through a SHARED LIBERTY, the
// same neighbour set the best-single-join one-hot considers.  Shared with
// train-health.js so the trained and the scored friend set cannot drift.
function friendsOfChain(cells, nbr, gid, byGid, r, out) {
  const start = out.length, owner = r.c, chainGid = r.gid, libs = r.libs;
  for (let a = 0; a < libs.length; a++) {
    const b4 = libs[a] * 4;
    for (let d = 0; d < 4; d++) {
      const j = nbr[b4 + d];
      if (cells[j] !== owner) continue;
      const gj = gid[j];
      if (gj === chainGid) continue;
      const k = byGid[gj].idx;
      let dup = false;
      for (let q = start; q < out.length; q++) if (out[q] === k) { dup = true; break; }
      if (!dup) out.push(k);
    }
  }
}

// The enemy chains in CONTACT with a chain, as indices into `chains`, appended
// to `out`.  Shared with train-health.js.
function foesOfChain(cells, nbr, gid, byGid, r, out) {
  const start = out.length, owner = r.c, stones = r.stones;
  for (let a = 0; a < stones.length; a++) {
    const b4 = stones[a] * 4;
    for (let d = 0; d < 4; d++) {
      const j = nbr[b4 + d];
      const c = cells[j];
      if (c === 0 || c === owner) continue;
      const k = byGid[gid[j]].idx;
      let dup = false;
      for (let q = start; q < out.length; q++) if (out[q] === k) { dup = true; break; }
      if (!dup) out.push(k);
    }
  }
}

// Scratch for one position's neighbour relations, reused across propagation
// passes (they are a function of the POSITION, not of the health estimates) and
// across positions.  Callers own their own instance so the trainer and the
// scorer cannot tread on each other.
function makeNeighbourhoods() {
  return { frFlat: [], frStart: [], frLen: [],
           foFlat: [], foStart: [], foLen: [] };
}
function neighbourhoodsOf(cells, nbr, gid, chains, byGid, wantFriends, wantFoes, nb) {
  nb.frFlat.length = 0; nb.frStart.length = 0; nb.frLen.length = 0;
  nb.foFlat.length = 0; nb.foStart.length = 0; nb.foLen.length = 0;
  for (let i = 0; i < chains.length; i++) {
    const r = chains[i];
    if (wantFriends) {
      const s = nb.frFlat.length;
      friendsOfChain(cells, nbr, gid, byGid, r, nb.frFlat);
      nb.frStart.push(s); nb.frLen.push(nb.frFlat.length - s);
    }
    if (wantFoes) {
      const s = nb.foFlat.length;
      foesOfChain(cells, nbr, gid, byGid, r, nb.foFlat);
      nb.foStart.push(s); nb.foLen.push(nb.foFlat.length - s);
    }
  }
  return nb;
}

// The neighbour-health keys for chain i under the current health estimates,
// written into `out` (capacity 2); returns how many.  `cfg` carries the bucket
// counts under the same field names a health model uses, so a scorer can pass
// the model itself.  A chain with no
// joinable friend, or none in contact with an enemy, emits nothing for that
// family — absence is distinguishable from any bucket by the weight not being
// there.  The trainer and the scorer both go through this, so what is trained
// and what is scored cannot drift.
function neighbourHealthKeys(nb, i, health, cfg, out) {
  const friendBuckets = cfg.friendHealthMaxBuckets;
  const foeMinBuckets = cfg.foeHealthMinBuckets;
  let n = 0;
  if (friendBuckets > 0) {
    const s = nb.frStart[i], len = nb.frLen[i];
    let best = -1;
    for (let a = 0; a < len; a++) { const v = health[nb.frFlat[s + a]]; if (v > best) best = v; }
    if (best >= 0) out[n++] = chainFriendHealthKey(best, friendBuckets);
  }
  if (foeMinBuckets > 0) {
    const s = nb.foStart[i], len = nb.foLen[i];
    let worst = 2;
    for (let a = 0; a < len; a++) { const v = health[nb.foFlat[s + a]]; if (v < worst) worst = v; }
    if (len > 0) out[n++] = chainFoeMinHealthKey(worst, foeMinBuckets);
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
// against ~180us for the scoring it guards.
const _eyeG = [], _eyeN = [];
function markLiveChains(cells, nbr, gid, dnbr, chains, byGid) {
  const area = cells.length;
  _eyeG.length = 0; _eyeN.length = 0;
  for (let p = 0; p < area; p++) {
    if (cells[p] !== 0) continue;
    const b4 = p * 4;
    let col = 0, n = 0, ok = true, firstGid = -2, sameGroup = 0;
    let a0 = -1, a1 = -1, a2 = -1, a3 = -1;
    for (let d = 0; d < 4; d++) {
      const j = nbr[b4 + d], c = cells[j];
      if (c === 0) { ok = false; break; }
      if (col === 0) col = c; else if (c !== col) { ok = false; break; }
      const q = gid[j];
      if (firstGid === -2) { firstGid = q; sameGroup = 1; } else if (q === firstGid) sameGroup++;
      if (q === a0 || q === a1 || q === a2 || q === a3) continue;
      // insertion sort into the four slots, so identical groups compare equal
      if (q < a0 || a0 < 0) { a3 = a2; a2 = a1; a1 = a0; a0 = q; }
      else if (q < a1 || a1 < 0) { a3 = a2; a2 = a1; a1 = q; }
      else if (q < a2 || a2 < 0) { a3 = a2; a2 = q; }
      else a3 = q;
      n++;
    }
    if (!ok) continue;
    // The opponent can never fill it, but the OWNER can unless the playout
    // policy declines to — so the point must also be an eye by THE rule
    // (game2.isTrueEye), which ppat-lib's move filter now shares.  A first version
    // of this check omitted the test entirely and 2.4% of the chains it marked
    // died in playouts.
    let enemyDiag = 0;
    for (let d = 0; d < 4; d++) if (cells[dnbr[b4 + d]] === -col) enemyDiag++;
    if (!isTrueEye(4, 0, sameGroup, enemyDiag)) continue;
    _eyeG.push(a0, a1, a2, a3); _eyeN.push(n);
  }
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
const _nb = makeNeighbourhoods(), _nbKeys = new Int32Array(2);
function chainHealthAll(model, cells, nbr, dnbr, gid, ls, chains, byGid) {
  const n = chains.length, w = model.weights;
  if (_baseZ.length < n) {
    _baseZ = new Float64Array(n * 2);
  }
  for (let i = 0; i < n; i++) {
    const r = chains[i];
    _survScratch.length = 0;
    chainSurvKeys(cells, nbr, dnbr, gid, r.c, r.gid, r.libs, r.stones,
                  model.stoneSalt, _survScratch,
                  model.maxLibs, model.maxJoinLibs, byGid, model.stoneNinecells,
                  model.libertyNinecells, model.maxStones,
                  model.maxLibNinecells);
    let z = model.bias;
    for (let j = 0; j < _survScratch.length; j++) z += w.get(_survScratch[j]) || 0;
    _baseZ[i] = z;
  }
  markLiveChains(cells, nbr, gid, dnbr, chains, byGid);
  const fhb = model.friendHealthMaxBuckets, fnb = model.foeHealthMinBuckets;
  if (fhb <= 0 && fnb <= 0) {
    for (let i = 0; i < n; i++) chains[i].p = chains[i].live ? 1 : 1 / (1 + Math.exp(-_baseZ[i]));
    return chains;
  }
  neighbourhoodsOf(cells, nbr, gid, chains, byGid, fhb > 0, fnb > 0, _nb);
  if (_live.length < n) { _live = new Uint8Array(n * 2); _keys = new Int32Array(n * 2 * NB_SLOTS); }
  for (let i = 0; i < n; i++) _live[i] = chains[i].live ? 1 : 0;
  propagateHealth(model, w, _baseZ, _live, n, _nb, _keys);
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
  return { bias: raw.bias, maxLibs: raw.maxLibs || 0,
           maxJoinLibs: raw.maxJoinLibs || 0, friendHealthMaxBuckets: raw.friendHealthMaxBuckets || 0,
           foeHealthMinBuckets: raw.foeHealthMinBuckets || 0, iterations: raw.iterations || 1,
           initHealth: raw.initHealth !== undefined ? raw.initHealth : 0.5,
           stoneNinecells: raw.stoneNinecells !== false,
           libertyNinecells: raw.libertyNinecells !== false,
           maxStones: raw.maxStones || 0,
           maxLibNinecells: raw.maxLibNinecells || 0,
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
function propagateHealth(model, weights, baseZ, live, n, nb, outKeys) {
  if (_pCur.length < n) { _pCur = new Float64Array(n * 2); _pNext = new Float64Array(n * 2); }
  for (let i = 0; i < n; i++) _pCur[i] = live[i] ? 1 : model.initHealth;
  for (let it = 1; ; it++) {
    for (let i = 0; i < n; i++) {
      const nk = neighbourHealthKeys(nb, i, _pCur, model, _propKeys);
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

// ── Exports ───────────────────────────────────────────────────────────────────

const HealthLib = {
  makeWeights,
  uh,
  xh4,
  ninecellId,
  chainLibCountKey,
  chainStoneCountKey,
  chainJoinLibsKey,
  chainFriendHealthKey,
  chainFoeMinHealthKey,
  chainSurvKeys,
  chainsOf,
  markLiveChains,
  friendsOfChain,
  foesOfChain,
  makeNeighbourhoods,
  neighbourhoodsOf,
  neighbourHealthKeys,
  propagateHealth,
  chainHealthAll,
  resolveHealthModel,
  NB_SLOTS,
};

if (typeof module !== 'undefined') module.exports = HealthLib;
else window.HealthLib = HealthLib;

})();
