'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.
// Wrapped in an IIFE to avoid polluting the global namespace.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

const { BLACK, EMPTY, PASS, isTrueEye } = _isNode ? require('./game2.js') : window.game;
const { makeIntFloatMap } = _isNode ? require('./int-map.js') : window.IntMap;
const { game3FromGame2 } = _isNode ? require('./game3.js') : window.Game3;
const VLibPat = _isNode ? require('./vlibpat.js') : window.VLibPat;

// Chain health lives in health-lib.js — see the note at its head for why the
// ninecell hash and its x-hash primitives went with it.
const HL = _isNode ? require('./health-lib.js') : window.HealthLib;
const { makeWeights, uh, xh4, ninecellHash,
        chainsOf, chainHealthAll, resolveHealthModel } = HL;


// ── Constants ─────────────────────────────────────────────────────────────────

// Cell state encoding (signed, color-canonicalized):
//   0            = empty
//  +1 .. +maxLibs = BLACK stone with that many liberties (capped)
//  -1 .. -maxLibs = WHITE stone with that many liberties (capped)




// Unordered eye-pair key over two ninecellHash values (uh is symmetric).
// Shared by the E family and train-health.js.  The optional rel code
// types the pair by the two points' toroidal relationship (same-eyespace
// neighbours mean something different from separated liberties); rel 0 is
// the untyped key.
function pairKey(ha, hb, rel) {
  let k = (Math.imul(uh(ha, hb) + 1, 2654435761) ^ 0x3E9A17) | 0;
  if (rel) k = (k ^ Math.imul(rel, 0x27d4eb2d)) | 0;
  return k === 0 ? 1 : k;
}

// Fold the spec tag ((maxLibs << 3) | sizeCode) into a window hash so
// different spec spaces cannot collide in the shared weight map.
function mixTag(h, tag) {
  return uh(h, tag);
}
// Spec size → 3-bit tag code.  Sizes 1-4 are themselves; the rectangle pairs
// take the free codes — 34 (the 3×4 ∪ 4×3 pair) is 5, 23 (the 2×3 ∪ 3×2 pair)
// is 7, with 0 and 6 already spoken for by the C and E families.  maxLibs 0 is the
// LADDER-CODED family ('size:L' in the trainers): raw is vlibpat's 7-state
// turn-independent tactical alphabet (0 empty, ±1 alive, ±2 dead, ±3
// unsettled) instead of capped liberty counts — structurally identical to
// an ml=3 encoding, so the whole plane/hash/34 machinery is shared.
// Render a spec back to its command-line token — the inverse of the trainers'
// --spec parser, so what a run prints can be pasted into the next one.
function specToken(sp) {
  const ph = sp.phaseBins > 1 ? 'p' + sp.phaseBins : '';
  if (sp.size === 0) return 'C' + (sp.caps ? sp.caps.join('.') : '') + ph;
  if (sp.size === 5) return 't' + ph;
  if (sp.size === 6) return 'E' + (sp.libGate || 8) + ph;
  const body = sp.maxLibs === 0 ? 'L'
             : sp.maxLibs < 0 ? 'H' + (-sp.maxLibs)
             : String(sp.maxLibs);
  return sp.size + ':' + body + ph;
}
function specString(specs) { return specs.map(specToken).join(','); }

function sizeCode(size) { return size === 34 ? 5 : size === 23 ? 7 : size; }
// The turn family (spec size 5, token 't') has no alphabet and no window, so it
// cannot share a tag base with a pattern spec.  tagBaseOf never returns 16 —
// non-negative maxLibs give 0-15 and health families give 17+ — so tag base 16
// is permanently free, and the family takes code 0 within it.
const TURN_TAG = 16 << 3;
const TURN_SALT = 0x5ce7a13b | 0;
function specTag(spec) {
  if (spec.size === 5) return TURN_TAG;
  return (tagBaseOf(spec.maxLibs) << 3) | sizeCode(spec.size);
}

// Leaf mapping: leaf(raw) is chosen so that 1 + leaf is PRIME.  uh's core is
// (1+a)(1+b), so unordered leaf pairs at the 2×2 level collide exactly when
// products coincide (3·8 = 4·6 ...): with the naive raw+ml+1 mapping the 2×2
// layer lost 12-35% of distinct orbits at maxLibs >= 2.  Prime leaves make
// pair products unique by factorisation — measured 2×2 fidelity 100% at every
// alphabet, and 3×3 fidelity 87.7% -> 94.7% (ml=2), 91.4% -> 97.2% (ml=3).
// Index: raw + maxLibs ∈ [0, 2·maxLibs]; supports maxLibs <= 15.
const _PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
                 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127];
const _leafTab = _PRIMES.map(p => p - 1);   // leaf = prime - 1, so 1+leaf is prime

// ── Core encoding ─────────────────────────────────────────────────────────────

// Returns the raw state of the cell at idx: 0 for empty, 1-maxLibs for BLACK
// with that many liberties (capped), maxLibs+1 .. 2*maxLibs for WHITE.
function rawState(game, maxLibs, idx) {
  const color = game.cells[idx];
  if (maxLibs === 1 || color === 0) return color;
  const libs = Math.min(game.groupLibertyCount(game.groupIdAt(idx)), maxLibs);
  return color * libs;
}

// Per-stone group liberty counts computed from `cells` alone (flood fill over
// game._nbr).  extractFeatures MUST use this instead of the game's _gid/_ls:
// the speculative doSetNext path mutates cells only, so the incremental group
// structures are stale there — under them the speculative stone read as gid
// -1 → NaN → 0 (invisible) and neighbour liberties were pre-move, which
// corrupted every maxLibs>=2 candidate evaluation (found 2026-08-22).
function _cellLibCounts(game, out) {
  const cells = game.cells, nbr = game._nbr, cap = game.N * game.N;
  const stack = _libStack.length >= cap ? _libStack : (_libStack = new Int32Array(cap));
  const seen  = _libSeen.length  >= cap ? _libSeen  : (_libSeen  = new Int32Array(cap));
  const mark  = _libMark.length  >= cap ? _libMark  : (_libMark  = new Int32Array(cap));
  const group = _libGroup.length >= cap ? _libGroup : (_libGroup = new Int32Array(cap));
  // seen/mark are Int32Arrays: past 2^31 the stored stamp wraps while the JS
  // counter doesn't, so equality never holds again and the flood loops
  // forever (hit live 2026-08-22 after ~35min of training).  Reset with
  // headroom for this call's per-group increments (<= cap + 1 << 65535).
  if (_libStampVal > 0x7fff0000) {
    _libSeen.fill(0); _libMark.fill(0);
    _libStampVal = 0;
  }
  const seenStamp = ++_libStampVal;   // per call: which stones are flooded
  for (let i = 0; i < cap; i++) {
    if (cells[i] === 0) { out[i] = 0; continue; }
    if (seen[i] === seenStamp) continue;
    // Flood this group from i: collect members, count distinct empty nbrs.
    // mark[] dedupes liberties with a per-GROUP stamp, so an empty cell
    // shared by two groups counts for both.
    const markStamp = ++_libStampVal;
    const color = cells[i];
    let top = 0, size = 0, libs = 0;
    stack[top++] = i; seen[i] = seenStamp;
    while (top > 0) {
      const c = stack[--top];
      group[size++] = c;
      const b = c * 4;
      for (let k = 0; k < 4; k++) {
        const nb = nbr[b + k];
        if (cells[nb] === 0) {
          if (mark[nb] !== markStamp) { mark[nb] = markStamp; libs++; }
        } else if (cells[nb] === color && seen[nb] !== seenStamp) {
          seen[nb] = seenStamp; stack[top++] = nb;
        }
      }
    }
    for (let s = 0; s < size; s++) out[group[s]] = libs;
  }
}
let _libStack = new Int32Array(0), _libSeen = new Int32Array(0),
    _libMark = new Int32Array(0), _libGroup = new Int32Array(0);
// deltaZ scratch (module-level, reused; stamped arrays reset on wrap).
let _dzStamp = 0;
let _dzMark2 = new Int32Array(0), _dzMark3 = new Int32Array(0), _dzMark4 = new Int32Array(0),
    _dzMarkC = new Int32Array(0);
let _dzOvN = new Int32Array(0), _dzOvI = new Int32Array(0);       // 2×2 hash overrides
let _dzOv3N = new Int32Array(0), _dzOv3I = new Int32Array(0);     // 3×3 hash overrides
let _dzLeafN = new Int32Array(0), _dzLeafI = new Int32Array(0);   // leaf overrides
let _dzUnion = new Int32Array(0);
let _libCounts = new Int32Array(0);
// Health-coded specs ('size:H<N>') are stored with maxLibs = -N so they group
// and sort apart from liberty-coded ones; their tag base is lifted clear of the
// liberty range so the two never share a key space.
function tagBaseOf(maxLibs) { return maxLibs < 0 ? 16 - maxLibs : maxLibs; }
const _phSaltScratch = new Int32Array(256);
let _libStampVal = 0;

// ── Multi-spec extraction ─────────────────────────────────────────────────────

// Given an array of specs [{ size: 1|2|3, maxLibs: N }, ...], scans every cell
// prepareSpecs: convert a specs array into the internal structure used by
// extractFeatures.  Call once per unique specs array and reuse the result.
// Also precomputes lookup tables for size:2 and size:3:
//   lut2/lut3: Map<maxLibs, { keys: Int32Array, pols: Int8Array, base, b2, b3[, ...], ml }>
//   Index = Σ (cell[i]+maxLibs) * base^i.  pols[i]===0 → skip (symmetric/empty).
function prepareSpecs(specs, opts) {
  const byMaxLibs = new Map();
  // size 0 = the chain-attribute family (spec token 'C', optionally
  // 'C<stones>.<libs>.<adjE>.<sec>[.<joinable>[.<weakestAdj>[.<eyes>
  // [.<sharedLibs>[.<bestFriendLibs>]]]]]'
  // caps, default 8.8.4.8; a fifth cap enables joinable-friendly-chains, a
  // sixth weakest-adjacent-enemy (min liberties over adjacent enemy chains,
  // 0 = none — the race signal), a seventh true-eye liberties (the life
  // signal; game2's eye rule replicated for the CHAIN's colour)): per-chain
  // keyed features, not windowed — kept out of the window machinery
  // entirely.  One C spec per model (the first wins).
  const chainSpec = specs.find(sp => sp.size === 0);
  const hasChains = chainSpec !== undefined;
  // size 6 = the eye-pair family (token 'E[<libGate>][pN]'): for each chain
  // with at most libGate liberties, one feature per unordered pair of
  // NON-ADJACENT liberties, keyed by the pair of owner-relative t-hashed
  // 3x3 neighbourhoods.  Two-eye safety is a conjunction of two separated
  // local shapes; no window family can see conjunctions.  Non-incremental.
  const eyeSpec = specs.find(sp => sp.size === 6);
  const hasEyePairs = eyeSpec !== undefined;
  const eyePairGate = hasEyePairs ? (eyeSpec.libGate || 8) : 0;
  const eyePairPhaseBins = hasEyePairs ? (eyeSpec.phaseBins || 1) : 1;
  const chainCaps = hasChains ? (chainSpec.caps || [8, 8, 4, 8]) : null;
  // Optional phase bucketing (token suffix pN): chain keys additionally
  // keyed by floor(phase * N) over [0, 1] — every chain in a position
  // shares the bucket.
  const chainPhaseBins = hasChains ? (chainSpec.phaseBins || 1) : 1;
  // Per-PATTERN-spec phase bins (token suffix pN on size:maxLibs): emitted
  // keys are salted by floor(phase * N), giving each spec its own phase-
  // conditioned weight planes.  Non-incremental (bucket crossings invalidate
  // every key): deltaZ and doSetNext refuse.
  const patPhaseBins = new Int32Array(256);
  let hasPhasedPatterns = false;
  for (const sp of specs) {
    if (sp.size !== 0 && sp.size !== 5 && sp.phaseBins > 1) {
      patPhaseBins[(tagBaseOf(sp.maxLibs) << 3) | sizeCode(sp.size)] = sp.phaseBins;
      hasPhasedPatterns = true;
    }
  }
  // The TURN feature (size 5): one antisymmetric feature per position, +1 when
  // BLACK is to move, keyed by phase bucket.  z has no tempo term otherwise,
  // and the value of holding the move plainly varies with fullness.  It lives
  // at its own tag base, so it registers its bins directly.
  const turnSpec = specs.find(sp => sp.size === 5);
  const hasTurn = turnSpec !== undefined;
  const turnPhaseBins = hasTurn ? (turnSpec.phaseBins || 1) : 1;
  if (turnPhaseBins > 1) { patPhaseBins[TURN_TAG] = turnPhaseBins; hasPhasedPatterns = true; }
  for (const spec of specs) {
    if (spec.size === 0 || spec.size === 5 || spec.size === 6) continue;
    if (!byMaxLibs.has(spec.maxLibs)) byMaxLibs.set(spec.maxLibs, []);
    byMaxLibs.get(spec.maxLibs).push(spec.size);
  }
  const sortedMaxLibs = [...byMaxLibs.keys()].sort((a, b) => b - a);



  let totalSizes = 0;   // feature slots per cell (a rectangle pair emits 2: one per orientation)
  for (const sizes of byMaxLibs.values())
    for (const s of sizes) totalSizes += (s === 34 || s === 23) ? 2 : 1;

  // maxLibs 0 = the ladder-coded family (needs a game3 tactical pass; not
  // incremental — deltaZ and speculative extraction refuse it).
  const hasLadder = byMaxLibs.has(0);
  // Negative group keys are the HEALTH-coded families (maxLibs = -N): each
  // stone is coded by its chain's frozen-model survival bucket instead of its
  // liberty count.  Needs the game's _gid/_ls, so like the ladder family it is
  // not incremental and refuses speculative extraction.
  const hasHealth = sortedMaxLibs.some(m => m < 0);

  // Health model, resolved once per prepared-specs object: needed by the
  // 'size:H<N>' families and by the C survival attribute.
  const needHealth = sortedMaxLibs.some(m => m < 0) ||
                     (hasChains && chainCaps.length > 12 && chainCaps[12] >= 2);
  const healthModel = needHealth ? resolveHealthModel(opts && opts.health) : null;
  return { byMaxLibs, sortedMaxLibs, healthModel,
           totalSizes: totalSizes + (hasChains ? 1 : 0) + (hasEyePairs ? 4 : 0) + (hasTurn ? 1 : 0),
           hasLadder, hasHealth, hasChains, chainCaps, chainPhaseBins, patPhaseBins, hasPhasedPatterns,
           hasEyePairs, eyePairGate, eyePairPhaseBins, hasTurn };
}

// and returns a flat array of { key, polarity } for all matching patterns.
//
// Optimisations vs calling pattern1/2/3 individually:
//   - Raw cell states are precomputed once per unique maxLibs value.
//   - size:2 and size:3 hash via whole-board X-hash planes (see above).
//   - pattern1 is inlined (raw[idx] already holds the capped liberty count).
function extractFeatures(game, prepSpecs, doSetNext, nextMove) {
  const cells = game.cells;
  const cap   = game.N * game.N;
  const N     = game.N;

  // Pre-allocate flat typed output arrays; max one feature per cell per size entry.
  const maxF   = cap * prepSpecs.totalSizes;
  const outKeys = new Int32Array(maxF);
  const outPols = new Int8Array(maxF);
  const outTags = new Int16Array(maxF);  // spec tag: (tagBase << 3) | size
  let   count   = 0;

  if (nextMove === PASS) doSetNext = false;
  // Ladder codes classify via a game3 rebuild of the CURRENT cells; under a
  // speculative mutation the mover/turn bookkeeping is wrong, so refuse
  // before touching the board.
  if (doSetNext && prepSpecs.hasLadder) {
    throw new Error('vpatterns: ladder-coded specs (size:L) do not support speculative extraction (doSetNext)');
  }
  if (doSetNext && prepSpecs.hasPhasedPatterns) {
    throw new Error('vpatterns: phase-binned pattern specs (pN) do not support speculative extraction (doSetNext)');
  }
  if (doSetNext && prepSpecs.hasHealth) {
    throw new Error('vpatterns: health-coded specs (size:H<N>) do not support speculative extraction (doSetNext)');
  }
  if (doSetNext && prepSpecs.hasEyePairs) {
    throw new Error('vpatterns: eye-pair specs (E) do not support speculative extraction (doSetNext)');
  }
  // A speculative mutation is the position AFTER the move, where the side to
  // move is the opponent — game.current still says otherwise.
  if (doSetNext && prepSpecs.hasTurn) {
    throw new Error('vpatterns: the turn spec (t) does not support speculative extraction (doSetNext)');
  }
  if (doSetNext && prepSpecs.hasChains) {
    throw new Error('vpatterns: chain-attribute specs (C) do not support speculative extraction (doSetNext) — the group structures are stale under the mutation');
  }
  let captures;
  if (doSetNext) {
    captures = game.captureList(nextMove);
    for (let c = 0; c < captures.length; c++) {
      cells[captures[c]] = EMPTY;
    }
    cells[nextMove] = game.current;
  }

  const { byMaxLibs, sortedMaxLibs } = prepSpecs;

  // X-hash planes: stored per prepSpecs per maxLibs so deltaZ can read the
  // base position after extraction (the hpatterns buffer contract: the caller
  // must have run extractFeatures on the current position, and nothing may
  // overwrite the planes between that call and deltaZ).
  const planes = prepSpecs._planes || (prepSpecs._planes = new Map());
  // Chain survival is identical across health groups — only the bucketing
  // differs — so compute it once per POSITION and let every 'size:H<N>' group
  // reuse it.  Rebuilt on each extractFeatures call; never cached across
  // positions.
  let survChains = null;

  const phSalt = _phSaltScratch;
  if (prepSpecs.hasPhasedPatterns) {
    const bins = prepSpecs.patPhaseBins, ph = 1 - game.emptyCount / cap;
    // over the WHOLE tag space: health families sit at lifted tag bases, and
    // a short loop would silently drop their phase salt
    for (let t = 0; t < bins.length; t++) {
      if (bins[t] > 1) {
        let b = Math.floor(ph * bins[t]);
        if (b >= bins[t]) b = bins[t] - 1;
        phSalt[t] = Math.imul((b + 1) ^ Math.imul(t + 1, 131), 0x9E3779B1) | 0;
      } else phSalt[t] = 0;
    }
  } else phSalt.fill(0);

  let raw = null;
  for (const maxLibs of sortedMaxLibs) {
    const isLadder = maxLibs === 0;   // sorts last (descending), after the liberty chain
    const isHealth = maxLibs < 0;     // 'size:H<N>' — stored as -N, sorts after the ladder
    const hb = isHealth ? -maxLibs : 0;
    if (isLadder) {
      // vlibpat's turn-independent 7-state tactical alphabet, from a fresh
      // game3 tactical pass over the current cells.
      raw = VLibPat.computeLadderCodes(game3FromGame2(game), null);
    } else if (isHealth) {
      // One survival probability per chain, bucketed to 1..hb and signed by
      // colour — the same alphabet shape as liberty counts, but the levels
      // mean "how likely is this chain to live" instead of "how many
      // liberties".  Uses the game's incremental chain structures, so this
      // family cannot run under doSetNext.
      raw = new Int8Array(cap);
      if (survChains === null) {
        const gidH = game._gid, lsH = game._ls, nbrH = game._nbr, dnbrH = game._dnbr;
        const cs = chainsOf(cells, nbrH, gidH);
        survChains = cs.chains;
        chainHealthAll(prepSpecs.healthModel, cells, nbrH, dnbrH, gidH, lsH, cs.chains, cs.byGid);
      }
      for (let k = 0; k < survChains.length; k++) {
        const r = survChains[k];
        let b = Math.floor(r.p * hb) + 1;
        if (b > hb) b = hb;
        const v = r.c * b;
        for (let j = 0; j < r.stones.length; j++) raw[r.stones[j]] = v;
      }
    } else if (raw === null) {
      raw = new Int8Array(cap);
      if (maxLibs === 1) {
        for (let i = 0; i < cap; i++) raw[i] = cells[i];
      } else {
        // Liberty counts from cells alone (NOT the game's _gid/_ls): under
        // doSetNext the cells are speculatively mutated and the incremental
        // structures are stale — see _cellLibCounts.
        const libs = _libCounts.length >= cap ? _libCounts : (_libCounts = new Int32Array(cap));
        _cellLibCounts(game, libs);
        for (let i = 0; i < cap; i++) {
          const c = cells[i];
          raw[i] = c === 0 ? 0 : (libs[i] < maxLibs ? c * libs[i] : c * maxLibs);
        }
      }
    } else {
      // Clamp in-place: rawState(maxLibs) = sign * min(|rawState(prevMaxLibs)|, maxLibs).
      for (let i = 0; i < cap; i++) {
        if      (raw[i] >  maxLibs) raw[i] =  maxLibs;
        else if (raw[i] < -maxLibs) raw[i] = -maxLibs;
      }
    }
    // Ladder family: leaf offset 3 (codes span ±3); size-1 keys in their own
    // block (131*16 — a real maxLibs cannot exceed 15).  Health families use
    // their bucket count as the offset, and a lifted tag base.
    const effOff = isLadder ? 3 : (isHealth ? hb : maxLibs);
    const tagBase = tagBaseOf(maxLibs);
    const sizes = byMaxLibs.get(maxLibs);
    const do1   = sizes.includes(1);
    const do2   = sizes.includes(2);
    const do3   = sizes.includes(3);
    const do4   = sizes.includes(4);
    const do34  = sizes.includes(34);   // 3×4 ∪ 4×3 rectangle pair
    const do23  = sizes.includes(23);   // 2×3 ∪ 3×2 rectangle pair

    if (do1) {
      const k1base = 131 * (isLadder ? 16 : tagBase);
      for (let idx = 0; idx < cap; idx++) {
        const s = raw[idx];
        if (s !== 0) {
          const libs = s > 0 ? s : -s;
          outKeys[count] = (libs + k1base) ^ phSalt[(tagBase << 3) | 1];
          outPols[count] = s > 0 ? 1 : -1;
          outTags[count] = (tagBase << 3) | 1;
          count++;
        }
      }
    }

    if (do2 || do3 || do4 || do34 || do23) {
      let pl = planes.get(maxLibs);
      if (!pl || pl.lN.length < cap) {
        pl = { lN: new Int32Array(cap), lI: new Int32Array(cap),
               h2N: new Int32Array(cap), h2I: new Int32Array(cap),
               h3N: new Int32Array(cap), h3I: new Int32Array(cap) };
        planes.set(maxLibs, pl);
      }
      const lN = pl.lN, lI = pl.lI, h2N = pl.h2N, h2I = pl.h2I;
      // Leaf planes: the prime mapping over raw (normal) and -raw (colour-
      // inverted).  Then the 2×2 X-hash plane over both colourings.
      const off = effOff;
      for (let i = 0; i < cap; i++) { lN[i] = _leafTab[raw[i] + off]; lI[i] = _leafTab[off - raw[i]]; }
      for (let y = 0; y < N; y++) {
        const r0 = y * N, r1 = (y + 1 < N ? y + 1 : 0) * N;
        for (let x = 0; x < N; x++) {
          const x1 = x + 1 < N ? x + 1 : 0;
          const i = r0 + x;
          h2N[i] = xh4(lN[r0+x], lN[r0+x1], lN[r1+x], lN[r1+x1]);
          h2I[i] = xh4(lI[r0+x], lI[r0+x1], lI[r1+x], lI[r1+x1]);
        }
      }
      if (do2) {
        const tag = (tagBase << 3) | 2;
        const pS2 = phSalt[tag];
        for (let i = 0; i < cap; i++) {
          const kN = h2N[i], kI = h2I[i];
          if (kN === kI) continue;   // colour-twin (incl. all-empty): zero value
          outKeys[count] = mixTag(kN < kI ? kN : kI, tag) ^ pS2;
          outPols[count] = kN < kI ? 1 : -1;
          outTags[count] = tag;
          count++;
        }
      }
      if (do23) {
        // 2×3 ∪ 3×2 = uh of two adjacent 2×2 sub-hashes (sharing a 2×1 core),
        // the same construction size 34 uses one level up.  BOTH orientations
        // under ONE tag, mandatorily: a 90° board rotation carries each
        // horizontal window onto a vertical one with an equal hash (each 2×2
        // child is D4-invariant, uh unordered), so only the union of the two
        // families keeps the feature multiset D4-invariant.
        const tag23 = (tagBase << 3) | 7;
        const pS23 = phSalt[tag23];
        for (let y = 0; y < N; y++) {
          const r0 = y * N, r1 = (y + 1 < N ? y + 1 : 0) * N;
          for (let x = 0; x < N; x++) {
            const x1 = x + 1 < N ? x + 1 : 0;
            const i = r0 + x;
            // Horizontal: 2 rows × 3 cols (children at i and one right).
            let kN = uh(h2N[i], h2N[r0 + x1]), kI = uh(h2I[i], h2I[r0 + x1]);
            if (kN !== kI) {
              outKeys[count] = mixTag(kN < kI ? kN : kI, tag23) ^ pS23;
              outPols[count] = kN < kI ? 1 : -1;
              outTags[count] = tag23;
              count++;
            }
            // Vertical: 3 rows × 2 cols (children at i and one down).
            kN = uh(h2N[i], h2N[r1 + x]); kI = uh(h2I[i], h2I[r1 + x]);
            if (kN !== kI) {
              outKeys[count] = mixTag(kN < kI ? kN : kI, tag23) ^ pS23;
              outPols[count] = kN < kI ? 1 : -1;
              outTags[count] = tag23;
              count++;
            }
          }
        }
      }
      if (do3 || do4 || do34) {
        // 3×3 = X of the four corner 2×2 sub-windows (anchors i, right, down,
        // down-right), exactly hpatterns' recursion; the plane is stored so
        // 4×4 (and deltaZ) can read it.
        const h3N = pl.h3N, h3I = pl.h3I;
        const tag = (tagBase << 3) | 3;
        const pS3 = phSalt[tag];
        for (let y = 0; y < N; y++) {
          const r0 = y * N, r1 = (y + 1 < N ? y + 1 : 0) * N;
          for (let x = 0; x < N; x++) {
            const x1 = x + 1 < N ? x + 1 : 0;
            const i = r0 + x;
            const kN = xh4(h2N[r0+x], h2N[r0+x1], h2N[r1+x], h2N[r1+x1]);
            const kI = xh4(h2I[r0+x], h2I[r0+x1], h2I[r1+x], h2I[r1+x1]);
            h3N[i] = kN; h3I[i] = kI;
            if (!do3 || kN === kI) continue;
            outKeys[count] = mixTag(kN < kI ? kN : kI, tag) ^ pS3;
            outPols[count] = kN < kI ? 1 : -1;
            outTags[count] = tag;
            count++;
          }
        }
        if (do4) {
          // 4×4 = X of the four corner 3×3 sub-windows.
          const tag4 = (tagBase << 3) | 4;
          const pS4 = phSalt[tag4];
          for (let y = 0; y < N; y++) {
            const r0 = y * N, r1 = (y + 1 < N ? y + 1 : 0) * N;
            for (let x = 0; x < N; x++) {
              const x1 = x + 1 < N ? x + 1 : 0;
              const kN = xh4(h3N[r0+x], h3N[r0+x1], h3N[r1+x], h3N[r1+x1]);
              const kI = xh4(h3I[r0+x], h3I[r0+x1], h3I[r1+x], h3I[r1+x1]);
              if (kN === kI) continue;
              outKeys[count] = mixTag(kN < kI ? kN : kI, tag4) ^ pS4;
              outPols[count] = kN < kI ? 1 : -1;
              outTags[count] = tag4;
              count++;
            }
          }
        }
        if (do34) {
          // 3×4 ∪ 4×3 = uh of two adjacent 3×3 sub-hashes (sharing a 3×2
          // core).  BOTH orientations under ONE tag, mandatorily: a 90°
          // board rotation carries each horizontal window onto a vertical
          // one with an equal hash (each 3×3 child is D4-invariant, uh
          // unordered), so only the union of the two families keeps the
          // feature multiset D4-invariant.
          const tag34 = (tagBase << 3) | 5;
          const pS34 = phSalt[tag34];
          for (let y = 0; y < N; y++) {
            const r0 = y * N, r1 = (y + 1 < N ? y + 1 : 0) * N;
            for (let x = 0; x < N; x++) {
              const x1 = x + 1 < N ? x + 1 : 0;
              const i = r0 + x;
              // Horizontal: 3 rows × 4 cols (children at i and one right).
              let kN = uh(h3N[i], h3N[r0 + x1]), kI = uh(h3I[i], h3I[r0 + x1]);
              if (kN !== kI) {
                outKeys[count] = mixTag(kN < kI ? kN : kI, tag34) ^ pS34;
                outPols[count] = kN < kI ? 1 : -1;
                outTags[count] = tag34;
                count++;
              }
              // Vertical: 4 rows × 3 cols (children at i and one down).
              kN = uh(h3N[i], h3N[r1 + x]); kI = uh(h3I[i], h3I[r1 + x]);
              if (kN !== kI) {
                outKeys[count] = mixTag(kN < kI ? kN : kI, tag34) ^ pS34;
                outPols[count] = kN < kI ? 1 : -1;
                outTags[count] = tag34;
                count++;
              }
            }
          }
        }
      }
    }

  }
  if (doSetNext) {
    for (let c = 0; c < captures.length; c++) {
      cells[captures[c]] = -game.current;
    }
    cells[nextMove] = EMPTY;
  }
  // Chain-attribute family (spec 'C'): one antisymmetric feature per chain,
  // keyed by (stones, liberties, adjacent enemy chains, secondary liberties),
  // each clamped.  Secondary liberties = distinct empties adjacent to the
  // chain's liberties that are not themselves liberties (a cheap eye-space
  // proxy).  Polarity = the owner, so a WHITE chain with the same attributes
  // contributes -w — the antisymmetric convention the komi machinery needs.
  // Turn: one feature for the whole position, polarity by side to move, keyed
  // by phase bucket.  A colour flip flips the sign, so it obeys the same
  // antisymmetric convention the pattern families do.
  if (prepSpecs.hasTurn) {
    outKeys[count] = mixTag(TURN_SALT, TURN_TAG) ^ phSalt[TURN_TAG];
    outPols[count] = game.current === BLACK ? 1 : -1;
    outTags[count] = TURN_TAG;
    count++;
  }
  if (prepSpecs.hasChains) {
    const gid = game._gid, nbr = game._nbr;
    const { chains, byGid } = chainsOf(cells, nbr, gid);
    const [CS, CL, CA, CE] = prepSpecs.chainCaps;
    const phBins = prepSpecs.chainPhaseBins;
    let phSalt = 0;
    if (phBins > 1) {
      const ph = 1 - game.emptyCount / cap;
      let b = Math.floor(ph * phBins);
      if (b >= phBins) b = phBins - 1;
      phSalt = Math.imul(b + 1, 0x9E3779B1) | 0;
    }
    const capCode = CS + 32 * (CL + 32 * (CA + 32 * CE));
    const CJ = prepSpecs.chainCaps.length > 4 ? prepSpecs.chainCaps[4] : -1;
    const CW = prepSpecs.chainCaps.length > 5 ? prepSpecs.chainCaps[5] : -1;
    const CY = prepSpecs.chainCaps.length > 6 ? prepSpecs.chainCaps[6] : -1;
    const CH = prepSpecs.chainCaps.length > 7 ? prepSpecs.chainCaps[7] : -1;   // shared (enemy-contested) liberties
    const CF = prepSpecs.chainCaps.length > 8 ? prepSpecs.chainCaps[8] : -1;   // strongest joinable friend's liberties
    const CP = prepSpecs.chainCaps.length > 9 ? prepSpecs.chainCaps[9] : -1;   // connection points (liberties adjacent to another friendly chain)
    const CD = prepSpecs.chainCaps.length > 10 ? prepSpecs.chainCaps[10] : -1; // density: floor(4*libs/stones) bucketed (blob low, string high)
    const CI = prepSpecs.chainCaps.length > 11 ? prepSpecs.chainCaps[11] : -1; // interior stones (all 4 neighbours same chain)
    // Slot 13 is the only one in the dot-list that is a COUNT, not a cap:
    // it is the NUMBER OF HEALTH BUCKETS the frozen chain-survival model's
    // P(survive) is split into (2 = two buckets, either side of p = 0.5).
    // 0 or 1 = off.
    const CV = prepSpecs.chainCaps.length > 12 ? prepSpecs.chainCaps[12] : 0;
    const survModel = CV >= 2 ? prepSpecs.healthModel : null;
    const ls = game._ls, dnbr = game._dnbr;
    // Health is a POSITION-level quantity — with a neighbour feature on, the
    // model iterates over the whole board — so every chain's p is computed
    // once here, before the attribute loop reads it.
    if (survModel) chainHealthAll(survModel, cells, nbr, dnbr, gid, ls, chains, byGid);
    // Eye rule for the per-chain attribute (Chris, 2026-09-08): a liberty
    // is an eye of THIS chain iff all 4 orthogonals belong to this chain.
    // Multi-chain eyes are not counted — whether those chains connect is
    // the joinable attribute's department — and no diagonal heuristic.
    const trueEyeFor = (idx, g) => {
      const base = idx * 4;
      for (let i = 0; i < 4; i++) if (gid[nbr[base + i]] !== g) return false;
      return true;
    };
    for (const r of chains) {
      const g = r.gid;
      const libSet = new Set(r.libs), adj = new Set();
      for (const idx of r.stones) {
        const base = idx * 4;
        for (let d = 0; d < 4; d++) {
          const n = nbr[base + d], nc = cells[n];
          if (nc !== 0 && nc !== r.c) adj.add(gid[n]);
        }
      }
      let sec = 0;
      const secSeen = new Set(), friends = new Set();
      for (const l of libSet) {
        const base = l * 4;
        for (let d = 0; d < 4; d++) {
          const n = nbr[base + d], nc = cells[n];
          if (nc === 0) { if (!libSet.has(n) && !secSeen.has(n)) { secSeen.add(n); sec++; } }
          else if (CJ >= 0 && nc === r.c && gid[n] !== g) friends.add(gid[n]);
        }
      }
      const nStones = r.stones.length;
      const stones = nStones > CS ? CS : nStones;
      const libs   = libSet.size > CL ? CL : libSet.size;
      const adjE   = adj.size > CA ? CA : adj.size;
      const secL   = sec > CE ? CE : sec;
      const joinF  = CJ >= 0 ? (friends.size > CJ ? CJ : friends.size) : 0;
      let weakest = 0;
      if (CW >= 0 && adj.size > 0) {
        weakest = Infinity;
        for (const eg of adj) { const el = ls[eg]; if (el < weakest) weakest = el; }
        if (weakest > CW) weakest = CW;
      }
      let eyes = 0;
      // >= 1, not >= 0: these count-and-break attributes would otherwise
      // leave a stray 1 at cap 0, since the first hit satisfies `>= 0`.  Cap 0
      // must mean OFF, exactly as it does for the clamp-after-loop attributes.
      if (CY >= 1) {
        for (const l of libSet) if (trueEyeFor(l, g)) { eyes++; if (eyes >= CY) break; }
      }
      let shared = 0;
      if (CH >= 0) {
        for (const l of libSet) {
          const base = l * 4;
          for (let d = 0; d < 4; d++) { const nc = cells[nbr[base + d]]; if (nc !== 0 && nc !== r.c) { shared++; break; } }
        }
        if (shared > CH) shared = CH;
      }
      let bestF = 0;
      if (CF >= 0) {
        for (const fg of friends) { const fl = ls[fg]; if (fl > bestF) bestF = fl; }
        if (bestF > CF) bestF = CF;
      }
      let interior = 0;
      if (CI >= 1) {   // see the eyes note above: cap 0 means off
        for (const idx of r.stones) {
          const base = idx * 4;
          let own = 0;
          for (let d = 0; d < 4; d++) if (gid[nbr[base + d]] === g && cells[nbr[base + d]] !== 0) own++;
          if (own === 4) { interior++; if (interior >= CI) break; }
        }
      }
      let connP = 0;
      if (CP >= 0) {
        for (const l of libSet) {
          const base = l * 4;
          for (let d = 0; d < 4; d++) { const n = nbr[base + d]; if (cells[n] === r.c && gid[n] !== g) { connP++; break; } }
        }
        if (connP > CP) connP = CP;
      }
      let dens = 0;
      if (CD >= 0) { dens = Math.floor(4 * libSet.size / nStones); if (dens > CD) dens = CD; }
      let surv = 0;
      if (CV >= 2) {
        surv = Math.floor(r.p * CV);        // CV buckets: 0 .. CV-1
        if (surv > CV - 1) surv = CV - 1;
      }
      const pack   = stones + (CS + 1) * (libs + (CL + 1) * (adjE + (CA + 1) * (secL + (CE + 1) * (joinF + (CJ >= 0 ? CJ + 1 : 1) * (weakest + (CW >= 0 ? CW + 1 : 1) * (eyes + (CY >= 0 ? CY + 1 : 1) * (shared + (CH >= 0 ? CH + 1 : 1) * (bestF + (CF >= 0 ? CF + 1 : 1) * (connP + (CP >= 0 ? CP + 1 : 1) * (dens + (CD >= 0 ? CD + 1 : 1) * (interior + (CI >= 0 ? CI + 1 : 1) * surv))))))))))); 
      let key = (Math.imul(pack + 1, 2654435761) ^ Math.imul(capCode + 1, 0x45d9f3b) ^ phSalt) | 0;
      if (key === 0) key = 1;    // int-map reserves key 0
      outKeys[count] = key;
      outPols[count] = r.c;
      outTags[count] = 0;        // specTag({size:0, maxLibs:0})
      count++;
    }
  }

  // Eye-pair family (spec 'E'): for each chain with <= libGate liberties,
  // one antisymmetric feature per unordered pair of NON-ADJACENT liberties,
  // keyed by the two owner-relative t-hashed 3x3 neighbourhoods (uh makes
  // the pair unordered; each region hash is D4-invariant by construction,
  // t-hash fidelity caveats as usual).  Pair geometry beyond non-adjacency
  // is deliberately discarded: two real eyes anywhere alive the chain.
  if (prepSpecs.hasEyePairs) {
    const gid = game._gid, nbr = game._nbr, dnbr = game._dnbr;
    const gate = prepSpecs.eyePairGate;
    let phSaltE = 0;
    const eBins = prepSpecs.eyePairPhaseBins;
    if (eBins > 1) {
      const ph = 1 - game.emptyCount / cap;
      let b = Math.floor(ph * eBins);
      if (b >= eBins) b = eBins - 1;
      phSaltE = Math.imul(b + 1, 0x85ebca6b) | 0;
    }
    const r3 = (l, owner) => ninecellHash(cells, nbr, dnbr, l, owner);
    const seenG = new Set();
    const libs = [];
    for (let idx = 0; idx < cap; idx++) {
      const c = cells[idx];
      if (c === 0) continue;
      const g0 = gid[idx];
      if (seenG.has(g0)) continue;
      seenG.add(g0);
      // collect this chain's liberties (walk its cells via gid match)
      libs.length = 0;
      for (let j = 0; j < cap; j++) {
        if (cells[j] !== 0) continue;
        const b4 = j * 4;
        for (let d = 0; d < 4; d++) if (gid[nbr[b4 + d]] === g0 && cells[nbr[b4 + d]] !== 0) { libs.push(j); break; }
      }
      if (libs.length < 2 || libs.length > gate) continue;
      for (let a = 0; a < libs.length; a++) {
        const ha = r3(libs[a], c);
        for (let b = a + 1; b < libs.length; b++) {
          const la = libs[a], lb = libs[b];
          // non-adjacent only: orthogonally adjacent liberties are one eyespace
          const b4 = la * 4;
          if (nbr[b4] === lb || nbr[b4 + 1] === lb || nbr[b4 + 2] === lb || nbr[b4 + 3] === lb) continue;
          let key = pairKey(ha, r3(lb, c)) ^ phSaltE;
          if (key === 0) key = 1;
          outKeys[count] = key;
          outPols[count] = c;
          outTags[count] = 6;
          count++;
        }
      }
    }
  }

  return { keys: outKeys, pols: outPols, tags: outTags, count, val: 0.5 };
}

// ── Speculative-incremental 1-ply evaluation ─────────────────────────────────
//
// deltaZ(game, prepSpecs, weights, move) → Δz for placing game.current at
// `move`, reading the base-position planes extractFeatures left on prepSpecs
// (the caller must have extracted the CURRENT position immediately before,
// with this prepSpecs, and not extracted anything else since).  V(after) =
// σ(z_base + Δz), with z_base from evaluateFeatures (features.z).
//
// Liberty bookkeeping: the pre-move incremental structures (game._gid/_ls/
// _lw/_sw) exactly describe the base position, and the post-move liberty
// counts follow analytically — an adjacent enemy group loses the placed
// point (ls-1); the placed/merged friendly group's count is the popcount of
// the merged liberty-bitset union minus the placed point.  A cell's raw
// value only changes when its group's CAPPED count changes, so groups far
// from the maxLibs cap contribute nothing however large they are.  Captures
// dirty far more (removed stones + their neighbours' liberties): NaN, and
// the caller falls back to full extraction — on a SEPARATE prepSpecs, so
// the base planes here survive (see vpatsearch).
function deltaZ(game, prepSpecs, weights, move) {
  // Ladder codes update non-locally (one stone can flip a whole ladder
  // path), so the incremental contract cannot hold — fail loudly.
  if (prepSpecs.hasLadder) {
    throw new Error('vpatterns deltaZ: ladder-coded specs (size:L) are not incremental');
  }
  if (prepSpecs.hasChains) {
    throw new Error('vpatterns deltaZ: chain-attribute specs (C) are not incremental');
  }
  if (prepSpecs.hasHealth) {
    throw new Error('vpatterns deltaZ: health-coded specs (size:H<N>) are not incremental');
  }
  if (prepSpecs.hasEyePairs) {
    throw new Error('vpatterns deltaZ: eye-pair specs (E) are not incremental');
  }
  if (prepSpecs.hasPhasedPatterns) {
    throw new Error('vpatterns deltaZ: phase-binned pattern specs (pN) are not incremental');
  }
  if (prepSpecs.hasTurn) {
    throw new Error('vpatterns deltaZ: the turn spec (t) is not incremental — the move flips it');
  }
  if (move === PASS) return 0;
  if (game.captureList(move).length > 0) return NaN;
  const N = game.N, cap = N * N, cells = game.cells, cur = game.current;
  const gid = game._gid, ls = game._ls, lw = game._lw, sw = game._sw, W = game._W;
  const nbr = game._nbr;

  if (_dzMark2.length < cap) {
    _dzMark2 = new Int32Array(cap); _dzMark3 = new Int32Array(cap); _dzMark4 = new Int32Array(cap);
    _dzMarkC = new Int32Array(cap);
    _dzOvN = new Int32Array(cap); _dzOvI = new Int32Array(cap);
    _dzOv3N = new Int32Array(cap); _dzOv3I = new Int32Array(cap);
    _dzLeafN = new Int32Array(cap); _dzLeafI = new Int32Array(cap);
    _dzDirtyCells = new Int32Array(cap); _dzA2 = new Int32Array(cap); _dzA3 = new Int32Array(cap);
  }
  if (_dzUnion.length < W) _dzUnion = new Int32Array(W);
  if (_dzStamp > 0x7ffffff0) { _dzMark2.fill(0); _dzMark3.fill(0); _dzMark4.fill(0); _dzMarkC.fill(0); _dzStamp = 0; }

  // Adjacent groups (deduped) and the placed/merged group's liberty count.
  const b4 = move * 4;
  let nF = 0, nE = 0;
  const fG = _dzFG, eG = _dzEG;
  const un = _dzUnion;
  for (let w = 0; w < W; w++) un[w] = 0;
  for (let d = 0; d < 4; d++) {
    const nb = nbr[b4 + d];
    const c = cells[nb];
    if (c === 0) { un[nb >> 5] |= 1 << (nb & 31); continue; }
    const g = gid[nb];
    if (c === cur) {
      let dup = false;
      for (let j = 0; j < nF; j++) if (fG[j] === g) { dup = true; break; }
      if (!dup) fG[nF++] = g;
    } else {
      let dup = false;
      for (let j = 0; j < nE; j++) if (eG[j] === g) { dup = true; break; }
      if (!dup) eG[nE++] = g;
    }
  }
  for (let j = 0; j < nF; j++) {
    const b = fG[j] * W;
    for (let w = 0; w < W; w++) un[w] |= lw[b + w];
  }
  un[move >> 5] &= ~(1 << (move & 31));
  let newFLibs = 0;
  for (let w = 0; w < W; w++) {
    let v = un[w];
    v = v - ((v >>> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
    newFLibs += (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
  }

  const { byMaxLibs, sortedMaxLibs } = prepSpecs;
  const planes = prepSpecs._planes;
  let delta = 0;

  for (const maxLibs of sortedMaxLibs) {
    const sizes = byMaxLibs.get(maxLibs);
    const do1 = sizes.includes(1), do2 = sizes.includes(2), do3 = sizes.includes(3),
          do4 = sizes.includes(4), do34 = sizes.includes(34), do23 = sizes.includes(23);
    const off = maxLibs;

    // ── Dirty cells for this maxLibs: the placed stone, plus every stone of a
    // group whose capped liberty count changes.  Leaf overrides + size-1 Δ.
    const cellStamp = ++_dzStamp;
    const k1base = 131 * maxLibs;
    let nDirty = 0;
    const dirty = _dzDirtyCells;
    const addDirty = (i, newRaw, oldRaw) => {
      _dzMarkC[i] = cellStamp;
      _dzLeafN[i] = _leafTab[newRaw + off];
      _dzLeafI[i] = _leafTab[off - newRaw];
      dirty[nDirty++] = i;
      if (do1) {
        if (oldRaw !== 0) {
          const w = weights.get((oldRaw > 0 ? oldRaw : -oldRaw) + k1base) ?? 0;
          delta -= (oldRaw > 0 ? 1 : -1) * w;
        }
        if (newRaw !== 0) {
          const w = weights.get((newRaw > 0 ? newRaw : -newRaw) + k1base) ?? 0;
          delta += (newRaw > 0 ? 1 : -1) * w;
        }
      }
    };
    const capd = (v) => v < maxLibs ? v : maxLibs;
    // The placed stone: 0 → cur * min(newFLibs, ml).
    addDirty(move, cur * capd(newFLibs), 0);
    // Friendly groups: min(ls, ml) → min(newFLibs, ml).
    for (let j = 0; j < nF; j++) {
      const g = fG[j];
      const oldC = capd(ls[g]), newC = capd(newFLibs);
      if (oldC === newC) continue;
      const b = g * W;
      for (let w = 0; w < W; w++) {
        let v = sw[b + w];
        while (v) {
          const i = (w << 5) + (31 - Math.clz32(v & -v));
          if (i < cap) addDirty(i, cur * newC, cur * oldC);
          v &= v - 1;
        }
      }
    }
    // Enemy groups: min(ls, ml) → min(ls - 1, ml).
    for (let j = 0; j < nE; j++) {
      const g = eG[j];
      const oldC = capd(ls[g]), newC = capd(ls[g] - 1);
      if (oldC === newC) continue;
      const b = g * W;
      for (let w = 0; w < W; w++) {
        let v = sw[b + w];
        while (v) {
          const i = (w << 5) + (31 - Math.clz32(v & -v));
          if (i < cap) addDirty(i, -cur * newC, -cur * oldC);
          v &= v - 1;
        }
      }
    }

    if (!(do2 || do3 || do4 || do34 || do23)) continue;
    const pl = planes.get(maxLibs);
    const lN = pl.lN, lI = pl.lI, h2N = pl.h2N, h2I = pl.h2I;
    const leafN = (i) => _dzMarkC[i] === cellStamp ? _dzLeafN[i] : lN[i];
    const leafI = (i) => _dzMarkC[i] === cellStamp ? _dzLeafI[i] : lI[i];

    // ── Affected 2×2 anchors: the 4 windows containing each dirty cell.
    const a2Stamp = ++_dzStamp;
    let nA2 = 0;
    const a2 = _dzA2;
    const tag2 = (maxLibs << 3) | 2;
    for (let di = 0; di < nDirty; di++) {
      const i = dirty[di];
      const r = (i / N) | 0, c = i % N;
      const rU = r === 0 ? N - 1 : r - 1, cL = c === 0 ? N - 1 : c - 1;
      const anchors4 = [r * N + c, r * N + cL, rU * N + c, rU * N + cL];
      for (let k = 0; k < 4; k++) {
        const a = anchors4[k];
        if (_dzMark2[a] === a2Stamp) continue;
        _dzMark2[a] = a2Stamp;
        a2[nA2++] = a;
      }
    }
    for (let k = 0; k < nA2; k++) {
      const a = a2[k];
      const r = (a / N) | 0, c = a % N;
      const rD = (r + 1 < N ? r + 1 : 0) * N, r0 = r * N;
      const cR = c + 1 < N ? c + 1 : 0;
      const kN = xh4(leafN(r0 + c), leafN(r0 + cR), leafN(rD + c), leafN(rD + cR));
      const kI = xh4(leafI(r0 + c), leafI(r0 + cR), leafI(rD + c), leafI(rD + cR));
      _dzOvN[a] = kN; _dzOvI[a] = kI;   // valid for anchors marked a2Stamp
      if (do2) {
        const oN = h2N[a], oI = h2I[a];
        if (oN !== oI) delta -= (oN < oI ? 1 : -1) * (weights.get(mixTag(oN < oI ? oN : oI, tag2)) ?? 0);
        if (kN !== kI) delta += (kN < kI ? 1 : -1) * (weights.get(mixTag(kN < kI ? kN : kI, tag2)) ?? 0);
      }
    }

    if (do23) {
      // ── Affected 2×3/3×2 pair anchors: each affected 2×2 anchor sits in two
      // horizontal pairs (anchored at it and one left) and two vertical pairs
      // (at it and one up).  Mirrors the 3×4/4×3 pass one level down, and
      // reuses _dzMark3 under fresh stamps — the 3×3 pass below takes its own.
      const h2atN = (i) => _dzMark2[i] === a2Stamp ? _dzOvN[i] : h2N[i];
      const h2atI0 = (i) => _dzMark2[i] === a2Stamp ? _dzOvI[i] : h2I[i];
      const tag23 = (maxLibs << 3) | 7;
      for (let ori = 0; ori < 2; ori++) {   // 0 = horizontal, 1 = vertical
        const oStamp = ++_dzStamp;
        for (let k = 0; k < nA2; k++) {
          const i = a2[k];
          const r = (i / N) | 0, c = i % N;
          const aPrev = ori === 0 ? r * N + (c === 0 ? N - 1 : c - 1)
                                  : (r === 0 ? N - 1 : r - 1) * N + c;
          for (let m = 0; m < 2; m++) {
            const a = m === 0 ? i : aPrev;
            if (_dzMark3[a] === oStamp) continue;
            _dzMark3[a] = oStamp;
            const ar = (a / N) | 0, ac = a % N;
            const aNext = ori === 0 ? ar * N + (ac + 1 < N ? ac + 1 : 0)
                                    : (ar + 1 < N ? ar + 1 : 0) * N + ac;
            const oN = uh(h2N[a], h2N[aNext]), oI = uh(h2I[a], h2I[aNext]);
            const kN = uh(h2atN(a), h2atN(aNext)), kI = uh(h2atI0(a), h2atI0(aNext));
            if (oN !== oI) delta -= (oN < oI ? 1 : -1) * (weights.get(mixTag(oN < oI ? oN : oI, tag23)) ?? 0);
            if (kN !== kI) delta += (kN < kI ? 1 : -1) * (weights.get(mixTag(kN < kI ? kN : kI, tag23)) ?? 0);
          }
        }
      }
    }

    if (!(do3 || do4 || do34)) continue;
    // ── Affected 3×3 anchors: the 4 windows whose corner children include an
    // affected 2×2 anchor.  New hashes are recorded as overrides for level 4;
    // old hashes come from the stored h3 planes.
    const h3N = pl.h3N, h3I = pl.h3I;
    const a3Stamp = ++_dzStamp;
    let nA3 = 0;
    const a3 = _dzA3;
    const tag3 = (maxLibs << 3) | 3;
    const h2at = (i) => _dzMark2[i] === a2Stamp ? _dzOvN[i] : h2N[i];
    const h2atI = (i) => _dzMark2[i] === a2Stamp ? _dzOvI[i] : h2I[i];
    for (let k = 0; k < nA2; k++) {
      const i = a2[k];
      const r = (i / N) | 0, c = i % N;
      const rU = r === 0 ? N - 1 : r - 1, cL = c === 0 ? N - 1 : c - 1;
      const anchors4 = [r * N + c, r * N + cL, rU * N + c, rU * N + cL];
      for (let m = 0; m < 4; m++) {
        const a = anchors4[m];
        if (_dzMark3[a] === a3Stamp) continue;
        _dzMark3[a] = a3Stamp;
        a3[nA3++] = a;
        const ar = (a / N) | 0, ac = a % N;
        const rD = (ar + 1 < N ? ar + 1 : 0) * N, r0 = ar * N;
        const cR = ac + 1 < N ? ac + 1 : 0;
        const oN = h3N[a], oI = h3I[a];
        const kN = xh4(h2at(r0 + ac), h2at(r0 + cR), h2at(rD + ac), h2at(rD + cR));
        const kI = xh4(h2atI(r0 + ac), h2atI(r0 + cR), h2atI(rD + ac), h2atI(rD + cR));
        _dzOv3N[a] = kN; _dzOv3I[a] = kI;   // valid for anchors marked a3Stamp
        if (do3) {
          if (oN !== oI) delta -= (oN < oI ? 1 : -1) * (weights.get(mixTag(oN < oI ? oN : oI, tag3)) ?? 0);
          if (kN !== kI) delta += (kN < kI ? 1 : -1) * (weights.get(mixTag(kN < kI ? kN : kI, tag3)) ?? 0);
        }
      }
    }

    if (!(do4 || do34)) continue;
    const h3at = (i) => _dzMark3[i] === a3Stamp ? _dzOv3N[i] : h3N[i];
    const h3atI = (i) => _dzMark3[i] === a3Stamp ? _dzOv3I[i] : h3I[i];
    if (do4) {
      // ── Affected 4×4 anchors: the spread of the affected 3×3 anchors.
      const a4Stamp = ++_dzStamp;
      const tag4 = (maxLibs << 3) | 4;
      for (let k = 0; k < nA3; k++) {
        const i = a3[k];
        const r = (i / N) | 0, c = i % N;
        const rU = r === 0 ? N - 1 : r - 1, cL = c === 0 ? N - 1 : c - 1;
        const anchors4 = [r * N + c, r * N + cL, rU * N + c, rU * N + cL];
        for (let m = 0; m < 4; m++) {
          const a = anchors4[m];
          if (_dzMark4[a] === a4Stamp) continue;
          _dzMark4[a] = a4Stamp;
          const ar = (a / N) | 0, ac = a % N;
          const rD = (ar + 1 < N ? ar + 1 : 0) * N, r0 = ar * N;
          const cR = ac + 1 < N ? ac + 1 : 0;
          const oN = xh4(h3N[r0 + ac], h3N[r0 + cR], h3N[rD + ac], h3N[rD + cR]);
          const oI = xh4(h3I[r0 + ac], h3I[r0 + cR], h3I[rD + ac], h3I[rD + cR]);
          const kN = xh4(h3at(r0 + ac), h3at(r0 + cR), h3at(rD + ac), h3at(rD + cR));
          const kI = xh4(h3atI(r0 + ac), h3atI(r0 + cR), h3atI(rD + ac), h3atI(rD + cR));
          if (oN !== oI) delta -= (oN < oI ? 1 : -1) * (weights.get(mixTag(oN < oI ? oN : oI, tag4)) ?? 0);
          if (kN !== kI) delta += (kN < kI ? 1 : -1) * (weights.get(mixTag(kN < kI ? kN : kI, tag4)) ?? 0);
        }
      }
    }
    if (do34) {
      // ── Affected 3×4/4×3 pair anchors: each affected 3×3 anchor sits in
      // two horizontal pairs (anchored at it and one left) and two vertical
      // pairs (at it and one up).  Sequential per-orientation passes reuse
      // _dzMark4 under fresh stamps.
      const tag34 = (maxLibs << 3) | 5;
      for (let ori = 0; ori < 2; ori++) {   // 0 = horizontal, 1 = vertical
        const oStamp = ++_dzStamp;
        for (let k = 0; k < nA3; k++) {
          const i = a3[k];
          const r = (i / N) | 0, c = i % N;
          const aPrev = ori === 0 ? r * N + (c === 0 ? N - 1 : c - 1)
                                  : (r === 0 ? N - 1 : r - 1) * N + c;
          for (let m = 0; m < 2; m++) {
            const a = m === 0 ? i : aPrev;
            if (_dzMark4[a] === oStamp) continue;
            _dzMark4[a] = oStamp;
            const ar = (a / N) | 0, ac = a % N;
            const aNext = ori === 0 ? ar * N + (ac + 1 < N ? ac + 1 : 0)
                                    : (ar + 1 < N ? ar + 1 : 0) * N + ac;
            const oN = uh(h3N[a], h3N[aNext]), oI = uh(h3I[a], h3I[aNext]);
            const kN = uh(h3at(a), h3at(aNext)), kI = uh(h3atI(a), h3atI(aNext));
            if (oN !== oI) delta -= (oN < oI ? 1 : -1) * (weights.get(mixTag(oN < oI ? oN : oI, tag34)) ?? 0);
            if (kN !== kI) delta += (kN < kI ? 1 : -1) * (weights.get(mixTag(kN < kI ? kN : kI, tag34)) ?? 0);
          }
        }
      }
    }
  }
  return delta;
}
const _dzFG = new Int32Array(4), _dzEG = new Int32Array(4);
let _dzDirtyCells = new Int32Array(0), _dzA2 = new Int32Array(0), _dzA3 = new Int32Array(0);

// ── Value function (pure) ─────────────────────────────────────────────────────

// V(s) = σ(Σ polarity_i · w[key_i]) = P(BLACK wins)
// features: { keys: Int32Array, pols: Int8Array, count, val }  (from extractFeatures)
// weights: Map<key, float>  (missing keys treated as 0)
function evaluateFeatures(features, weights) {
  let z = 0;
  const { keys, pols, count } = features;
  for (let i = 0; i < count; i++) {
    const w = weights.get(keys[i]) ?? 0;
    z += pols[i] * w;
  }
  features.z   = z;   // logit, for the zBase + deltaZ fast path
  features.val = 1 / (1 + Math.exp(-z));
  return features.val;
}

// Convenience: extract features and evaluate in one call.
// model must have a preparedSpecs property (see prepareSpecs).
function evaluate(game, model) {
  return evaluateFeatures(extractFeatures(game, model.preparedSpecs), model.weights);
}

// ── Persistence ───────────────────────────────────────────────────────────────

// Loads a model JS file and returns { weights: Map<number,float>, specs: [...] }.
// Always returns a fresh copy so multiple callers don't share the same Map.
// health: path to (or already-loaded) health model, required when the specs
// are health-coded or use the C survival attribute.
function loadWeights(filePath, health) {
  const raw = require(require('path').resolve(filePath));
  const specs = raw.specs;
  const weights = makeWeights(Math.max(1024, (raw.weights.size ?? raw.weights.length) * 2));
  for (const [k, v] of raw.weights) weights.set(k, v);
  const preparedSpecs = prepareSpecs(specs, { health });
  if (raw.health) checkHealthMatch(raw.health, filePath, preparedSpecs.healthModel);
  return { specs, preparedSpecs, weights, komi: raw.komi };
}

// A health-coded model's keys are bucket indices produced by whichever health
// model was loaded when it trained, so pairing it with a different one
// re-indexes every weight — no error, just wrong.  We do NOT match file names
// (a model may legitimately be copied or renamed); we compare the parameters
// the training run recorded against the health model now in use.  Alphabet
// differences change the key space outright and are fatal; the rest change
// the bucket boundaries and are reported loudly.
function checkHealthMatch(want, filePath, got) {
  const fatal = [];
  for (const k of ['otherMaxLibs', 'maxLibs', 'maxJoinLibs', 'friendHealthMaxBuckets',
                   'foeHealthMinBuckets', 'stoneSalt']) {
    if (want[k] !== undefined && want[k] !== got[k]) fatal.push(`${k}: trained ${want[k]}, loaded ${got[k]}`);
  }
  if (fatal.length) {
    throw new Error(`vpatterns: ${filePath} was trained against a health model with a different ALPHABET ` +
                    `— its weights index a different key space (${fatal.join('; ')})`);
  }
  const soft = [];
  for (const k of ['minPhase', 'maxPhase', 'delta', 'iterations', 'initHealth', 'bias', 'nWeights']) {
    if (want[k] !== undefined && got[k] !== undefined && want[k] !== got[k]) {
      soft.push(`${k}: trained ${want[k]}, loaded ${got[k]}`);
    }
  }
  if (soft.length) {
    console.error(`vpatterns WARNING: ${filePath} was trained against a DIFFERENT health model ` +
                  `(${soft.join('; ')}) — same alphabet, but the survival probabilities and hence the ` +
                  `bucket assignments differ, so its weights are mis-indexed.`);
  }
}

// Writes a model { weights, specs } to a JS file (browser-includable).
function saveWeights(filePath, model) {
  const fs         = require('fs');
  const specStr    = JSON.stringify(model.specs);
  // Health-coded models record the health model's PARAMETERS (never its path)
  // so a later pairing can be sanity-checked — see checkHealthMatch.
  let healthStr = '';
  if (model.specs.some(sp => sp.maxLibs < 0)) {
    const h = model.preparedSpecs.healthModel;
    healthStr = `, health: { minPhase: ${h.minPhase}, maxPhase: ${h.maxPhase}, delta: ${h.delta}, ` +
                `otherMaxLibs: ${h.otherMaxLibs}, maxLibs: ${h.maxLibs}, maxJoinLibs: ${h.maxJoinLibs}, ` +
                `friendHealthMaxBuckets: ${h.friendHealthMaxBuckets}, ` +
                `foeHealthMinBuckets: ${h.foeHealthMinBuckets}, iterations: ${h.iterations}, ` +
                `initHealth: ${h.initHealth}, ` +
                `stoneSalt: ${h.stoneSalt}, ` +
                `bias: ${h.bias}, nWeights: ${h.weights.size} }`;
  }
  const pairs = [];
  model.weights.forEach((k, v) => pairs.push(`[${k},${+v.toFixed(6)}]`));
  const weightsStr = '[' + pairs.join(',') + ']';
  const src = [
    "'use strict';",
    '// Auto-generated by train-vpatterns.js — do not edit by hand.',
    `const vpatternsModel = { specs: ${specStr}${healthStr}, weights: new Map(${weightsStr})` +
      (model.komi !== undefined ? `, komi: ${model.komi}` : '') + ` };`,
    "if (typeof module !== 'undefined') module.exports = vpatternsModel;",
    "else window.vpatternsModel = vpatternsModel;",
  ].join('\n') + '\n';
  fs.writeFileSync(filePath, src);
}

// ── Exports ───────────────────────────────────────────────────────────────────

const Patterns = {
  rawState,
  makeWeights,
  specTag,
  prepareSpecs,
  extractFeatures,
  evaluateFeatures,
  evaluate,
  deltaZ,
  loadWeights,
  saveWeights,
  specToken,
  specString,
  ninecellHash,
  pairKey,
};

if (typeof module !== 'undefined') module.exports = Patterns;
else window.VPatterns = Patterns;

})();
