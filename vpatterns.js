'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.
// Wrapped in an IIFE to avoid polluting the global namespace.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

const { BLACK, EMPTY, PASS } = _isNode ? require('./game2.js') : window.game;
const { makeIntFloatMap } = _isNode ? require('./int-map.js') : window.IntMap;
const { game3FromGame2 } = _isNode ? require('./game3.js') : window.Game3;
const VLibPat = _isNode ? require('./vlibpat.js') : window.VLibPat;

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

// ── Constants ─────────────────────────────────────────────────────────────────

// Cell state encoding (signed, color-canonicalized):
//   0            = empty
//  +1 .. +maxLibs = BLACK stone with that many liberties (capped)
//  -1 .. -maxLibs = WHITE stone with that many liberties (capped)


// ── X-hash (hpatterns' scheme) ────────────────────────────────────────────────
// Symmetry-invariant hierarchical "X" hash, replacing the min-over-16-
// transforms canonicalisation: uh(a,b) is an unordered combiner; a 2×2
// window hashes its two diagonals unordered, which is invariant to exactly
// D4; 3×3 recurses on its four corner 2×2 sub-windows in the same X
// arrangement, and 4×4 on its four corner 3×3 sub-windows (deliberately
// lossy above 2×2 — distinct shapes may share weights, the trade hpatterns
// measured at ~83% key fidelity for 3-state 3×3).
// Colour canonicalisation compares the hash of the board against the hash of
// the colour-inverted board: equal → colour-twin (zero value by symmetry,
// dropped — all-empty and self-inverse-under-D4 patterns); else key = min,
// polarity says which colouring won.  Leaves enter as raw + maxLibs + 1 (≥ 1: uh has an
// absorbing element at -1, and 0 is unsafe as a map key downstream).
function uh(a, b) {
  return (1234567 + a + b + Math.imul(a, b)) | 0;
}
function xh4(tl, tr, bl, br) {
  return uh(uh(tl, br), uh(tr, bl));
}
// Fold the spec tag ((maxLibs << 3) | sizeCode) into a window hash so
// different spec spaces cannot collide in the shared weight map.
function mixTag(h, tag) {
  return uh(h, tag);
}
// Spec size → 3-bit tag code.  Sizes 1-4 are themselves; spec size 34 (the
// 3×4 ∪ 4×3 rectangle pair) takes the free code 5.  maxLibs 0 is the
// LADDER-CODED family ('size:L' in the trainers): raw is vlibpat's 7-state
// turn-independent tactical alphabet (0 empty, ±1 alive, ±2 dead, ±3
// unsettled) instead of capped liberty counts — structurally identical to
// an ml=3 encoding, so the whole plane/hash/34 machinery is shared.
function specTag(spec) {
  return (spec.maxLibs << 3) | (spec.size === 34 ? 5 : spec.size);
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
const _phSaltScratch = new Int32Array(64);
let _libStampVal = 0;

// ── Multi-spec extraction ─────────────────────────────────────────────────────

// Given an array of specs [{ size: 1|2|3, maxLibs: N }, ...], scans every cell
// prepareSpecs: convert a specs array into the internal structure used by
// extractFeatures.  Call once per unique specs array and reuse the result.
// Also precomputes lookup tables for size:2 and size:3:
//   lut2/lut3: Map<maxLibs, { keys: Int32Array, pols: Int8Array, base, b2, b3[, ...], ml }>
//   Index = Σ (cell[i]+maxLibs) * base^i.  pols[i]===0 → skip (symmetric/empty).
function prepareSpecs(specs) {
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
  const chainCaps = hasChains ? (chainSpec.caps || [8, 8, 4, 8]) : null;
  // Optional phase bucketing (token suffix pN): chain keys additionally
  // keyed by floor(phase * N) over [0, 1] — every chain in a position
  // shares the bucket.
  const chainPhaseBins = hasChains ? (chainSpec.phaseBins || 1) : 1;
  // Per-PATTERN-spec phase bins (token suffix pN on size:maxLibs): emitted
  // keys are salted by floor(phase * N), giving each spec its own phase-
  // conditioned weight planes.  Non-incremental (bucket crossings invalidate
  // every key): deltaZ and doSetNext refuse.
  const patPhaseBins = new Int32Array(64);
  let hasPhasedPatterns = false;
  for (const sp of specs) {
    if (sp.size !== 0 && sp.phaseBins > 1) {
      patPhaseBins[(sp.maxLibs << 3) | (sp.size === 34 ? 5 : sp.size)] = sp.phaseBins;
      hasPhasedPatterns = true;
    }
  }
  for (const spec of specs) {
    if (spec.size === 0) continue;
    if (!byMaxLibs.has(spec.maxLibs)) byMaxLibs.set(spec.maxLibs, []);
    byMaxLibs.get(spec.maxLibs).push(spec.size);
  }
  const sortedMaxLibs = [...byMaxLibs.keys()].sort((a, b) => b - a);



  let totalSizes = 0;   // feature slots per cell (size 34 emits 2: one per orientation)
  for (const sizes of byMaxLibs.values())
    for (const s of sizes) totalSizes += s === 34 ? 2 : 1;

  // maxLibs 0 = the ladder-coded family (needs a game3 tactical pass; not
  // incremental — deltaZ and speculative extraction refuse it).
  const hasLadder = byMaxLibs.has(0);

  return { byMaxLibs, sortedMaxLibs, totalSizes: totalSizes + (hasChains ? 1 : 0), hasLadder, hasChains, chainCaps, chainPhaseBins, patPhaseBins, hasPhasedPatterns };
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
  const outTags = new Int8Array(maxF);   // spec tag: (maxLibs << 3) | size
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

  const phSalt = _phSaltScratch;
  if (prepSpecs.hasPhasedPatterns) {
    const bins = prepSpecs.patPhaseBins, ph = 1 - game.emptyCount / cap;
    for (let t = 0; t < 64; t++) {
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
    if (isLadder) {
      // vlibpat's turn-independent 7-state tactical alphabet, from a fresh
      // game3 tactical pass over the current cells.
      raw = VLibPat.computeLadderCodes(game3FromGame2(game), null);
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
    // block (131*16 — a real maxLibs cannot exceed 15).
    const effOff = isLadder ? 3 : maxLibs;
    const sizes = byMaxLibs.get(maxLibs);
    const do1   = sizes.includes(1);
    const do2   = sizes.includes(2);
    const do3   = sizes.includes(3);
    const do4   = sizes.includes(4);
    const do34  = sizes.includes(34);   // 3×4 ∪ 4×3 rectangle pair

    if (do1) {
      const k1base = 131 * (isLadder ? 16 : maxLibs);
      for (let idx = 0; idx < cap; idx++) {
        const s = raw[idx];
        if (s !== 0) {
          const libs = s > 0 ? s : -s;
          outKeys[count] = (libs + k1base) ^ phSalt[(maxLibs << 3) | 1];
          outPols[count] = s > 0 ? 1 : -1;
          outTags[count] = (maxLibs << 3) | 1;
          count++;
        }
      }
    }

    if (do2 || do3 || do4 || do34) {
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
        const tag = (maxLibs << 3) | 2;
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
      if (do3 || do4 || do34) {
        // 3×3 = X of the four corner 2×2 sub-windows (anchors i, right, down,
        // down-right), exactly hpatterns' recursion; the plane is stored so
        // 4×4 (and deltaZ) can read it.
        const h3N = pl.h3N, h3I = pl.h3I;
        const tag = (maxLibs << 3) | 3;
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
          const tag4 = (maxLibs << 3) | 4;
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
          const tag34 = (maxLibs << 3) | 5;
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
  if (prepSpecs.hasChains) {
    const gid = game._gid, nbr = game._nbr;
    const chains = new Map();   // gid -> { c, stones, cells }
    for (let idx = 0; idx < cap; idx++) {
      const c = cells[idx];
      if (c === 0) continue;
      let r = chains.get(gid[idx]);
      if (!r) { r = { c, stones: 0, cells: [] }; chains.set(gid[idx], r); }
      r.stones++;
      r.cells.push(idx);
    }
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
    const ls = game._ls, dnbr = game._dnbr;
    // Eye rule for the per-chain attribute (Chris, 2026-09-08): a liberty
    // is an eye of THIS chain iff all 4 orthogonals belong to this chain.
    // Multi-chain eyes are not counted — whether those chains connect is
    // the joinable attribute's department — and no diagonal heuristic.
    const trueEyeFor = (idx, g) => {
      const base = idx * 4;
      for (let i = 0; i < 4; i++) if (gid[nbr[base + i]] !== g) return false;
      return true;
    };
    for (const [g, r] of chains) {
      const libSet = new Set(), adj = new Set();
      for (const idx of r.cells) {
        const base = idx * 4;
        for (let d = 0; d < 4; d++) {
          const n = nbr[base + d], nc = cells[n];
          if (nc === 0) libSet.add(n);
          else if (nc !== r.c) adj.add(gid[n]);
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
      const stones = r.stones > CS ? CS : r.stones;
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
      if (CY >= 0) {
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
      if (CI >= 0) {
        for (const idx of r.cells) {
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
      if (CD >= 0) { dens = Math.floor(4 * libSet.size / r.stones); if (dens > CD) dens = CD; }
      const pack   = stones + (CS + 1) * (libs + (CL + 1) * (adjE + (CA + 1) * (secL + (CE + 1) * (joinF + (CJ >= 0 ? CJ + 1 : 1) * (weakest + (CW >= 0 ? CW + 1 : 1) * (eyes + (CY >= 0 ? CY + 1 : 1) * (shared + (CH >= 0 ? CH + 1 : 1) * (bestF + (CF >= 0 ? CF + 1 : 1) * (connP + (CP >= 0 ? CP + 1 : 1) * (dens + (CD >= 0 ? CD + 1 : 1) * interior)))))))))); 
      let key = (Math.imul(pack + 1, 2654435761) ^ Math.imul(capCode + 1, 0x45d9f3b) ^ phSalt) | 0;
      if (key === 0) key = 1;    // int-map reserves key 0
      outKeys[count] = key;
      outPols[count] = r.c;
      outTags[count] = 0;        // specTag({size:0, maxLibs:0})
      count++;
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
  if (prepSpecs.hasPhasedPatterns) {
    throw new Error('vpatterns deltaZ: phase-binned pattern specs (pN) are not incremental');
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
          do4 = sizes.includes(4), do34 = sizes.includes(34);
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

    if (!(do2 || do3 || do4 || do34)) continue;
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
function loadWeights(filePath) {
  const raw = require(require('path').resolve(filePath));
  const specs = raw.specs;
  const weights = makeWeights(Math.max(1024, (raw.weights.size ?? raw.weights.length) * 2));
  for (const [k, v] of raw.weights) weights.set(k, v);
  return { specs, preparedSpecs: prepareSpecs(specs), weights, komi: raw.komi };
}

// Writes a model { weights, specs } to a JS file (browser-includable).
function saveWeights(filePath, model) {
  const fs         = require('fs');
  const specStr    = JSON.stringify(model.specs);
  const pairs = [];
  model.weights.forEach((k, v) => pairs.push(`[${k},${+v.toFixed(6)}]`));
  const weightsStr = '[' + pairs.join(',') + ']';
  const src = [
    "'use strict';",
    '// Auto-generated by train-vpatterns.js — do not edit by hand.',
    `const vpatternsModel = { specs: ${specStr}, weights: new Map(${weightsStr})` +
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
};

if (typeof module !== 'undefined') module.exports = Patterns;
else window.VPatterns = Patterns;

})();
