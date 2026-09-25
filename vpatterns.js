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

// extractFeatures rebuilds a Game3 for the ladder pass when no synced one is
// supplied (the slow path).  Warn once per process, with a stack, so those
// call sites get noticed and can pass a game3 — without flooding hot loops.
let _warnedLadderRebuild = false;

// Chain health lives in health-lib.js — see the note at its head for why the
// ninecell hash and its x-hash primitives went with it.
const HL = _isNode ? require('./health-lib.js') : window.HealthLib;
const { makeWeights, uh, xh4, ninecellId,
        chainsOf, chainHealthAll, resolveHealthModel } = HL;


// ── Constants ─────────────────────────────────────────────────────────────────

// Cell state encoding (signed, color-canonicalized):
//   0            = empty
//  +1 .. +maxLibs = BLACK stone with that many liberties (capped)
//  -1 .. -maxLibs = WHITE stone with that many liberties (capped)





// Fold the spec tag ((maxLibs << 3) | sizeCode) into a window hash so
// different spec spaces cannot collide in the shared weight map.
function mixTag(h, tag) {
  return uh(h, tag);
}
// Spec size → 3-bit tag code.  Sizes 1-4 are themselves; the rectangle pairs
// take the free codes — 34 (the 3×4 ∪ 4×3 pair) is 5, 23 (the 2×3 ∪ 3×2 pair)
// is 7 (6 is currently unused).  maxLibs 0 is the
// LADDER-CODED family ('size:L' in the trainers): raw is vlibpat's 7-state
// turn-independent tactical alphabet (0 empty, ±1 alive, ±2 dead, ±3
// unsettled) instead of capped liberty counts — structurally identical to
// an ml=3 encoding, so the whole plane/hash/34 machinery is shared.
// Render a spec back to its command-line token — the inverse of the trainers'
// --spec parser, so what a run prints can be pasted into the next one.
function specToken(sp) {
  const ph = sp.phaseBins > 1 ? 'p' + sp.phaseBins : '';
  if (sp.turn) return 't' + ph;
  const body = sp.maxLibs === 0 ? 'L'
             : sp.maxLibs < 0 ? 'H' + (-sp.maxLibs)
             : String(sp.maxLibs);
  return sp.size + ':' + body + ph;
}
function specString(specs) { return specs.map(specToken).join(','); }

function sizeCode(size) { return size === 34 ? 5 : size === 23 ? 7 : size; }
// The turn family (spec {turn:true}, token 't') has no alphabet and no window, so it
// cannot share a tag base with a pattern spec.  tagBaseOf never returns 16 —
// non-negative maxLibs give 0-15 and health families give 17+ — so tag base 16
// is permanently free, and the family takes code 0 within it.
const TURN_TAG = 16 << 3;
const TURN_SALT = 0x5ce7a13b | 0;
function specTag(spec) {
  if (spec.turn) return TURN_TAG;
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

// 3×3 centre mix.  The recursive X-hash over the four corner 2×2 sub-windows is
// D4-exact but lossy at the top combine — distinct 3×3 shapes can collide.  The
// centre cell sits in all four corner 2×2's yet the combine cannot isolate it,
// so multiplying the combined hash by (odd base + centre leaf) folds it back in:
// ml=3 3×3 fidelity 97.4% → 98.5% (collisions −43%) for one imul.  Ordered — the
// centre is a fixed, distinguished slot, not a symmetric operand, so this is
// cheaper than a uh() and holds the same fidelity.  Baked into the stored h3
// plane, so 4×4 / 3×4 inherit it.  Odd base keeps the multiply near-bijective
// (only centre leaf 1 makes the factor even).
const _NC3_CTR_MIX = 2649461;

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

// Non-speculative fast path.  When cells have NOT been speculatively mutated
// (doSetNext is false), the game's incremental chain structures are current, so
// a stone's liberty count is simply _ls[_gid[i]] — one array read per cell, no
// flood (the same source rawState() already trusts).  Used for every real-
// position extraction (search base, playout-truncation eval, training); the
// flood in _cellLibCounts is kept only for the doSetNext path, where the
// speculative cells make _gid/_ls stale.
function _cellLibCountsFast(game, out) {
  const cells = game.cells, gid = game._gid, ls = game._ls, cap = game.N * game.N;
  for (let i = 0; i < cap; i++) out[i] = cells[i] === 0 ? 0 : ls[gid[i]];
}
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
  // size 0 was the chain-attribute family (spec token 'C'), removed 2026-09-11.
  // Models trained with one still carry it in their specs array, and without
  // this they would fall through into the window machinery as a size-0 pattern
  // spec and score silently wrong.
  if (specs.some(sp => sp.size === 0)) {
    throw new Error('vpatterns: this model uses the chain-attribute family (spec C), which was ' +
                    'removed on 2026-09-11 — retrain it without the C term');
  }
  // size 5 was the turn family; it is now an explicit {turn:true} spec.  A model
  // that still carries {size:5} predates that change and would misload as a 5x5
  // window, so reject it rather than score silently wrong.
  if (specs.some(sp => sp.size === 5)) {
    throw new Error('vpatterns: size 5 was the turn family, now spec {turn:true} — ' +
                    'this model predates that change; retrain it');
  }
  const byMaxLibs = new Map();
  // Per-PATTERN-spec phase bins (token suffix pN on size:maxLibs): emitted
  // keys are salted by floor(phase * N), giving each spec its own phase-
  // conditioned weight planes.  Non-incremental (bucket crossings invalidate
  // every key): deltaZ and doSetNext refuse.
  const patPhaseBins = new Int32Array(256);
  let hasPhasedPatterns = false;
  for (const sp of specs) {
    if (!sp.turn && sp.size !== 0 && sp.phaseBins > 1) {
      patPhaseBins[(tagBaseOf(sp.maxLibs) << 3) | sizeCode(sp.size)] = sp.phaseBins;
      hasPhasedPatterns = true;
    }
  }
  // The TURN feature (spec {turn:true}): one antisymmetric feature per position, +1 when
  // BLACK is to move, keyed by phase bucket.  z has no tempo term otherwise,
  // and the value of holding the move plainly varies with fullness.  It lives
  // at its own tag base, so it registers its bins directly.
  const turnSpec = specs.find(sp => sp.turn);
  const hasTurn = turnSpec !== undefined;
  const turnPhaseBins = hasTurn ? (turnSpec.phaseBins || 1) : 1;
  if (turnPhaseBins > 1) { patPhaseBins[TURN_TAG] = turnPhaseBins; hasPhasedPatterns = true; }
  for (const spec of specs) {
    if (spec.turn) continue;
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
  // 'size:H<N>' families.
  const healthModel = hasHealth ? resolveHealthModel(opts && opts.health) : null;
  return { byMaxLibs, sortedMaxLibs, healthModel,
           totalSizes: totalSizes + (hasTurn ? 1 : 0),
           hasLadder, hasHealth, patPhaseBins, hasPhasedPatterns,
           hasTurn };
}

// and returns a flat array of { key, polarity } for all matching patterns.
//
// Optimisations vs calling pattern1/2/3 individually:
//   - Raw cell states are precomputed once per unique maxLibs value.
//   - size:2 and size:3 hash via whole-board X-hash planes (see above).
//   - pattern1 is inlined (raw[idx] already holds the capped liberty count).
// `game3` (optional): a caller-supplied Game3 already synced to `game`, reused
// for the ladder-code pass instead of rebuilding one with game3FromGame2.  The
// ladder read borrows it non-destructively (play/undo balanced), so it comes
// back unchanged.  Must match `game`.
// `game3RebuildOk` (optional): set by callers that legitimately have no Game3
// to supply (e.g. trainers replaying a Game2), acknowledging the ladder-pass
// rebuild so it stays silent.  The warning is then reserved for UNacknowledged
// rebuilds — a caller that should have passed a synced Game3.
// Whether extractFeatures on these prepared specs must build a Game3: only the
// ladder-coded family (size:L) needs one — liberty/health/turn specs read the
// Game2 directly.  A caller doing per-candidate evaluation can build ONE Game3
// and pass it as `game3` (advancing it with play/undo) instead of rebuilding.
function needsGame3(prepSpecs) { return !!(prepSpecs && prepSpecs.hasLadder); }

function extractFeatures(game, prepSpecs, doSetNext, nextMove, reuse, game3, game3RebuildOk) {
  const cells = game.cells;
  const cap   = game.N * game.N;
  const N     = game.N;

  // Flat typed output arrays; max one feature per cell per size entry.  A hot
  // inference caller that consumes the result before its NEXT extract on this
  // prepSpecs can pass reuse=true to draw them from a scratch buffer kept on
  // prepSpecs — no per-call allocation, no GC churn.  Only up to `count` is ever
  // read, so stale tail entries from a prior call are harmless; base and
  // fallback searches use different prepSpecs, so their buffers never collide.
  // Default (reuse falsy) allocates fresh, safe for callers that RETAIN the
  // result across later extracts (the trainer's bias pairs / eager test cache).
  const maxF   = cap * prepSpecs.totalSizes;
  let outKeys, outPols, outTags;
  if (reuse) {
    let sc = prepSpecs._out;
    if (!sc || sc.keys.length < maxF) {
      sc = { keys: new Int32Array(maxF), pols: new Int8Array(maxF), tags: new Int16Array(maxF) };
      prepSpecs._out = sc;
    }
    outKeys = sc.keys; outPols = sc.pols; outTags = sc.tags;
  } else {
    outKeys = new Int32Array(maxF);
    outPols = new Int8Array(maxF);
    outTags = new Int16Array(maxF);
  }
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
  // A speculative mutation is the position AFTER the move, where the side to
  // move is the opponent — game.current still says otherwise.
  if (doSetNext && prepSpecs.hasTurn) {
    throw new Error('vpatterns: the turn spec (t) does not support speculative extraction (doSetNext)');
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
      if (game3 && game3.emptyCount !== game.emptyCount)
        throw new Error('vpatterns.extractFeatures: supplied game3 does not match game');
      if (!game3 && !game3RebuildOk && !_warnedLadderRebuild) {
        _warnedLadderRebuild = true;
        console.error('vpatterns.extractFeatures: no game3 supplied — building one for the ladder pass ' +
          '(slow path; pass a synced game3 to reuse it).  First occurrence:\n' +
          (new Error().stack || '').split('\n').slice(2, 7).join('\n'));
      }
      raw = VLibPat.computeLadderCodes(game3 || game3FromGame2(game), null);
    } else if (isHealth) {
      // One survival probability per chain, bucketed to 1..hb and signed by
      // colour — the same alphabet shape as liberty counts, but the levels
      // mean "how likely is this chain to live" instead of "how many
      // liberties".  Uses the game's incremental chain structures, so this
      // family cannot run under doSetNext.
      raw = new Int8Array(cap);
      if (survChains === null) {
        // The health system takes the games and returns each chain with its
        // survival probability in .p (and its stone list, for the bucket
        // painting below); it builds a Game3 itself if a ladder2 model needs one.
        survChains = chainHealthAll(prepSpecs.healthModel, game, game3, game3RebuildOk);
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
        // Liberty counts.  Under doSetNext the cells are speculatively mutated
        // and the incremental _gid/_ls are stale, so flood from cells alone;
        // otherwise read the current incremental structures directly.
        const libs = _libCounts.length >= cap ? _libCounts : (_libCounts = new Int32Array(cap));
        if (doSetNext) _cellLibCounts(game, libs);
        else           _cellLibCountsFast(game, libs);
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
            // Fold the centre cell (r1+x1) back in — see _NC3_CTR_MIX.
            const kN = Math.imul(xh4(h2N[r0+x], h2N[r0+x1], h2N[r1+x], h2N[r1+x1]), _NC3_CTR_MIX + lN[r1 + x1]);
            const kI = Math.imul(xh4(h2I[r0+x], h2I[r0+x1], h2I[r1+x], h2I[r1+x1]), _NC3_CTR_MIX + lI[r1 + x1]);
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
  // Turn: one feature for the whole position, polarity by side to move, keyed
  // by phase bucket.  A colour flip flips the sign, so it obeys the same
  // antisymmetric convention the pattern families do.
  if (prepSpecs.hasTurn) {
    outKeys[count] = mixTag(TURN_SALT, TURN_TAG) ^ phSalt[TURN_TAG];
    outPols[count] = game.current === BLACK ? 1 : -1;
    outTags[count] = TURN_TAG;
    count++;
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
  if (prepSpecs.hasHealth) {
    throw new Error('vpatterns deltaZ: health-coded specs (size:H<N>) are not incremental');
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
        // Fold the centre cell (rD+cR) back in — see _NC3_CTR_MIX; the centre
        // may itself be dirty, so read it through the override-aware leaf.
        const kN = Math.imul(xh4(h2at(r0 + ac), h2at(r0 + cR), h2at(rD + ac), h2at(rD + cR)), _NC3_CTR_MIX + leafN(rD + cR));
        const kI = Math.imul(xh4(h2atI(r0 + ac), h2atI(r0 + cR), h2atI(rD + ac), h2atI(rD + cR)), _NC3_CTR_MIX + leafI(rD + cR));
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
// A caller with a Game3 already synced to `game` (e.g. a search maintaining one)
// may pass it to avoid a ladder-spec rebuild.  Without one, the rebuild here is
// the expected cost of the one-call convenience, so it is acknowledged rather
// than warned (callers wanting reuse use extractFeatures with a game3 directly).
function evaluate(game, model, game3) {
  return evaluateFeatures(extractFeatures(game, model.preparedSpecs, false, undefined, true, game3, !game3), model.weights);
}

// ── Persistence ───────────────────────────────────────────────────────────────

// Loads a model JS file and returns { weights: Map<number,float>, specs: [...] }.
// Always returns a fresh copy so multiple callers don't share the same Map.
// health: path to (or already-loaded) health model, required when the specs
// are health-coded or use the C survival attribute.
function loadWeights(filePath, health) {
  const raw = require(require('path').resolve(filePath));
  return modelFromRaw(raw, health, filePath);
}

// Build a runtime model from an already-loaded raw object (the module export a
// vpat file produces, or the `vpat` field embedded in a featurepol file), rather
// than from a file path.  Same processing as loadWeights: a fresh makeWeights
// table so callers don't share a Map, prepared specs, and the health-alphabet
// check when the model is health-coded.
function modelFromRaw(raw, health, srcName) {
  const specs = raw.specs;
  const weights = makeWeights(Math.max(1024, (raw.weights.size ?? raw.weights.length) * 2));
  for (const [k, v] of raw.weights) weights.set(k, v);
  // Prefer the health model embedded in the file (it travels with the vpat model
  // it was trained against, so loading needs no external HEALTH_DATA); fall back
  // to the caller-supplied one for older files that recorded only parameters.
  const preparedSpecs = prepareSpecs(specs, { health: raw.healthModel || health });
  // Only sanity-check an EXTERNAL pairing (older files with a `health` params
  // block): an embedded model is by definition the one training used.
  if (!raw.healthModel && raw.health) checkHealthMatch(raw.health, srcName || '<embedded>', preparedSpecs.healthModel);
  return { specs, preparedSpecs, weights, komi: raw.komi, trunc: raw.trunc };
}

// A health-coded model's keys are bucket indices produced by whichever health
// model was loaded when it trained, so pairing it with a different one
// re-indexes every weight — no error, just wrong.  We do NOT match file names
// (a model may legitimately be copied or renamed); we compare the parameters
// the training run recorded against the health model now in use and warn loudly
// on any difference — alphabet or bucket boundary — but do not block: the caller
// may knowingly be pairing them.
function checkHealthMatch(want, filePath, got) {
  const diff = [];
  for (const k of ['maxLibs', 'maxJoinLibs', 'friendHealthMaxBuckets',
                   'foeHealthMinBuckets', 'stoneSalt',
                   'minPhase', 'maxPhase', 'delta', 'iterations', 'initHealth', 'bias', 'nWeights']) {
    if (want[k] !== undefined && got[k] !== undefined && want[k] !== got[k]) {
      diff.push(`${k}: trained ${want[k]}, loaded ${got[k]}`);
    }
  }
  if (diff.length) {
    console.error(`vpatterns WARNING: ${filePath} was trained against a DIFFERENT health model ` +
                  `(${diff.join('; ')}) — its weights may be mis-indexed against the one now loaded.`);
  }
}

// The `{ specs, weights: new Map(...), komi, healthModel?, trunc? }` object literal
// for a model — the payload of a vpat file, and also what a featurepol file
// embeds under its `vpat` field so the rank model travels with the policy it was
// trained against.
function modelLiteral(model) {
  const specStr = JSON.stringify(model.specs);
  // Health-coded models EMBED the full health model they were trained against,
  // so the vpat weights' key space travels with it: loading needs no external
  // HEALTH_DATA and cannot be paired with a mismatched health model (see
  // modelFromRaw).  Reconstructed on load via resolveHealthModel.
  let healthStr = '';
  if (model.specs.some(sp => sp.maxLibs < 0)) {
    healthStr = `, healthModel: ${HL.modelLiteral(model.preparedSpecs.healthModel)}`;
  }
  // Truncation default for consumers (puct-ppat-fp-trunc, mc-ppat): the delta
  // the model was fitted for, so a model carries its own inference config
  // instead of it being passed alongside every time.  An env var still overrides.
  let truncStr = '';
  if (model.trunc && model.trunc.delta != null) {
    truncStr = `, trunc: { delta: ${model.trunc.delta} }`;
  }
  const pairs = [];
  // Skip weights that quantize to zero: they read back as 0 anyway (a missing
  // key looks up to 0 in every consumer), so writing them is pure file bloat.
  model.weights.forEach((k, v) => { const q = +v.toFixed(6); if (q !== 0) pairs.push(`[${k},${q}]`); });
  const weightsStr = '[' + pairs.join(',') + ']';
  return `{ specs: ${specStr}${healthStr}${truncStr}, weights: new Map(${weightsStr})` +
         (model.komi !== undefined ? `, komi: ${model.komi}` : '') + ` }`;
}

// Writes a model { weights, specs } to a JS file (browser-includable).
function saveWeights(filePath, model) {
  const fs = require('fs');
  const src = [
    "'use strict';",
    '// Auto-generated by train-vpatterns.js — do not edit by hand.',
    `const vpatternsModel = ${modelLiteral(model)};`,
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
  needsGame3,
  evaluateFeatures,
  evaluate,
  deltaZ,
  loadWeights,
  modelFromRaw,
  modelLiteral,
  saveWeights,
  specToken,
  specString,
  ninecellId,
  // Key primitives, exposed for offline tooling (vpat-fold) that must
  // reproduce the exact output keys a size-2/size-3 extraction emits.
  mixTag,
  tagBaseOf,
  sizeCode,
};

if (typeof module !== 'undefined') module.exports = Patterns;
else window.VPatterns = Patterns;

})();
