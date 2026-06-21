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
//               stones8/stones12 are adaptive: if the inner ring holds <2 stones
//               it expands to the next ring (8→12, 12→20), hash-perturbed so it
//               stays distinct, for more context.
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
//   ladderStatus 4-bit ladder presence mask at the move from ladder2
//               (urgent-kill/urgent-save/wasted-extend/wasted-attack)
//   urgentKill<n> / urgentSave<n> / wastedExtend<n> / wastedAttack<n>
//               one ladder flag's summed chain stone-count (cumulative, up to n)
//
// BROWSER-COMPATIBLE: no Node-only APIs at top level.

(function () {

const Util = (typeof require === 'function') ? require('./util.js') : window.Util;
const { PASS }                 = Util.load('./game2.js', 'Game2');
const { game3FromGame2 }       = Util.load('./game3.js', 'Game3');
const { getAllLadderStatuses } = Util.load('./ladder2.js', 'Ladder2');

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

// Canonical encoding of the first n cell values in cv: min base-3 value over the
// 8 D4 symmetries (so the result is D4-invariant).  n must be a closed prefix.
function _canonStones(cv, n) {
  let best = Infinity;
  for (let s = 0; s < 8; s++) {
    const po = s * NEAR_MAX;
    let raw = 0;
    for (let i = 0; i < n; i++) raw = raw * 3 + cv[_NEAR_PERM[po + i]];
    if (raw < best) best = raw;
  }
  return best;
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

// Build one feature term { str, kind, param, salt, evalFn, needsLadder } from a
// token like "capture6" / "stones4" / "ladderStatus".
function _makeTerm(str) {
  const m = /^([a-zA-Z]+)(\d*)$/.exec(str);
  if (!m) throw new Error(`featurepol: bad feature term "${str}"`);
  const kind = m[1];
  const param = m[2] ? parseInt(m[2], 10) : null;
  const salt = _hashStr(str);
  // A size<n> term is CUMULATIVE: it carries sizeFn (the raw size) and is expanded
  // by parseSpec into n additive "≥k present" indicator spaces (thermometer
  // encoding) so the logit sums over size, like npat's tactical slots.  Other
  // terms set evalFn directly (one value → one key).
  let evalFn = null, sizeFn = null, cumulative = false, needsLadder = false;
  switch (kind) {
    case 'stones': {
      if (!_STONES_N.has(param)) throw new Error(`featurepol: stones<n> needs n in {4,8,12,20} (D4-closed), got "${str}"`);
      const n = param;
      if (n === 8 || n === 12) {
        // Adaptive: if the inner n cells hold fewer than 2 stones (little nearby
        // info), expand to the next ring (8→12, 12→20).  The expanded encoding is
        // run through the hash mixer (seeded with the term's existing integer
        // salt) so it avalanches away from any plain stones<n> encoding — no
        // carved-out value ranges, just entropy.
        const expandN = n === 8 ? 12 : 20;
        evalFn = (ctx, idx) => {
          const nn = ctx.nearNbr, base = idx * NEAR_MAX, cells = ctx.game.cells, cur = ctx.cur, cv = _cvScratch;
          let nStones = 0;
          for (let i = 0; i < n; i++) { const c = cells[nn[base + i]]; cv[i] = c === 0 ? 0 : c === cur ? 1 : 2; if (cv[i]) nStones++; }
          if (nStones >= 2) return _canonStones(cv, n);
          for (let i = n; i < expandN; i++) { const c = cells[nn[base + i]]; cv[i] = c === 0 ? 0 : c === cur ? 1 : 2; }
          return _hashCombine(salt, _canonStones(cv, expandN));
        };
      } else {
        evalFn = (ctx, idx) => {
          const nn = ctx.nearNbr, base = idx * NEAR_MAX, cells = ctx.game.cells, cur = ctx.cur, cv = _cvScratch;
          for (let i = 0; i < n; i++) { const c = cells[nn[base + i]]; cv[i] = c === 0 ? 0 : c === cur ? 1 : 2; }
          return _canonStones(cv, n);
        };
      }
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
    case 'ko': {
      // Binary: 1 iff the move creates a ko (captures one lone stone into a ko shape).
      evalFn = (ctx, idx) => (ctx.game.createsKo(idx) ? 1 : 0);
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
  return { str, kind, param, salt, evalFn, sizeFn, cumulative, maxLevel: param, needsLadder };
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
  const slotOf = new Map();   // term salt → memo slot
  const computers = [];       // computers[slot] = value fn (sizeFn for cumulative, else evalFn)
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
      const slot = slotFor(t);
      if (t.cumulative)       { gate.push(slot); cumTerms.push({ salt: t.salt, slot, maxLevel: t.maxLevel }); }
      else if (t.kind === 'ko') { gate.push(slot); baseTerms.push({ salt: t.salt, slot, bin: true }); }
      else                    { baseTerms.push({ salt: t.salt, slot, bin: false }); }
    }
    let maxKeys = 1;
    for (const c of cumTerms) maxKeys *= c.maxLevel;
    spaces.push({ str: spaceStr, salt: _hashStr('space:' + spaceStr), gate, baseTerms, cumTerms, maxKeys });
  }
  if (spaces.length === 0) throw new Error(`featurepol: no feature spaces in "${str}"`);
  let maxKeysPerMove = 0;
  for (const sp of spaces) maxKeysPerMove += sp.maxKeys;
  return { str, spaces, computers, numSlots: computers.length, maxKeysPerMove, needsLadder };
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
    delta: new Float32Array(initialCapacity),  // reusable reinforce buffer
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
  // Toroidal nearest-cell table: nearNbr[idx*NEAR_MAX + k] = flat index of the
  // k-th nearest cell to idx.
  const nearNbr = new Int32Array(cap * NEAR_MAX);
  for (let idx = 0; idx < cap; idx++) {
    const r = (idx / N) | 0, c = idx - r * N, base = idx * NEAR_MAX;
    for (let k = 0; k < NEAR_MAX; k++) {
      const dy = _NEAR_OFFSETS[k][0], dx = _NEAR_OFFSETS[k][1];
      nearNbr[base + k] = _wrap(r + dy, N) * N + _wrap(c + dx, N);
    }
  }
  return {
    N,
    nearNbr,
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
  const ctx = { game, cur: game.current, nearNbr: state.nearNbr, ladderSizes: null, memo };
  if (spec.needsLadder) {
    const g3 = game3 || game3FromGame2(game);
    ctx.ladderSizes = _buildLadderSizes(game, g3, state.ladderSizes);
  }
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

// ── REINFORCE ─────────────────────────────────────────────────────────────────
//
// Each feature space contributes exactly one key per move, so the gradient of
// log π for each touched weight is feat_chosen − Σ_i π_i feat_i.  weightDecay
// applies decoupled L2 shrink (w ← w + Δw − lr·decay·w).

// Optional `stats` (an object with absSum/count) accumulates |weight| and a
// count for every genuine weight update (frequency-weighted over the run), for
// the trainer's avgW column.
function reinforceUpdate(state, chosenIndex, advantage, weights, lr, weightDecay = 0, stats) {
  const n = state.count;
  if (n === 0 || chosenIndex < 0) return 0;
  const step = lr * advantage;
  if (step === 0) return 0;
  const decayStep = lr * weightDecay;
  const keys = state.keys, keyOff = state.keyOff, probs = state.probs;
  const vals = weights.vals, delta = weights.delta, touched = state.touched;
  let tc = 0;
  for (let k = keyOff[chosenIndex], e = keyOff[chosenIndex + 1]; k < e; k++) {
    const idx = keys[k]; touched[tc++] = idx; delta[idx] += step;
  }
  for (let i = 0; i < n; i++) {
    const pi = probs[i];
    if (pi === 0) continue;
    const sub = step * pi;
    for (let k = keyOff[i], e = keyOff[i + 1]; k < e; k++) {
      const idx = keys[k]; touched[tc++] = idx; delta[idx] -= sub;
    }
  }
  // Apply each touched weight's net delta (+ decoupled decay) exactly once.  The
  // `d !== 0` guard (matching npat) skips duplicate visits — delta is zeroed on
  // the first — so a feature shared across moves is decayed once, not once per
  // occurrence.  (A touched weight whose net delta is 0 isn't decayed this step.)
  for (let i = 0; i < tc; i++) {
    const idx = touched[i], d = delta[idx];
    if (d !== 0) {
      vals[idx] += d - decayStep * vals[idx];
      delta[idx] = 0;
      if (stats) { stats.absSum += Math.abs(vals[idx]); stats.count++; }
    }
  }
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
    '// Auto-generated by train-featurepol.js — do not edit by hand.',
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
  policyMove,
  greedyMove,
  reinforceUpdate,
  modelWeights,
  serialize,
  loadModel,
  NEAR_MAX,
  // exposed for tests
  _hashStr, _captureCount, _atariStones,
};

if (typeof module !== 'undefined') module.exports = FeaturePol;
else window.FeaturePol = FeaturePol;

})();
