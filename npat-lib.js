'use strict';

// npat-lib.js — "nine-pattern" policy features with ladder-aware cell encoding.
//
// For each candidate move, we extract the NINE 3×3 windows that overlap the
// candidate point.  Each window contributes a linear feature indexed by a
// canonical pattern key.  The move's logit is the sum of those 9 feature
// weights; moves are sampled by softmax over logits.
//
// Cell encoding (mover-relative, 6 values):
//   0 = EMPTY           empty, not an urgent liberty
//   1 = EMPTY_URGENT    empty; playing here resolves some ladder (saves a
//                       friend chain or kills a foe chain).  Comes from
//                       ladder2.getAllLadderStatuses(...).urgentLibs.
//   2 = FRIEND          own stone, chain is not in a tactical ladder
//   3 = FRIEND_URGENT   own stone, chain has 1–2 liberties AND the mover has
//                       a move that saves it (ladder status: urgentLibs ≠ [])
//   4 = FOE             opponent stone, chain is not in a tactical ladder
//   5 = FOE_URGENT      opponent stone, chain has 1–2 liberties AND the mover
//                       has a move that kills it
//
// The board is toroidal (game2's neighbour tables wrap), so there is no
// off-board state to encode.
//
// Raw pattern index for a single window:
//   raw = relPos * 6^9 + Σ_i cells[i] * 6^i                        (i = 0..8)
// where relPos ∈ 0..8 is the candidate's position within the window.
// Range: [0, 9 * 10_077_696) = [0, 90_699_264).  Fits in 32 bits.
//
// Canonical key: min over 8 D4 symmetries of the transformed raw.  Under σ,
// new_cells[σ(i)] = cells[i] and new_relPos = σ(relPos).  This is O(8×9) per
// window and happens at extraction time; no global canonical-ID table is
// precomputed.  Sparse weights live in Map<rawCanon, weight>.
//
// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
const { PASS }          = _isNode ? require('./game2.js')  : window.Game2;
const { game3FromGame2 } = _isNode ? require('./game3.js') : window.Game3;
const { getAllLadderStatuses } = _isNode ? require('./ladder2.js') : window.Ladder2;

// ── D4 position permutations on a 3×3 grid ────────────────────────────────────
// Positions linearised as i = row*3 + col, row,col ∈ {0,1,2}.
// perm[i] is the new position of the value originally at position i.

function _mkPerm(fn) {
  const p = new Int32Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
    const [nr, nc] = fn(r, c);
    p[r * 3 + c] = nr * 3 + nc;
  }
  return p;
}

const _D4 = [
  _mkPerm((r, c) => [r, c]),         // 0: Identity
  _mkPerm((r, c) => [c, 2 - r]),     // 1: Rot90CW
  _mkPerm((r, c) => [2 - r, 2 - c]), // 2: Rot180
  _mkPerm((r, c) => [2 - c, r]),     // 3: Rot270CW
  _mkPerm((r, c) => [r, 2 - c]),     // 4: FlipH
  _mkPerm((r, c) => [2 - r, c]),     // 5: FlipV
  _mkPerm((r, c) => [c, r]),         // 6: TransposeMD
  _mkPerm((r, c) => [2 - c, 2 - r]), // 7: TransposeAD
];

// ── Encoding constants ───────────────────────────────────────────────────────

const CELL_BASE    = 6;
const CELLS_BASE   = 10077696; // 6^9

// Cell-state labels (exported for readability in other modules / tests).
const CELL_EMPTY          = 0;
const CELL_EMPTY_URGENT   = 1;
const CELL_FRIEND         = 2;
const CELL_FRIEND_URGENT  = 3;
const CELL_FOE            = 4;
const CELL_FOE_URGENT     = 5;

// ── Ladder-status annotation ─────────────────────────────────────────────────
//
// Build two per-cell flag arrays for the current game position:
//   stoneUrgent[idx] = 1 if the stone at idx belongs to a chain whose ladder
//                      status has non-empty urgentLibs (i.e. mover can resolve
//                      it with a single move), else 0.
//   libUrgent[idx]   = 1 if the empty cell at idx appears in some chain's
//                      urgentLibs, else 0.
//
// Both arrays are Uint8Array(N*N), reused across calls when provided.
//
// Uses ladder2's Game3-based analysis: convert the Game2 board to Game3, run
// getAllLadderStatuses, then walk each urgent chain's stones by iterating the
// chain's group bitset in Game3.

function annotateLadders(game, out, game3) {
  const N   = game.N;
  const cap = N * N;

  // Lazy-allocate output buffers.
  if (!out) out = { stoneUrgent: new Uint8Array(cap), libUrgent: new Uint8Array(cap) };
  out.stoneUrgent.fill(0);
  out.libUrgent.fill(0);

  // Empty-board fast path.
  if (game.emptyCount === cap) return out;

  // Game3 view for ladder2.  Callers that maintain a Game3 in lockstep with
  // `game` can pass it in to skip the rebuild; otherwise we fall back to
  // game3FromGame2 (which preserves `current`).
  const g3 = game3 || game3FromGame2(game);
  const infos = getAllLadderStatuses(g3);

  for (const info of infos) {
    if (!info.status) continue;
    const { urgentLibs } = info.status;
    if (urgentLibs.length === 0) continue;

    // Mark urgent liberty cells (empty points).
    for (const lib of urgentLibs) out.libUrgent[lib] = 1;

    // Mark every stone of the urgent chain.  Game3 stores group stones in a
    // bitset at _sw[gid * W + wi]; walk it here.
    const gid = info.gid;
    const W   = g3._W;
    const sw  = g3._sw;
    const gb  = gid * W;
    for (let wi = 0; wi < W; wi++) {
      let w = sw[gb + wi];
      while (w) {
        const lsb = w & -w;
        const si = wi * 32 + (31 - Math.clz32(lsb));
        if (si < cap) out.stoneUrgent[si] = 1;
        w ^= lsb;
      }
    }
  }

  return out;
}

// ── State ─────────────────────────────────────────────────────────────────────
//
// moves   [count]        flat board index of move i
// patIds  [count * 9]    canonical pattern keys per window (w ∈ 0..8)
// logits  [count]        scratch buffer for softmax
// probs   [count]        scratch buffer for softmax
// ladder  { stoneUrgent, libUrgent }   reusable ladder-annotation buffers
//
// Window order w = (wr+2)*3 + (wc+2) where (wr, wc) ∈ {-2,-1,0}² is the
// offset from the candidate to the window's top-left corner.

function createState(N) {
  const cap = N * N;
  // Precomputed toroidal 5×5 neighbour table.  For each board index idx and
  // each offset (pr, pc) ∈ [-2, 2]², patchNbr[idx*25 + (pr+2)*5 + (pc+2)]
  // gives the flat board index of (r+pr mod N, c+pc mod N).  This replaces
  // 25 `_wrap` calls per candidate in extractFeatures with one indexed load.
  const patchNbr = new Int32Array(cap * 25);
  for (let idx = 0; idx < cap; idx++) {
    const r = (idx / N) | 0;
    const c = idx - r * N;
    const base = idx * 25;
    for (let pr = -2; pr <= 2; pr++) {
      const br = _wrap(r + pr, N);
      const rowBase = br * N;
      const prBase  = (pr + 2) * 5;
      for (let pc = -2; pc <= 2; pc++) {
        patchNbr[base + prBase + (pc + 2)] = rowBase + _wrap(c + pc, N);
      }
    }
  }
  return {
    N,
    moves:    new Int32Array(cap),
    patIds:   new Int32Array(cap * 9),
    logits:   new Float64Array(cap),
    probs:    new Float64Array(cap),
    ladder:   { stoneUrgent: new Uint8Array(cap), libUrgent: new Uint8Array(cap) },
    patchNbr,
    count:    0,
  };
}

// ── Runtime D4 canonicalisation ──────────────────────────────────────────────
//
// canonKey(relPos, cells[9]) returns the minimum raw-int over 8 D4 symmetries.
//
// For each D4 permutation σ we have tCells[σ(i)] = cells[i], so
//   enc_σ = Σ_j tCells[j] · 6^j = Σ_i cells[i] · 6^{σ(i)}
// and the full raw is σ(relPos) · CELLS_BASE + enc_σ.  Precompute:
//   _D4cw[σ*9 + i]      = 6^{σ(i)}               (cell weight)
//   _D4rp[σ*9 + relPos] = σ(relPos) · CELLS_BASE (relPos offset)
// so each permutation becomes a flat 9-term dot product plus one add.

const _D4cw = new Int32Array(72);
const _D4rp = new Int32Array(72);
(function () {
  const pow6 = new Int32Array(10);
  pow6[0] = 1;
  for (let k = 1; k < 10; k++) pow6[k] = pow6[k - 1] * CELL_BASE;
  for (let di = 0; di < 8; di++) {
    const perm = _D4[di];
    for (let i = 0; i < 9; i++) {
      _D4cw[di * 9 + i] = pow6[perm[i]];
      _D4rp[di * 9 + i] = perm[i] * CELLS_BASE;
    }
  }
})();

function canonKey(relPos, cells) {
  const c0 = cells[0], c1 = cells[1], c2 = cells[2];
  const c3 = cells[3], c4 = cells[4], c5 = cells[5];
  const c6 = cells[6], c7 = cells[7], c8 = cells[8];
  const cw = _D4cw, rp = _D4rp;
  let best = 0x7fffffff;
  for (let di = 0; di < 8; di++) {
    const o = di * 9;
    const raw = rp[o + relPos]
      + c0 * cw[o]     + c1 * cw[o + 1] + c2 * cw[o + 2]
      + c3 * cw[o + 3] + c4 * cw[o + 4] + c5 * cw[o + 5]
      + c6 * cw[o + 6] + c7 * cw[o + 7] + c8 * cw[o + 8];
    if (raw < best) best = raw;
  }
  return best;
}

const _windowCells  = new Int32Array(9);
const _patch5       = new Int32Array(25);

// ── Core feature extraction ───────────────────────────────────────────────────
//
// Toroidal wrap — game2's _nbr only covers ±1 offsets, but we need ±2.

function _wrap(x, N) { x %= N; return x < 0 ? x + N : x; }

// Fill state with 9 canonical pattern keys for every legal non-true-eye move.
// Optional ladderInfo skips re-running the ladder analysis (useful if the
// caller already computed it for this board).  Optional game3 is a Game3
// kept in lockstep with `game`; passing it avoids a game3FromGame2 rebuild.
function extractFeatures(game, state, ladderInfo, game3) {
  const N      = game.N;
  const cap    = N * N;
  const cells  = game.cells;
  const cur    = game.current;
  const emC    = game._emptyCells;
  const ec     = game.emptyCount;

  const li = ladderInfo || annotateLadders(game, state.ladder, game3);
  const stoneUrgent = li.stoneUrgent;
  const libUrgent   = li.libUrgent;

  let count = 0;
  const moves    = state.moves;
  const patIds   = state.patIds;
  const patchNbr = state.patchNbr;
  const wc9      = _windowCells;
  const patch    = _patch5;

  for (let ei = 0; ei < ec; ei++) {
    const idx = emC[ei];
    if (!game.isLegal(idx) || game.isTrueEye(idx)) continue;

    // Build the 5×5 patch of encoded cell values centred on the candidate.
    // patchNbr pre-stores the 25 toroidal flat indices for each idx; 25
    // indexed loads replace 25 _wrap calls.
    const nbrBase = idx * 25;
    for (let p = 0; p < 25; p++) {
      const bi = patchNbr[nbrBase + p];
      const ci = cells[bi];
      let v;
      if (ci === 0) {
        v = libUrgent[bi] ? CELL_EMPTY_URGENT : CELL_EMPTY;
      } else if (ci === cur) {
        v = stoneUrgent[bi] ? CELL_FRIEND_URGENT : CELL_FRIEND;
      } else {
        v = stoneUrgent[bi] ? CELL_FOE_URGENT : CELL_FOE;
      }
      patch[p] = v;
    }

    const off = count * 9;
    for (let wr = -2; wr <= 0; wr++) {
      for (let wc = -2; wc <= 0; wc++) {
        const relRow = -wr;
        const relCol = -wc;
        const relPos = relRow * 3 + relCol;

        // Fill wc9 from the patch — no board indexing at all here.
        for (let i = 0; i < 9; i++) {
          const dr = (i / 3) | 0;
          const dc = i - dr * 3;
          wc9[i] = patch[(wr + dr + 2) * 5 + (wc + dc + 2)];
        }

        patIds[off + (wr + 2) * 3 + (wc + 2)] = canonKey(relPos, wc9);
      }
    }

    moves[count] = idx;
    count++;
  }

  state.count = count;
}

// ── Scoring helpers ───────────────────────────────────────────────────────────

function _score(patIds, i, weights) {
  const off = i * 9;
  const isMap = weights instanceof Map;
  let s = 0;
  if (isMap) {
    for (let w = 0; w < 9; w++) s += weights.get(patIds[off + w]) || 0;
  } else {
    for (let w = 0; w < 9; w++) s += weights[patIds[off + w]] || 0;
  }
  return s;
}

// Evaluate all moves and return them sorted by score descending.
function evaluate(game, state, weights) {
  extractFeatures(game, state);
  const out = [];
  for (let i = 0; i < state.count; i++) {
    out.push({ move: state.moves[i], score: _score(state.patIds, i, weights) });
  }
  return out.sort((a, b) => b.score - a.score);
}

function _computeSoftmax(state, weights) {
  const n   = state.count;
  if (n === 0) return 0;
  const pid = state.patIds;
  const lg  = state.logits;
  const pr  = state.probs;
  const isMap = weights instanceof Map;

  let maxL = -Infinity;
  for (let i = 0; i < n; i++) {
    const off = i * 9;
    let s = 0;
    if (isMap) {
      for (let w = 0; w < 9; w++) s += weights.get(pid[off + w]) || 0;
    } else {
      for (let w = 0; w < 9; w++) s += weights[pid[off + w]] || 0;
    }
    lg[i] = s;
    if (s > maxL) maxL = s;
  }
  let sum = 0;
  for (let i = 0; i < n; i++) { pr[i] = Math.exp(lg[i] - maxL); sum += pr[i]; }
  const inv = 1 / sum;
  for (let i = 0; i < n; i++) pr[i] *= inv;
  return n;
}

// Extract features, softmax over logits, sample an action.
// Returns { move, index, prob }.  Optional game3 kept in lockstep with game.
function policyMove(game, state, weights, rng, game3) {
  extractFeatures(game, state, undefined, game3);
  const n = state.count;
  if (n === 0) return { move: PASS, index: -1, prob: 1 };

  _computeSoftmax(state, weights);
  const probs = state.probs;

  const R = (rng || Math).random();
  let r = R, chosen = n - 1;
  for (let i = 0; i < n; i++) {
    r -= probs[i];
    if (r <= 0) { chosen = i; break; }
  }
  return { move: state.moves[chosen], index: chosen, prob: probs[chosen] };
}

// Greedy argmax move (no sampling).  Optional game3 kept in lockstep.
function greedyMove(game, state, weights, game3) {
  extractFeatures(game, state, undefined, game3);
  const n = state.count;
  if (n === 0) return PASS;
  let best = -Infinity, bestI = 0;
  for (let i = 0; i < n; i++) {
    const s = _score(state.patIds, i, weights);
    if (s > best) { best = s; bestI = i; }
  }
  return state.moves[bestI];
}

// ── REINFORCE step ────────────────────────────────────────────────────────────
//
// For each canonical pattern key k, ∂ log π_m / ∂ w[k] = count(k in f_m) −
// Σ_i π_i · count(k in f_i).  Update: Δw[k] = lr · adv · ∂ log π_m / ∂ w[k].
//
// state.probs must be populated (by policyMove / _computeSoftmax) before the
// call.

function reinforceUpdate(state, chosenIndex, advantage, weights, lr) {
  const n = state.count;
  if (n === 0 || chosenIndex < 0) return;

  const step  = lr * advantage;
  if (step === 0) return;

  const patIds = state.patIds;
  const probs  = state.probs;
  const isMap  = weights instanceof Map;
  const delta  = new Map();

  {
    const off = chosenIndex * 9;
    for (let k = 0; k < 9; k++) {
      const pid = patIds[off + k];
      delta.set(pid, (delta.get(pid) ?? 0) + step);
    }
  }
  for (let i = 0; i < n; i++) {
    const pi = probs[i];
    if (pi === 0) continue;
    const sub = step * pi;
    const off = i * 9;
    for (let k = 0; k < 9; k++) {
      const pid = patIds[off + k];
      delta.set(pid, (delta.get(pid) ?? 0) - sub);
    }
  }
  if (isMap) {
    for (const [pid, d] of delta) {
      if (d === 0) continue;
      weights.set(pid, (weights.get(pid) ?? 0) + d);
    }
  } else {
    for (const [pid, d] of delta) weights[pid] += d;
  }
}

// ── Module exports ────────────────────────────────────────────────────────────

const NPatterns = {
  createState,
  extractFeatures,
  evaluate,
  policyMove,
  greedyMove,
  reinforceUpdate,
  annotateLadders,
  canonKey,
  // Constants.
  CELL_BASE,
  CELLS_BASE,
  CELL_EMPTY, CELL_EMPTY_URGENT,
  CELL_FRIEND, CELL_FRIEND_URGENT,
  CELL_FOE, CELL_FOE_URGENT,
  // Exposed for tests.
  _D4,
};

if (typeof module !== 'undefined') module.exports = NPatterns;
else window.NPatterns = NPatterns;

})();
