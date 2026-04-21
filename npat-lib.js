'use strict';

// npat-lib.js — "nine-pattern" policy features.
//
// For each candidate move, we extract all NINE 3×3 windows that overlap the
// candidate point.  Each window contributes a linear feature indexed by a
// canonical pattern ID.  The move's logit is the sum of those 9 feature
// weights; moves are sampled by softmax over logits.
//
// Cell encoding (mover-relative):
//   0 = EMPTY
//   1 = FRIEND  (same colour as the player to move)
//   2 = FOE     (opposite colour)
//
// The board is toroidal (game2's neighbour tables wrap), so there is no
// off-board state to encode.
//
// Raw pattern index for a single window:
//   raw = relPos * 3^9 + Σ_i cells[i] * 3^i                       (i = 0..8)
// where relPos ∈ 0..8 is the candidate's position within the window
// (relPos = relRow*3 + relCol, relRow = candidateRow - windowTopRow ∈ 0..2).
// Range: [0, 9 * 19683) = [0, 177147).
//
// D4 canonicalisation: for each of the 8 D4 symmetries σ, transform both the
// cells (new_cells[σ(i)] = cells[i]) and the candidate's relative position
// (new_relPos = σ(relPos)), re-encode, and keep the minimum.  Color swap is
// NOT a symmetry (encoding already absorbs it via FRIEND/FOE).
//
// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
const { PASS } = _isNode ? require('./game2.js') : window.Game2;

// ── D4 position permutations on a 3×3 grid ────────────────────────────────────
// Positions linearised as i = row*3 + col, row,col ∈ {0,1,2}.
// Each entry is the permutation σ where new_pos = σ[old_pos].

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
  _mkPerm((r, c) => [c, 2 - r]),     // 1: Rot90CW       (r,c) → (c, 2−r)
  _mkPerm((r, c) => [2 - r, 2 - c]), // 2: Rot180        (r,c) → (2−r, 2−c)
  _mkPerm((r, c) => [2 - c, r]),     // 3: Rot270CW      (r,c) → (2−c, r)
  _mkPerm((r, c) => [r, 2 - c]),     // 4: FlipH         (r,c) → (r, 2−c)
  _mkPerm((r, c) => [2 - r, c]),     // 5: FlipV         (r,c) → (2−r, c)
  _mkPerm((r, c) => [c, r]),         // 6: TransposeMD   (r,c) → (c, r)
  _mkPerm((r, c) => [2 - c, 2 - r]), // 7: TransposeAD   (r,c) → (2−c, 2−r)
];

// ── Load-time canonicalisation ────────────────────────────────────────────────
//
// Build _CANON_ID: Int32Array[177147] mapping raw → dense canonical ID.

const _CELLS_BASE = 19683; // 3^9
const _RAW_RANGE  = 9 * _CELLS_BASE;

const _CANON_ID = new Int32Array(_RAW_RANGE);

const NUM_PATTERNS = (function _buildCanonTable() {
  const cells   = new Int32Array(9);
  const tCells  = new Int32Array(9);
  const idMap   = new Map(); // minRaw → canonical ID
  let nextId = 0;

  for (let raw = 0; raw < _RAW_RANGE; raw++) {
    let relPos   = (raw / _CELLS_BASE) | 0;
    let packed   = raw - relPos * _CELLS_BASE;

    // Decode base-3 cells.
    let p = packed;
    for (let i = 0; i < 9; i++) { cells[i] = p % 3; p = (p / 3) | 0; }

    // Invariant: the candidate cell is always EMPTY.  Skip impossible raws
    // (they just waste a slot in _CANON_ID, but their canonical mapping is
    // unused — assign them to an arbitrary id to keep the array dense-ish).
    if (cells[relPos] !== 0) {
      _CANON_ID[raw] = -1;
      continue;
    }

    let minRaw = raw;
    for (let di = 0; di < 8; di++) {
      const perm = _D4[di];
      // new_cells[perm[i]] = cells[i]
      for (let i = 0; i < 9; i++) tCells[perm[i]] = cells[i];
      const nRelPos = perm[relPos];
      // Re-encode.
      let enc = 0;
      for (let i = 8; i >= 0; i--) enc = enc * 3 + tCells[i];
      const nRaw = nRelPos * _CELLS_BASE + enc;
      if (nRaw < minRaw) minRaw = nRaw;
    }

    if (!idMap.has(minRaw)) idMap.set(minRaw, nextId++);
    _CANON_ID[raw] = idMap.get(minRaw);
  }

  return nextId;
}());

// ── State ─────────────────────────────────────────────────────────────────────
//
// For a board of size N:
//   moves   [count]        flat board index of move i
//   patIds  [count * 9]    9 canonical pattern IDs per move (row-major: 9 ids)
//   logits  [count]        scratch buffer for softmax
//   probs   [count]        scratch buffer for softmax
//
// The 9 pattern IDs are stored in a fixed window order keyed by (wr, wc),
// wr,wc ∈ {-2,-1,0} (the offset from the candidate to the window's top-left
// corner).  Order: wi = (wr+2)*3 + (wc+2) ∈ 0..8.

function createState(N) {
  const cap = N * N;
  return {
    moves:  new Int32Array(cap),
    patIds: new Int32Array(cap * 9),
    logits: new Float64Array(cap),
    probs:  new Float64Array(cap),
    count:  0,
  };
}

// ── Core feature extraction ───────────────────────────────────────────────────
//
// Board coordinate wrap: game2's _nbr / _dnbr are toroidal, but those tables
// only cover offsets in {-1,0,+1}.  For 3×3 windows anchored at a corner two
// steps away from the candidate, we need offsets up to ±2, so we compute the
// wrapped (row,col) ourselves.

function _wrap(x, N) { x %= N; return x < 0 ? x + N : x; }

// Extract 9-pattern features for all legal non-true-eye moves from game into
// state.  After this call:
//   state.count              = number of candidate moves
//   state.moves[i]           = flat board index of candidate i  (0 ≤ i < count)
//   state.patIds[i*9 + w]    = canonical pattern ID for window w (w ∈ 0..8)
function extractFeatures(game, state) {
  const N      = game.N;
  const cap    = N * N;
  const cells  = game.cells;
  const cur    = game.current;
  const emC    = game._emptyCells;
  const ec     = game.emptyCount;

  let count = 0;
  const moves  = state.moves;
  const patIds = state.patIds;

  for (let ei = 0; ei < ec; ei++) {
    const idx = emC[ei];
    if (!game.isLegal(idx) || game.isTrueEye(idx)) continue;

    const r = (idx / N) | 0;
    const c = idx - r * N;

    const off = count * 9;
    // Iterate the 9 windows that overlap the candidate.  wr, wc ∈ {-2,-1,0}
    // are the offsets of the window's top-left corner relative to the
    // candidate.  relRow = -wr, relCol = -wc are the candidate's
    // (row, col) within the 3×3 window.
    for (let wr = -2; wr <= 0; wr++) {
      for (let wc = -2; wc <= 0; wc++) {
        const relRow = -wr;
        const relCol = -wc;
        const relPos = relRow * 3 + relCol;

        // Pack the 9 cells of the window in base 3.  Cell order: i = dr*3+dc
        // with dr,dc ∈ {0,1,2}.  Candidate cell (at relPos) is always EMPTY=0.
        let packed = 0;
        // Base-3 accumulation from high index to low (i = 8..0).
        for (let i = 8; i >= 0; i--) {
          const dr = (i / 3) | 0;
          const dc = i - dr * 3;
          let v;
          if (i === relPos) {
            v = 0;
          } else {
            const br = _wrap(r + wr + dr, N);
            const bc = _wrap(c + wc + dc, N);
            const ci = cells[br * N + bc];
            v = ci === 0 ? 0 : (ci === cur ? 1 : 2);
          }
          packed = packed * 3 + v;
        }

        const raw = relPos * _CELLS_BASE + packed;
        const wi  = (wr + 2) * 3 + (wc + 2);
        patIds[off + wi] = _CANON_ID[raw];
      }
    }

    moves[count] = idx;
    count++;
  }

  state.count = count;
}

// ── Scoring helpers ───────────────────────────────────────────────────────────

// score(i) = Σ_{w=0..8} w[ patIds[i*9+w] ]
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

// Compute softmax probabilities over legal moves and write them into
// state.logits / state.probs.  Returns state.count.
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
// weights: Float32Array/Float64Array indexed by canonical pattern ID, or a Map.
// rng: object with .random(): [0,1).  Defaults to Math.
// Returns { move, index, prob }:
//   move  — flat board index of the chosen move (or PASS if no legal non-eye)
//   index — i in [0, state.count) (or -1 for PASS)
//   prob  — π(move | state)       (or 1 for PASS)
function policyMove(game, state, weights, rng) {
  extractFeatures(game, state);
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

// Greedy argmax move (no sampling).  Useful for evaluation.
function greedyMove(game, state, weights) {
  extractFeatures(game, state);
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
// Policy gradient for a single (state, chosen_move, advantage) triple.
//
// Let f_i ∈ ℕ^9 be the 9 pattern IDs of legal move i.  The logit is
//   ℓ_i = Σ_k w[ f_i[k] ]
// and the policy is π_i = softmax(ℓ)_i.
//
// For a canonical pattern id `p`, the gradient of log π_m w.r.t. w[p] is
//   ∂ log π_m / ∂ w[p] = count(p in f_m) − Σ_i π_i · count(p in f_i)
//
// REINFORCE with advantage A (e.g. episode return, or return minus baseline):
//   Δw[p] = lr · A · (count(p in f_m) − Σ_i π_i · count(p in f_i))
//
// Because feature ids are unbounded in count (we use a Map to stay sparse),
// we accumulate updates into the `weights` Map.
//
// state must have been populated by extractFeatures AND _computeSoftmax; this
// function re-uses state.probs so the caller should not mutate it between
// extractFeatures() and reinforceUpdate().

function reinforceUpdate(state, chosenIndex, advantage, weights, lr) {
  const n = state.count;
  if (n === 0 || chosenIndex < 0) return;

  const step  = lr * advantage;
  if (step === 0) return;

  const patIds = state.patIds;
  const probs  = state.probs;
  const isMap  = weights instanceof Map;

  // Positive term: +step per occurrence of pid in f_{chosen}.
  // Negative term: −step · π_i per occurrence of pid in f_i.
  //
  // Efficient path: for each pid appearing in any legal move, its net delta
  // is step · (count_in_chosen − Σ_{i: pid∈f_i} π_i · count(pid, f_i)).
  //
  // Use a small scratch Map<pid, delta>.  9*n ops.
  const delta = new Map();

  // + step per count in chosen.
  {
    const off = chosenIndex * 9;
    for (let k = 0; k < 9; k++) {
      const pid = patIds[off + k];
      delta.set(pid, (delta.get(pid) ?? 0) + step);
    }
  }

  // − step · π_i per count in each move i.
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
  NUM_PATTERNS,
  // Exposed for tests.
  _CANON_ID,
  _CELLS_BASE,
  _D4,
};

if (typeof module !== 'undefined') module.exports = NPatterns;
else window.NPatterns = NPatterns;

})();
