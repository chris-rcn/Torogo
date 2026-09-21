'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.
// Wrapped in an IIFE to avoid polluting the global namespace.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

// Board symmetry for a toroidal Go position.  The board is a torus, so its full
// automorphism group is (translations Z_N x Z_N) semidirect D4 = 8*N^2 — every
// one of the 8 square symmetries about ANY centre/axis, not just the board
// centre.  That matters here: symmetric play routinely builds a reflection or
// rotation about a translated (even half-integer) axis — e.g. two mirrored
// columns share a horizontal axis BETWEEN two rows, which no centre-only D4
// check would catch.
//
// We detect the position's STABILISER (the subgroup fixing it, colour-preserving
// and ko-fixing) and use it to prune duplicate root moves: two moves related by
// a stabiliser element lead to positions identical up to that symmetry, so they
// have exactly equal value and interchangeable subtrees — evaluating one covers
// both.  The stabiliser is trivial for a generic position (so this costs and
// does nothing mid-game) and rich near the symmetric opening (8-fold for the
// lone centre stone, the whole group for an empty board), which is exactly where
// search budget is otherwise spread across equivalent moves.
//
// Interface:  const sym = Symmetry.of(game);
//   sym.hasSymmetry()          -> bool: any non-identity element fixes the board
//   sym.distinctMoves(moves)   -> one move per orbit (input order kept; out-of-
//                                 range entries such as PASS pass through as-is)
//   sym.canonical(cell)        -> the cell's orbit representative (min index)
//   sym.orbit(cell)            -> all cells equivalent to `cell`
//   sym.ops()                  -> the stabiliser elements {op, tr, tc}
// Colour-preserving only (a colour swap changes the side to move, so it is not a
// like-for-like move).  Exact for board + ko; positional-superko history could
// in principle break the equivalence, but not in the opening where the win lives.

(function () {

const Util = (typeof require === 'function') ? require('./util.js') : window.Util;
const { EMPTY, PASS } = Util.load('./game2.js', 'Game2');

// The 8 D4 point operations about the origin, as signed coordinate permutations
// of (r, c) mod N.  These are the only linear maps preserving the 4-neighbour
// set {+/-e_r, +/-e_c}, hence the complete point group of the square-lattice
// torus — the torus adds the translation factor, applied separately.
const NEG = (x, N) => (N - x) % N;
const POINT_OPS = [
  (r, c, N) => [r, c],                     // identity
  (r, c, N) => [r, NEG(c, N)],             // mirror over the r-axis
  (r, c, N) => [NEG(r, N), c],             // mirror over the c-axis
  (r, c, N) => [NEG(r, N), NEG(c, N)],     // 180 rotation
  (r, c, N) => [c, r],                     // main-diagonal transpose
  (r, c, N) => [c, NEG(r, N)],             // rotation
  (r, c, N) => [NEG(c, N), r],             // rotation
  (r, c, N) => [NEG(c, N), NEG(r, N)],     // anti-diagonal transpose
];

function of(game) {
  const N = game.N, area = N * N, cells = game.cells;
  const ko = game.ko;                          // a cell index, or PASS for none
  const hasKo = ko !== PASS && ko >= 0 && ko < area;

  const stones = [];
  for (let i = 0; i < area; i++) if (cells[i] !== EMPTY) stones.push(i);

  // image of `cell` under element (op, tr, tc)
  function image(op, tr, tc, cell) {
    const rc = POINT_OPS[op]((cell / N) | 0, cell % N, N);
    return ((rc[0] + tr) % N) * N + (rc[1] + tc) % N;
  }

  // Does (op, tr, tc) fix the position?  g is a bijection of the torus, so if
  // every stone maps to a same-colour cell (necessarily a stone) it is a
  // permutation within each colour class and empties fall onto empties — so
  // checking the stones (plus the ko point) is sufficient.
  function fixes(op, tr, tc) {
    for (let k = 0; k < stones.length; k++) {
      const s = stones[k];
      if (cells[image(op, tr, tc, s)] !== cells[s]) return false;
    }
    if (hasKo && image(op, tr, tc, ko) !== ko) return false;
    return true;
  }

  // Stabiliser: identity, plus every (op, tr, tc) found to fix the position.
  const stab = [{ op: 0, tr: 0, tc: 0 }];
  let fullGroup = false;   // empty board, no ko: the whole 8*N^2 group (all cells one orbit)

  if (stones.length === 0 && !hasKo) {
    fullGroup = true;
  } else {
    // A non-identity op is a symmetry only via a translation that lands a fixed
    // reference cell onto a matching cell — so the candidate translations come
    // straight from the stone positions, not a full N^2 scan.
    const ref = stones.length ? stones[0] : ko;
    const refCol = stones.length ? cells[ref] : null;
    const targets = stones.length ? stones : [ko];
    for (let op = 1; op < 8; op++) {
      const p = POINT_OPS[op]((ref / N) | 0, ref % N, N);   // op-image of the reference
      const seen = new Set();
      for (let t = 0; t < targets.length; t++) {
        const s = targets[t];
        if (stones.length && cells[s] !== refCol) continue;
        const tr = (((s / N) | 0) - p[0] + N) % N;
        const tc = ((s % N) - p[1] + N) % N;
        const key = tr * N + tc;
        if (seen.has(key)) continue;
        seen.add(key);
        if (fixes(op, tr, tc)) stab.push({ op, tr, tc });
      }
    }
  }

  function canonical(cell) {
    if (fullGroup) return 0;
    let best = cell;
    for (let i = 1; i < stab.length; i++) {
      const img = image(stab[i].op, stab[i].tr, stab[i].tc, cell);
      if (img < best) best = img;
    }
    return best;
  }

  function orbit(cell) {
    if (fullGroup) { const a = new Array(area); for (let i = 0; i < area; i++) a[i] = i; return a; }
    const set = new Set();
    for (let i = 0; i < stab.length; i++) set.add(image(stab[i].op, stab[i].tr, stab[i].tc, cell));
    return [...set];
  }

  function hasSymmetry() { return fullGroup || stab.length > 1; }

  function distinctMoves(moves) {
    if (!hasSymmetry()) return Array.from(moves);
    const out = [];
    if (fullGroup) {
      let tookCell = false;
      for (const m of moves) {
        if (m < 0 || m >= area) out.push(m);            // PASS etc.: its own orbit
        else if (!tookCell) { out.push(m); tookCell = true; }
      }
      return out;
    }
    const seen = new Set();
    for (const m of moves) {
      if (m < 0 || m >= area) { out.push(m); continue; }  // PASS etc.: its own orbit
      const k = canonical(m);
      if (!seen.has(k)) { seen.add(k); out.push(m); }
    }
    return out;
  }

  function ops() { return stab.map(e => ({ op: e.op, tr: e.tr, tc: e.tc })); }

  return { hasSymmetry, distinctMoves, canonical, orbit, ops };
}

const Symmetry = { of };
if (typeof module !== 'undefined') module.exports = Symmetry;
else window.Symmetry = Symmetry;

})();
