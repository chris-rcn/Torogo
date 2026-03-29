'use strict';

// book.js — Opening book with D4 symmetry via Zobrist hashing.
//
// Book entries are keyed by the canonical (minimum-hash) Zobrist hash over all
// 8 D4 transforms.  Moves are stored in canonical coordinates with a selection
// count; lookup un-applies the normalising transform before returning the move.
//
// Structure: Map<BigInt, Map<int, int>>
//   Outer key: canonical Zobrist hash
//   Inner key: canonical flat move index (or PASS = -1)
//   Inner value: number of times this move was selected by the agent
//
// API:
//   addToBook(book, game2, move)      → void
//   lookupBook(book, game2)           → move object or null
//   serializeBook(book)               → JSON string
//   deserializeBook(json)             → book

const PASS = -1;
const BLACK = 1;

// ── D4 transforms ─────────────────────────────────────────────────────────────
// applyTransform(t, x, y, N) → [x', y'], t in 0..7.
// Order: identity, rotate90CCW, rotate180, rotate270CCW, flipH, flipV, flipMain, flipAnti.

function applyTransform(t, x, y, N) {
  const m = N - 1;
  switch (t) {
    case 0: return [x,     y    ];   // identity
    case 1: return [m - y, x    ];   // rotate 90° CCW
    case 2: return [m - x, m - y];   // rotate 180°
    case 3: return [y,     m - x];   // rotate 270° CCW
    case 4: return [m - x, y    ];   // reflect across vertical axis
    case 5: return [x,     m - y];   // reflect across horizontal axis
    case 6: return [y,     x    ];   // reflect across main diagonal
    case 7: return [m - y, m - x];   // reflect across anti-diagonal
  }
}

// inverseTransform[t] gives the index of the inverse of transform t.
// Rotation inverses: rot90CCW↔rot270CCW; reflections are self-inverse.
const INV_T = [0, 3, 2, 1, 4, 5, 6, 7];

// ── Seeded PRNG (mulberry32) ──────────────────────────────────────────────────

function mulberry32(seed) {
  let s = seed >>> 0;
  return function () {
    s += 0x6D2B79F5;
    let x = Math.imul(s ^ (s >>> 15), 1 | s);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0);   // returns uint32
  };
}

// ── Zobrist tables (64-bit via two concatenated uint32s) ──────────────────────

const _tableCache = new Map();

function getZobristTable(N) {
  if (_tableCache.has(N)) return _tableCache.get(N);
  const cap  = N * N;
  const rand = mulberry32(0xDEADBEEF ^ N);
  const r64  = () => (BigInt(rand()) << 32n) | BigInt(rand());
  const black = new Array(cap).fill(null).map(r64);
  const white = new Array(cap).fill(null).map(r64);
  const tbl   = { black, white };
  _tableCache.set(N, tbl);
  return tbl;
}

// ── Canonical hash ────────────────────────────────────────────────────────────

// Hash `cells` (Int8Array, length N*N) under D4 transform `t`.
function hashWithTransform(cells, N, t, tbl) {
  let h = 0n;
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const c = cells[y * N + x];
      if (c === 0) continue;
      const [tx, ty] = applyTransform(t, x, y, N);
      const ti = ty * N + tx;
      h ^= (c === BLACK ? tbl.black[ti] : tbl.white[ti]);
    }
  }
  return h;
}

// Returns { hash: BigInt, t: int } where t is the transform that minimises the hash.
function canonicalHash(cells, N) {
  const tbl = getZobristTable(N);
  let minHash = null, minT = 0;
  for (let t = 0; t < 8; t++) {
    const h = hashWithTransform(cells, N, t, tbl);
    if (minHash === null || h < minHash) { minHash = h; minT = t; }
  }
  return { hash: minHash, t: minT };
}

// ── Canonical move helpers ────────────────────────────────────────────────────

function toCanonMove(move, t, N) {
  if (move === PASS) return PASS;
  const mx = move % N, my = (move / N) | 0;
  const [cx, cy] = applyTransform(t, mx, my, N);
  return cy * N + cx;
}

function fromCanonMove(canonMove, t, N) {
  if (canonMove === PASS) return PASS;
  const cx = canonMove % N, cy = (canonMove / N) | 0;
  const [ax, ay] = applyTransform(INV_T[t], cx, cy, N);
  return ay * N + ax;
}

// ── Book operations ───────────────────────────────────────────────────────────

// Increment the selection count for (position, move).
// move is a flat index or PASS.
//
// Uses joint (hash, canonMove) minimization so that D4-equivalent moves at
// symmetric positions (e.g. the root with only the center stone) are merged
// into one canonical entry rather than stored separately.
function addToBook(book, game2, move) {
  const N    = game2.N;
  const tbl  = getZobristTable(N);
  const cells = game2.cells;

  let bestHash = null, bestCm = null;
  for (let t = 0; t < 8; t++) {
    const h  = hashWithTransform(cells, N, t, tbl);
    const cm = toCanonMove(move, t, N);
    if (bestHash === null || h < bestHash || (h === bestHash && cm < bestCm)) {
      bestHash = h; bestCm = cm;
    }
  }

  let entry = book.get(bestHash);
  if (!entry) { entry = new Map(); book.set(bestHash, entry); }
  entry.set(bestCm, (entry.get(bestCm) || 0) + 1);

  if (game2.emptyCount < book.minEmptyCount) book.minEmptyCount = game2.emptyCount;
}

// Return a move sampled by softmax over selection counts, or null if unknown.
// Returns { type, move, x, y } in game2's coordinate system.
// minExperience: minimum total selections for the position before returning a move.
function lookupBook(book, game2, minExperience = 2) {
  if (game2.emptyCount < book.minEmptyCount) return null;
  const N = game2.N;
  const tbl = getZobristTable(N);
  const cells = game2.cells;

  // Find the canonical hash and all transforms that achieve it.
  const { hash } = canonicalHash(game2.cells, N);
  const entry = book.get(hash);
  if (!entry) return null;

  let totalExp = 0;
  for (const count of entry.values()) totalExp += count;
  if (totalExp < minExperience) return null;

  // Sample proportionally to selection counts.
  const moves = [...entry.keys()];
  const counts = moves.map(m => entry.get(m));
  const total = counts.reduce((a, b) => a + b, 0);

  let r = Math.random() * total;
  let bestCanon = moves[moves.length - 1];
  for (let i = 0; i < moves.length; i++) {
    r -= counts[i];
    if (r <= 0) { bestCanon = moves[i]; break; }
  }

  // Collect all transforms that achieve the canonical hash, then pick one at
  // random so that symmetric positions return a uniformly random equivalent move.
  const symTs = [];
  for (let t = 0; t < 8; t++) {
    if (hashWithTransform(cells, N, t, tbl) === hash) symTs.push(t);
  }
  const chosenT = symTs[Math.floor(Math.random() * symTs.length)];

  const idx = fromCanonMove(bestCanon, chosenT, N);
  if (idx === PASS) return { type: 'pass', move: PASS };
  if (!game2.isLegal(idx)) return null;  // guard against hash collision
  return { type: 'place', move: idx, x: idx % N, y: (idx / N) | 0 };
}

// ── Serialization ─────────────────────────────────────────────────────────────
//
// The book is saved as a .js file loadable via <script> in the browser or
// require() in Node.  Format:
//   var BookData = [[hashStr, [[canonMove, count], ...]], ...];
//   if (typeof module !== 'undefined') module.exports = BookData;
//   else window.BookData = BookData;

function serializeBook(book) {
  const entries = [];
  for (const [hash, moves] of book) {
    const movesArr = [];
    for (const [move, count] of moves) movesArr.push([move, count]);
    entries.push([hash.toString(), movesArr]);
  }
  const payload = { minEmptyCount: book.minEmptyCount, entries };
  const data = JSON.stringify(payload);
  return `var BookData = ${data};\nif (typeof module !== 'undefined') module.exports = BookData;\nelse window.BookData = BookData;\n`;
}

// `data` is the raw object (BookData), not a string.
function deserializeBook(data) {
  const book = new Map();
  book.minEmptyCount = data.minEmptyCount ?? Infinity;
  for (const [hashStr, movesArr] of data.entries) {
    const moves = new Map();
    for (const [move, count] of movesArr) moves.set(move, count);
    book.set(BigInt(hashStr), moves);
  }
  return book;
}

module.exports = {
  applyTransform, INV_T,
  canonicalHash,
  addToBook, lookupBook,
  serializeBook, deserializeBook,
};
