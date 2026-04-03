'use strict';

// Group storage: flat typed arrays indexed by gid, bitsets for stones/liberties.
//
// Interface:
//   new Game2(size)
//   game.play(move)     → true/false
//   game.isLegal(move)  → boolean (non-mutating)
//   game.isTrueEye(idx)       → boolean for game.current
//   game.randomLegalMove()    → idx or PASS  (uniform random non-eye legal move)
//   game.emptyCount     number of empty cells (updated incrementally)
//   game.gameOver       boolean
//   game.current        BLACK | WHITE
//   game.cells          Int8Array  (read-only)
//   game.N              board size

(function() {

const EMPTY = 0, BLACK = 1, WHITE = -1;
const PASS  = -1;
const _komiOverrides = new Map([
  [ 5, 24.5],
  [ 6, 35.5],
  [ 7, 48.5],
]);
const KOMI = size => _komiOverrides.get(size) ?? 3.5;
function setKomi(size, komi) { _komiOverrides.set(size, komi); }

// Shared neighbor-table cache (same design as game.js)
const topologyCache = new Map();
function getTopology(N) {
  const cached = topologyCache.get(N);
  if (cached !== undefined) return cached;
  const cap = N * N;
  const nbr  = new Int32Array(cap * 4);
  const dnbr = new Int32Array(cap * 4);
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const i = y * N + x;
      nbr[i*4+0] = ((y-1+N)%N)*N + x;
      nbr[i*4+1] = ((y+1  )%N)*N + x;
      nbr[i*4+2] = y*N + (x-1+N)%N;
      nbr[i*4+3] = y*N + (x+1  )%N;
      dnbr[i*4+0] = ((y-1+N)%N)*N + (x-1+N)%N;
      dnbr[i*4+1] = ((y-1+N)%N)*N + (x+1  )%N;
      dnbr[i*4+2] = ((y+1  )%N)*N + (x-1+N)%N;
      dnbr[i*4+3] = ((y+1  )%N)*N + (x+1  )%N;
    }
  }
  const allCells = new Int32Array(cap);
  for (let i = 0; i < cap; i++) allCells[i] = i;

  const t = { nbr, dnbr, allCells };
  topologyCache.set(N, t);
  return t;
}

// Popcount for a 32-bit integer
function _pop32(x) {
  x = x - ((x >>> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  x = (x + (x >>> 4)) & 0x0f0f0f0f;
  return (Math.imul(x, 0x01010101) >>> 24);
}

class Game2 {
  constructor(size, applyFirstMove = true) {
    const N = size;
    const cap = N * N;
    this.N        = N;
    this.boardSize = N;

    this.cells = new Int8Array(cap);
    this._gid  = new Int32Array(cap).fill(-1);
    this._nextGid = 0;

    // Bitset width: words per group bitset (ceil(cap/32))
    const W = (cap + 31) >> 5;
    this._W = W;
    const MAX_G = 4 * cap + 4;
    this._gc = new Uint8Array(MAX_G);       // group color
    this._sw = new Int32Array(MAX_G * W);   // stones bitset words
    this._ss = new Int32Array(MAX_G);       // stones size
    this._lw = new Int32Array(MAX_G * W);   // liberty bitset words
    this._ls = new Int32Array(MAX_G);       // liberty size

    const tables = getTopology(N);
    this._nbr     = tables.nbr;
    this._dnbr    = tables.dnbr;
    this.nbr      = tables.nbr;      // public read-only
    this._allCells = tables.allCells; // all cell indices [0..N*N-1], shared across instances

    this.current = BLACK;
    this.ko      = PASS;
    this.consecutivePasses = 0;
    this.gameOver = false;
    this.moveCount = 0;
    this.lastMove = PASS;
    this.emptyCount = N * N;

    if (applyFirstMove) {
      const center = (N >> 1) * N + (N >> 1);
      this._place(center, BLACK);
      this.current   = WHITE;
      this.moveCount = 1;
    }
  }

  // ── Reset (reuse instance across games) ───────────────────────────────────

  reset() {
    const cap = this.N * this.N;
    this.cells.fill(0);
    this._gid.fill(-1);
    const used = this._nextGid * this._W;
    this._sw.fill(0, 0, used);
    this._ss.fill(0, 0, this._nextGid);
    this._lw.fill(0, 0, used);
    this._ls.fill(0, 0, this._nextGid);
    this._nextGid = 0;
    this.current = BLACK;
    this.ko = PASS;
    this.consecutivePasses = 0;
    this.gameOver = false;
    this.moveCount = 0;
    this.emptyCount = this.N * this.N;
    const center = (this.N >> 1) * this.N + (this.N >> 1);
    this._place(center, BLACK);
    this.current   = WHITE;
    this.moveCount = 1;
  }

  // ── Group tracking ─────────────────────────────────────────────────────────

  _place(idx, color) {
    const cells  = this.cells;
    const gidArr = this._gid;
    const nbr    = this._nbr;
    const base   = idx * 4;
    const W      = this._W;
    const lw     = this._lw;
    const ls     = this._ls;
    const sw     = this._sw;
    const ss     = this._ss;

    cells[idx] = color;
    this.emptyCount--;

    // Remove idx from liberties of adjacent groups
    let s0 = -1, s1 = -1, s2 = -1, s3 = -1;
    for (let i = 0; i < 4; i++) {
      const ni  = nbr[base + i];
      const gid = gidArr[ni];
      if (gid === -1 || gid === s0 || gid === s1 || gid === s2 || gid === s3) continue;
      if      (s0 === -1) s0 = gid;
      else if (s1 === -1) s1 = gid;
      else if (s2 === -1) s2 = gid;
      else                s3 = gid;
      const m = 1 << (idx & 31);
      const wi = idx >> 5;
      const lb = gid * W;
      if (lw[lb + wi] & m) { lw[lb + wi] &= ~m; ls[gid]--; }
    }

    // Collect same-color neighbor groups (sg0..sg3) and own liberties (ml0..ml3)
    let sg0 = -1, sg1 = -1, sg2 = -1, sg3 = -1, ns = 0;
    let ml0 = -1, ml1 = -1, ml2 = -1, ml3 = -1;
    for (let i = 0; i < 4; i++) {
      const ni = nbr[base + i];
      const c  = cells[ni];
      if (c === color) {
        const gid = gidArr[ni];
        if (gid !== sg0 && gid !== sg1 && gid !== sg2 && gid !== sg3) {
          if      (sg0 === -1) sg0 = gid;
          else if (sg1 === -1) sg1 = gid;
          else if (sg2 === -1) sg2 = gid;
          else                 sg3 = gid;
          ns++;
        }
      } else if (c === EMPTY) {
        if      (ml0 === -1) ml0 = ni;
        else if (ml1 === -1) ml1 = ni;
        else if (ml2 === -1) ml2 = ni;
        else                 ml3 = ni;
      }
    }

    if (ns === 0) {
      const gid = this._nextGid++;
      gidArr[idx] = gid;
      this._gc[gid] = color;
      sw[gid * W + (idx >> 5)] |= (1 << (idx & 31));
      ss[gid] = 1;
      const lb = gid * W;
      let lsCount = 0, m, wi;
      if (ml0 !== -1) { m = 1<<(ml0&31); wi = ml0>>5; if (!(lw[lb+wi]&m)) { lw[lb+wi]|=m; lsCount++; } }
      if (ml1 !== -1) { m = 1<<(ml1&31); wi = ml1>>5; if (!(lw[lb+wi]&m)) { lw[lb+wi]|=m; lsCount++; } }
      if (ml2 !== -1) { m = 1<<(ml2&31); wi = ml2>>5; if (!(lw[lb+wi]&m)) { lw[lb+wi]|=m; lsCount++; } }
      if (ml3 !== -1) { m = 1<<(ml3&31); wi = ml3>>5; if (!(lw[lb+wi]&m)) { lw[lb+wi]|=m; lsCount++; } }
      ls[gid] = lsCount;
      return gid;
    }

    // Merge into largest same-color neighbor group
    let mainGid = sg0;
    if (sg1 !== -1 && ss[sg1] > ss[mainGid]) mainGid = sg1;
    if (sg2 !== -1 && ss[sg2] > ss[mainGid]) mainGid = sg2;
    if (sg3 !== -1 && ss[sg3] > ss[mainGid]) mainGid = sg3;
    const gb = mainGid * W;  // group base (same for sw and lw)

    sw[gb + (idx >> 5)] |= (1 << (idx & 31));
    ss[mainGid]++;
    gidArr[idx] = mainGid;

    let m2, wi2;
    if (ml0 !== -1) { m2=1<<(ml0&31); wi2=ml0>>5; if (!(lw[gb+wi2]&m2)) { lw[gb+wi2]|=m2; ls[mainGid]++; } }
    if (ml1 !== -1) { m2=1<<(ml1&31); wi2=ml1>>5; if (!(lw[gb+wi2]&m2)) { lw[gb+wi2]|=m2; ls[mainGid]++; } }
    if (ml2 !== -1) { m2=1<<(ml2&31); wi2=ml2>>5; if (!(lw[gb+wi2]&m2)) { lw[gb+wi2]|=m2; ls[mainGid]++; } }
    if (ml3 !== -1) { m2=1<<(ml3&31); wi2=ml3>>5; if (!(lw[gb+wi2]&m2)) { lw[gb+wi2]|=m2; ls[mainGid]++; } }

    let needRecount = false;
    if (sg0 !== mainGid && sg0 !== -1) { needRecount = true; const ob = sg0 * W;
      for (let wi = 0; wi < W; wi++) {
        let w = sw[ob + wi];
        while (w) { const bit = 31 - Math.clz32(w & -w); gidArr[wi * 32 + bit] = mainGid; w &= w - 1; }
        sw[gb + wi] |= sw[ob + wi];
      }
      ss[mainGid] += ss[sg0];
      for (let wi = 0; wi < W; wi++) lw[gb + wi] |= lw[ob + wi];
    }
    if (sg1 !== mainGid && sg1 !== -1) { needRecount = true; const ob = sg1 * W;
      for (let wi = 0; wi < W; wi++) {
        let w = sw[ob + wi];
        while (w) { const bit = 31 - Math.clz32(w & -w); gidArr[wi * 32 + bit] = mainGid; w &= w - 1; }
        sw[gb + wi] |= sw[ob + wi];
      }
      ss[mainGid] += ss[sg1];
      for (let wi = 0; wi < W; wi++) lw[gb + wi] |= lw[ob + wi];
    }
    if (sg2 !== mainGid && sg2 !== -1) { needRecount = true; const ob = sg2 * W;
      for (let wi = 0; wi < W; wi++) {
        let w = sw[ob + wi];
        while (w) { const bit = 31 - Math.clz32(w & -w); gidArr[wi * 32 + bit] = mainGid; w &= w - 1; }
        sw[gb + wi] |= sw[ob + wi];
      }
      ss[mainGid] += ss[sg2];
      for (let wi = 0; wi < W; wi++) lw[gb + wi] |= lw[ob + wi];
    }
    if (sg3 !== mainGid && sg3 !== -1) { needRecount = true; const ob = sg3 * W;
      for (let wi = 0; wi < W; wi++) {
        let w = sw[ob + wi];
        while (w) { const bit = 31 - Math.clz32(w & -w); gidArr[wi * 32 + bit] = mainGid; w &= w - 1; }
        sw[gb + wi] |= sw[ob + wi];
      }
      ss[mainGid] += ss[sg3];
      for (let wi = 0; wi < W; wi++) lw[gb + wi] |= lw[ob + wi];
    }
    if (needRecount) {
      let newLs = 0;
      for (let wi = 0; wi < W; wi++) newLs += _pop32(lw[gb + wi]);
      ls[mainGid] = newLs;
    }
    return mainGid;
  }

  _remove(gid) {
    const W      = this._W;
    const sw     = this._sw;
    const lw     = this._lw;
    const ls     = this._ls;
    const cells  = this.cells;
    const gidArr = this._gid;
    const nbr    = this._nbr;
    const sb     = gid * W;
    let count = 0;

    for (let wi = 0; wi < W; wi++) {
      let w = sw[sb + wi];
      while (w) {
        const bit = 31 - Math.clz32(w & -w);
        const idx = wi * 32 + bit;
        cells[idx] = EMPTY;
        gidArr[idx] = -1;
        const base = idx * 4;
        for (let i = 0; i < 4; i++) {
          const nGid = gidArr[nbr[base + i]];
          if (nGid !== -1 && nGid !== gid) {
            const nlb = nGid * W;
            const m = 1 << (idx & 31);
            const nwi = idx >> 5;
            if (!(lw[nlb + nwi] & m)) { lw[nlb + nwi] |= m; ls[nGid]++; }
          }
        }
        count++;
        w &= w - 1;
      }
    }
    this.emptyCount += count;
    return count;
  }

  // ── Public group info ──────────────────────────────────────────────────────

  /** Group id of the stone at idx, or -1 if empty. */
  groupIdAt(idx)          { return this._gid[idx]; }
  /** Number of stones in group gid. */
  groupSize(gid)          { return this._ss[gid]; }
  /** Number of liberties of group gid. */
  groupLibertyCount(gid)  { return this._ls[gid]; }
  /** Blended distance between two board indices: 0.4*manhattan + 0.6*euclidean,
   *  both computed on the torus. */
  distance(a, b) {
    const N = this.N;
    const ax = a % N, ay = (a / N) | 0;
    const bx = b % N, by = (b / N) | 0;
    const dx = Math.min(Math.abs(ax - bx), N - Math.abs(ax - bx));
    const dy = Math.min(Math.abs(ay - by), N - Math.abs(ay - by));
    return 0.4 * (dx + dy) + 0.6 * Math.sqrt(dx * dx + dy * dy);
  }
  /** All liberty indices of the group containing stone at idx, as an Int32Array. */
  groupLibs(idx) {
    const gid = this._gid[idx];
    if (gid === -1) return new Int32Array(0);
    const lc  = this._ls[gid];
    const out = new Int32Array(lc);
    const cap = this.N * this.N;
    const lb  = gid * this._W;
    let found = 0;
    for (let wi = 0; wi < this._W && found < lc; wi++) {
      let w = this._lw[lb + wi];
      while (w && found < lc) {
        const i = wi * 32 + (31 - Math.clz32(w & -w));
        if (i < cap) out[found++] = i;
        w &= w - 1;
      }
    }
    return out;
  }

  // ── Non-mutating legality checks ───────────────────────────────────────────

  _isSingleSuicide(idx, color) {
    const cells  = this.cells;
    const gidArr = this._gid;
    const nbr    = this._nbr;
    const base   = idx * 4;
    const ls     = this._ls;
    for (let i = 0; i < 4; i++) {
      const ni = nbr[base + i];
      const c  = cells[ni];
      if (c === EMPTY) return false;
      const gid = gidArr[ni];
      if (c === color) { if (ls[gid] > 1) return false; }
      else             { if (ls[gid] === 1) return false; }
    }
    return true;
  }

  _isMultiSuicide(idx, color) {
    const cells  = this.cells;
    const gidArr = this._gid;
    const nbr    = this._nbr;
    const base   = idx * 4;
    const ls     = this._ls;
    const lw     = this._lw;
    const W      = this._W;
    let hasFriendly = false;
    let s0 = -1, s1 = -1, s2 = -1, s3 = -1;
    for (let i = 0; i < 4; i++) {
      const ni = nbr[base + i];
      const c  = cells[ni];
      if (c === EMPTY) return false;
      const gid = gidArr[ni];
      if (gid === s0 || gid === s1 || gid === s2 || gid === s3) continue;
      if      (s0 === -1) s0 = gid;
      else if (s1 === -1) s1 = gid;
      else if (s2 === -1) s2 = gid;
      else                s3 = gid;
      if (c === color) {
        hasFriendly = true;
        if (ls[gid] > 1) return false;
        if (ls[gid] === 1 && !((lw[gid * W + (idx >> 5)] >>> (idx & 31)) & 1)) return false;
      } else {
        if (ls[gid] === 1 && ((lw[gid * W + (idx >> 5)] >>> (idx & 31)) & 1)) return false;
      }
    }
    return hasFriendly;
  }

  _isKo(idx, color) {
    if (idx !== this.ko) return false;
    const opp    = -color;
    const cells  = this.cells;
    const gidArr = this._gid;
    const nbr    = this._nbr;
    const base   = idx * 4;
    const ls     = this._ls;
    const lw     = this._lw;
    const ss     = this._ss;
    const W      = this._W;
    let s0 = -1, s1 = -1, s2 = -1, s3 = -1;
    let captured = 0;
    for (let i = 0; i < 4; i++) {
      const ni = nbr[base + i];
      if (cells[ni] !== opp) continue;
      const gid = gidArr[ni];
      if (gid === s0 || gid === s1 || gid === s2 || gid === s3) continue;
      if      (s0 === -1) s0 = gid;
      else if (s1 === -1) s1 = gid;
      else if (s2 === -1) s2 = gid;
      else                s3 = gid;
      if (ls[gid] === 1 && ((lw[gid * W + (idx >> 5)] >>> (idx & 31)) & 1))
        captured += ss[gid];
    }
    return captured === 1;
  }

  isLegal(idx) {
    if (idx === PASS) return true;
    if (this.cells[idx] !== EMPTY) return false;
    if (this._isSingleSuicide(idx, this.current)) return false;
    if (this._isMultiSuicide(idx, this.current))  return false;
    if (this._isKo(idx, this.current))             return false;
    return true;
  }

  // Returns true if playing at idx (for this.current) would capture at least one opponent stone.
  isCapture(idx) {
    if (idx === PASS) return false;
    const opp = this.current === BLACK ? WHITE : BLACK;
    const nbr = this._nbr;
    for (let d = 0; d < 4; d++) {
      const ni = nbr[idx * 4 + d];
      if (this.cells[ni] === opp && this._ls[this._gid[ni]] === 1) return true;
    }
    return false;
  }

  // Returns an array of cell indices that would be captured by playing idx for this.current.
  // Returns [] if idx is PASS or no captures would occur.
  captureList(idx) {
    if (idx === PASS) return [];
    const opp    = -this.current;
    const nbr    = this._nbr;
    const cells  = this.cells;
    const gidArr = this._gid;
    const ls     = this._ls;
    const sw     = this._sw;
    const W      = this._W;
    const result = [];
    let seen0 = -1, seen1 = -1, seen2 = -1, seen3 = -1;
    for (let d = 0; d < 4; d++) {
      const ni  = nbr[idx * 4 + d];
      if (cells[ni] !== opp) continue;
      const gid = gidArr[ni];
      if (ls[gid] !== 1) continue;
      if (gid === seen0 || gid === seen1 || gid === seen2 || gid === seen3) continue;
      if      (seen0 === -1) seen0 = gid;
      else if (seen1 === -1) seen1 = gid;
      else if (seen2 === -1) seen2 = gid;
      else                   seen3 = gid;
      const gb = gid * W;
      for (let wi = 0; wi < W; wi++) {
        let w = sw[gb + wi];
        while (w) {
          const lsb = w & -w;
          result.push(wi * 32 + (31 - Math.clz32(lsb)));
          w ^= lsb;
        }
      }
    }
    return result;
  }

  isTrueEye(idx) {
    const color  = this.current;
    const cells  = this.cells;
    const gidArr = this._gid;
    const nbr    = this._nbr;
    const dnbr   = this._dnbr;
    const base   = idx * 4;
    let firstGid = -2, friendCount = 0, emptyCount = 0, sameGroup = 0;
    for (let i = 0; i < 4; i++) {
      const ni = nbr[base + i];
      const c  = cells[ni];
      if (c === color) {
        friendCount++;
        const gid = gidArr[ni];
        if (firstGid === -2) { firstGid = gid; sameGroup = 1; }
        else if (gid === firstGid) sameGroup++;
      } else if (c === EMPTY) {
        emptyCount++;
      }
    }
    // 3 same-group friends + 1 empty: proto-eye, treat as true eye
    if (friendCount === 3 && emptyCount === 1 && sameGroup === 3) return true;
    if (friendCount < 4) return false;
    if (sameGroup === 4) return true;
    let dc = 0;
    for (let i = 0; i < 4; i++) if (cells[dnbr[base + i]] === color) dc++;
    return dc >= 3;
  }

  // ── Main move interface ────────────────────────────────────────────────────

  play(idx) {
    return this.playInfo(idx).success;
  }

  playInfo(idx) {
    if (this.gameOver) return { success: false };
    const color = this.current;
    const opp   = -color;

    if (idx === PASS) {
      this.consecutivePasses++;
      if (this.consecutivePasses >= 2) this.gameOver = true;
      this.current = opp;
      this.ko = PASS;
      this.moveCount++;
      this.lastMove = PASS;
      return { success: true };
    }

    if (!this.isLegal(idx)) return { success: false };

    this._place(idx, color);

    const cells  = this.cells;
    const gidArr = this._gid;
    const nbr    = this._nbr;
    const base   = idx * 4;
    const ls     = this._ls;
    const ss     = this._ss;
    const sw     = this._sw;
    const W      = this._W;
    let c0 = -1, c1 = -1, c2 = -1, c3 = -1;
    let capturedCount = 0;
    let capturedIdx   = PASS;

    for (let i = 0; i < 4; i++) {
      const ni  = nbr[base + i];
      if (cells[ni] !== opp) continue;
      const gid = gidArr[ni];
      if (gid === -1 || gid === c0 || gid === c1 || gid === c2 || gid === c3) continue;
      if      (c0 === -1) c0 = gid;
      else if (c1 === -1) c1 = gid;
      else if (c2 === -1) c2 = gid;
      else                c3 = gid;
      if (ls[gid] === 0) {
        if (ss[gid] === 1) {
          const gb = gid * W;
          for (let wi = 0; wi < W; wi++) {
            if (sw[gb + wi]) {
              capturedIdx = wi * 32 + (31 - Math.clz32(sw[gb + wi] & -sw[gb + wi]));
              break;
            }
          }
        }
        capturedCount += this._remove(gid);
      }
    }

    this.ko = PASS;
    if (capturedCount === 1 && capturedIdx !== PASS) {
      const myGid = gidArr[idx];
      const lw    = this._lw;
      if (ss[myGid] === 1 && ls[myGid] === 1 &&
          ((lw[myGid * W + (capturedIdx >> 5)] >>> (capturedIdx & 31)) & 1)) {
        this.ko = capturedIdx;
      }
    }

    this.consecutivePasses = 0;
    this.current = opp;
    this.moveCount++;
    this.lastMove = idx;
    if (this.moveCount >= 4 * this.N * this.N) this.gameOver = true;
    return { capturedCount, success: true };
  }

  // ── Stone / liberty queries ───────────────────────────────────────────────

  // Returns an array of flat cell indices for every stone in the group with
  // the given gid.
  groupStones(gid) {
    const W   = this._W;
    const sw  = this._sw;
    const cap = this.N * this.N;
    const wb  = gid * W;
    const out = [];
    for (let wi = 0; wi < W; wi++) {
      let w = sw[wb + wi];
      while (w) {
        const i = wi * 32 + (31 - Math.clz32(w & -w));
        if (i < cap) out.push(i);
        w &= w - 1;
      }
    }
    return out;
  }



  // Returns the liberty count of the group containing idx, plus the first two
  // liberty indices (lib0, lib1; -1 if absent).  Returns count=0 when idx is
  // empty or has no group.
  groupLibs2(idx) {
    const gid = this._gid[idx];
    if (gid === -1) return { count: 0, lib0: -1, lib1: -1 };
    const lc = this._ls[gid];
    if (lc === 0) return { count: 0, lib0: -1, lib1: -1 };
    const W  = this._W;
    const lw = this._lw;
    const lb = gid * W;
    const cap = this.N * this.N;
    let lib0 = -1, lib1 = -1, found = 0;
    for (let wi = 0; wi < W; wi++) {
      let w = lw[lb + wi];
      while (w) {
        const i = wi * 32 + (31 - Math.clz32(w & -w));
        if (i < cap) {
          if (found === 0) lib0 = i;
          else             lib1 = i;
          if (++found === 2) return { count: lc, lib0, lib1 };
        }
        w &= w - 1;
      }
    }
    return { count: lc, lib0, lib1 };
  }

  // ── Display ───────────────────────────────────────────────────────────────

  // Render the board as a ● ○ · string.  Optional 'markIdx' marks that
  // cell with bracket separators: · ·(●)· · — row width is unchanged.
  toString(markIdx = this.lastMove, { centerAt = null } = {}) {
    const N = this.N;
    const cells = this.cells;
    const markX = (markIdx !== PASS) ? markIdx % N : -1;
    const markY = (markIdx !== PASS) ? (markIdx / N | 0) : -1;

    const half = (N / 2) | 0;
    const x0 = (centerAt !== null && centerAt !== PASS)
      ? ((centerAt % N) - half + N) % N : 0;
    const y0 = (centerAt !== null && centerAt !== PASS)
      ? (((centerAt / N) | 0) - half + N) % N : 0;

    // Convert board-space mark to display-space coordinates.
    const dmx = markX >= 0 ? (markX - x0 + N) % N : -1;
    const dmy = markY >= 0 ? (markY - y0 + N) % N : -1;

    const rows = [];
    for (let dy = 0; dy < N; dy++) {
      const by = (y0 + dy) % N;
      const mx = (dy === dmy) ? dmx : -1;
      let row = (mx === 0) ? '(' : ' ';
      for (let dx = 0; dx < N; dx++) {
        const bx = (x0 + dx) % N;
        const c = cells[by * N + bx];
        const ch = c === BLACK ? '●' : c === WHITE ? '○' : '·';
        if (dx > 0) row += (dx === mx) ? '(' : (dx - 1 === mx) ? ')' : ' ';
        row += ch;
      }
      row += (mx === N - 1) ? ')' : ' ';
      rows.push(row);
    }
    return rows.join('\n');
  }

  // ── Clone ─────────────────────────────────────────────────────────────────

  clone() {
    const g = Object.create(Game2.prototype);
    g.N         = this.N;
    g.boardSize = this.N;
    g.cells = new Int8Array(this.cells);
    g._gid  = new Int32Array(this._gid);
    g._nextGid = this._nextGid;
    g._W    = this._W;
    g._gc   = new Uint8Array(this._gc);
    g._sw   = new Int32Array(this._sw);
    g._ss   = new Int32Array(this._ss);
    g._lw   = new Int32Array(this._lw);
    g._ls   = new Int32Array(this._ls);
    g._nbr      = this._nbr;      // immutable, share
    g._dnbr     = this._dnbr;     // immutable, share
    g._allCells = this._allCells; // shared scratch buffer — shuffled in-place by randomLegalMove
    g.nbr       = this._nbr;      // public alias
    g.current           = this.current;
    g.ko                = this.ko;
    g.consecutivePasses = this.consecutivePasses;
    g.gameOver          = this.gameOver;
    g.moveCount         = this.moveCount;
    g.lastMove          = this.lastMove;
    g.emptyCount        = this.emptyCount;
    return g;
  }

  // ── Scoring ───────────────────────────────────────────────────────────────

  // Accurate area score via flood-fill of empty regions.
  // Returns { black, white } where white already includes KOMI(N).
  calcScore() {
    const N      = this.N;
    const cap    = N * N;
    const cells  = this.cells;
    const nbr    = this._nbr;
    const visited = new Uint8Array(cap);
    let black = 0, white = 0;

    for (let i = 0; i < cap; i++) {
      if      (cells[i] === BLACK) black++;
      else if (cells[i] === WHITE) white++;
    }

    for (let start = 0; start < cap; start++) {
      if (cells[start] !== EMPTY || visited[start]) continue;
      let bBorder = false, wBorder = false;
      const region = [start];
      visited[start] = 1;
      for (let qi = 0; qi < region.length; qi++) {
        const i = region[qi];
        const b = i * 4;
        for (let k = 0; k < 4; k++) {
          const n = nbr[b + k];
          const c = cells[n];
          if      (c === EMPTY && !visited[n]) { visited[n] = 1; region.push(n); }
          else if (c === BLACK) bBorder = true;
          else if (c === WHITE) wBorder = true;
        }
      }
      if      (bBorder && !wBorder) black += region.length;
      else if (wBorder && !bBorder) white += region.length;
    }

    return { black, white: white + KOMI(N) };
  }

  // Accurate winner using flood-fill territory + komi.  Returns BLACK, WHITE, or null.
  calcWinner() {
    const sc = this.calcScore();
    return sc.black > sc.white ? BLACK : sc.white > sc.black ? WHITE : null;
  }

  // Returns a uniform random legal non-true-eye move, or PASS if none exists.
  // Fast path: try random cells.  Fallback: incremental Fisher-Yates over
  // _allCells in-place (no allocation; order on entry is irrelevant).
  randomLegalMove() {
    const cap = this._allCells.length;

    // Fallback: shuffle _allCells in-place, test each cell.
    const allCells = this._allCells;
    for (let end = cap - 1; end >= 0; end--) {
      const ri  = Math.floor(Math.random() * (end + 1));
      const idx = allCells[ri];
      if (this.isLegal(idx) && !this.isTrueEye(idx)) return idx;
      allCells[ri]  = allCells[end];
      allCells[end] = idx;
    }
    return PASS;
  }

  // Fast 1-step area estimate.  Returns 'black', 'white', or null.
  // Undercounts large interior empty regions; use only for playout rollouts.
  estimateWinner() {
    const N = this.N, cap = N * N;
    const cells = this.cells, nbr = this._nbr;
    let black = 0, white = 0;
    for (let i = 0; i < cap; i++) {
      const c = cells[i];
      if (c === BLACK) { black++; continue; }
      if (c === WHITE) { white++; continue; }
      const base = i * 4;
      let bAdj = false, wAdj = false;
      for (let k = 0; k < 4; k++) {
        const nc = cells[nbr[base + k]];
        if (nc === BLACK) bAdj = true;
        else if (nc === WHITE) wAdj = true;
      }
      if (bAdj && !wAdj) black++;
      else if (wAdj && !bAdj) white++;
    }
    return black > white + KOMI(N) ? BLACK
         : white + KOMI(N) > black ? WHITE
         : null;
  }

}

// Flat index → coordinate string, e.g. 10 on a 9×9 board → "b2".  PASS → 'pass'.
function coordStr(move, N) {
  if (move === PASS) return 'pass';
  return String.fromCharCode(97 + move % N) + ((move / N | 0) + 1);
}

// Coordinate string → flat index.  'pass' → PASS.
function parseMove(str, N) {
  if (str === 'pass') return PASS;
  const x = str.charCodeAt(0) - 97;
  const y = parseInt(str.slice(1), 10) - 1;
  return y * N + x;
}

// Agent move object { type, x, y } → flat index.
function agentMoveToIdx(agentMove, N) {
  return agentMove.type === 'pass' ? PASS : agentMove.y * N + agentMove.x;
}

// Parse a ● ○ · board string.  Mark decoration and alphanumeric labels are
// stripped.  Returns { size, stones } where stones is [[x, y, color], ...]
// and color is BLACK or WHITE.
function parseBoard(boardStr) {
  const rows = boardStr.trim().split('\n')
    .map(r => r.trim().replace(/[()a-zA-Z0-9]/g, ' ').split(/\s+/).filter(t => t))
    .filter(row => row.some(t => t === '●' || t === '○' || t === '·'));
  const size = rows.length;
  const stones = [];
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      if (rows[y][x] === '●') stones.push([x, y, BLACK]);
      else if (rows[y][x] === '○') stones.push([x, y, WHITE]);
    }
  return { size, stones };
}

const _exports = { Game2, PASS, BLACK, WHITE, KOMI, setKomi, coordStr, parseMove, agentMoveToIdx, parseBoard };
if (typeof module !== 'undefined') module.exports = _exports;
else window.Game2 = _exports;

})();
