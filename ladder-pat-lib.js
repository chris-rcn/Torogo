'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs (require, process, etc.).
// Loaded as a plain <script> tag; ladder2.js must be loaded first.

(function () {

// ── Feature toggles (edit to enable/disable for profiling or ablation) ───────
const F_LADDER     = 1;  // 283 ms
const F_PATTERN    = 1;  //  20 ms
const F_CHAIN_SIZE = 1;  //  12 ms
const F_DISTANCE   = 1;  //   9 ms
const F_SELF_ATARI = 1;  //   9 ms
const F_CAPTURE    = 1;  //   3 ms

const DEFAULT_MIN_STONES = 2;
const DEFAULT_MAX_STONES = 8;
const MAX_BOARD_SIZE = 13;
const MAX_CAP = MAX_BOARD_SIZE * MAX_BOARD_SIZE;
const MAX_PREV_DIST = 10;

const { getAllLadderStatuses: _getAllLadderStatuses } = typeof require === 'function'
  ? require('./ladder2.js') : window.Ladder2;
const { game3FromGame2 } = typeof require === 'function'
  ? require('./game3.js') : window.Game3;
const { makeZobrist } = typeof require === 'function'
  ? require('./util.js') : window.Util;

// ladder2.js operates on Game3 (play/undo). Convert from Game2 at the call site.
const getAllLadderStatuses = (game2, min) => _getAllLadderStatuses(game3FromGame2(game2), min);

// ── D4 position permutations ─────────────────────────────────────────────────
// Positions 0–7: N=0, E=1, S=2, W=3, NE=4, SE=5, SW=6, NW=7
const _D4 = [
  [0,1,2,3,4,5,6,7],
  [1,2,3,0,5,6,7,4],
  [2,3,0,1,6,7,4,5],
  [3,0,1,2,7,4,5,6],
  [0,3,2,1,7,6,5,4],
  [2,1,0,3,5,4,7,6],
  [3,2,1,0,6,5,4,7],
  [1,0,3,2,4,7,6,5],
];

// ── Pattern encoding ─────────────────────────────────────────────────────────
//
// Adj states: 0=empty, then for each of friend/foe: one state per lib bucket.
//   maxAdjLibs=1 → 3 states (empty/friend/foe)
//   maxAdjLibs=2 → 5 states (empty/friend/friend-atari/foe/foe-atari)
//   maxAdjLibs=3 → 7 states (empty/friend/friend-2lib/friend-atari/foe/…)
// Diag states (3): 0=empty, 1=friend, 2=foe
//
// Canonical key = min raw encoding across all 8 D4 transforms.
// Cached lazily on first encounter.

const DIAG_STATES = 3;

// ── Feature key hashing ──────────────────────────────────────────────────────
// All keys are full-range int32 to avoid artificial partitioning.
// Seeds for all feature types are derived from a single master table.
const _seeds = makeZobrist(12345, 8);
let _s = 0;
const _patSaltSeed    = _seeds[_s++];
const _ladderSeed     = _seeds[_s++];
const _captureSeed    = _seeds[_s++];
const _distSeed       = _seeds[_s++];
const _selfAtariSeed  = _seeds[_s++];
const _chainSizeSeed  = _seeds[_s++];

// Pattern salts: one per adj-lib threshold.
const _patSalts = makeZobrist(_patSaltSeed, 8);

// Ladder keys: 4 types × max levels — indexed as [type * MAX_LEVELS + level].
const MAX_LADDER_LEVELS = 8;
const _ladderKeys = makeZobrist(_ladderSeed, 4 * MAX_LADDER_LEVELS);
const TYPE_KILL_URGENT = 0;
const TYPE_SAVE_URGENT = 1;
const TYPE_KILL_WASTED = 2;
const TYPE_SAVE_WASTED = 3;

// Capture keys: unary per level (1+, 2+, 3+, 4+).
const MAX_CAPTURE_FEAT = 4;
const _captureKeys = makeZobrist(_captureSeed, MAX_CAPTURE_FEAT);

// Distance-to-previous-move keys: unary per level.
const _distKeys = makeZobrist(_distSeed, MAX_PREV_DIST);

// Self-atari keys: unary by group size.
const MAX_SELF_ATARI = 8;
const _selfAtariKeys = makeZobrist(_selfAtariSeed, MAX_SELF_ATARI);

// Resulting chain size keys: unary by merged group size after playing.
const MAX_CHAIN_SIZE = 8;
const _chainSizeKeys = makeZobrist(_chainSizeSeed, MAX_CHAIN_SIZE);

function normalizeKey(hash) {
  return (hash | 0) || 1;
}

// Create a reusable feature extractor.
// opts.maxAdjLibs: 1 = no lib info (3 adj states), 2 = atari (5), 3 = +2lib (7), etc.
// opts.minLadderStones: minimum chain size for ladder search (default 2).
// opts.maxLadderStones: 0 = skip ladder features entirely, >0 = unary up to this count.
function createLadderPat(opts) {
  opts = opts || {};
  const maxAdjLibs = opts.maxAdjLibs || 1;
  const maxLadderStones = opts.maxLadderStones !== undefined ? opts.maxLadderStones : DEFAULT_MAX_STONES;
  const minLadderStones = opts.minLadderStones !== undefined ? opts.minLadderStones : DEFAULT_MIN_STONES;

  // Build one canon function per threshold t (1..maxAdjLibs).
  // Each has its own adj state count, cache, and hash salt.
  const _canons = [];
  for (let t = 1; t <= maxAdjLibs; t++) {
    const A = 1 + 2 * t;
    const D = DIAG_STATES;
    const salt = _patSalts[t - 1];
    const cache = new Map();
    _canons.push(function canon(v0, v1, v2, v3, v4, v5, v6, v7) {
      const raw = v0 + A*(v1 + A*(v2 + A*(v3 + A*(v4 + D*(v5 + D*(v6 + D*v7))))));
      let c = cache.get(raw);
      if (c !== undefined) return c;

      const v = [v0, v1, v2, v3, v4, v5, v6, v7];
      let minV = raw;
      for (let di = 1; di < 8; di++) {
        const p = _D4[di];
        const enc = v[p[0]] + A*(v[p[1]] + A*(v[p[2]] + A*(v[p[3]]
          + A*(v[p[4]] + D*(v[p[5]] + D*(v[p[6]] + D*v[p[7]]))))));
        if (enc < minV) minV = enc;
      }
      const key = Math.imul(minV ^ salt, 0x5bd1e995) | 0;
      cache.set(raw, key);
      return key;
    });
  }

  // Compute adj state values for all thresholds at once.
  // Writes into buf[offset..offset+maxAdjLibs-1].
  // At threshold t: 0=empty, friend states 1..t, foe states t+1..2t
  const _adjBuf = new Int32Array(4 * 8); // 4 neighbors × max 8 thresholds
  function adjStates(cellColor, cur, lsArr, gid, buf, offset) {
    const libs = lsArr[gid];
    const isFriend = cellColor === cur;
    for (let ti = 0; ti < maxAdjLibs; ti++) {
      const t = ti + 1;
      const cl = Math.min(libs, t);
      const base = isFriend ? 0 : t;
      buf[offset + ti] = base + t + 1 - cl;
    }
  }

  // Reusable per-cell ladder arrays.
  const killUrgent = new Int32Array(MAX_CAP);
  const saveUrgent = new Int32Array(MAX_CAP);
  const killWasted = new Int32Array(MAX_CAP);
  const saveWasted = new Int32Array(MAX_CAP);

  function getFeatures(game) {
    const N      = game.N;
    if (N > MAX_BOARD_SIZE) throw new Error('ladder-pat: board size ' + N + ' exceeds MAX_BOARD_SIZE ' + MAX_BOARD_SIZE);
    const cells  = game.cells;
    const gidArr = game._gid;
    const lsArr  = game._ls;
    const nbr    = game._nbr;
    const dnbr   = game._dnbr;
    const cur    = game.current;
    const emC    = game._emptyCells;
    const ec     = game.emptyCount;
    const ko     = game.ko;
    const prev   = game.lastMove;
    const hasPrev = prev >= 0;
    const prevX  = hasPrev ? prev % N : 0;
    const prevY  = hasPrev ? (prev / N) | 0 : 0;
    const halfN  = N >> 1;

    if (F_LADDER && maxLadderStones > 0) {
      killUrgent.fill(0); saveUrgent.fill(0);
      killWasted.fill(0); saveWasted.fill(0);

      const statuses = getAllLadderStatuses(game, minLadderStones);
      for (const { gid, color, status } of statuses) {
        const chainSize = game.groupSize(gid);
        const isOpponent = color !== cur;
        if (status.moverSucceeds) {
          const arr = isOpponent ? killUrgent : saveUrgent;
          for (const lib of status.urgentLibs) arr[lib] += chainSize;
        } else {
          const arr = isOpponent ? killWasted : saveWasted;
          for (const lib of status.libs) arr[lib] += chainSize;
        }
      }
    }

    const candidates = [];

    for (let ei = 0; ei < ec; ei++) {
      const idx = emC[ei];
      const b4 = idx * 4;

      const niN = nbr[b4], niS = nbr[b4 + 1], niW = nbr[b4 + 2], niE = nbr[b4 + 3];
      const cN = cells[niN], cS = cells[niS], cW = cells[niW], cE = cells[niE];
      const anyEmpty = (cN === 0) | (cS === 0) | (cW === 0) | (cE === 0);
      if (anyEmpty) {
        if (idx === ko && game._isKo(idx, cur)) continue;
      } else {
        if (game._isSingleSuicide(idx, cur)) continue;
        if (game._isMultiSuicide(idx, cur)) continue;
        if (idx === ko && game._isKo(idx, cur)) continue;
      }

      // True-eye check + capture count + neighbor info
      let friendCount = 0, emptyNbr = 0, firstGid = -2, sameGroup = 0;
      let capCount = 0, capGid0 = -1, capGid1 = -1, capGid2 = -1;
      let maxFriendLibs = 0;
      // Store neighbor color and gid for pattern computation below.
      let gN = -1, gE = -1, gS = -1, gW = -1;
      if (cN === 0) { emptyNbr++; }
      else { gN = gidArr[niN]; if (cN === cur) { friendCount++; const fl = lsArr[gN]; if (fl > maxFriendLibs) maxFriendLibs = fl; if (firstGid === -2) { firstGid = gN; sameGroup = 1; } else if (gN === firstGid) sameGroup++; } else if (lsArr[gN] === 1 && gN !== capGid0 && gN !== capGid1 && gN !== capGid2) { if (capCount === 0) capGid0 = gN; else if (capCount === 1) capGid1 = gN; else if (capCount === 2) capGid2 = gN; capCount++; } }
      if (cE === 0) { emptyNbr++; }
      else { gE = gidArr[niE]; if (cE === cur) { friendCount++; const fl = lsArr[gE]; if (fl > maxFriendLibs) maxFriendLibs = fl; if (firstGid === -2) { firstGid = gE; sameGroup = 1; } else if (gE === firstGid) sameGroup++; } else if (lsArr[gE] === 1 && gE !== capGid0 && gE !== capGid1 && gE !== capGid2) { if (capCount === 0) capGid0 = gE; else if (capCount === 1) capGid1 = gE; else if (capCount === 2) capGid2 = gE; capCount++; } }
      if (cS === 0) { emptyNbr++; }
      else { gS = gidArr[niS]; if (cS === cur) { friendCount++; const fl = lsArr[gS]; if (fl > maxFriendLibs) maxFriendLibs = fl; if (firstGid === -2) { firstGid = gS; sameGroup = 1; } else if (gS === firstGid) sameGroup++; } else if (lsArr[gS] === 1 && gS !== capGid0 && gS !== capGid1 && gS !== capGid2) { if (capCount === 0) capGid0 = gS; else if (capCount === 1) capGid1 = gS; else if (capCount === 2) capGid2 = gS; capCount++; } }
      if (cW === 0) { emptyNbr++; }
      else { gW = gidArr[niW]; if (cW === cur) { friendCount++; const fl = lsArr[gW]; if (fl > maxFriendLibs) maxFriendLibs = fl; if (firstGid === -2) { firstGid = gW; sameGroup = 1; } else if (gW === firstGid) sameGroup++; } else if (lsArr[gW] === 1 && gW !== capGid0 && gW !== capGid1 && gW !== capGid2) { if (capCount === 0) capGid0 = gW; else if (capCount === 1) capGid1 = gW; else if (capCount === 2) capGid2 = gW; capCount++; } }

      if (friendCount === 3 && emptyNbr === 1 && sameGroup === 3) continue;
      if (friendCount === 4) {
        if (sameGroup === 4) continue;
        let dc = 0;
        if (cells[dnbr[b4]]     === cur) dc++;
        if (cells[dnbr[b4 + 1]] === cur) dc++;
        if (cells[dnbr[b4 + 2]] === cur) dc++;
        if (cells[dnbr[b4 + 3]] === cur) dc++;
        if (dc >= 3) continue;
      }

      // Diag values (NE=4, SE=5, SW=6, NW=7)
      const cNE = cells[dnbr[b4 + 1]], vNE = cNE === 0 ? 0 : (cNE === cur ? 1 : 2);
      const cSE = cells[dnbr[b4 + 3]], vSE = cSE === 0 ? 0 : (cSE === cur ? 1 : 2);
      const cSW = cells[dnbr[b4 + 2]], vSW = cSW === 0 ? 0 : (cSW === cur ? 1 : 2);
      const cNW = cells[dnbr[b4]],     vNW = cNW === 0 ? 0 : (cNW === cur ? 1 : 2);

      const keys = [];

      // Pattern features (one per lib threshold, unary)
      if (F_PATTERN) {
        const A = _adjBuf;
        const hasN = cN !== 0, hasE = cE !== 0, hasS = cS !== 0, hasW = cW !== 0;
        if (hasN) adjStates(cN, cur, lsArr, gN, A, 0);
        if (hasE) adjStates(cE, cur, lsArr, gE, A, 8);
        if (hasS) adjStates(cS, cur, lsArr, gS, A, 16);
        if (hasW) adjStates(cW, cur, lsArr, gW, A, 24);
        for (let ti = 0; ti < maxAdjLibs; ti++) {
          keys.push(normalizeKey(_canons[ti](
            hasN ? A[ti] : 0, hasE ? A[8 + ti] : 0, hasS ? A[16 + ti] : 0, hasW ? A[24 + ti] : 0,
            vNE, vSE, vSW, vNW)));
        }
      }

      // Capture features (unary: captures 1+, 2+, 3+, 4+ distinct enemy groups)
      if (F_CAPTURE) {
        const caps = Math.min(capCount, MAX_CAPTURE_FEAT);
        for (let c = 0; c < caps; c++) keys.push(_captureKeys[c]);
      }

      // Distance to previous move (unary: dist ≤ MAX, dist ≤ MAX-1, ...)
      if (F_DISTANCE && hasPrev) {
        const dx = Math.abs(idx % N - prevX);
        const dy = Math.abs((idx / N | 0) - prevY);
        const dist = Math.min(dx, N - dx) + Math.min(dy, N - dy);
        if (dist <= MAX_PREV_DIST) {
          for (let d = dist - 1; d < MAX_PREV_DIST; d++) keys.push(_distKeys[d]);
        }
      }

      // Ladder features (unary)
      if (F_LADDER && maxLadderStones > 0) {
        const ku = Math.min(killUrgent[idx], maxLadderStones);
        const su = Math.min(saveUrgent[idx], maxLadderStones);
        const kw = Math.min(killWasted[idx], maxLadderStones);
        const sw = Math.min(saveWasted[idx], maxLadderStones);
        for (let c = 0; c < ku; c++) keys.push(_ladderKeys[TYPE_KILL_URGENT * MAX_LADDER_LEVELS + c]);
        for (let c = 0; c < su; c++) keys.push(_ladderKeys[TYPE_SAVE_URGENT * MAX_LADDER_LEVELS + c]);
        for (let c = 0; c < kw; c++) keys.push(_ladderKeys[TYPE_KILL_WASTED * MAX_LADDER_LEVELS + c]);
        for (let c = 0; c < sw; c++) keys.push(_ladderKeys[TYPE_SAVE_WASTED * MAX_LADDER_LEVELS + c]);
      }

      // Self-atari feature (unary by resulting group size)
      if (F_SELF_ATARI) {
        let saSize = 0;
        if (emptyNbr >= 2 || maxFriendLibs >= 3) {
          // definitely not self-atari
        } else if (capCount === 0 && emptyNbr === 1 && maxFriendLibs <= 1) {
          saSize = 1;
          if (gN >= 0 && cN === cur) saSize += game.groupSize(gN);
          if (gE >= 0 && cE === cur && gE !== gN) saSize += game.groupSize(gE);
          if (gS >= 0 && cS === cur && gS !== gN && gS !== gE) saSize += game.groupSize(gS);
          if (gW >= 0 && cW === cur && gW !== gN && gW !== gE && gW !== gS) saSize += game.groupSize(gW);
        } else {
          const cg = game.clone();
          cg.play(idx);
          const gid = cg._gid[idx];
          if (gid !== -1 && cg._ls[gid] === 1) saSize = cg.groupSize(gid);
        }
        if (saSize > 0) {
          const sa = Math.min(saSize, MAX_SELF_ATARI);
          for (let c = 0; c < sa; c++) keys.push(_selfAtariKeys[c]);
        }
      }

      // Resulting chain size (unary: 1 + sum of distinct adjacent friendly group sizes)
      if (F_CHAIN_SIZE) {
        let chainSize = 1;
        if (gN >= 0 && cN === cur) chainSize += game.groupSize(gN);
        if (gE >= 0 && cE === cur && gE !== gN) chainSize += game.groupSize(gE);
        if (gS >= 0 && cS === cur && gS !== gN && gS !== gE) chainSize += game.groupSize(gS);
        if (gW >= 0 && cW === cur && gW !== gN && gW !== gE && gW !== gS) chainSize += game.groupSize(gW);
        const cs = Math.min(chainSize, MAX_CHAIN_SIZE);
        for (let c = 0; c < cs; c++) keys.push(_chainSizeKeys[c]);
      }

      candidates.push({ move: idx, keys });
    }
    return candidates;
  }

  return { getFeatures };
}

// Compute unnormalized softmax values over candidates.
// Returns { vals: Float64Array, sum, max } where vals[i] = exp(logit[i] - max).
// Callers normalize as needed: vals[i] / sum gives the probability.
function softmax(candidates, weightFn) {
  const n = candidates.length;
  const vals = new Float64Array(n);
  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    let v = 0;
    for (const k of candidates[i].keys) v += weightFn(k);
    vals[i] = v;
    if (v > max) max = v;
  }
  let sum = 0;
  for (let i = 0; i < n; i++) { vals[i] = Math.exp(vals[i] - max); sum += vals[i]; }
  return { vals, sum, max };
}

const _exports = {
  createLadderPat, softmax, normalizeKey,
  // Exposed for testing:
  _patSalts, _ladderKeys, _captureKeys, _distKeys, _selfAtariKeys, _chainSizeKeys,
  MAX_LADDER_LEVELS, MAX_CAPTURE_FEAT, MAX_PREV_DIST, MAX_SELF_ATARI, MAX_CHAIN_SIZE,
  TYPE_KILL_URGENT, TYPE_SAVE_URGENT, TYPE_KILL_WASTED, TYPE_SAVE_WASTED,
};
if (typeof module !== 'undefined') module.exports = _exports;
else window.LadderPat = _exports;

})();
