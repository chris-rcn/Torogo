'use strict';

// ladder2-static.js — Fast ladder solver using virtual stone tracking.
// Reads the board without copying or modifying it.  Tracks defender
// extensions and attacker blocks in pre-allocated bitmask arrays,
// computing group liberties incrementally via flood fill.

(function() {

const { PASS, coordStr } = typeof require === 'function' ? require('./game2.js') : window.Game2;
const Ladder2 = typeof require === 'function' ? require('./ladder2.js') : window.Ladder2;

// ── Pre-allocated scratch buffers ───────────────────────────────────────────

const MAX_CAP = 361;

const _groupMask = new Uint8Array(MAX_CAP);
const _groupList = new Int16Array(MAX_CAP);
let _groupLen = 0;

const _vAtkMask = new Uint8Array(MAX_CAP);

const _capMask = new Uint8Array(MAX_CAP);
const _capList = new Int16Array(MAX_CAP * 4);
let _capLen = 0;

const _libSeen = new Int32Array(MAX_CAP);
let _libGen = 0;

const _stk = new Int16Array(MAX_CAP);

let _fallbackCount = 0;
let _totalCalls = 0;
let _debugVerify = false;
let _divergences = 0;
let _traceLog = false;
let _traceN = 0;

// ── Effective cell state ────────────────────────────────────────────────────

function _isVirtuallyEmpty(board, ni) {
  if (_groupMask[ni]) return false;
  if (_capMask[ni])   return true;
  if (_vAtkMask[ni])  return false;
  return board[ni] === 0;
}

function _isVirtuallyAtk(board, ni, atkColor) {
  if (_capMask[ni]) return false;
  return board[ni] === atkColor || _vAtkMask[ni];
}

// ── Liberty computation ─────────────────────────────────────────────────────

function _computeLibs(board, nbr) {
  _libGen++;
  const gen = _libGen;
  let count = 0, lib0 = -1, lib1 = -1;
  for (let gi = 0; gi < _groupLen; gi++) {
    const si = _groupList[gi];
    const base = si * 4;
    for (let d = 0; d < 4; d++) {
      const ni = nbr[base + d];
      if (_libSeen[ni] === gen) continue;
      _libSeen[ni] = gen;
      if (_isVirtuallyEmpty(board, ni)) {
        if (count === 0) lib0 = ni;
        else if (count === 1) lib1 = ni;
        if (++count >= 3) return 3 | (lib0 << 8) | (lib1 << 20);
      }
    }
  }
  return count | (lib0 << 8) | (lib1 << 20);
}

function _unpackCount(packed) { return packed & 0xFF; }
function _unpackLib0(packed)  { return (packed >> 8) & 0xFFF; }
function _unpackLib1(packed)  { return (packed >> 20) & 0xFFF; }

// ── Flood fill helpers ──────────────────────────────────────────────────────

// Check if the virtual attacker group at startIdx has any liberty.
function _atkGroupHasAnyLib(board, nbr, cap, startIdx, atkColor) {
  _libGen++;
  const gen = _libGen;
  _libSeen[startIdx] = gen;
  _stk[0] = startIdx;
  let si = 0, sn = 1;
  while (si < sn) {
    const ci = _stk[si++];
    const base = ci * 4;
    for (let d = 0; d < 4; d++) {
      const ni = nbr[base + d];
      if (_libSeen[ni] === gen) continue;
      _libSeen[ni] = gen;
      if (_isVirtuallyAtk(board, ni, atkColor)) {
        _stk[sn++] = ni;
      } else if (_isVirtuallyEmpty(board, ni)) {
        return true;
      }
    }
  }
  return false;
}

// Check if the virtual attacker group at startIdx has any liberty
// OTHER THAN excludeLib.
function _atkGroupHasOtherLib(board, nbr, cap, startIdx, excludeLib, atkColor) {
  _libGen++;
  const gen = _libGen;
  _libSeen[startIdx] = gen;
  _stk[0] = startIdx;
  let si = 0, sn = 1;
  while (si < sn) {
    const ci = _stk[si++];
    const base = ci * 4;
    for (let d = 0; d < 4; d++) {
      const ni = nbr[base + d];
      if (_libSeen[ni] === gen) continue;
      _libSeen[ni] = gen;
      if (_isVirtuallyAtk(board, ni, atkColor)) {
        _stk[sn++] = ni;
      } else if (ni !== excludeLib && _isVirtuallyEmpty(board, ni)) {
        return true;
      }
    }
  }
  return false;
}

// Flood fill and mark all stones in the virtual attacker group as captured.
function _doCapture(board, nbr, cap, startIdx, atkColor) {
  _libGen++;
  const gen = _libGen;
  _libSeen[startIdx] = gen;
  _stk[0] = startIdx;
  let si = 0, sn = 1;
  let count = 0;
  while (si < sn) {
    const ci = _stk[si++];
    if (!_capMask[ci]) {
      _capMask[ci] = 1;
      _capList[_capLen++] = ci;
      count++;
    }
    const base = ci * 4;
    for (let d = 0; d < 4; d++) {
      const ni = nbr[base + d];
      if (_libSeen[ni] === gen) continue;
      _libSeen[ni] = gen;
      if (_isVirtuallyAtk(board, ni, atkColor)) {
        _stk[sn++] = ni;
      }
    }
  }
  return count;
}

// After placing defender at lib, check adjacent attacker groups for captures.
// Counts new liberties gained from captures without modifying _capMask.
// Returns:  1 = capture gives 3+ libs (immediate escape)
//           0 = no captures
//          -1 = capture detected but < 3 libs (fall back)
function _handleCaptures(board, nbr, cap, lib, atkColor) {
  const base = lib * 4;
  let anyCap = false;
  // Count current libs first (before captures).
  const curPacked = _computeLibs(board, nbr);
  let curLibs = _unpackCount(curPacked);
  if (curLibs >= 3) return 0;  // already 3+ libs, no capture needed

  // For each adjacent attacker group with 0 remaining libs:
  // count how many of its stones are adjacent to the defender group
  // (those become new liberties after capture).
  // Use a generation to track which captured groups we've already counted.
  _libGen++;
  const capGen = _libGen;
  for (let d = 0; d < 4; d++) {
    const ni = nbr[base + d];
    if (!_isVirtuallyAtk(board, ni, atkColor)) continue;
    if (_libSeen[ni] === capGen) continue;  // already counted this group
    if (!_atkGroupHasOtherLib(board, nbr, cap, ni, lib, atkColor)) {
      // This group would be captured.  Count new libs and mark stones as seen.
      anyCap = true;
      curLibs += _countNewLibsFromCapture(board, nbr, cap, ni, atkColor, capGen);
      if (curLibs >= 3) return 1;  // escape!
    }
  }
  if (!anyCap) return 0;
  return -1;  // capture with < 3 libs — fall back
}

// Count how many cells in the captured attacker group are adjacent to
// the defender group (and not already counted as liberties).
// These cells become new liberties after capture.
// Flood fill the captured group from startIdx.  Count cells adjacent to
// _groupMask (these become new liberties after capture).  Mark all group
// stones with capGen to prevent double-counting across multiple captures.
function _countNewLibsFromCapture(board, nbr, cap, startIdx, atkColor, capGen) {
  _libGen++;
  const gen = _libGen;
  _libSeen[startIdx] = gen;
  _stk[0] = startIdx;
  let si = 0, sn = 1;
  let newLibs = 0;
  while (si < sn) {
    const ci = _stk[si++];
    _libSeen[ci] = capGen;  // mark for cross-group dedup
    // Check if this captured cell is adjacent to the defender group.
    const base2 = ci * 4;
    for (let d2 = 0; d2 < 4; d2++) {
      if (_groupMask[nbr[base2 + d2]]) { newLibs++; break; }
    }
    // Traverse the captured group.
    for (let d2 = 0; d2 < 4; d2++) {
      const ni2 = nbr[base2 + d2];
      if (_libSeen[ni2] === gen || _libSeen[ni2] === capGen) continue;
      _libSeen[ni2] = gen;
      if (_isVirtuallyAtk(board, ni2, atkColor)) {
        _stk[sn++] = ni2;
      }
    }
  }
  return newLibs;
}


// Would attacker playing at lib capture a non-main defender group?
// Only falls back when the group would actually be captured (0 libs after play).
// lib is adjacent to the group and is one of its liberties, so playing there
// reduces its lib count by 1.  Only dangerous if it had exactly 1 lib (= lib).
function _atkCapturesNonMainDef(game2, board, nbr, lib, defColor) {
  const base = lib * 4;
  for (let d = 0; d < 4; d++) {
    const ni = nbr[base + d];
    if (board[ni] === defColor && !_groupMask[ni] && !_capMask[ni]) {
      const gid = game2._gid[ni];
      if (game2._ls[gid] === 1) return true;  // would be captured
    }
  }
  return false;
}

// ── Group merge (ladder breaker) ────────────────────────────────────────────

function _mergeAdjacentFriendly(board, nbr, cap, lib, defColor) {
  let added = 0;
  const base = lib * 4;
  for (let d = 0; d < 4; d++) {
    const ni = nbr[base + d];
    if (board[ni] === defColor && !_groupMask[ni]) {
      _groupMask[ni] = 1;
      _groupList[_groupLen++] = ni;
      added++;
      let si = 0, sn = 1;
      _stk[0] = ni;
      while (si < sn) {
        const ci = _stk[si++];
        const b2 = ci * 4;
        for (let d2 = 0; d2 < 4; d2++) {
          const ni2 = nbr[b2 + d2];
          if (board[ni2] === defColor && !_groupMask[ni2]) {
            _groupMask[ni2] = 1;
            _groupList[_groupLen++] = ni2;
            added++;
            _stk[sn++] = ni2;
          }
        }
      }
    }
  }
  return added;
}

// ── Capture-escape detection ────────────────────────────────────────────────

// Is there an adjacent attacker group (on the actual board) in atari whose
// liberty is NOT lib0 or lib1?  If so, the defender can capture it without
// consuming any of its own libs → guaranteed 3+ libs.
// Check if any adjacent attacker group is in atari with liberty outside
// the defender's libs.  Uses game2's original data — accurate at depth 0,
// conservative at deeper depths.
function _hasCaptureEscape(game2, lib0, lib1) {
  const cells = game2.cells;
  const gidArr = game2._gid;
  const nbr = game2._nbr;
  const ls = game2._ls;
  const defColor = cells[_groupList[0]];
  const atkColor = -defColor;
  let seen0 = -1, seen1 = -1, seen2 = -1, seen3 = -1;
  for (let gi = 0; gi < _groupLen; gi++) {
    const si = _groupList[gi];
    if (cells[si] !== defColor) continue;
    for (let d = 0; d < 4; d++) {
      const ni = nbr[si * 4 + d];
      if (cells[ni] !== atkColor) continue;
      const agid = gidArr[ni];
      if (agid === seen0 || agid === seen1 || agid === seen2 || agid === seen3) continue;
      if      (seen0 === -1) seen0 = agid;
      else if (seen1 === -1) seen1 = agid;
      else if (seen2 === -1) seen2 = agid;
      else                   seen3 = agid;
      if (ls[agid] !== 1) continue;
      const { lib0: alib } = game2.groupLibs2(ni);
      if (alib !== lib0 && alib !== lib1) return true;
    }
  }
  return false;
}

// Conservative version: any adjacent attacker group with few libs might
// become capturable due to virtual stones.  For fallback (return null).
function _mightHaveCaptureEscape(game2) {
  const cells = game2.cells;
  const gidArr = game2._gid;
  const nbr = game2._nbr;
  const ls = game2._ls;
  const defColor = cells[_groupList[0]];
  const atkColor = -defColor;
  let seen0 = -1, seen1 = -1, seen2 = -1, seen3 = -1;
  for (let gi = 0; gi < _groupLen; gi++) {
    const si = _groupList[gi];
    if (cells[si] !== defColor) continue;
    for (let d = 0; d < 4; d++) {
      const ni = nbr[si * 4 + d];
      if (cells[ni] !== atkColor) continue;
      const agid = gidArr[ni];
      if (agid === seen0 || agid === seen1 || agid === seen2 || agid === seen3) continue;
      if      (seen0 === -1) seen0 = agid;
      else if (seen1 === -1) seen1 = agid;
      else if (seen2 === -1) seen2 = agid;
      else                   seen3 = agid;
      if (ls[agid] <= 5) return true;  // conservative: virtual stones might reduce to atari
    }
  }
  return false;
}

// Check if both lib0 and lib1 are adjacent to a friendly group (on the
// actual board, not in _groupMask) with 3+ libs.  If so, the defender
// can always extend to whichever the attacker doesn't block and merge,
// reaching 3+ libs regardless of attacker play.
function _bothLibsConnect(game2, lib0, lib1) {
  const cells = game2.cells;
  const nbr = game2._nbr;
  const gidArr = game2._gid;
  const ls = game2._ls;
  const defColor = cells[_groupList[0]];
  return _libConnects(nbr, cells, gidArr, ls, lib0, defColor) &&
         _libConnects(nbr, cells, gidArr, ls, lib1, defColor);
}

function _libConnects(nbr, cells, gidArr, ls, lib, defColor) {
  const base = lib * 4;
  for (let d = 0; d < 4; d++) {
    const ni = nbr[base + d];
    if (cells[ni] === defColor && !_groupMask[ni] && ls[gidArr[ni]] >= 3) return true;
  }
  return false;
}

// ── Core recursive tracer ───────────────────────────────────────────────────

function _traceRec(game2, board, nbr, cap, idx, defColor, atkColor, current, depth, ko) {
  if (depth > 100) return null;

  const packed = _computeLibs(board, nbr);
  const lc = _unpackCount(packed);
  const cs = _traceLog ? (i) => coordStr(i, _traceN) : null;
  if (_traceLog) console.log(`${'  '.repeat(depth)}d=${depth} libs=${lc}${lc >= 1 ? ' ' + cs(_unpackLib0(packed)) : ''}${lc >= 2 ? ' ' + cs(_unpackLib1(packed)) : ''} ${current === defColor ? 'DEF' : 'ATK'}`);
  if (lc >= 3) { if (_traceLog) console.log(`${'  '.repeat(depth)}→ escape (3+ libs)`); return true; }
  if (lc === 0) { if (_traceLog) console.log(`${'  '.repeat(depth)}→ captured (0 libs)`); return false; }

  const lib0 = _unpackLib0(packed);
  const lib1 = _unpackLib1(packed);

  if (current === defColor) {
    if (lc === 2) {
      // If both liberties connect to a friendly group, the defender can
      // always extend to one of them and merge — reaching 3+ libs.
      if (_bothLibsConnect(game2, lib0, lib1)) {
        if (_traceLog) console.log(`${'  '.repeat(depth)}→ escape (both libs connect)`);
        return true;
      }
      // Adjacent attacker in atari with liberty outside ours?
      // Capturing doesn't consume any of our libs → 2 + freed ≥ 3.
      if (depth === 0 && _hasCaptureEscape(game2, lib0, lib1)) {
        if (_traceLog) console.log(`${'  '.repeat(depth)}→ escape (capture-escape, 2 libs)`);
        return true;
      }
    }

    for (let li = 0; li < lc; li++) {
      const lib = li === 0 ? lib0 : lib1;
      if (_traceLog) console.log(`${'  '.repeat(depth)}DEF extend ${cs(lib)}`);

      const prevLen = _groupLen;
      const prevCapLen = _capLen;
      _groupMask[lib] = 1;
      _groupList[_groupLen++] = lib;

      const mergeCount = _mergeAdjacentFriendly(board, nbr, cap, lib, defColor);
      if (_traceLog && mergeCount > 0) console.log(`${'  '.repeat(depth)}  merged ${mergeCount} stones`);
      const capResult = _handleCaptures(board, nbr, cap, lib, atkColor);
      if (_traceLog && capResult !== 0) console.log(`${'  '.repeat(depth)}  capture → ${capResult === 1 ? 'escape (3+ libs)' : 'fallback'}`);

      const result = capResult === 1 ? true :
                     capResult === -1 ? null :
                     _traceRec(game2, board, nbr, cap, idx, defColor, atkColor, atkColor, depth + 1, -1);

      while (_capLen > prevCapLen) _capMask[_capList[--_capLen]] = 0;
      while (_groupLen > prevLen) _groupMask[_groupList[--_groupLen]] = 0;

      if (result === null) return null;
      if (result) return true;
    }
    // All branches failed. Check for capture escapes the static solver can't model.
    if (_mightHaveCaptureEscape(game2)) return null;
    return false;
  }

  // Attacker tries each liberty.
  for (let li = 0; li < lc; li++) {
    const lib = li === 0 ? lib0 : lib1;

    if (lib === ko) { if (_traceLog) console.log(`${'  '.repeat(depth)}ATK ${cs(lib)} ko → skip`); continue; }
    if (_atkCapturesNonMainDef(game2, board, nbr, lib, defColor)) { if (_traceLog) console.log(`${'  '.repeat(depth)}ATK ${cs(lib)} captures non-main def → fallback`); return null; }

    _vAtkMask[lib] = 1;

    const afterPacked = _computeLibs(board, nbr);
    const afterLc = _unpackCount(afterPacked);

    let fail = false;
    if (afterLc === 0) {
      if (_traceLog) console.log(`${'  '.repeat(depth)}ATK ${cs(lib)} captures group`);
      fail = true;
    } else {
      if (!_atkGroupHasAnyLib(board, nbr, cap, lib, atkColor)) {
        if (_traceLog) console.log(`${'  '.repeat(depth)}ATK ${cs(lib)} suicide → skip`);
        _vAtkMask[lib] = 0;
        continue;
      }
      if (_traceLog) console.log(`${'  '.repeat(depth)}ATK block ${cs(lib)}`);
      if (afterLc >= 2) { _vAtkMask[lib] = 0; return null; }
      const result = _traceRec(game2, board, nbr, cap, idx, defColor, atkColor, defColor, depth + 1, -1);
      if (result === null) { _vAtkMask[lib] = 0; return null; }
      if (!result) fail = true;
    }

    _vAtkMask[lib] = 0;
    if (fail) return false;
  }
  return true;
}

// ── Entry points ────────────────────────────────────────────────────────────

function _initGroup(game2, stoneIdx) {
  const board = game2.cells;
  const nbr = game2._nbr;
  const cap = game2.N * game2.N;
  const defColor = board[stoneIdx];

  _groupMask.fill(0, 0, cap);
  _vAtkMask.fill(0, 0, cap);
  _capMask.fill(0, 0, cap);
  _groupLen = 0;
  _capLen = 0;

  _stk[0] = stoneIdx;
  _groupMask[stoneIdx] = 1;
  _groupList[_groupLen++] = stoneIdx;
  let si = 0, sn = 1;
  while (si < sn) {
    const ci = _stk[si++];
    const base = ci * 4;
    for (let d = 0; d < 4; d++) {
      const ni = nbr[base + d];
      if (board[ni] === defColor && !_groupMask[ni]) {
        _groupMask[ni] = 1;
        _groupList[_groupLen++] = ni;
        _stk[sn++] = ni;
      }
    }
  }
}

function _staticCanReach3Libs(game2, stoneIdx, current) {
  _totalCalls++;
  _initGroup(game2, stoneIdx);
  const board = game2.cells;
  const nbr = game2._nbr;
  const cap = game2.N * game2.N;
  const defColor = board[stoneIdx];
  const atkColor = -defColor;

  const result = _traceRec(game2, board, nbr, cap, stoneIdx, defColor, atkColor, current, 0, -1);

  // DEBUG: verify against reference
  if (result !== null && _debugVerify) {
    const g = game2.clone();
    g.current = current;
    const ref = _fallbackCanReach3Libs(g, stoneIdx);
    if (result !== ref) {
      _divergences++;
      if (_divergences <= 1) {
        const N = game2.N;
        const cs = (idx) => coordStr(idx, N);
        console.log(`DIVERGE: static=${result} ref=${ref} stone=${cs(stoneIdx)} current=${current > 0 ? 'B' : 'W'}`);
        console.log(game2.toString(stoneIdx, { labels: true }));
        const { count: lc2, lib0: l0, lib1: l1 } = game2.groupLibs2(stoneIdx);
        console.log(`  groupSize=${game2.groupSize(game2._gid[stoneIdx])} libs=${lc2}: ${cs(l0)}${lc2 > 1 ? ' ' + cs(l1) : ''}`);
        // Re-run with tracing
        console.log('--- TRACE ---');
        _traceLog = true;
        _traceN = N;
        _initGroup(game2, stoneIdx);
        _traceRec(game2, board, nbr, cap, stoneIdx, defColor, atkColor, current, 0, -1);
        _traceLog = false;
      }
      return ref;  // use correct result
    }
  }

  if (result !== null) return result;

  _fallbackCount++;
  const g = game2.clone();
  g.current = current;
  return _fallbackCanReach3Libs(g, stoneIdx);
}

// Delegate to ladder2.js _canReach3Libs (includes capture-escape search).
function _fallbackCanReach3Libs(game2, idx) {
  return Ladder2._canReach3Libs(game2, idx);
}

// ── Public API ──────────────────────────────────────────────────────────────

function getLadderStatus2(game2, stoneIdx) {
  const { count: lc, lib0, lib1 } = game2.groupLibs2(stoneIdx);
  if (lc < 1 || lc > 2) {
    const N = game2.N;
    console.warn(`getLadderStatus2: group at ${stoneIdx % N},${(stoneIdx / N) | 0} has ${lc} liberties (expected ≤ 2)`);
    return null;
  }
  const atari = lc === 1;
  const libs = atari ? [lib0] : [lib0, lib1];
  const gColor = game2.cells[stoneIdx];
  const mover = game2.current;
  const defending = gColor === mover;

  let escape;
  if (defending && atari) {
    escape = false;
  } else {
    escape = _staticCanReach3Libs(game2, stoneIdx, -mover);
  }
  if (defending === escape) {
    return { libs, moverSucceeds: true, urgentLibs: [] };
  }

  let moverSucceeds = false;
  const urgentLibs = [];
  for (const libIdx of libs) {
    if (!game2.isLegal(libIdx)) continue;  // skip illegal (ko/suicide)
    if (!defending && atari) {
      escape = false;
    } else {
      escape = _staticCanReach3LibsAfterPlay(game2, stoneIdx, libIdx, mover);
    }
    if (defending === escape) {
      moverSucceeds = true;
      urgentLibs.push(libIdx);
    }
  }
  return { libs, moverSucceeds, urgentLibs };
}

function _staticCanReach3LibsAfterPlay(game2, stoneIdx, playIdx, mover) {
  _totalCalls++;
  if (!game2.isLegal(playIdx)) return false;
  const board = game2.cells;
  const nbr = game2._nbr;
  const cap = game2.N * game2.N;
  const defColor = board[stoneIdx];
  const atkColor = -defColor;
  const defending = defColor === mover;

  _initGroup(game2, stoneIdx);

  if (defending) {
    _groupMask[playIdx] = 1;
    _groupList[_groupLen++] = playIdx;
    _mergeAdjacentFriendly(board, nbr, cap, playIdx, defColor);
    const capResult = _handleCaptures(board, nbr, cap, playIdx, atkColor);
    if (capResult === 1) return true;   // capture gives 3+ libs → escape
    if (capResult === -1) {
      _fallbackCount++;
      const g2 = game2.clone();
      if (!g2.play(playIdx)) return false;
      if (g2.cells[stoneIdx] === 0) return false;
      return _fallbackCanReach3Libs(g2, stoneIdx);
    }
  } else {
    // Attacker plays: check for non-main defender interaction.
    if (_atkCapturesNonMainDef(game2, board, nbr, playIdx, defColor)) {
      _fallbackCount++;
      const g2 = game2.clone();
      if (!g2.play(playIdx)) return false;
      if (g2.cells[stoneIdx] === 0) return false;
      return _fallbackCanReach3Libs(g2, stoneIdx);
    }
    _vAtkMask[playIdx] = 1;
  }

  const result = _traceRec(game2, board, nbr, cap, stoneIdx, defColor, atkColor, -mover, 0, -1);

  if (!defending) _vAtkMask[playIdx] = 0;

  if (result !== null && _debugVerify) {
    const g2 = game2.clone();
    if (!g2.play(playIdx)) { if (result !== false) _divergences++; return false; }
    if (g2.cells[stoneIdx] === 0) { if (result !== false) _divergences++; return false; }
    const ref = _fallbackCanReach3Libs(g2, stoneIdx);
    if (result !== ref) {
      _divergences++;
      if (_divergences <= 1) {
        const N = game2.N;
        console.log(`DIVERGE(afterPlay): static=${result} ref=${ref} stone=${coordStr(stoneIdx,N)} play=${coordStr(playIdx,N)}`);
        console.log(game2.toString(stoneIdx, { labels: true }));
      }
      return ref;
    }
  }

  if (result !== null) return result;

  _fallbackCount++;
  const g = game2.clone();
  if (!g.play(playIdx)) return false;
  if (g.cells[stoneIdx] === 0) return false;
  return _fallbackCanReach3Libs(g, stoneIdx);
}

// ── Pure-static API (no fallback — returns null for unsolvable groups) ─────

function _pureStaticCanReach3Libs(game2, stoneIdx, current) {
  _totalCalls++;
  _initGroup(game2, stoneIdx);
  const board = game2.cells;
  const nbr = game2._nbr;
  const cap = game2.N * game2.N;
  const defColor = board[stoneIdx];
  const atkColor = -defColor;
  const result = _traceRec(game2, board, nbr, cap, stoneIdx, defColor, atkColor, current, 0, -1);
  if (result !== null && _debugVerify) {
    const g = game2.clone(); g.current = current;
    if (result !== _fallbackCanReach3Libs(g, stoneIdx)) { _divergences++; return null; }
  }
  return result;
}

function _pureStaticCanReach3LibsAfterPlay(game2, stoneIdx, playIdx, mover) {
  _totalCalls++;
  if (!game2.isLegal(playIdx)) return false;
  const board = game2.cells;
  const nbr = game2._nbr;
  const cap = game2.N * game2.N;
  const defColor = board[stoneIdx];
  const atkColor = -defColor;
  const defending = defColor === mover;

  _initGroup(game2, stoneIdx);

  if (defending) {
    _groupMask[playIdx] = 1;
    _groupList[_groupLen++] = playIdx;
    _mergeAdjacentFriendly(board, nbr, cap, playIdx, defColor);
    const capResult = _handleCaptures(board, nbr, cap, playIdx, atkColor);
    if (capResult === 1) return true;
    if (capResult === -1) return null;
  } else {
    if (_atkCapturesNonMainDef(game2, board, nbr, playIdx, defColor)) return null;
    _vAtkMask[playIdx] = 1;
  }

  const result = _traceRec(game2, board, nbr, cap, stoneIdx, defColor, atkColor, -mover, 0, -1);
  if (!defending) _vAtkMask[playIdx] = 0;
  if (result !== null && _debugVerify) {
    const g2 = game2.clone();
    if (!g2.play(playIdx)) { if (result !== false) { _divergences++; return null; } }
    else if (g2.cells[stoneIdx] === 0) { if (result !== false) { _divergences++; return null; } }
    else if (result !== _fallbackCanReach3Libs(g2, stoneIdx)) { _divergences++; return null; }
  }
  return result;
}

function getStaticLadderStatus2(game2, stoneIdx) {
  const { count: lc, lib0, lib1 } = game2.groupLibs2(stoneIdx);
  if (lc < 1 || lc > 2) return null;
  const atari = lc === 1;
  const libs = atari ? [lib0] : [lib0, lib1];
  const gColor = game2.cells[stoneIdx];
  const mover = game2.current;
  const defending = gColor === mover;

  let escape;
  if (defending && atari) {
    escape = false;
  } else {
    escape = _pureStaticCanReach3Libs(game2, stoneIdx, -mover);
    if (escape === null) return null;
  }
  if (defending === escape) {
    return { libs, moverSucceeds: true, urgentLibs: [] };
  }

  let moverSucceeds = false;
  const urgentLibs = [];
  for (const libIdx of libs) {
    if (!game2.isLegal(libIdx)) continue;  // skip illegal (ko/suicide)
    if (!defending && atari) {
      escape = false;
    } else {
      escape = _pureStaticCanReach3LibsAfterPlay(game2, stoneIdx, libIdx, mover);
      if (escape === null) return null;
    }
    if (defending === escape) {
      moverSucceeds = true;
      urgentLibs.push(libIdx);
    }
  }
  return { libs, moverSucceeds, urgentLibs };
}

function getAllStaticLadderStatuses(game2, minChainSize = 1) {
  const cap = game2.N * game2.N;
  const results = [];
  const visited = new Set();
  for (let i = 0; i < cap; i++) {
    if (game2.cells[i] === 0) continue;
    const gid = game2._gid[i];
    if (visited.has(gid)) continue;
    visited.add(gid);
    if (game2.groupSize(gid) < minChainSize) continue;
    const { count: lc } = game2.groupLibs2(i);
    if (lc === 0 || lc > 2) continue;
    const status = getStaticLadderStatus2(game2, i);
    if (status !== null) {
      results.push({ gid, color: game2.cells[i], status });
    }
  }
  return results;
}

// ── Fallback API (always returns a result) ──────────────────────────────────

function getAllLadderStatuses(game2, minChainSize = 1) {
  const cap = game2.N * game2.N;
  const results = [];
  const visited = new Set();
  for (let i = 0; i < cap; i++) {
    if (game2.cells[i] === 0) continue;
    const gid = game2._gid[i];
    if (visited.has(gid)) continue;
    visited.add(gid);
    if (game2.groupSize(gid) < minChainSize) continue;
    const { count: lc } = game2.groupLibs2(i);
    if (lc === 0 || lc > 2) continue;
    const status = getLadderStatus2(game2, i);
    results.push({ gid, color: game2.cells[i], status });
  }
  return results;
}

function getFallbackCount() { return _fallbackCount; }
function getTotalCalls() { return _totalCalls; }
function resetFallbackCount() { _fallbackCount = 0; _totalCalls = 0; }
function setDebugVerify(v) { _debugVerify = v; _divergences = 0; }
function getDivergences() { return _divergences; }

const _exports = {
  getLadderStatus2, getAllLadderStatuses,
  getStaticLadderStatus2, getAllStaticLadderStatuses,
  getFallbackCount, getTotalCalls, resetFallbackCount, setDebugVerify, getDivergences,
};
if (typeof module !== 'undefined') module.exports = _exports;
else window.Ladder2Static = _exports;

})();
