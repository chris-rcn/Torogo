'use strict';

(function() {

const Util = (typeof require === 'function') ? require('./util.js') : window.Util;
const { PASS } = Util.load('./game3.js', 'Game3');

// Moves the defender can play to capture an adjacent enemy chain in atari:
// the single remaining liberty of each opponent group touching the chased
// group at `idx`.  Capturing frees liberties for the chased group.  Iterates
// the chased group's own stones via its bitset (O(chain size), not O(board)).
function _defenderCaptureMoves(game, idx) {
  const enemy = -game.cells[idx], gid = game._gid[idx];
  const W = game._W, sw = game._sw, nbr = game._nbr, cells = game.cells, gidArr = game._gid, ls = game._ls;
  const moves = new Set();
  const base = gid * W;
  for (let w = 0; w < W; w++) {
    let bits = sw[base + w];
    while (bits !== 0) {
      const b = bits & -bits;                       // lowest set bit
      const stone = (w << 5) + (31 - Math.clz32(b));  // a stone of the chased group
      bits ^= b;
      const nb = stone * 4;
      for (let d = 0; d < 4; d++) {
        const ni = nbr[nb + d];
        if (cells[ni] !== enemy) continue;          // adjacent enemy stone
        // O(1) atari check via the stored per-chain liberty count; only fetch
        // the actual liberty (the capture move) when the enemy is in atari.
        if (ls[gidArr[ni]] === 1) moves.add(game.groupLibs2(ni).lib0);
      }
    }
  }
  return moves;
}

// Per-probe reading budget, with restart-with-reshuffle (Las Vegas
// restarts).  On a toroidal board ladders have no edges to die against, so
// a chase can wrap and branch into an astronomically large tree — one hung
// a training run for 3+ hours on 2026-08-20.  Measured on a captured
// pathological probe, read cost was heavy-tailed in the (random) candidate
// ordering — but the openness move-ordering heuristic (see _emptyNbrs)
// collapsed that variance, so retry-with-reshuffle stopped paying and the
// budget is now SINGLE-SHOT: NODE_START === NODE_MAX = 5000 — with the
// depth limit killing capture-cycles at the source, every observed hard
// read finishes in a few hundred to ~2.3k nodes, so 5k covers them with
// margin while bounding the unknown worst case tightly.  Ordinary reads (a handful of nodes)
// never notice.  A read still capped reports ok (= "not proven capturable
// within budget"), the same truncation semantics as any depth-limited
// reader; the doubling machinery remains for _setBudgets experiments.
let NODE_START = 5000;
let NODE_MAX   = 5000;

// Fisher-Yates with Math.random.  Candidate iteration order is randomized so
// no systematic correlation (e.g. liberty-enumeration order tracking the
// chase direction on this edgeless board) can repeatedly steer the
// early-exit search into a worst-case subtree; expected effort becomes the
// mean over orderings (randomized-quicksort argument).  Verdicts and the
// urgentLibs SET are order-independent, so features and agent behavior are
// unchanged — only effort (readNodes) varies run to run.
function _shuffle(a) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = (Math.random() * (i + 1)) | 0;
    const t = a[i]; a[i] = a[j]; a[j] = t;
  }
  return a;
}

// Empty (wrapped) neighbours of a cell — the move-ordering key.  Mined from
// 2.7M logged read nodes over 22 hard 13x13 positions (2026-08-21): at
// defender nodes the resolving move has the max empty-neighbour count 96.8%
// of the time, at attacker nodes 74.3% — and resolver-first is the
// cost-optimal objective at both (the resolver's subtree is paid regardless
// of order; ordering only decides how many sibling subtrees are added).
function _emptyNbrs(game, c) {
  const N = game.N, x = c % N, y = (c / N) | 0, cells = game.cells;
  let n = 0;
  if (cells[((y + N - 1) % N) * N + x] === 0) n++;
  if (cells[((y + 1) % N) * N + x] === 0) n++;
  if (cells[y * N + (x + N - 1) % N] === 0) n++;
  if (cells[y * N + (x + 1) % N] === 0) n++;
  return n;
}

// Order candidates by openness (desc).  A fractional dither on the integer
// key randomizes ties (it can never reorder distinct counts), so no fixed
// ordering can be systematically adversarial.
function _orderOpen(game, a) {
  const k = a.map(m => _emptyNbrs(game, m) + Math.random());
  for (let i = 1; i < a.length; i++) {
    const mv = a[i], kv = k[i];
    let j = i - 1;
    while (j >= 0 && k[j] < kv) { a[j + 1] = a[j]; k[j + 1] = k[j]; j--; }
    a[j + 1] = mv; k[j + 1] = kv;
  }
  return a;
}
// Period-6 move-cycle prune.  game3 has no superko, so on the edgeless torus
// a capture-recapture chase can repeat positions forever; every real repeat
// observed across the hard-position corpus (2026-08-21 sweep) had period
// exactly 6 — the 6-move capture cycle.  A move whose last-6 move indexes
// equal the 6 before them is an exact position repeat (14,718/14,718 fires
// on the dump-3756203 repro) and is treated as illegal: undo and skip.
// Corpus effect: verdicts unchanged 98/98; the cycle-driven reads collapse
// (repro #97: ~6k -> ~560 median nodes, depth 163 -> 34) and no read hits
// the 2x-area depth limit any more (max acyclic depth seen: 118).
const _movePath = [];

function _cyclePlay(game, idx) {
  if (!game.play(idx)) return false;
  const p = _movePath;
  p.push(idx);
  const d = p.length;
  if (d >= 12 &&
      p[d - 1] === p[d - 7] && p[d - 2] === p[d - 8] &&
      p[d - 3] === p[d - 9] && p[d - 4] === p[d - 10] &&
      p[d - 5] === p[d - 11] && p[d - 6] === p[d - 12]) {
    p.pop();
    game.undo();
    return false;
  }
  return true;
}

function _cycleUndo(game) {
  _movePath.pop();
  game.undo();
}

let _budget = NODE_START;
let _capTripped = false;   // dump forensics once per process
// Probe-root snapshot, stashed at each budget reset.  The cap can expire deep
// in an already-boring line, so dumping the state at trip time captures the
// wrong frame — the root that actually spent the budget is what a repro needs.
let _probeRoot = null;

function _onCapTrip(game, idx) {
  if (_capTripped) return;
  _capTripped = true;
  console.error(`ladder2: read still capped at NODE_MAX (${NODE_MAX}) after retries for group at idx ${idx} — returning unproven-ok (further reports silent)`);
  const root = _probeRoot;
  if (typeof process !== 'undefined' && process.versions && process.versions.node) {
    try {
      const fs = require('fs');
      // Internal structures too: a rebuilt game3 on identical cells reads tiny
      // trees (ordering-hypothesis sweep, 2026-08-20), so if the live read
      // explodes, the divergence must be in structure, not cells.
      const groups = {};
      if (game._gid) {
        for (let i = 0; i < game.cells.length; i++) {
          if (game.cells[i] === 0) continue;
          const gid = game._gid[i];
          if (!(gid in groups)) {
            let size = null, libs = null;
            try { size = game.groupSize(gid); } catch {}
            try { libs = game.groupLibs2(i).count; } catch {}
            groups[gid] = { anchor: i, size, libs };
          }
        }
      }
      fs.writeFileSync(`out/ladder2-cap-dump-${process.pid}.json`, JSON.stringify({
        when: new Date().toISOString(), idx, N: game.N, current: game.current,
        // The probe root: state at budget reset — replay THIS to reproduce.
        root,
        // Trip-time state (mid-read, wherever the counter expired) for context.
        cells: Array.from(game.cells),
        gid: game._gid ? Array.from(game._gid) : null,
        groups,
        opStack: game._opStack ? game._opStack.length : null,
      }));
    } catch {}   // forensics only — never let the dump break the read
  }
}

// Reads whether the group at stoneIdx can reach 3+ liberties despite best
// attacker play. Uses play/undo instead of clone.  Returns the search effort
// alongside the verdict:
//   { ok: boolean, nodes: number, maxDepth: number }
// where `nodes` is the total number of positions read in this subtree and
// `maxDepth` is the deepest recursion level reached (the root counts as 1).
// `depth` is the current node's depth (callers start at 1).
function _canReach3Libs(game, idx, depth = 1) {
  let nodes = 1;
  let maxDepth = depth;
  let capped = false;
  if (--_budget <= 0) {
    return { ok: true, nodes, maxDepth, capped: true };
  }
  // Depth limit: 2x board area.  Pure backstop now that the period-6 cycle
  // prune (see _cyclePlay) kills capture-recapture repeats at the source —
  // with the prune, no corpus read exceeds depth 118, so this should never
  // fire; if it does, it returns unresolved-ok like any depth-limited reader.
  if (depth > 2 * game.N * game.N) {
    return { ok: true, nodes, maxDepth, capped };
  }
  // Fold a child's effort into this node's running totals.
  const acc = r => { nodes += r.nodes; if (r.maxDepth > maxDepth) maxDepth = r.maxDepth; if (r.capped) capped = true; };

  const { count: lc, lib0, lib1 } = game.groupLibs2(idx);
  if (lc >= 3) return { ok: true,  nodes, maxDepth, capped };
  if (lc === 0) return { ok: false, nodes, maxDepth, capped };

  const defColor = game.cells[idx];

  if (game.current === defColor) {
    // Defender's turn: extend onto a liberty, or capture an adjacent enemy
    // chain in atari; succeed if any leads to safety.
    const libs = lc === 1 ? [lib0] : [lib0, lib1];
    const moves = new Set(libs);
    for (const m of _defenderCaptureMoves(game, idx)) moves.add(m);
    // Sorted iteration: candidate order (and so search effort) is a pure
    // function of the position, not of the game3's construction history —
    // liberty/group enumeration order varies with how the object was built,
    // which makes unsorted reads irreproducible from a rebuilt board.
    for (const moveIdx of _orderOpen(game, [...moves])) {
      if (!_cyclePlay(game, moveIdx)) continue;   // suicide/illegal/cycle — skip
      const captured = game.cells[idx] === 0;
      let ok = false;
      if (!captured) {
        const r = _canReach3Libs(game, idx, depth + 1);
        acc(r);
        ok = r.ok;
      }
      _cycleUndo(game);
      if (ok) return { ok: true, nodes, maxDepth, capped };
    }
    return { ok: false, nodes, maxDepth, capped };
  }

  // Attacker's turn (1 or 2 libs): tries each liberty; succeeds if any leads
  // to capture.  Sorted for position-determined iteration (see defender note).
  let libs;
  if (lc === 1) libs = [lib0];
  else {
    // Same dithered-openness key as _orderOpen, inlined for the 2-lib case.
    libs = _emptyNbrs(game, lib1) + Math.random() > _emptyNbrs(game, lib0) + Math.random()
      ? [lib1, lib0] : [lib0, lib1];
  }
  for (const libIdx of libs) {
    if (!_cyclePlay(game, libIdx)) continue;   // illegal/cycle for attacker — skip
    const captured = game.cells[idx] === 0;
    if (captured) {
      _cycleUndo(game);
      return { ok: false, nodes, maxDepth, capped };
    }
    const afterLc = game.groupLibs2(idx).count;
    if (afterLc === 0) {
      _cycleUndo(game);
      return { ok: false, nodes, maxDepth, capped };
    }
    // Only pursue the chase when the attacker's move reduced the group to a
    // single liberty.  This is also the termination guard: a snapback that
    // bounces the group back to 2+ liberties does NOT recurse, so the mutual
    // attacker/defender recursion can't cycle forever.
    if (afterLc === 1) {
      const r = _canReach3Libs(game, idx, depth + 1);
      acc(r);
      _cycleUndo(game);
      if (!r.ok) return { ok: false, nodes, maxDepth, capped };
    } else {
      _cycleUndo(game);
    }
  }

  return { ok: true, nodes, maxDepth, capped };
}

// getAllLadderStatuses(game, minChainSize) — run getLadderStatus on every
// group with 1 or 2 liberties and return an array of { gid, color, status }
// objects, one per group (groups with 0 or 3+ liberties are skipped).
// minChainSize: skip groups smaller than this (default 1).
function getAllLadderStatuses(game, minChainSize = 1) {
  const cap  = game.N * game.N;
  const results = [];
  const visited = new Set();
  for (let i = 0; i < cap; i++) {
    if (game.cells[i] === 0) continue;
    const gid = game._gid[i];
    if (visited.has(gid)) continue;
    visited.add(gid);
    if (game.groupSize(gid) < minChainSize) continue;
    const { count: lc } = game.groupLibs2(i);
    if (lc === 0 || lc > 2) continue;
    const status = getLadderStatus(game, i);
    results.push({ gid, color: game.cells[i], status });
  }
  return results;
}

// Examines the group containing the stone at stoneIdx (must have 1 or 2
// liberties).  For each liberty, simulates both colours playing it first.
//
// Returns { libs: [], moverSucceeds: boolean, urgentLibs: [] }
//
// Logs a warning and returns null when the group has more than 2 liberties.
function getLadderStatus(game, stoneIdx) {
  const { count: lc, lib0, lib1 } = game.groupLibs2(stoneIdx);
  if (lc < 1 || lc > 2) {
    const N = game.N;
    console.warn(`getLadderStatus: group at ${stoneIdx % N},${(stoneIdx / N) | 0} has ${lc} liberties (expected ≤ 2)`);
    return null;
  }
  const atari = lc === 1;
  const libs = atari ? [lib0] : [lib0, lib1];
  _movePath.length = 0;   // cycle-prune path: fresh per read
  const gColor = game.cells[stoneIdx];
  const mover = game.current;   // BLACK or WHITE
  const defending = gColor === mover;

  // Total reading effort spent across every _canReach3Libs probe below: summed
  // node count, and the deepest recursion reached by any probe.
  let readNodes = 0, readDepth = 0;
  const probe = () => {
    _probeRoot = { cells: Array.from(game.cells), current: game.current, stoneIdx };
    // Restart with reshuffle: budget expiry means this ordering drew a huge
    // subtree; a fresh shuffle at double the budget usually finds a short one.
    for (let budget = NODE_START; ; budget *= 2) {
      _budget = budget;
      const r = _canReach3Libs(game, stoneIdx, 1);
      readNodes += r.nodes;
      if (r.maxDepth > readDepth) readDepth = r.maxDepth;
      if (!r.capped) return r.ok;
      if (budget >= NODE_MAX) { _onCapTrip(game, stoneIdx); return r.ok; }
    }
  };

  let escape;
  // Try opponent playing first.
  if (defending && atari) {
    escape = false;
  } else {
    _cyclePlay(game, PASS);
    escape = probe();
    _cycleUndo(game);
  }
  if (defending === escape) {
    // group is not urgent
    return { libs, moverSucceeds: true, urgentLibs: [], readNodes, readDepth };
  }

  // Try mover playing first.  When defending, the saving moves also include
  // captures of adjacent atari'd enemy chains (not just the group's liberties).
  let moverSucceeds = false;
  let urgentLibs = [];
  const moverMoves = _shuffle(defending ? [...new Set([...libs, ..._defenderCaptureMoves(game, stoneIdx)])] : [...libs]);
  for (const moveIdx of moverMoves) {
    if (!defending && atari) {
      escape = false;
    } else {
      if (!_cyclePlay(game, moveIdx)) {
        continue;
      }
      escape = probe();
      _cycleUndo(game);
    }
    if (defending === escape) {
      moverSucceeds = true;
      urgentLibs.push(moveIdx);
    }
  }
  return { libs, moverSucceeds, urgentLibs, readNodes, readDepth };
}

// _setBudgets: test/measurement hook (e.g. emulate a fixed single budget
// with start === max); returns the previous values.
function _setBudgets(start, max) {
  const prev = [NODE_START, NODE_MAX];
  NODE_START = start; NODE_MAX = max;
  return prev;
}

const _exports = { getLadderStatus, getAllLadderStatuses, _canReach3Libs, _setBudgets };
if (typeof module !== 'undefined') module.exports = _exports;
else window.Ladder2 = _exports;

})();
