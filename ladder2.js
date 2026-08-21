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

// Per-probe reading budget.  On a toroidal board ladders have no edges to
// die against, so a chase can wrap and branch into an astronomically large
// (hours-of-CPU) tree — one did, hanging a training run for 3+ hours on
// 2026-08-20.  Each root probe gets NODE_CAP nodes; an exhausted read stops
// and reports ok (= "not proven capturable within budget"), the same
// truncation semantics as any depth-limited reader.  Every position a cap
// changes is one the uncapped code could never finish, so no completed game
// ever depended on the difference.
const NODE_CAP = 10000;

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
let _budget = NODE_CAP;
let _capTripped = false;   // dump forensics once per process

function _onCapTrip(game, idx) {
  if (_capTripped) return;
  _capTripped = true;
  console.error(`ladder2: NODE_CAP (${NODE_CAP}) exhausted reading group at idx ${idx} — truncating (further trips silent)`);
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
  if (--_budget <= 0) {
    _onCapTrip(game, idx);
    return { ok: true, nodes, maxDepth, capped: true };
  }
  // Fold a child's effort into this node's running totals.
  const acc = r => { nodes += r.nodes; if (r.maxDepth > maxDepth) maxDepth = r.maxDepth; };

  const { count: lc, lib0, lib1 } = game.groupLibs2(idx);
  if (lc >= 3) return { ok: true,  nodes, maxDepth };
  if (lc === 0) return { ok: false, nodes, maxDepth };

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
    for (const moveIdx of _shuffle([...moves])) {
      if (!game.play(moveIdx)) continue;     // suicide/illegal — skip
      const captured = game.cells[idx] === 0;
      let ok = false;
      if (!captured) {
        const r = _canReach3Libs(game, idx, depth + 1);
        acc(r);
        ok = r.ok;
      }
      game.undo();
      if (ok) return { ok: true, nodes, maxDepth };
    }
    return { ok: false, nodes, maxDepth };
  }

  // Attacker's turn (1 or 2 libs): tries each liberty; succeeds if any leads
  // to capture.  Sorted for position-determined iteration (see defender note).
  const libs = lc === 1 ? [lib0] : (Math.random() < 0.5 ? [lib0, lib1] : [lib1, lib0]);
  for (const libIdx of libs) {
    if (!game.play(libIdx)) continue;        // illegal for attacker — skip
    const captured = game.cells[idx] === 0;
    if (captured) {
      game.undo();
      return { ok: false, nodes, maxDepth };
    }
    const afterLc = game.groupLibs2(idx).count;
    if (afterLc === 0) {
      game.undo();
      return { ok: false, nodes, maxDepth };
    }
    // Only pursue the chase when the attacker's move reduced the group to a
    // single liberty.  This is also the termination guard: a snapback that
    // bounces the group back to 2+ liberties does NOT recurse, so the mutual
    // attacker/defender recursion can't cycle forever.
    if (afterLc === 1) {
      const r = _canReach3Libs(game, idx, depth + 1);
      acc(r);
      game.undo();
      if (!r.ok) return { ok: false, nodes, maxDepth };
    } else {
      game.undo();
    }
  }

  return { ok: true, nodes, maxDepth };
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
  const gColor = game.cells[stoneIdx];
  const mover = game.current;   // BLACK or WHITE
  const defending = gColor === mover;

  // Total reading effort spent across every _canReach3Libs probe below: summed
  // node count, and the deepest recursion reached by any probe.
  let readNodes = 0, readDepth = 0;
  const probe = () => {
    _budget = NODE_CAP;
    const r = _canReach3Libs(game, stoneIdx, 1);
    readNodes += r.nodes;
    if (r.maxDepth > readDepth) readDepth = r.maxDepth;
    return r.ok;
  };

  let escape;
  // Try opponent playing first.
  if (defending && atari) {
    escape = false;
  } else {
    game.play(PASS);
    escape = probe();
    game.undo();
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
      if (!game.play(moveIdx)) {
        continue;
      }
      escape = probe();
      game.undo();
    }
    if (defending === escape) {
      moverSucceeds = true;
      urgentLibs.push(moveIdx);
    }
  }
  return { libs, moverSucceeds, urgentLibs, readNodes, readDepth };
}

const _exports = { getLadderStatus, getAllLadderStatuses, _canReach3Libs };
if (typeof module !== 'undefined') module.exports = _exports;
else window.Ladder2 = _exports;

})();
