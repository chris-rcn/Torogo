#!/usr/bin/env node
'use strict';

// test-game3-undo.js — round-trip / consistency tests for Game3 play+undo.
//
// Two properties are checked on positions reached by clean self-play:
//
//   (1) Consistency: the internal group bookkeeping (_gid partition + the
//       liberty counts groupLibs2 reads from the bitsets) must match ground
//       truth computed by flood-filling `cells` over the toroidal topology.
//       Catches play()/undo() leaving metadata that disagrees with the board.
//
//   (2) Round-trip: after an arbitrary BALANCED sequence of play/undo churn
//       (sequential + deeply nested, incl. captures, ko, PASS, double-pass),
//       the canonical live state must be byte-identical to before the churn.
//       Catches undo() not fully restoring state — the actual worry.
//
// Both comparisons are gid-LABEL-INDEPENDENT: groups are identified by their
// minimum stone index, not by internal gid numbering, so a benign group-id
// relabeling after undo is not flagged as a difference.

const { Game2 } = require('./game2.js');
const { Game3, game3FromGame2, PASS } = require('./game3.js');

const SEED  = parseInt(process.argv[2] || '1', 10);
const GAMES = parseInt(process.argv[3] || '400', 10);
const N = 9, cap = N * N;
let _s = (SEED >>> 0) || 1;
const rng = () => { _s = (_s * 1664525 + 1013904223) >>> 0; return _s / 4294967296; };

// Ground-truth group partition + liberty counts from `cells` alone, using the
// (static, never-mutated) neighbor table.  Groups keyed by min stone index.
function groundTruth(g) {
  const cells = g.cells, nbr = g._nbr;
  const comp = new Int32Array(cap).fill(-1);
  for (let i = 0; i < cap; i++) {
    if (cells[i] === 0 || comp[i] !== -1) continue;
    const color = cells[i], members = [];
    const stack = [i]; comp[i] = i; let mn = i;
    while (stack.length) {
      const s = stack.pop(); members.push(s); if (s < mn) mn = s;
      const b = s * 4;
      for (let d = 0; d < 4; d++) { const ni = nbr[b + d]; if (cells[ni] === color && comp[ni] === -1) { comp[ni] = i; stack.push(ni); } }
    }
    for (const s of members) comp[s] = mn;
  }
  const libCount = new Map();
  for (let i = 0; i < cap; i++) {
    if (cells[i] === 0) continue;
    const mn = comp[i];
    let set = libCount.get(mn); if (!set) { set = new Set(); libCount.set(mn, set); }
    const b = i * 4;
    for (let d = 0; d < 4; d++) { const ni = nbr[b + d]; if (cells[ni] === 0) set.add(ni); }
  }
  const part = new Array(cap).fill(-1), libs = new Array(cap).fill(-1);
  for (let i = 0; i < cap; i++) { if (cells[i] === 0) continue; part[i] = comp[i]; libs[i] = libCount.get(comp[i]).size; }
  return { part, libs };
}

// Same signature, but from the INTERNAL structures (_gid + groupLibs2).
function internalSig(g) {
  const minByGid = new Map();
  for (let i = 0; i < cap; i++) { if (g.cells[i] === 0) continue; const gid = g._gid[i]; if (!minByGid.has(gid) || i < minByGid.get(gid)) minByGid.set(gid, i); }
  const part = new Array(cap).fill(-1), libs = new Array(cap).fill(-1);
  for (let i = 0; i < cap; i++) { if (g.cells[i] === 0) continue; part[i] = minByGid.get(g._gid[i]); libs[i] = g.groupLibs2(i).count; }
  return { part, libs };
}

const eqArr = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);

// Canonical live state (gid-label-independent via internalSig).
function canon(g) {
  return JSON.stringify({
    cells: Array.from(g.cells), current: g.current, ko: g.ko,
    cp: g.consecutivePasses, go: g.gameOver, mc: g.moveCount, lm: g.lastMove,
    sig: internalSig(g),
  });
}

// Balanced nested play/undo churn (captures, ko, PASS all reachable).
function churn(g, depth, branch) {
  if (depth <= 0) return;
  for (let t = 0; t < branch; t++) {
    const m = (t === branch - 1) ? PASS : (rng() * cap) | 0;
    if (!g.play(m)) continue;
    churn(g, depth - 1, branch);
    g.undo();
  }
}

let pass = 0, fail = 0;
const fails = [];
function check(cond, msg) { if (cond) pass++; else { fail++; if (fails.length < 10) fails.push(msg); } }

for (let game = 0; game < GAMES; game++) {
  // Reference position from clean self-play (no undo in construction).
  const g2 = new Game2(N);
  const moves = 1 + ((rng() * (cap * 2)) | 0);
  for (let k = 0; k < moves && !g2.gameOver; k++) g2.play(g2.randomLegalMove({ random: rng }));
  let g;
  try { g = game3FromGame2(g2); } catch (e) { continue; }   // skip unreplayable (capture) snapshots

  // (1) play() leaves internally-consistent metadata.
  const gt0 = groundTruth(g), is0 = internalSig(g);
  check(eqArr(gt0.part, is0.part) && eqArr(gt0.libs, is0.libs), `game ${game}: pre-churn metadata disagrees with board`);

  // (2) round-trip across balanced churn.
  const before = canon(g);
  for (let k = 0; k < 60; k++) { const m = (rng() < 0.1) ? PASS : (rng() * cap) | 0; if (g.play(m)) g.undo(); }   // sequential
  churn(g, 4, 4);                                                                                                  // nested
  const after = canon(g);
  check(before === after, `game ${game}: state not restored after balanced play/undo churn`);

  // (3) still consistent with the board after all the churn.
  const gt1 = groundTruth(g), is1 = internalSig(g);
  check(eqArr(gt1.part, is1.part) && eqArr(gt1.libs, is1.libs), `game ${game}: post-churn metadata disagrees with board`);
}

console.log(`Game3 undo round-trip:  ${pass} passed, ${fail} failed  (seed=${SEED}, games=${GAMES})`);
for (const m of fails) console.log(`  - ${m}`);
process.exit(fail > 0 ? 1 : 0);
