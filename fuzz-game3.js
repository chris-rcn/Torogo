#!/usr/bin/env node
'use strict';

// fuzz-game3.js — play/undo invariant fuzzer for Game3.
//
// Game3's core principle is that every operation has an exact undo.  This
// tool hunts violations: per seed it builds a realistic random base
// position, then drives search-like nested play/undo excursions (the usage
// pattern of ladder reads and ab-search), auditing the incremental
// structures against cells-derived ground truth after EVERY play and EVERY
// undo:
//   - _gid is constant across each flood-fill component,
//   - _ss[gid] equals the component's true stone count,
//   - _ls[gid] equals the component's true liberty count.
//
// --seed is a true master seed: it seeds a stream from which each
// iteration's own seed is drawn, so any two master seeds explore disjoint
// iteration sequences (safe to run several instances in parallel).  A
// reported failure names the ITERATION seed; --replay <iterSeed> reruns
// exactly that case.
//
// Motivated by a wild corruption capture (2026-08-21): a long-lived game3
// in a training run reported a group size of -13 while its cells were sane
// (out/ladder2-cap-dump-3664672.json), the likely mechanism behind
// irreproducible ladder-read explosions.
//
// Usage:
//   node fuzz-game3.js [--size 13] [--seed 1] [--limit 0] [--steps 4000] [--depth 24]
//   node fuzz-game3.js --replay <iterSeed> [--size ...] [--steps ...] [--depth ...]
//
//   --size    board size                                  (default 13)
//   --seed    master seed for the iteration-seed stream   (default 1)
//   --replay  run exactly one iteration by its seed
//   --limit   number of iterations; 0 = run forever       (default 0)
//   --steps   play/undo excursion steps per iteration     (default 4000)
//   --depth   max excursion nesting depth                 (default 24)
//
// Exits 1 with the seed and failing check on the first violation.

const { Game2, PASS } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');
const { makeRng } = require('./xorshift.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['size', 'seed', 'replay', 'limit', 'steps', 'depth']);
if (opts.help) {
  console.log('Usage: node fuzz-game3.js [--size 13] [--seed 1] [--limit 0] [--steps 4000] [--depth 24]');
  console.log('       node fuzz-game3.js --replay <iterSeed>');
  process.exit(0);
}
const SIZE     = opts.getInt('size', 13);
const SEED0    = opts.getInt('seed', 1);
const LIMIT    = opts.getInt('limit', 0);
const MAX_STEP = opts.getInt('steps', 4000);
const MAX_DEEP = opts.getInt('depth', 24);

// Audit game3's incremental structures against ground truth recomputed from
// cells alone.  Returns null when consistent, else a description.
function auditStructures(g3, ctx) {
  const N = g3.N, cap = N * N, cells = g3.cells;
  const nbr = i => {
    const x = i % N, y = (i / N) | 0;
    return [((y + N - 1) % N) * N + x, ((y + 1) % N) * N + x,
            y * N + (x + N - 1) % N, y * N + (x + 1) % N];
  };
  const seen = new Int8Array(cap);
  for (let i = 0; i < cap; i++) {
    if (cells[i] === 0 || seen[i]) continue;
    const color = cells[i], gid = g3._gid[i];
    const comp = [i], libs = new Set();
    seen[i] = 1;
    for (let s = 0; s < comp.length; s++) {
      for (const n of nbr(comp[s])) {
        if (cells[n] === color && !seen[n]) { seen[n] = 1; comp.push(n); }
        else if (cells[n] === 0) libs.add(n);
      }
    }
    for (const c of comp) {
      if (g3._gid[c] !== gid) return `${ctx}: component of ${i} spans gids ${gid} and ${g3._gid[c]} (cell ${c})`;
    }
    if (g3.groupSize(gid) !== comp.length) return `${ctx}: groupSize(${gid}): ${g3.groupSize(gid)} != true size ${comp.length} (anchor ${i})`;
    if (g3.groupLibertyCount(gid) !== libs.size) return `${ctx}: groupLibertyCount(${gid}): ${g3.groupLibertyCount(gid)} != true libs ${libs.size} (anchor ${i})`;
  }
  return null;
}

// One deterministic fuzz iteration.  Returns null (clean) or failure info.
function fuzzSeed(seed) {
  const rng = makeRng(seed);
  const g2 = new Game2(SIZE);
  const cap = SIZE * SIZE;
  const target = (cap * 0.25 | 0) + ((rng.random() * cap * 0.65) | 0);
  for (let m = 0; m < target && !g2.gameOver; m++) g2.play(g2.randomLegalMove(rng));
  const g3 = game3FromGame2(g2);
  let fail = auditStructures(g3, 'baseline');
  if (fail) return { fail, step: -1 };
  let step = 0;
  function excursion(depth) {
    if (step >= MAX_STEP || fail) return;
    const tries = 1 + ((rng.random() * 3) | 0);
    for (let t = 0; t < tries && !fail; t++) {
      const mv = rng.random() < 0.03 ? PASS : (rng.random() * cap) | 0;
      if (!g3.play(mv)) continue;
      step++;
      fail = auditStructures(g3, `after play ${mv} @step ${step}`);
      if (!fail && depth < MAX_DEEP && rng.random() < 0.65) excursion(depth + 1);
      g3.undo();
      if (!fail) fail = auditStructures(g3, `after undo of ${mv} @step ${step}`);
    }
  }
  while (step < MAX_STEP && !fail) excursion(0);
  return fail ? { fail, step } : null;
}

const REPLAY = opts.getInt('replay', 0);
if (REPLAY) {
  console.log(`fuzz-game3 replay: iterSeed: ${REPLAY}  size: ${SIZE}  steps: ${MAX_STEP}  depth: ${MAX_DEEP}`);
  const r = fuzzSeed(REPLAY);
  if (r) { console.log(`FAILURE iterSeed: ${REPLAY}  detail: ${r.fail}`); process.exit(1); }
  console.log('clean');
  process.exit(0);
}

const t0 = Date.now();
const master = makeRng(SEED0);
let done = 0;
console.log(`fuzz-game3: size: ${SIZE}  masterSeed: ${SEED0}  limit: ${LIMIT || 'none'}  steps/iter: ${MAX_STEP}  depth: ${MAX_DEEP}`);
for (;;) {
  const iterSeed = ((master.random() * 0x7fffffff) | 0) || 1;
  const r = fuzzSeed(iterSeed);
  if (r) {
    console.log(`FAILURE iterSeed: ${iterSeed}  (masterSeed ${SEED0}, iteration ${done + 1})  detail: ${r.fail}`);
    console.log(`replay: node fuzz-game3.js --replay ${iterSeed} --size ${SIZE} --steps ${MAX_STEP} --depth ${MAX_DEEP}`);
    process.exit(1);
  }
  done++;
  if (done % 200 === 0) {
    const secs = ((Date.now() - t0) / 1000) | 0;
    console.log(`iterations clean: ${done}  elapsed: ${secs}s  ops audited: ~${(done * MAX_STEP / 1e6).toFixed(1)}M`);
  }
  if (LIMIT && done >= LIMIT) {
    console.log(`no failures in ${done} iterations (~${done * MAX_STEP} audited ops)`);
    break;
  }
}
