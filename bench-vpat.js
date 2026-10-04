'use strict';

// bench-vpat.js — vpat evaluation cost: plays uniformly random games and, at
// every position, evaluates each model non-incrementally (full extraction +
// evaluation, the call puct-trunc's truncated playouts make).  Every model
// sees the same positions; the order rotates per position.  Only positions
// with --min-phase <= phase <= --max-phase are evaluated; each game ends once
// the board is fuller than --max-phase.
//
// --f32 (prototype): extract once per position, then time the weight lookup
// three ways on the same features: int-map's get() (int32 keys and float64
// values in two arrays); 'split', the same two-array layout probed inline;
// and 'f32', an interleaved table (int32 key + float32 value in one 8-byte
// slot).  All three share int-map's hash, probing and capacity.
//
//   node bench-vpat.js --model A.js[,B.js...] [--games 200] [--size 13]
//                      [--min-phase 0] [--max-phase 1] [--seed 1] [--f32]

const { performance } = require('perf_hooks');
const Util = require('./util.js');
const VPat = require('./vpatterns.js');
const { Game2 } = require('./game2.js');
const { makeRng } = require('./xorshift.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help', 'f32'], ['model', 'games', 'size', 'min-phase', 'max-phase', 'seed']);
if (opts.help || !opts.model) {
  console.log('Usage: node bench-vpat.js --model A.js[,B.js...] [--games 200] [--size 13] [--min-phase 0] [--max-phase 1] [--seed 1] [--f32]');
  process.exit(opts.help ? 0 : 1);
}
const GAMES = parseInt(opts.games || '200', 10);
const SIZE  = parseInt(opts.size || '13', 10);
const MIN_PHASE = parseFloat(opts['min-phase'] ?? '0');
const MAX_PHASE = parseFloat(opts['max-phase'] ?? '1');
if (!(MIN_PHASE >= 0 && MIN_PHASE <= MAX_PHASE && MAX_PHASE <= 1)) {
  console.error('--min-phase and --max-phase must satisfy 0 <= min <= max <= 1'); process.exit(1);
}
const rng   = makeRng(parseInt(opts.seed || '1', 10));
const F32 = !!opts.f32;
const models = opts.model.split(',').map(f => ({ file: f, m: VPat.loadWeights(f), ms: 0, n: 0, sink: 0,
                                                 tbl: null, spl: null, exMs: 0, mapMs: 0, splMs: 0, f32Ms: 0, maxDiff: 0 }));

// --f32: int-map's hash and triangular probing over one buffer of 8-byte
// slots, key at 2i and float32 value at 2i+1; capacity as modelFromRaw sizes
// int-map (the power of two >= 2x the weight count).
function makeF32Table(weights) {
  let cap = 1;
  while (cap < Math.max(1024, 2 * weights.size)) cap <<= 1;
  const buf = new ArrayBuffer(cap * 8);
  const k32 = new Int32Array(buf), f32 = new Float32Array(buf), mask = cap - 1;
  weights.forEach((key, val) => {
    let i = (Math.imul(796154621, key) ^ Math.imul(862632693, key >> 16)) & mask, skip = 1;
    while (k32[i << 1] !== 0) { i = (i + skip) & mask; skip++; }
    k32[i << 1] = key; f32[(i << 1) | 1] = val;
  });
  return { k32, f32, mask };
}
function makeSplitTable(weights) {
  let cap = 1;
  while (cap < Math.max(1024, 2 * weights.size)) cap <<= 1;
  const ks = new Int32Array(cap), vs = new Float64Array(cap), mask = cap - 1;
  weights.forEach((key, val) => {
    let i = (Math.imul(796154621, key) ^ Math.imul(862632693, key >> 16)) & mask, skip = 1;
    while (ks[i] !== 0) { i = (i + skip) & mask; skip++; }
    ks[i] = key; vs[i] = val;
  });
  return { ks, vs, mask };
}
function evalSplit(features, t) {
  const { keys, pols, count } = features, { ks, vs, mask } = t;
  let z = 0;
  for (let n = 0; n < count; n++) {
    const key = keys[n];
    let i = (Math.imul(796154621, key) ^ Math.imul(862632693, key >> 16)) & mask, skip = 1;
    for (;;) {
      const k = ks[i];
      if (k === key) { z += pols[n] * vs[i]; break; }
      if (k === 0) break;
      i = (i + skip) & mask; skip++;
    }
  }
  return 1 / (1 + Math.exp(-z));
}
function evalF32(features, t) {
  const { keys, pols, count } = features, { k32, f32, mask } = t;
  let z = 0;
  for (let n = 0; n < count; n++) {
    const key = keys[n];
    let i = (Math.imul(796154621, key) ^ Math.imul(862632693, key >> 16)) & mask, skip = 1;
    for (;;) {
      const k = k32[i << 1];
      if (k === key) { z += pols[n] * f32[(i << 1) | 1]; break; }
      if (k === 0) break;
      i = (i + skip) & mask; skip++;
    }
  }
  return 1 / (1 + Math.exp(-z));
}
// vpatterns' evaluateFeatures loop, copied so both lookups are timed in the
// same kind of call.
function evalMap(features, weights) {
  const { keys, pols, count } = features;
  let z = 0;
  for (let n = 0; n < count; n++) z += pols[n] * (weights.get(keys[n]) ?? 0);
  return 1 / (1 + Math.exp(-z));
}
if (F32) for (const x of models) { x.tbl = makeF32Table(x.m.weights); x.spl = makeSplitTable(x.m.weights); }

// puct-trunc's vpatValueB shape: a small helper, so V8 inlines as it does there.
function valueB(game2, m) {
  return VPat.evaluateFeatures(VPat.extractFeatures(game2, m.preparedSpecs, false, undefined, true), m.weights);
}

const WARMUP_GAMES = Math.min(10, GAMES);
let pos = 0;
for (let gi = 0; gi < WARMUP_GAMES + GAMES; gi++) {
  const timed = gi >= WARMUP_GAMES;
  const g = new Game2(SIZE, false);
  const maxMoves = 4 * SIZE * SIZE;
  for (let mv = 0; !g.gameOver && mv < maxMoves && g.phase() <= MAX_PHASE; mv++) {
    if (g.phase() >= MIN_PHASE) {
      for (let k = 0; k < models.length; k++) {
        const x = models[(pos + k) % models.length];
        if (F32) {
          const t0 = performance.now();
          const f = VPat.extractFeatures(g, x.m.preparedSpecs, false, undefined, true);
          const t1 = performance.now();
          // The three lookups in a rotating order, so none always runs first.
          let vm = 0, vs = 0, vf = 0, tm = 0, ts = 0, tf = 0;
          for (let r = 0; r < 3; r++) {
            const which = (pos + r) % 3, a = performance.now();
            if (which === 0) { vm = evalMap(f, x.m.weights); tm = performance.now() - a; }
            else if (which === 1) { vs = evalSplit(f, x.spl); ts = performance.now() - a; }
            else { vf = evalF32(f, x.tbl); tf = performance.now() - a; }
          }
          x.sink += vm + vs + vf;
          if (vs !== vm) throw new Error(`split table disagrees with int-map: ${vs} vs ${vm}`);
          if (timed) { x.exMs += t1 - t0; x.mapMs += tm; x.splMs += ts; x.f32Ms += tf; x.n++; x.maxDiff = Math.max(x.maxDiff, Math.abs(vm - vf)); }
          continue;
        }
        const t = performance.now();
        x.sink += valueB(g, x.m);
        if (timed) { x.ms += performance.now() - t; x.n++; }
      }
      pos++;
    }
    g.play(g.randomLegalMove(rng));
  }
}

console.log(`games: ${GAMES} (+${WARMUP_GAMES} warm-up)  size: ${SIZE}  min-phase: ${MIN_PHASE}  max-phase: ${MAX_PHASE}`);
for (const x of models) {
  const head = `${x.file}  spec: ${VPat.specString(x.m.specs)}  weights: ${x.m.weights.size}  evals: ${x.n}`;
  const us = ms => (1000 * ms / x.n).toFixed(2);
  console.log(F32 ? `${head}  extract us: ${us(x.exMs)}  int-map us: ${us(x.mapMs)}  split us: ${us(x.splMs)}  f32 us: ${us(x.f32Ms)}  max|dV|: ${x.maxDiff.toExponential(1)}`
                  : `${head}  us/eval: ${us(x.ms)}`);
}
