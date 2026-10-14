'use strict';
// test-wasm-playout.js — the module's playout (ppat-wasm.js playout, c/wasm_ppat.c)
// must be the same policy as puct-trunc.js's untruncated playout.  The two use
// different random streams and float arithmetic, so the check is statistical:
// from random roots across phases, K playouts each in a JS copy of puct-trunc's
// loop and in the module, comparing per-position mean results (sum of z^2 over
// positions should be near the position count), mean ppat / uniform move counts
// and trace sizes; plus the trace's own invariants.  Also times both.  Build
// first with c/build-wasm.sh.
//
//   node c/test-wasm-playout.js [--model out/ppat-data-1623136.js] [--positions 120] [--playouts 150]
//                               [--min-phase 0.55] [--seed 1]

const path = require('path');
const R = path.join(__dirname, '..');
const Util = require(path.join(R, 'util.js'));
const { Game2, PASS, BLACK } = require(path.join(R, 'game2.js'));
const { makeRng } = require(path.join(R, 'xorshift.js'));
const PPat = require(path.join(R, 'ppat-lib.js'));
const PpatWasm = require(path.join(R, 'ppat-wasm.js'));

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['model', 'positions', 'playouts', 'min-phase', 'seed']);
if (opts.help) { console.log('usage: node c/test-wasm-playout.js [--model F] [--positions 120] [--playouts 150] [--min-phase 0.55] [--seed 1]'); process.exit(0); }
const MODEL = path.resolve(opts.model || path.join(R, 'out', 'ppat-data-1623136.js'));
const M = parseInt(opts.positions || '120', 10), K = parseInt(opts.playouts || '150', 10);
const UBP = parseFloat(opts['min-phase'] ?? '0.55'), SEED = parseInt(opts.seed || '1', 10);
const SIZE = 13, AREA = SIZE * SIZE;

const model = PPat.loadWeights(MODEL);
model.ppatMinPhase = UBP;
const pw = PpatWasm.loadSync(path.join(__dirname, 'ppat-playout-test.wasm'));
pw.init(SIZE);
pw.setModel(model);

// puct-trunc.js's playout with truncation off: uniform moves until the first
// ppat move, then everything through the incremental cache.
const inc = PPat.createIncremental(SIZE);
function jsPlayout(game2, played, rng, out) {
  const cap = game2.N * game2.N, moveLimit = 3 * game2.emptyCount + 20, weightStep = 1 / cap;
  let moves = 0, uniformMoves = 0, weight = 1.0, incLive = false;
  while (!game2.gameOver && moves < moveLimit) {
    const current = game2.current;
    if (UBP > 0 && (cap - game2.emptyCount) / cap < UBP) uniformMoves++;
    let idx;
    if (!incLive && UBP > 0 && (cap - game2.emptyCount) / cap < UBP) { idx = game2.randomLegalMove(rng); game2.play(idx); }
    else {
      if (!incLive) { PPat.incrementalReset(inc); incLive = true; }
      idx = PPat.ppatMoveIncremental(game2, inc, model, rng);
      PPat.incrementalPlay(game2, inc, idx);
    }
    if (idx !== PASS && weight > 0 && played[idx] === 0) played[idx] = current === BLACK ? weight : -weight;
    moves++;
    weight -= weightStep;
  }
  out.ppat = moves - uniformMoves; out.uniform = uniformMoves;
  return game2.estimateWinner() === BLACK ? 1 : 0;
}

// Roots: random games stopped at a phase drawn from [0.15, 0.75].
const rng = makeRng(SEED);
const roots = [];
while (roots.length < M) {
  const g = new Game2(SIZE, false), target = 0.15 + 0.6 * rng.random();
  while (!g.gameOver && g.phase() < target) g.play(g.randomLegalMove(rng));
  if (!g.gameOver) roots.push(g);
}

const played = new Float32Array(AREA), o = {};
let chi2 = 0, dSum = 0, varSum = 0, traceBad = 0;
const zs = [];
const tot = { js: { ppat: 0, uniform: 0, trace: 0 }, wasm: { ppat: 0, uniform: 0, trace: 0 } };
let jsMs = 0, wasmMs = 0;
const nz = t => { let n = 0; for (let i = 0; i < AREA; i++) if (t[i] !== 0) n++; return n; };
for (const root of roots) {
  let sj = 0, sw = 0;
  let t0 = performance.now();
  for (let k = 0; k < K; k++) {
    played.fill(0);
    sj += jsPlayout(root.clone(), played, rng, o);
    tot.js.ppat += o.ppat; tot.js.uniform += o.uniform; tot.js.trace += nz(played);
  }
  jsMs += performance.now() - t0;
  pw.setRoot(root);
  t0 = performance.now();
  for (let k = 0; k < K; k++) {
    pw.descend([], 0);
    sw += pw.playout((rng.random() * 4294967296) >>> 0, UBP);
    const tr = pw.trace(), n = nz(tr), moves = pw.ppatMoves + pw.uniformMoves;
    tot.wasm.ppat += pw.ppatMoves; tot.wasm.uniform += pw.uniformMoves; tot.wasm.trace += n;
    // Trace invariants: at most one entry per move, |w| in (0, 1], w = 1 - j/area for a move j.
    if (n > moves) traceBad++;
    for (let i = 0; i < AREA; i++) {
      const w = Math.abs(tr[i]);
      if (w === 0) continue;
      const j = Math.round((1 - w) * AREA);
      if (w > 1 || j < 0 || j >= moves || Math.abs(w - (1 - j / AREA)) > 1e-5) { traceBad++; break; }
    }
  }
  wasmMs += performance.now() - t0;
  const pj = sj / K, pv = sw / K;
  // The two-proportion test's pooled variance: each side's own p(1-p) is badly
  // calibrated near 0 or 1 at small K (a 1.00 vs 0.97 split scores z^2 ~ 3).
  const pp = (sj + sw) / (2 * K), v = 2 * pp * (1 - pp) / K;
  dSum += pv - pj; varSum += v;
  if (v > 0) { const z2 = (pv - pj) ** 2 / v; chi2 += z2; zs.push({ z2, pj, pv, phase: root.phase(), idx: roots.indexOf(root) }); }
}
zs.sort((a, b) => b.z2 - a.z2);
const n = M * K;
const d = dSum / M, se = Math.sqrt(varSum) / M;
console.log(`model: ${path.basename(MODEL)}, positions: ${M}, playouts each: ${K}, gate: ${UBP}`);
console.log(`mean P(black) wasm - js: ${d >= 0 ? '+' : ''}${d.toFixed(4)} +- ${se.toFixed(4)}   sum z^2 over positions: ${chi2.toFixed(1)} (expect ~${M} +- ${Math.round(Math.sqrt(2 * M))})`);
for (const k of ['ppat', 'uniform', 'trace'])
  console.log(`mean ${k.padEnd(7)} moves/entries per playout: js ${(tot.js[k] / n).toFixed(2)}  wasm ${(tot.wasm[k] / n).toFixed(2)}`);
console.log(`largest z^2: ${zs.slice(0, 4).map(z => `${z.z2.toFixed(1)} (#${z.idx}: js ${z.pj.toFixed(2)} wasm ${z.pv.toFixed(2)} @ phase ${z.phase.toFixed(2)})`).join(', ')}`);
console.log(`trace invariant violations: ${traceBad}`);
console.log(`time per playout: js ${(1000 * jsMs / n).toFixed(1)} us, wasm (descend + playout + trace read) ${(1000 * wasmMs / n).toFixed(1)} us`);
const ok = traceBad === 0 && Math.abs(d) < 3 * se + 1e-9 && Math.abs(chi2 - M) < 4 * Math.sqrt(2 * M);
console.log(ok ? 'PASS' : 'FAIL');
process.exit(ok ? 0 : 1);
