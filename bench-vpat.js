'use strict';

// bench-vpat.js — vpat evaluation cost: plays uniformly random games and, at
// every position, evaluates each model non-incrementally (full extraction +
// evaluation, the call puct-trunc's truncated playouts make).  Every model
// sees the same positions; the order rotates per position.  Only positions
// with --min-phase <= phase <= --max-phase are evaluated; each game ends once
// the board is fuller than --max-phase.
//
//   node bench-vpat.js --model A.js[,B.js...] [--games 200] [--size 13]
//                      [--min-phase 0] [--max-phase 1] [--seed 1]

const { performance } = require('perf_hooks');
const Util = require('./util.js');
const VPat = require('./vpatterns.js');
const { Game2 } = require('./game2.js');
const { makeRng } = require('./xorshift.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['model', 'games', 'size', 'min-phase', 'max-phase', 'seed']);
if (opts.help || !opts.model) {
  console.log('Usage: node bench-vpat.js --model A.js[,B.js...] [--games 200] [--size 13] [--min-phase 0] [--max-phase 1] [--seed 1]');
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
const models = opts.model.split(',').map(f => ({ file: f, m: VPat.loadWeights(f), ms: 0, n: 0, sink: 0 }));

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
for (const x of models)
  console.log(`${x.file}  spec: ${VPat.specString(x.m.specs)}  weights: ${x.m.weights.size}  evals: ${x.n}  us/eval: ${(1000 * x.ms / x.n).toFixed(2)}`);
