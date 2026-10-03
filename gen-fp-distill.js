'use strict';

// gen-fp-distill.js — distillation targets for a featurepol PUCT prior.
//
// Positions come from fp-heavy self-play from an empty board opened with
// RANDOM_STONES uniformly random moves (no free centre stone), cut at a
// random phase (half drawn from [0.4, 0.8], where pruning costs most; half
// from [0, 0.8]).  At each,
// the model's top --cands (60) moves (vpat ranking off, as in puct-trunc) are each
// played and searched once by puct-trunc at --playouts, so moves top-K pruning
// would drop still get their own win ratio.  NDJSON to stdout, one record per
// position; progress to stderr:
//
//   {"size":13,"moves":[..],"phase":0.52,"po":50,"cands":["d4",..],"wr":[0.613,..]}
//     moves: coordStr history from the empty board (Game2(size, false)),
//            the random opening included
//     wr[i]: the position's mover's win ratio after cands[i]
//
// Usage: node gen-fp-distill.js [--positions 1000] [--cands 60] [--playouts 50]
//                               [--model ref/ref-fp2-data.js] [--size 13] [--seed N] > out/x.ndjson

const Util = require('./util.js');
const FP = require('./featurepol-lib.js');
const { Game2, PASS, coordStr } = require('./game2.js');
const { makeRng } = require('./xorshift.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['positions', 'cands', 'playouts', 'model', 'size', 'seed']);
if (opts.help) {
  console.log(`Usage: node gen-fp-distill.js [options] > out/records.ndjson

Per-candidate puct-trunc win ratios for featurepol distillation.

  --positions N   positions to record                          (default 1000)
  --cands N       candidates per position, the model's top N   (default 60)
  --playouts N    puct-trunc playouts per candidate            (default 50)
  --model FILE    featurepol model ranking the candidates      (default ref/ref-fp2-data.js)
  --size N        board size                                   (default 13)
  --seed N        rng seed                                     (default: random, logged)
  --help          show this message`);
  process.exit(0);
}
const POSITIONS = parseInt(opts.positions || '1000', 10);
const CANDS     = parseInt(opts.cands || '60', 10);
const PLAYOUTS  = parseInt(opts.playouts || '50', 10);
const SIZE      = parseInt(opts.size || '13', 10);
const MODEL     = opts.model || 'ref/ref-fp2-data.js';
const SEED      = opts.seed !== undefined ? parseInt(opts.seed, 10) : Util.randomSeed();
for (const [k, v] of [['positions', POSITIONS], ['cands', CANDS], ['playouts', PLAYOUTS], ['size', SIZE]])
  if (!(v >= 1)) { console.error(`--${k} must be a positive integer`); process.exit(1); }

const rng = makeRng(SEED);
const { weights } = FP.loadModel({ path: MODEL });
if (weights.spec.rankSpaces && weights.spec.rankSpaces.length) weights.rankTopN = 0;   // as puct-trunc
const fpState = FP.createState(SIZE, weights.spec);
const fpHeavy = require('./ai/ref-fp-heavy.js');
const teacher = require('./ai/puct-trunc.js').create(Util.makeCfg(null));
console.error(`gen-fp-distill: positions ${POSITIONS}, cands ${CANDS}, playouts ${PLAYOUTS}, model ${MODEL}, size ${SIZE}, seed ${SEED}`);

const RANDOM_STONES = 4;   // uniformly random opening moves, in place of the free centre stone
const t0 = Date.now();
let done = 0;
while (done < POSITIONS) {
  // Position: fp-heavy self-play to a sampled phase.
  const target = rng.random() < 0.5 ? 0.4 + 0.4 * rng.random() : 0.8 * rng.random();
  const g = new Game2(SIZE, false);
  const moves = [];
  for (let k = 0; k < RANDOM_STONES; k++) {
    const m = g.randomLegalMove(rng);
    g.play(m);
    moves.push(coordStr(m, SIZE));
  }
  while (!g.gameOver && g.phase() < target) {
    const m = fpHeavy.getMove(g, 0, { rng }).move;
    g.play(m);
    moves.push(coordStr(m, SIZE));
  }
  if (g.gameOver) continue;

  // Candidates: the model's top CANDS placements.
  FP.extractFeatures(g, fpState, weights);
  FP.computeSoftmax(fpState, weights);
  const order = Array.from({ length: fpState.count }, (_, i) => i)
    .sort((a, b) => fpState.probs[b] - fpState.probs[a]).slice(0, CANDS).map(i => fpState.moves[i]);
  if (order.length < 2) continue;

  // Teacher: one search per candidate, from the position after it.
  const wr = [];
  for (const m of order) {
    const c = g.clone();
    c.play(m);
    let w;
    if (c.gameOver) w = c.calcWinner() === g.current ? 1 : 0;
    else {
      const res = teacher.getMove(c, 1e9, { rng, playoutLimit: PLAYOUTS });
      w = 1 - res.rootWinRatio;   // rootWinRatio is the child's mover's
    }
    wr.push(+w.toFixed(3));
  }
  process.stdout.write(JSON.stringify({ size: SIZE, moves, phase: +g.phase().toFixed(3), po: PLAYOUTS,
                                        cands: order.map(m => coordStr(m, SIZE)), wr }) + '\n');
  done++;
  if (done % 10 === 0 || done === POSITIONS)
    console.error(`  ${done}/${POSITIONS} positions  ${((Date.now() - t0) / 1000).toFixed(0)}s`);
}
