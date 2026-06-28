'use strict';

// fp-train-oracle.js — deterministic REINFORCE mini-run for verifying the C port.
//
// Trains from zero over `--steps` seeded-random positions with an ORDER-INDEPENDENT
// deterministic move choice (so JS and C agree without sharing an RNG or depending on
// candidate order), then optionally serializes the model and dumps final weights.
//
//   node fp-train-oracle.js --spec S --size 9 --steps 200 --seed 1 [--save model.js]
//
// stdout: 'P <csv moves>' per training position (for C to replay), then
//         'W <hash> <weight>' for every weight, sorted by hash.

const fs = require('fs');
const FeaturePol = require('./featurepol-lib.js');
const { Game2 } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');
const { makeRng } = require('./xorshift.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), [], ['spec', 'size', 'steps', 'seed', 'save']);
const SPEC = opts.spec, SIZE = parseInt(opts.size || '9', 10);
const STEPS = parseInt(opts.steps || '200', 10), SEED = parseInt(opts.seed || '1', 10);
const SAVE = opts.save || null;
const LR = 0.05, WD = 0.000002;
if (!SPEC) { console.error('need --spec'); process.exit(1); }

const weights = FeaturePol.createWeights({ spec: SPEC });
const needLadder = weights.spec.needsLadder;
const state = FeaturePol.createState(SIZE, weights.spec);
const rng = makeRng(SEED);
const wStats = { absSum: 0, count: 0 };
let ema = 0, totalUpdates = 0;

const out = [];
for (let i = 0; i < STEPS; i++) {
  const game = new Game2(SIZE);
  const g3 = needLadder ? game3FromGame2(game) : undefined;
  const nSetup = 2 + ((rng.random() * (SIZE * SIZE - 4)) | 0);
  const moves = [];
  for (let j = 0; j < nSetup && !game.gameOver; j++) {
    const mv = game.randomLegalMove(rng); game.play(mv); if (g3) g3.play(mv); moves.push(mv);
  }
  out.push('P ' + moves.join(','));

  FeaturePol.extractFeatures(game, state, weights, g3);
  const n = state.count;
  if (n === 0) continue;
  FeaturePol.computeSoftmax(state, weights, 1);
  // deterministic, order-independent chosen: (i % n)-th smallest move index
  const sorted = Array.from(state.moves.subarray(0, n)).sort((a, b) => a - b);
  const chosenMove = sorted[i % n];
  let ci = 0; for (let k = 0; k < n; k++) if (state.moves[k] === chosenMove) { ci = k; break; }
  const adv = (i % 2 ? 1 : -1) * 0.5;
  totalUpdates += FeaturePol.reinforceUpdate(state, ci, adv, weights, LR, WD, wStats) || 0;
  ema = 0.99 * ema + 0.01 * adv;
}

if (SAVE) fs.writeFileSync(SAVE, FeaturePol.serialize(weights, { spec: weights.spec.str, ema, totalUpdates }));

const pairs = [];
for (const [hash, idx] of weights.map) pairs.push([hash >>> 0, weights.vals[idx]]);
pairs.sort((a, b) => a[0] - b[0]);
for (const [h, v] of pairs) out.push('W ' + h + ' ' + v.toPrecision(17));
process.stdout.write(out.join('\n') + '\n');
