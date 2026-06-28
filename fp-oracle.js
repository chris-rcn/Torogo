'use strict';

// fp-oracle.js — ground-truth dumper for the C featurepol port.
//
// For a given --spec, replays --positions seeded-random positions and prints, per
// position, every candidate move's 32-bit feature-key HASHES (not dense indices) and
// the softmax probabilities under a deterministic weight assignment w(hash).  The C
// engine must reproduce these exactly (same hashes, same probs to full double precision).
//
//   node fp-oracle.js --spec stones8 [--size 9] [--positions 40] [--seed 1] [--temp 1]
//
// Output: NDJSON, one record per position:
//   {"size":9,"pos":"c3,...","cands":[{"m":<idx>,"k":[<hash>,...],"p":<prob>},...]}
// plus a first '#' line echoing the spec.  Keys per move are sorted ascending so the
// comparison is order-independent (extraction order can differ but the SET must match).

const FeaturePol = require('./featurepol-lib.js');
const { Game2 } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');
const { makeRng } = require('./xorshift.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), [], ['spec', 'size', 'positions', 'seed', 'temp']);
const SPEC = opts.spec;
const SIZE = parseInt(opts.size || '9', 10);
const NPOS = parseInt(opts.positions || '40', 10);
const SEED = parseInt(opts.seed || '1', 10);
const TEMP = opts.temp !== undefined ? parseFloat(opts.temp) : 1;
if (!SPEC) { console.error('need --spec'); process.exit(1); }

// Deterministic weight as a function of the 32-bit key hash — reproduced identically in C.
function mix32(x) {
  x = x >>> 0;
  x = Math.imul(x ^ (x >>> 16), 0x45d9f3b);
  x = Math.imul(x ^ (x >>> 16), 0x45d9f3b);
  return (x ^ (x >>> 16)) >>> 0;
}
function wForHash(h) { return (mix32((h ^ 0xABCD1234) >>> 0) / 4294967296 - 0.5) * 0.2; }

const weights = FeaturePol.createWeights({ spec: SPEC });
const spec = weights.spec;
const needLadder = spec.needsLadder;
const state = FeaturePol.createState(SIZE, spec);

process.stdout.write(`# spec=${spec.str} size=${SIZE} temp=${TEMP} needsLadder=${needLadder}\n`);

const rng = makeRng(SEED);
for (let p = 0; p < NPOS; p++) {
  const game = new Game2(SIZE);
  const g3 = needLadder ? game3FromGame2(game) : undefined;
  const nSetup = 2 + ((rng.random() * (SIZE * SIZE - 4)) | 0);   // varied board fullness
  const playedMoves = [];
  for (let i = 0; i < nSetup && !game.gameOver; i++) {
    const mv = game.randomLegalMove(rng);
    game.play(mv); if (g3) g3.play(mv);
    playedMoves.push(mv);
  }
  FeaturePol.extractFeatures(game, state, weights, g3);
  // Assign deterministic weights to every interned key (idempotent across positions).
  for (const [hash, idx] of weights.map) weights.vals[idx] = wForHash(hash);
  FeaturePol.computeSoftmax(state, weights, TEMP);

  // invert dense idx -> hash for this dump
  const invMap = new Map();
  for (const [hash, idx] of weights.map) invMap.set(idx, hash);

  const cands = [];
  for (let i = 0; i < state.count; i++) {
    const k0 = state.keyOff[i], k1 = state.keyOff[i + 1];
    const hs = [];
    for (let k = k0; k < k1; k++) hs.push(invMap.get(state.keys[k]) >>> 0);
    hs.sort((a, b) => a - b);
    cands.push({ m: state.moves[i], k: hs, p: state.probs[i] });
  }
  process.stdout.write(JSON.stringify({ size: SIZE, pos: playedMoves.join(','), cands }) + '\n');
}
