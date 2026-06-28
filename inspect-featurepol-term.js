'use strict';

// inspect-featurepol-term.js — visualise what one featurepol spec term's keys
// look like.  Plays softmax self-play games with a loaded model; for the chosen
// term (one spec space, possibly a conjunction), the FIRST time each distinct
// key is encountered it prints the board (centred on the candidate move that
// produced the key) and that key's learned logit (the model weight).
//
// Usage:
//   node inspect-featurepol-term.js --data <featurepol.js> --term '<spec term>'
//        [--size 13] [--games 50] [--max-keys 100] [--seed 1] [--temp 1]
//
//   --term   a SINGLE spec space, e.g. 'capture9', 'adjLib4', 'stones8+lib4'
//            (must be one of the model's spaces, else its keys are unlearned/0).

const path = require('path');
const FP = require('./featurepol-lib.js');
const { Game2, PASS, coordStr } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');
const { makeRng } = require('./xorshift.js');
const Util = require('./util.js');

const opts = Util.parseArgs(process.argv.slice(2), ['help'], ['data', 'games', 'max-keys', 'seed', 'size', 'temp', 'term']);
if (opts.help || !opts.data || !opts.term) {
  console.log("Usage: node inspect-featurepol-term.js --data <featurepol.js> --term '<spec term>' [--size 13] [--games 50] [--max-keys 100] [--seed 1] [--temp 1]");
  process.exit(opts.help ? 0 : 1);
}
const N        = parseInt(opts.size || '13', 10);
const MAXGAMES = parseInt(opts.games || '50', 10);
const MAXKEYS  = parseInt(opts['max-keys'] || '100', 10);
const SEED     = parseInt(opts.seed || '1', 10);
const TEMP     = opts.temp !== undefined ? parseFloat(opts.temp) : 1;
const TERM     = String(opts.term).trim();

const { weights: model } = FP.loadModel({ name: 'inspect', path: opts.data });

const termSpec = FP.parseSpec(TERM);
if (termSpec.spaces.length !== 1) {
  console.error(`--term must be a single spec space (no commas); got ${termSpec.spaces.length}`);
  process.exit(1);
}
const modelSpaces = new Set(model.spec.spaces.map(s => s.str));
if (!modelSpaces.has(TERM)) {
  console.error(`ERROR: "${TERM}" is not a space in the model spec.`);
  console.error(`  model spaces: ${[...modelSpaces].join(', ')}`);
  process.exit(1);
}

const rng         = makeRng(SEED);
const modelState  = FP.createState(N, model.spec);
const termState   = FP.createState(N, termSpec);
const termWeights = FP.createWeights({ spec: termSpec });   // just to intern the term's keys → raw hashes
const needLadder  = model.spec.needsLadder || termSpec.needsLadder;

const seen = new Set();
const idx2hash = new Map();
let lastTermSize = 0, shown = 0;

console.error(`inspect: model ${model.size}w from ${path.basename(opts.data)}  term='${TERM}'  size=${N}  temp=${TEMP}  seed=${SEED}`);

outer:
for (let g = 0; g < MAXGAMES; g++) {
  const game  = new Game2(N);
  const game3 = game3FromGame2(game);
  let mv = 0;
  while (!game.gameOver && mv < N * N * 4) {
    // Term's keys for every candidate move at this position.
    FP.extractFeatures(game, termState, termWeights, termSpec.needsLadder ? game3 : undefined);
    if (termWeights.size !== lastTermSize) {             // keep dense-idx → raw-hash map current
      for (const [h, d] of termWeights.map) if (!idx2hash.has(d)) idx2hash.set(d, h);
      lastTermSize = termWeights.size;
    }
    for (let i = 0; i < termState.count; i++) {
      const idx = termState.moves[i];
      for (let k = termState.keyOff[i]; k < termState.keyOff[i + 1]; k++) {
        const hash = idx2hash.get(termState.keys[k]);
        if (seen.has(hash)) continue;
        seen.add(hash);
        const di = model.map.get(hash);
        const logit = di !== undefined ? model.vals[di] : 0;
        shown++;
        console.log(`\n=== key #${shown}  game ${g + 1} move ${mv + 1}  point=${coordStr(idx, N)}  logit=${logit.toFixed(4)}${di === undefined ? '  (UNLEARNED)' : ''} ===`);
        console.log(game.toString(idx, { centerAt: idx, labels: true }));
        if (shown >= MAXKEYS) { console.error(`reached --max-keys ${MAXKEYS}`); break outer; }
      }
    }
    // softmax move from the full model, then advance.
    const choice = FP.policyMove(game, modelState, model, rng, needLadder ? game3 : undefined, TEMP);
    if (choice.index < 0 || choice.move === PASS) break;
    game.play(choice.move);
    game3.play(choice.move);
    mv++;
  }
}
console.error(`done: ${shown} distinct keys shown across up to ${MAXGAMES} games`);
