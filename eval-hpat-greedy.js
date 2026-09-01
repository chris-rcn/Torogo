'use strict';
// Direct eval of the PURE hpat greedy policy against a reference agent.
// Replicates train-featurepol-reinforce.js's evalVsReference() exactly:
//   3 random opening moves, alternating colours, move limit N*N*4, calcWinner().
// The policy is argmax over legal non-true-eye moves of the mover-relative
// hpat logit z of the resulting position — identical to featurepol's hpat rank 1.

const HP = require('./hpatterns.js');
const { Game2, BLACK, PASS } = require('./game2.js');

const argv = process.argv.slice(2);
const opt = (k, d) => { const i = argv.indexOf('--' + k); return i >= 0 ? argv[i + 1] : d; };
const N      = parseInt(opt('size', '8'), 10);
const GAMES  = parseInt(opt('games', '2000'), 10);
const REF    = opt('ref', 'ref-npat-softmax');
const DATA   = opt('data', 'out/hpat-s8-spec3.js');

const refGetMove = require('./ai/' + REF + '.js').getMove;
const raw   = require(require('path').resolve(DATA));
const model = HP.createModel(raw.maxStones, raw.maxSize);
model.weights = HP.weightsMap(raw);

function zOf(f, w) {
  let z = 0;
  for (let i = 0; i < f.count; i++) { const v = w.get(f.keys[i]); if (v !== undefined) z += f.pols[i] * v; }
  return z;
}

function hpatGreedy(game) {
  const black = game.current === BLACK;
  const emC = game._emptyCells, ec = game.emptyCount, w = model.weights;
  let best = PASS, bestS = -Infinity;
  for (let ei = 0; ei < ec; ei++) {
    const idx = emC[ei];
    if (!game.isLegal(idx) || game.isTrueEye(idx)) continue;
    const z = zOf(HP.extractFeatures(game, model, undefined, idx), w);
    const s = black ? z : -z;
    if (s > bestS) { bestS = s; best = idx; }
  }
  return best;
}

let wins = 0;
const t0 = Date.now();
for (let g = 0; g < GAMES; g++) {
  const policyIsBlack = (g % 2 === 0);
  const game = new Game2(N);
  for (let r = 0; r < 3 && !game.gameOver; r++) game.play(game.randomLegalMove());
  let m = 0;
  while (!game.gameOver && m++ < N * N * 4) {
    let idx;
    if ((game.current === BLACK) === policyIsBlack) {
      idx = hpatGreedy(game);
    } else {
      const mv = refGetMove(game);
      idx = mv && mv.move !== undefined ? mv.move : PASS;
    }
    game.play(idx);
  }
  if ((game.calcWinner() === BLACK) === policyIsBlack) wins++;
  if ((g + 1) % 250 === 0) {
    const se = Math.sqrt((wins / (g + 1)) * (1 - wins / (g + 1)) / (g + 1));
    console.log(`${g + 1} games  wins=${wins}  wr=${(100 * wins / (g + 1)).toFixed(2)}%  +/-${(196 * se).toFixed(2)}  ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }
}
const wr = wins / GAMES, se = Math.sqrt(wr * (1 - wr) / GAMES);
console.log(`\nFINAL  hpat-greedy vs ${REF}  size=${N}  ${wins}/${GAMES} = ${(100 * wr).toFixed(2)}%  95%CI +/-${(196 * se).toFixed(2)}`);
