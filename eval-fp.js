'use strict';
// Standalone eval of a saved featurepol model against a reference agent.
// Mirrors train-featurepol-reinforce.js evalVsReference() exactly:
//   3 random opening moves, alternating colours, move limit N*N*4, calcWinner().
// The hpat model comes from FP_HPAT_DATA, so the same fp weights can be scored
// against different ranking models.
const path = require('path');
const FeaturePol = require('./featurepol-lib.js');
const { Game2, BLACK, PASS, setKomi, KOMI } = require('./game2.js');
const { game3FromGame2 } = require('./game3.js');

const argv = process.argv.slice(2);
const opt = (k, d) => { const i = argv.indexOf('--' + k); return i >= 0 ? argv[i + 1] : d; };
const N     = parseInt(opt('size', '7'), 10);
const GAMES = parseInt(opt('games', '10000'), 10);
const REF   = opt('ref', 'ref-npat-softmax');
const MODEL = opt('model');
const TOPN  = parseInt(opt('topn', '2'), 10);
if (opt('komi') !== undefined) setKomi(N, parseFloat(opt('komi')));

const evalGetMove = require('./ai/' + REF + '.js').getMove;
const loaded = FeaturePol.loadModel({ path: MODEL });
const weights = loaded.weights;
if (TOPN > 0) FeaturePol.setHpatTopN(TOPN);
FeaturePol.setHpatPositionRatio(1);

const state = FeaturePol.createState(N, weights.spec);
const needTac = weights.spec.needsLadder;
console.log(`model=${MODEL}  spec='${weights.spec.str}'  FP_HPAT_DATA=${process.env.FP_HPAT_DATA || '(none)'}`);
console.log(`size=${N}  komi=${KOMI(N)}  ref=${REF}  eval-hpat-topn=${TOPN}  games=${GAMES}`);

let wins = 0;
const t0 = Date.now();
for (let g = 0; g < GAMES; g++) {
  const policyIsBlack = (g % 2 === 0);
  const game = new Game2(N);
  const game3 = game3FromGame2(game);
  for (let r = 0; r < 3 && !game.gameOver; r++) { const mv = game.randomLegalMove(); game.play(mv); game3.play(mv); }
  let m = 0;
  while (!game.gameOver && m++ < N * N * 4) {
    let idx;
    if ((game.current === BLACK) === policyIsBlack) {
      idx = FeaturePol.greedyMove(game, state, weights, needTac ? game3 : undefined);
    } else {
      const mv = evalGetMove(game);
      idx = mv && mv.move !== undefined ? mv.move : PASS;
    }
    game.play(idx); game3.play(idx);
  }
  if ((game.calcWinner() === BLACK) === policyIsBlack) wins++;
  if ((g + 1) % 1000 === 0) {
    const p = wins / (g + 1), se = Math.sqrt(p * (1 - p) / (g + 1));
    console.log(`${g + 1} games  wr=${(100 * p).toFixed(2)}%  +/-${(196 * se).toFixed(2)}  ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }
}
const p = wins / GAMES, se = Math.sqrt(p * (1 - p) / GAMES);
console.log(`\nFINAL  ${path.basename(MODEL)}  hpat=${path.basename(process.env.FP_HPAT_DATA || 'none')}  ${wins}/${GAMES} = ${(100 * p).toFixed(2)}%  95%CI +/-${(196 * se).toFixed(2)}`);
