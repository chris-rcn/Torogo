'use strict';
// Fixture for c/test_fpol.bin: random-game positions scored by featurepol-lib.js.
//
//   node c/test_fpol_gen.js <model.js> [positions] [seed] > fixture.txt
//
// One line per position: N nMoves m1..mN nCand (move logit prob)*nCand, where the
// moves replay from Game2(N, true) (the free centre stone, as C g2_new) and the
// candidates are featurepol's, with logits and temperature-1 probabilities.
// Positions are spread over both board sizes the agents use and every phase.

const path = require('path');
const FeaturePol = require(path.join(__dirname, '..', 'featurepol-lib.js'));
const { Game2, PASS } = require(path.join(__dirname, '..', 'game2.js'));
const { makeRng } = require(path.join(__dirname, '..', 'xorshift.js'));

const [modelPath, nPos = '400', seed = '1'] = process.argv.slice(2);
if (!modelPath) { console.error('usage: node c/test_fpol_gen.js <model.js> [positions] [seed]'); process.exit(1); }
const { weights } = FeaturePol.loadModel({ name: 'test_fpol', path: modelPath });
const rng = makeRng(parseInt(seed, 10));
const states = new Map();
const out = [];
for (let p = 0; p < parseInt(nPos, 10); p++) {
  const N = p % 2 ? 13 : 9;
  const g = new Game2(N, true);
  const target = Math.floor(rng.random() * N * N * 1.2);   // some positions run into the endgame
  const moves = [];
  while (moves.length < target && !g.gameOver) {
    const mv = g.randomLegalMove(rng);
    if (mv === PASS) break;
    g.play(mv); moves.push(mv);
  }
  if (g.gameOver) continue;
  let st = states.get(N);
  if (!st) { st = FeaturePol.createState(N, weights.spec); states.set(N, st); }
  FeaturePol.extractFeatures(g, st, weights);
  FeaturePol.computeSoftmax(st, weights, 1);
  const cand = [];
  for (let i = 0; i < st.count; i++) cand.push(`${st.moves[i]} ${st.logits[i]} ${st.probs[i]}`);
  out.push(`${N} ${moves.length} ${moves.join(' ')} ${st.count} ${cand.join(' ')}`);
}
process.stdout.write(out.join('\n') + '\n');
