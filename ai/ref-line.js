'use strict';

// ref-line — frozen reference: ai/fp-line-vpat.js as of 2026-10-07 (last
// changed in 378da09) at DEPTH=7, WIDTH=4, DITHER=0.05, with the featurepol
// model featurepol-cbk7wa32.js (tracked) and the vpat model frozen as
// ref/ref-line-vpat.js (out/vpat-wabotxn6.js).
//
// Frozen once fielded: any strength-affecting change gets a new name.
// All parameters are hardcoded.  This script reads no environment variables.
//
// fp-line-vpat: featurepol filters the root moves, a featurepol line values
// each one.  Featurepol's top WIDTH moves are the candidates.  Each is
// played, then featurepol's top move for the side to move (width 1 beyond the
// root), until the line is DEPTH plies long, root move included.  The vpat
// evaluator scores the position after every ply of the line, and the
// candidate's value is the mean of those scores, in the mover's view, plus
// uniform noise in [0, DITHER·(1 − root phase)).  The best-valued candidate is
// played (ties: featurepol's order).
//
// A line that ends early — the game is over, or featurepol lists no move —
// averages over the positions it reached; a finished game scores
// estimateWinner.

const path = require('path');
const { PASS, BLACK } = require('../game2.js');
const { game3FromGame2 } = require('../game3.js');
const FeaturePol = require('../featurepol-lib.js');
const VPat = require('../vpatterns.js');
const { makeRng } = require('../xorshift.js');

// Hardcoded configuration (fp-line-vpat's names):
const DEPTH     = 7;      // plies per line, the root move included
const WIDTH     = 4;      // featurepol candidates at the root
const DITHER    = 0.05;   // root-candidate noise, scaled by (1 - root phase)
const FPOL_PATH = path.join(__dirname, '..', 'featurepol-cbk7wa32.js');    // FPOL_DATA
const VPAT_PATH = path.join(__dirname, '..', 'ref', 'ref-line-vpat.js');   // VPAT_DATA

function create() {
  const ownRng = makeRng();
  const { weights: fpWeights, modelName: fpName } = FeaturePol.loadModel({ name: 'ref-line', path: FPOL_PATH });
  const vpatModel = VPat.modelFromRaw(require(VPAT_PATH), undefined, { float32: true });

  // One Game3 per decision, advanced in lockstep with each line (play, then undo
  // back to the root), when featurepol's ladder features or the vpat model need it.
  const needG3 = fpWeights.spec.needsLadder || VPat.needsGame3(vpatModel.preparedSpecs);

  console.error(`ref-line: fp-line-vpat at depth: ${DEPTH}, width: ${WIDTH}, dither: ${DITHER}, ` +
                `fp: ${fpName} (${fpWeights.map.size} weights), ` +
                `vpat: ${path.basename(VPAT_PATH)} (${VPat.specString(vpatModel.specs)})`);

  let fpState = null, fpScores = null, stateN = 0;
  function ensureState(N) {
    if (stateN === N) return;
    fpState  = FeaturePol.createState(N, fpWeights.spec);
    fpScores = new Float64Array(N * N + 1);
    stateN   = N;
  }

  // Featurepol's candidates of g, best-first, copied into `out` (at most
  // out.length of them).  Returns the count.
  function fpRanked(g, game3, out) {
    FeaturePol.extractFeatures(g, fpState, fpWeights, game3);
    const n = FeaturePol.scoreAll(fpState, fpWeights, fpScores);
    const order = new Array(n);
    for (let i = 0; i < n; i++) order[i] = i;
    order.sort((a, b) => fpScores[b] - fpScores[a]);
    const k = Math.min(out.length, n);
    for (let j = 0; j < k; j++) out[j] = fpState.moves[order[j]];
    return k;
  }

  // Featurepol's top move of g, or PASS when it lists none.
  function fpBest(g, game3) {
    FeaturePol.extractFeatures(g, fpState, fpWeights, game3);
    const n = FeaturePol.scoreAll(fpState, fpWeights, fpScores);
    if (n === 0) return PASS;
    let bi = 0;
    for (let i = 1; i < n; i++) if (fpScores[i] > fpScores[bi]) bi = i;
    return fpState.moves[bi];
  }

  // Mean P(BLACK wins) over the positions after each ply of the line that
  // starts with rootMove.  game3 (if any) is back at the root on return.
  function lineValue(game, rootMove, game3) {
    const g = game.clone();
    let move = rootMove, sum = 0, n = 0;
    for (let d = 0; d < DEPTH; d++) {
      g.play(move);
      if (game3) game3.play(move);
      sum += g.gameOver ? (g.estimateWinner() === BLACK ? 1 : 0) : VPat.evaluate(g, vpatModel, game3);
      n++;
      if (g.gameOver || d === DEPTH - 1) break;
      move = fpBest(g, game3);
      if (move === PASS) break;
    }
    if (game3) for (let i = 0; i < n; i++) game3.undo();
    return sum / n;
  }

  function getMove(game, _budgetMs, options = {}) {
    if (game.gameOver) return { move: PASS };
    const rng = options.rng || ownRng;
    const dither = DITHER * (1 - game.phase());
    ensureState(game.N);
    const g3 = needG3 ? game3FromGame2(game) : undefined;
    const cand = new Int32Array(WIDTH);
    const k = fpRanked(game, g3, cand);
    if (k === 0) return { move: PASS };
    const mover = game.current;
    let best = cand[0], bestV = -Infinity;
    for (let j = 0; j < k; j++) {
      const v = lineValue(game, cand[j], g3);
      const mv = (mover === BLACK ? v : 1 - v) + (dither > 0 ? rng.random() * dither : 0);
      if (mv > bestV) { bestV = mv; best = cand[j]; }
    }
    return { move: best, info: `line=${bestV.toFixed(3)} d${DEPTH}w${WIDTH}` };
  }

  return { getMove };
}

let _default = null;
function _def() { return _default || (_default = create()); }
module.exports = { create, getMove: (g, b, o) => _def().getMove(g, b, o) };
