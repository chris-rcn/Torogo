'use strict';

// ref-ply8 — frozen reference: ai/fp-ply-vpat.js as of 2026-10-08 (last changed
// in 7999f7e) at PL_WIDTH=8, PL_TEMP=3, PL_EVAL_TOP_K=10, PL_DITHER=0.01, with
// the featurepol model featurepol-cbk7wa32.js (tracked) and the vpat model
// frozen as ref/ref-ply8-vpat.js (out/vpat-24-34.js).
//
// Frozen once fielded: any strength-affecting change gets a new name.
// All parameters are hardcoded.  This script reads no environment variables.
//
// fp-ply-vpat: featurepol filters the root moves, a featurepol-weighted
// one-ply expectation values each one.  Featurepol's top PL_WIDTH moves are
// the candidates.  Each is played, and the resulting position is valued by
// valueB: the mean of the vpat evaluations of EVERY next-ply position (each
// move featurepol lists for the side to move), weighted by featurepol's
// softmax probability of that move.  The candidate's value, in the mover's
// view plus uniform noise in [0, PL_DITHER·(1 − root phase)), picks the move
// (ties: featurepol's order).
//
// The next-ply values are incremental (vpatterns deltaZ off one extraction of
// the position) when the vpat model allows it; a capturing move, or every
// move for a model with ladder, health, phased or turn terms, takes a full
// extraction instead.  A position where featurepol lists no move is valued by
// its pass: estimateWinner if that ends the game, else the vpat evaluation.

const path = require('path');
const { PASS, BLACK } = require('../game2.js');
const { game3FromGame2 } = require('../game3.js');
const FeaturePol = require('../featurepol-lib.js');
const VPat = require('../vpatterns.js');
const { makeRng } = require('../xorshift.js');

// Hardcoded configuration (fp-ply-vpat's names):
const PL_WIDTH      = 8;      // featurepol candidates at the root
const PL_TEMP       = 3;      // softmax temperature weighting the next-ply values
const PL_EVAL_TOP_K = 10;     // next-ply moves that contribute (renormalised)
const PL_DITHER     = 0.01;   // root-candidate noise, scaled by (1 - root phase)
const FPOL_PATH = path.join(__dirname, '..', 'featurepol-cbk7wa32.js');    // PL_FPOL_DATA
const VPAT_PATH = path.join(__dirname, '..', 'ref', 'ref-ply8-vpat.js');   // PL_VPAT_DATA

function create() {
  const ownRng = makeRng();
  const { weights: fpWeights, modelName: fpName } = FeaturePol.loadModel({ name: 'ref-ply8', path: FPOL_PATH });
  const vpatModel = VPat.modelFromRaw(require(VPAT_PATH), undefined, { float32: true });
  const prep = vpatModel.preparedSpecs;
  // deltaZ's contract (see vpatterns.deltaZ): none of these term kinds.
  const incremental = !(prep.hasLadder || prep.hasHealth || prep.hasPhasedPatterns || prep.hasTurn);
  // Full extractions go through a SEPARATE prepSpecs, so the base planes
  // deltaZ reads stay valid for the remaining moves.
  const fbPrep = VPat.prepareSpecs(vpatModel.specs, { health: prep.healthModel || undefined,
                                                      ladderMinChain: prep.ladderMinChain });

  // One Game3 per decision, advanced in lockstep (play, then undo back to the
  // root), when featurepol's ladder features or the vpat model need it.
  const vpatG3 = VPat.needsGame3(prep);
  const needG3 = fpWeights.spec.needsLadder || vpatG3;

  console.error(`ref-ply8: fp-ply-vpat at width: ${PL_WIDTH}, temp: ${PL_TEMP}, eval-top-k: ${PL_EVAL_TOP_K}, dither: ${PL_DITHER}, ` +
                `fp: ${fpName} (${fpWeights.map.size} weights), ` +
                `vpat: ${path.basename(VPAT_PATH)} (${VPat.specString(vpatModel.specs)}), ` +
                `incremental: ${incremental ? 'yes' : 'no'}`);

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

  // P(BLACK wins) of `game`: the featurepol-softmax-weighted (PL_TEMP) mean of the vpat
  // values of every next-ply position (featurepol's PL_EVAL_TOP_K most probable,
  // renormalised, when set).  options.game3: a Game3 synced to
  // `game`, back where it was on return; built here when needed and not
  // supplied.  `game` is not changed.
  function valueB(game, options = {}) {
    if (game.gameOver) return game.estimateWinner() === BLACK ? 1 : 0;
    ensureState(game.N);
    const game3 = needG3 ? (options.game3 || game3FromGame2(game)) : undefined;
    FeaturePol.extractFeatures(game, fpState, fpWeights, game3);
    const n = FeaturePol.computeSoftmax(fpState, fpWeights, PL_TEMP);
    if (n === 0) {
      const g = game.clone();
      g.play(PASS);
      return g.gameOver ? (g.estimateWinner() === BLACK ? 1 : 0) : VPat.evaluate(g, vpatModel, game3);
    }
    const g3v = vpatG3 ? game3 : undefined;
    const f = VPat.extractFeatures(game, prep, false, undefined, true, g3v);
    VPat.evaluateFeatures(f, vpatModel.weights);
    const zBase = f.z;
    // The contributing moves: all n, or the K most probable.
    let idx = null, k = n;
    if (PL_EVAL_TOP_K > 0 && PL_EVAL_TOP_K < n) {
      idx = new Array(n);
      for (let i = 0; i < n; i++) idx[i] = i;
      idx.sort((a, b) => fpState.probs[b] - fpState.probs[a]);
      k = PL_EVAL_TOP_K;
    }
    let sum = 0, mass = 0;
    for (let j = 0; j < k; j++) {
      const i = idx ? idx[j] : j;
      const m = fpState.moves[i];
      const d = incremental ? VPat.deltaZ(game, prep, vpatModel.weights, m) : NaN;
      let v;
      if (d === d) {
        v = 1 / (1 + Math.exp(-(zBase + d)));
      } else {
        const g = game.clone();
        g.play(m);
        if (g3v) g3v.play(m);
        v = VPat.evaluateFeatures(VPat.extractFeatures(g, fbPrep, false, undefined, true, g3v), vpatModel.weights);
        if (g3v) g3v.undo();
      }
      sum += fpState.probs[i] * v;
      mass += fpState.probs[i];
    }
    return idx ? sum / mass : sum;
  }

  function getMove(game, _budgetMs, options = {}) {
    if (game.gameOver) return { move: PASS };
    const rng = options.rng || ownRng;
    const dither = PL_DITHER * (1 - game.phase());
    ensureState(game.N);
    const g3 = needG3 ? game3FromGame2(game) : undefined;
    const cand = new Int32Array(PL_WIDTH);
    const k = fpRanked(game, g3, cand);
    if (k === 0) return { move: PASS };
    const mover = game.current;
    let best = cand[0], bestV = -Infinity;
    for (let j = 0; j < k; j++) {
      const child = game.clone();
      child.play(cand[j]);
      if (g3) g3.play(cand[j]);
      const v = valueB(child, { game3: g3 });
      if (g3) g3.undo();
      const mv = (mover === BLACK ? v : 1 - v) + (dither > 0 ? rng.random() * dither : 0);
      if (mv > bestV) { bestV = mv; best = cand[j]; }
    }
    return { move: best, info: `ply=${bestV.toFixed(3)} w${PL_WIDTH}` };
  }

  return { getMove, valueB };
}

let _default = null;
function _def() { return _default || (_default = create()); }
module.exports = { create, getMove: (g, b, o) => _def().getMove(g, b, o), valueB: (g, o) => _def().valueB(g, o) };
