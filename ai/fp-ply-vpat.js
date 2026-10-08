'use strict';
// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.

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
//
// Config (cfg reader, slot-aware):
//   PL_WIDTH        featurepol candidates at the root; 0 = every one   (default 4)
//   PL_FPOL_DATA    featurepol weights             (default featurepol-cbk7wa32.js)
//   PL_VPAT_DATA    evaluator                      (default out/vpat-24-34.js)
//   PL_DITHER       uniform noise in [0, PL_DITHER·(1 − phase)) added to each
//                   root candidate's value, phase = the root's fullness: full on
//                   an empty board, zero at phase 1; 0 = off         (default 0.001)
// In the browser the models come from window.featurepolModel and
// window.vpatternsModel.

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
const Util = (typeof require === 'function') ? require('../util.js') : window.Util;
const { PASS, BLACK } = Util.load('./game2.js', 'Game2');
const { game3FromGame2 } = Util.load('./game3.js', 'Game3');
const FeaturePol = Util.load('./featurepol-lib.js', 'FeaturePol');
const VPat = Util.load('./vpatterns.js', 'VPatterns');
const { makeRng } = Util.load('./xorshift.js', 'XorShift');

function create(cfg) {
  cfg = cfg || Util.makeCfg();

  const PL_WIDTH   = cfg.int('PL_WIDTH', 4);
  if (!(PL_WIDTH >= 0))   throw new Error(`fp-ply-vpat: PL_WIDTH must be >= 0, got ${PL_WIDTH}`);
  const PL_DITHER  = cfg.float('PL_DITHER', 0.001);
  if (!(PL_DITHER >= 0))  throw new Error(`fp-ply-vpat: PL_DITHER must be >= 0, got ${PL_DITHER}`);
  const ownRng = makeRng();

  const fpPath = _isNode ? cfg.str('PL_FPOL_DATA', require('path').join(__dirname, '..', 'featurepol-cbk7wa32.js')) : undefined;
  const { weights: fpWeights, modelName: fpName } = FeaturePol.loadModel({ name: 'fp-ply-vpat', path: fpPath });

  const vpatPath = _isNode ? cfg.str('PL_VPAT_DATA', require('path').join(__dirname, '..', 'out', 'vpat-24-34.js')) : null;
  const vpatRaw = _isNode ? require(require('path').resolve(vpatPath))
                          : (typeof window !== 'undefined' && window.vpatternsModel) || null;
  if (!vpatRaw) throw new Error(`fp-ply-vpat: cannot load the vpat model from ${_isNode ? vpatPath : 'window.vpatternsModel'}`);
  const vpatModel = VPat.modelFromRaw(vpatRaw, undefined, { float32: true });
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

  console.error(`fp-ply-vpat[${cfg.slot != null ? cfg.slot : '-'}]: width: ${PL_WIDTH === 0 ? 'all' : PL_WIDTH}, dither: ${PL_DITHER}, ` +
                `fp: ${fpName} (${fpWeights.map.size} weights), ` +
                `vpat: ${_isNode ? require('path').basename(vpatPath) : 'window.vpatternsModel'} (${VPat.specString(vpatModel.specs)}), ` +
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

  // P(BLACK wins) of `game`: the featurepol-softmax-weighted mean of the vpat
  // values of every next-ply position.  options.game3: a Game3 synced to
  // `game`, back where it was on return; built here when needed and not
  // supplied.  `game` is not changed.
  function valueB(game, options = {}) {
    if (game.gameOver) return game.estimateWinner() === BLACK ? 1 : 0;
    ensureState(game.N);
    const game3 = needG3 ? (options.game3 || game3FromGame2(game)) : undefined;
    FeaturePol.extractFeatures(game, fpState, fpWeights, game3);
    const n = FeaturePol.computeSoftmax(fpState, fpWeights);
    if (n === 0) {
      const g = game.clone();
      g.play(PASS);
      return g.gameOver ? (g.estimateWinner() === BLACK ? 1 : 0) : VPat.evaluate(g, vpatModel, game3);
    }
    const g3v = vpatG3 ? game3 : undefined;
    const f = VPat.extractFeatures(game, prep, false, undefined, true, g3v);
    VPat.evaluateFeatures(f, vpatModel.weights);
    const zBase = f.z;
    let sum = 0;
    for (let i = 0; i < n; i++) {
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
    }
    return sum;
  }

  function getMove(game, _budgetMs, options = {}) {
    if (game.gameOver) return { move: PASS };
    const rng = options.rng || ownRng;
    const dither = PL_DITHER * (1 - game.phase());
    ensureState(game.N);
    const g3 = needG3 ? game3FromGame2(game) : undefined;
    const cand = new Int32Array(PL_WIDTH || game.N * game.N);
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
function _def() { return _default || (_default = create(Util.makeCfg())); }
if (typeof module !== 'undefined') module.exports = { create, getMove: (g, b, o) => _def().getMove(g, b, o), valueB: (g, o) => _def().valueB(g, o) };
else { window.getMove = (g, b, o) => _def().getMove(g, b, o); window.valueB = (g, o) => _def().valueB(g, o); }

})();
