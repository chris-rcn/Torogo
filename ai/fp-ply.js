'use strict';
// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.

// fp-ply: fp-ply-vpat with an AGENT as the evaluator.  Featurepol filters the
// root moves, a featurepol-weighted one-ply expectation values each one.
// Featurepol's top PL_WIDTH moves are the candidates.  Each is played, and the
// resulting position is valued by valueB: the mean of VALUE_AGENT's
// valueB(game, { game3, rng }) -> P(BLACK wins) over EVERY next-ply position
// (each move featurepol lists for the side to move), weighted by featurepol's
// softmax probability of that move.  The candidate's value, in the mover's
// view plus uniform noise in [0, PL_DITHER·(1 − root phase)), picks the move
// (ties: featurepol's order).  A position where featurepol lists no move is
// valued by its pass: estimateWinner if that ends the game, else the value
// agent's valueB.
//
// Config (cfg reader, slot-aware):
//   VALUE_AGENT     ai/<name>.js whose valueB values the next-ply positions;
//                   it reads its own knobs from the same (slot-aware) cfg
//                   (required; in the browser, window.valueAgent)
//   PL_WIDTH        featurepol candidates at the root; 0 = every one   (default 8)
//   PL_TEMP         temperature of the featurepol softmax that weights the
//                   next-ply values; 0 = all weight on featurepol's top move
//                                                                    (default 3)
//   PL_EVAL_TOP_K   next-ply moves that contribute: featurepol's K most probable,
//                   their probabilities renormalised to sum 1; 0 = all (default 10)
//   PL_FPOL_DATA    featurepol weights             (default featurepol-cbk7wa32.js)
//   PL_DITHER       uniform noise in [0, PL_DITHER·(1 − phase)) added to each
//                   root candidate's value, phase = the root's fullness: full on
//                   an empty board, zero at phase 1; 0 = off         (default 0.01)
// In the browser the featurepol model comes from window.featurepolModel.

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
const Util = (typeof require === 'function') ? require('../util.js') : window.Util;
const { PASS, BLACK } = Util.load('./game2.js', 'Game2');
const { game3FromGame2 } = Util.load('./game3.js', 'Game3');
const FeaturePol = Util.load('./featurepol-lib.js', 'FeaturePol');
const { makeRng } = Util.load('./xorshift.js', 'XorShift');

function create(cfg) {
  cfg = cfg || Util.makeCfg();

  const PL_WIDTH   = cfg.int('PL_WIDTH', 8);
  if (!(PL_WIDTH >= 0))   throw new Error(`fp-ply: PL_WIDTH must be >= 0, got ${PL_WIDTH}`);
  const PL_TEMP    = cfg.float('PL_TEMP', 3);
  if (!(PL_TEMP >= 0))    throw new Error(`fp-ply: PL_TEMP must be >= 0, got ${PL_TEMP}`);
  const PL_EVAL_TOP_K = cfg.int('PL_EVAL_TOP_K', 10);
  if (!(PL_EVAL_TOP_K >= 0)) throw new Error(`fp-ply: PL_EVAL_TOP_K must be >= 0, got ${PL_EVAL_TOP_K}`);
  const PL_DITHER  = cfg.float('PL_DITHER', 0.01);
  if (!(PL_DITHER >= 0))  throw new Error(`fp-ply: PL_DITHER must be >= 0, got ${PL_DITHER}`);
  const ownRng = makeRng();

  const fpPath = _isNode ? cfg.str('PL_FPOL_DATA', require('path').join(__dirname, '..', 'featurepol-cbk7wa32.js')) : undefined;
  const { weights: fpWeights, modelName: fpName } = FeaturePol.loadModel({ name: 'fp-ply', path: fpPath });

  // The evaluator: an agent module (create(cfg) factory or a bare module).
  const VALUE_AGENT = _isNode ? cfg.str('VALUE_AGENT', null) : 'window.valueAgent';
  if (!VALUE_AGENT) throw new Error('fp-ply: VALUE_AGENT is required (ai/<name>.js whose valueB values the positions)');
  const valueMod = _isNode ? require(require('path').join(__dirname, VALUE_AGENT + '.js'))
                           : (typeof window !== 'undefined' && window.valueAgent) || null;
  if (!valueMod) throw new Error('fp-ply: window.valueAgent is not set');
  const valueAgent = typeof valueMod.create === 'function' ? valueMod.create(cfg) : valueMod;
  if (typeof valueAgent.valueB !== 'function') throw new Error(`fp-ply: VALUE_AGENT ${VALUE_AGENT} has no valueB`);

  // One Game3 per decision, advanced in lockstep (play, then undo back to the
  // root): featurepol's ladder features read it, and the value agent gets it.
  console.error(`fp-ply[${cfg.slot != null ? cfg.slot : '-'}]: width: ${PL_WIDTH === 0 ? 'all' : PL_WIDTH}, temp: ${PL_TEMP}, eval-top-k: ${PL_EVAL_TOP_K === 0 ? 'all' : PL_EVAL_TOP_K}, dither: ${PL_DITHER}, ` +
                `fp: ${fpName} (${fpWeights.map.size} weights), value-agent: ${VALUE_AGENT}`);

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

  // The value agent's P(BLACK wins) of g (exact when it is over).
  function leafValue(g, game3, rng) {
    if (g.gameOver) return g.estimateWinner() === BLACK ? 1 : 0;
    return valueAgent.valueB(g, { game3, rng });
  }

  // P(BLACK wins) of `game`: the featurepol-softmax-weighted (PL_TEMP) mean of
  // the value agent's values of every next-ply position (featurepol's
  // PL_EVAL_TOP_K most probable, renormalised, when set).  options.game3: a
  // Game3 synced to `game`, back where it was on return; built here when not
  // supplied.  `game` is not changed.
  function valueB(game, options = {}) {
    if (game.gameOver) return game.estimateWinner() === BLACK ? 1 : 0;
    const rng = options.rng || ownRng;
    ensureState(game.N);
    const game3 = options.game3 || game3FromGame2(game);
    FeaturePol.extractFeatures(game, fpState, fpWeights, game3);
    const n = FeaturePol.computeSoftmax(fpState, fpWeights, PL_TEMP);
    if (n === 0) {
      const g = game.clone();
      g.play(PASS);
      game3.play(PASS);
      const v = leafValue(g, game3, rng);
      game3.undo();
      return v;
    }
    // The contributing moves: all n, or the K most probable.  Copied out of
    // the shared featurepol state, which the value agent may overwrite.
    let idx = new Array(n);
    for (let i = 0; i < n; i++) idx[i] = i;
    if (PL_EVAL_TOP_K > 0 && PL_EVAL_TOP_K < n) {
      idx.sort((a, b) => fpState.probs[b] - fpState.probs[a]);
      idx = idx.slice(0, PL_EVAL_TOP_K);
    }
    const moves = idx.map(i => fpState.moves[i]), probs = idx.map(i => fpState.probs[i]);
    let sum = 0, mass = 0;
    for (let j = 0; j < moves.length; j++) {
      const g = game.clone();
      g.play(moves[j]);
      game3.play(moves[j]);
      sum += probs[j] * leafValue(g, game3, rng);
      game3.undo();
      mass += probs[j];
    }
    return sum / mass;
  }

  function getMove(game, _budgetMs, options = {}) {
    if (game.gameOver) return { move: PASS };
    const rng = options.rng || ownRng;
    const dither = PL_DITHER * (1 - game.phase());
    ensureState(game.N);
    const g3 = game3FromGame2(game);
    const cand = new Int32Array(PL_WIDTH || game.N * game.N);
    const k = fpRanked(game, g3, cand);
    if (k === 0) return { move: PASS };
    const mover = game.current;
    let best = cand[0], bestV = -Infinity;
    for (let j = 0; j < k; j++) {
      const child = game.clone();
      child.play(cand[j]);
      g3.play(cand[j]);
      const v = valueB(child, { game3: g3, rng });
      g3.undo();
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
else { window.create = create; window.getMove = (g, b, o) => _def().getMove(g, b, o); window.valueB = (g, o) => _def().valueB(g, o); }

})();
