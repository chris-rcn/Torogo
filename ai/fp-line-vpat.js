'use strict';
// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.

// fp-line-vpat: featurepol filters the root moves, a featurepol line values
// each one.  Featurepol's top LN_WIDTH moves are the candidates.  Each is
// played, then featurepol's top move for the side to move (width 1 beyond the
// root), until the line is LN_DEPTH plies long, root move included.  The vpat
// evaluator scores the position after every ply of the line, and the
// candidate's value is the mean of those scores, in the mover's view.  The
// best-valued candidate is played (ties: featurepol's order).
//
// valueB(game, options) is that line value for any position: the mean
// P(BLACK wins) over the position itself and the next LN_DEPTH − 1 positions of
// featurepol's line from it.  getMove values each candidate by valueB of the
// position after it.
//
// A line that ends early — the game is over, or featurepol lists no move —
// averages over the positions it reached; a finished game scores
// estimateWinner.
//
// Config (cfg reader, slot-aware):
//   LN_DEPTH        plies per line, the root move included; 1 = the root
//                   moves' own results                                (default 7)
//   LN_WIDTH        featurepol candidates at the root; 0 = every one   (default 4)
//   LN_FPOL_DATA    featurepol weights             (default featurepol-cbk7wa32.js)
//   LN_VPAT_DATA    evaluator                      (default out/vpat-wabotxn6.js)
//   LN_DITHER       uniform noise in [0, LN_DITHER·(1 − phase)) added to each
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

  const LN_DEPTH   = cfg.int('LN_DEPTH', 7);
  const LN_WIDTH   = cfg.int('LN_WIDTH', 4);
  if (!(LN_DEPTH >= 1))   throw new Error(`fp-line-vpat: LN_DEPTH must be >= 1, got ${LN_DEPTH}`);
  if (!(LN_WIDTH >= 0))   throw new Error(`fp-line-vpat: LN_WIDTH must be >= 0, got ${LN_WIDTH}`);
  const LN_DITHER  = cfg.float('LN_DITHER', 0.001);
  if (!(LN_DITHER >= 0))  throw new Error(`fp-line-vpat: LN_DITHER must be >= 0, got ${LN_DITHER}`);
  const ownRng = makeRng();

  const fpPath = _isNode ? cfg.str('LN_FPOL_DATA', require('path').join(__dirname, '..', 'featurepol-cbk7wa32.js')) : undefined;
  const { weights: fpWeights, modelName: fpName } = FeaturePol.loadModel({ name: 'fp-line-vpat', path: fpPath });

  const vpatPath = _isNode ? cfg.str('LN_VPAT_DATA', require('path').join(__dirname, '..', 'out', 'vpat-wabotxn6.js')) : null;
  const vpatRaw = _isNode ? require(require('path').resolve(vpatPath))
                          : (typeof window !== 'undefined' && window.vpatternsModel) || null;
  if (!vpatRaw) throw new Error(`fp-line-vpat: cannot load the vpat model from ${_isNode ? vpatPath : 'window.vpatternsModel'}`);
  const vpatModel = VPat.modelFromRaw(vpatRaw, undefined, { float32: true });

  // One Game3 per decision, advanced in lockstep with each line (play, then undo
  // back to the root), when featurepol's ladder features or the vpat model need it.
  const needG3 = fpWeights.spec.needsLadder || VPat.needsGame3(vpatModel.preparedSpecs);

  console.error(`fp-line-vpat[${cfg.slot != null ? cfg.slot : '-'}]: depth: ${LN_DEPTH}, width: ${LN_WIDTH === 0 ? 'all' : LN_WIDTH}, dither: ${LN_DITHER}, ` +
                `fp: ${fpName} (${fpWeights.map.size} weights), ` +
                `vpat: ${_isNode ? require('path').basename(vpatPath) : 'window.vpatternsModel'} (${VPat.specString(vpatModel.specs)})`);

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

  // Mean P(BLACK wins) over `game` and the next LN_DEPTH − 1 positions of
  // featurepol's line from it (fewer if the game ends or featurepol lists no
  // move).  options.game3: a Game3 synced to `game`, back where it was on
  // return; built here when needed and not supplied.  `game` is not changed.
  function valueB(game, options = {}) {
    const game3 = needG3 ? (options.game3 || game3FromGame2(game)) : undefined;
    if (stateN !== game.N) ensureState(game.N);
    const g = game.clone();
    let sum = 0, n = 0, plays = 0;
    for (let d = 0; ; d++) {
      sum += g.gameOver ? (g.estimateWinner() === BLACK ? 1 : 0) : VPat.evaluate(g, vpatModel, game3);
      n++;
      if (g.gameOver || d === LN_DEPTH - 1) break;
      const move = fpBest(g, game3);
      if (move === PASS) break;
      g.play(move);
      if (game3) { game3.play(move); plays++; }
    }
    if (game3) for (let i = 0; i < plays; i++) game3.undo();
    return sum / n;
  }

  function getMove(game, _budgetMs, options = {}) {
    if (game.gameOver) return { move: PASS };
    const rng = options.rng || ownRng;
    const dither = LN_DITHER * (1 - game.phase());
    ensureState(game.N);
    const g3 = needG3 ? game3FromGame2(game) : undefined;
    const cand = new Int32Array(LN_WIDTH || game.N * game.N);
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
    return { move: best, info: `line=${bestV.toFixed(3)} d${LN_DEPTH}w${LN_WIDTH}` };
  }

  return { getMove, valueB };
}

let _default = null;
function _def() { return _default || (_default = create(Util.makeCfg())); }
if (typeof module !== 'undefined') module.exports = { create, getMove: (g, b, o) => _def().getMove(g, b, o), valueB: (g, o) => _def().valueB(g, o) };
else { window.getMove = (g, b, o) => _def().getMove(g, b, o); window.valueB = (g, o) => _def().valueB(g, o); }

})();
