'use strict';
// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.

// fp-line-vpat: featurepol filters the root moves, a featurepol line values
// each one.  Featurepol's top WIDTH moves are the candidates.  Each is
// played, then featurepol's top move for the side to move (width 1 beyond the
// root), until the line is DEPTH plies long, root move included.  The vpat
// evaluator scores the position after every ply of the line, and the
// candidate's value is the mean of those scores, in the mover's view.  The
// best-valued candidate is played (ties: featurepol's order).
//
// A line that ends early — the game is over, or featurepol lists no move —
// averages over the positions it reached; a finished game scores
// estimateWinner.
//
// Config (cfg reader, slot-aware):
//   DEPTH           plies per line, the root move included; 1 = the root
//                   moves' own results                                (default 7)
//   DEPTH_MIN       each move draws its depth uniformly from DEPTH_MIN..DEPTH,
//                   inclusive                                    (default DEPTH)
//   WIDTH           featurepol candidates at the root; 0 = every one   (default 4)
//   WIDTH_MIN       each move draws its width uniformly from WIDTH_MIN..WIDTH,
//                   inclusive; with WIDTH 0 it must be 0         (default WIDTH)
//   FPOL_DATA       featurepol weights             (default featurepol-cbk7wa32.js)
//   VPAT_DATA       evaluator                      (default out/vpat-wabotxn6.js)
//   DITHER          uniform noise in [0, DITHER·(1 − phase)) added to each root
//                   candidate's value, phase = the root's fullness: full on an
//                   empty board, zero at phase 1; 0 = off            (default 0.001)
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

  const DEPTH      = cfg.int('DEPTH', 7);
  const WIDTH      = cfg.int('WIDTH', 4);
  if (!(DEPTH >= 1))      throw new Error(`fp-line-vpat: DEPTH must be >= 1, got ${DEPTH}`);
  const DEPTH_MIN  = cfg.int('DEPTH_MIN', DEPTH);
  if (!(DEPTH_MIN >= 1 && DEPTH_MIN <= DEPTH))
    throw new Error(`fp-line-vpat: DEPTH_MIN must be in 1..DEPTH, got ${DEPTH_MIN} with DEPTH ${DEPTH}`);
  if (!(WIDTH >= 0))      throw new Error(`fp-line-vpat: WIDTH must be >= 0, got ${WIDTH}`);
  const WIDTH_MIN  = cfg.int('WIDTH_MIN', WIDTH);
  if (WIDTH === 0 ? WIDTH_MIN !== 0 : !(WIDTH_MIN >= 1 && WIDTH_MIN <= WIDTH))
    throw new Error(`fp-line-vpat: WIDTH_MIN must be in 1..WIDTH (0 with WIDTH 0), got ${WIDTH_MIN} with WIDTH ${WIDTH}`);
  const DITHER     = cfg.float('DITHER', 0.001);
  if (!(DITHER >= 0))     throw new Error(`fp-line-vpat: DITHER must be >= 0, got ${DITHER}`);
  const ownRng = makeRng();

  const fpPath = _isNode ? cfg.str('FPOL_DATA', require('path').join(__dirname, '..', 'featurepol-cbk7wa32.js')) : undefined;
  const { weights: fpWeights, modelName: fpName } = FeaturePol.loadModel({ name: 'fp-line-vpat', path: fpPath });

  const vpatPath = _isNode ? cfg.str('VPAT_DATA', require('path').join(__dirname, '..', 'out', 'vpat-wabotxn6.js')) : null;
  const vpatRaw = _isNode ? require(require('path').resolve(vpatPath))
                          : (typeof window !== 'undefined' && window.vpatternsModel) || null;
  if (!vpatRaw) throw new Error(`fp-line-vpat: cannot load the vpat model from ${_isNode ? vpatPath : 'window.vpatternsModel'}`);
  const vpatModel = VPat.modelFromRaw(vpatRaw, undefined, { float32: true });

  // One Game3 per decision, advanced in lockstep with each line (play, then undo
  // back to the root), when featurepol's ladder features or the vpat model need it.
  const needG3 = fpWeights.spec.needsLadder || VPat.needsGame3(vpatModel.preparedSpecs);

  console.error(`fp-line-vpat[${cfg.slot != null ? cfg.slot : '-'}]: depth: ${DEPTH_MIN < DEPTH ? `${DEPTH_MIN}-${DEPTH}` : DEPTH}, width: ${WIDTH === 0 ? 'all' : WIDTH_MIN < WIDTH ? `${WIDTH_MIN}-${WIDTH}` : WIDTH}, dither: ${DITHER}, ` +
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

  // Mean P(BLACK wins) over the positions after each ply of the line that
  // starts with rootMove, `depth` plies long.  game3 (if any) is back at the
  // root on return.
  function lineValue(game, rootMove, game3, depth) {
    const g = game.clone();
    let move = rootMove, sum = 0, n = 0;
    for (let d = 0; d < depth; d++) {
      g.play(move);
      if (game3) game3.play(move);
      sum += g.gameOver ? (g.estimateWinner() === BLACK ? 1 : 0) : VPat.evaluate(g, vpatModel, game3);
      n++;
      if (g.gameOver || d === depth - 1) break;
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
    // This move's width and depth; each drawn only when it has a range, so a
    // fixed one consumes no rng.
    const w = WIDTH_MIN < WIDTH ? WIDTH_MIN + Math.floor(rng.random() * (WIDTH - WIDTH_MIN + 1)) : WIDTH;
    const depth = DEPTH_MIN < DEPTH ? DEPTH_MIN + Math.floor(rng.random() * (DEPTH - DEPTH_MIN + 1)) : DEPTH;
    const cand = new Int32Array(w || game.N * game.N);
    const k = fpRanked(game, g3, cand);
    if (k === 0) return { move: PASS };
    const mover = game.current;
    let best = cand[0], bestV = -Infinity;
    for (let j = 0; j < k; j++) {
      const v = lineValue(game, cand[j], g3, depth);
      const mv = (mover === BLACK ? v : 1 - v) + (dither > 0 ? rng.random() * dither : 0);
      if (mv > bestV) { bestV = mv; best = cand[j]; }
    }
    return { move: best, info: `line=${bestV.toFixed(3)} d${depth}w${w}` };
  }

  return { getMove };
}

let _default = null;
function _def() { return _default || (_default = create(Util.makeCfg())); }
if (typeof module !== 'undefined') module.exports = { create, getMove: (g, b, o) => _def().getMove(g, b, o) };
else window.getMove = (g, b, o) => _def().getMove(g, b, o);

})();
