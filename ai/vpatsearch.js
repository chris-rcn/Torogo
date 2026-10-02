'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.
// Wrapped in an IIFE to avoid polluting the global namespace.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

/**
 * Pattern-weight policy agent with alpha-beta search.
 *
 * Value function: V(s) = σ(Σ polarity_i · w[key_i]) = P(BLACK wins)
 * Move selection: full-width alpha-beta, BLACK maximises V, WHITE minimises V.
 *
 * create(cfg) builds an instance from a config reader (Util.makeCfg), so
 * selfplay can run two with different models (P1_VPAT_DATA / P2_VPAT_DATA).
 * Config: VPAT_DATA (model file, default out/ref13.js), SEARCH_DEPTH (1),
 * DITHER (0.002), and MIN_LIBS / MAX_LIBS (1, 1) for the specs used when no
 * model file loads (the browser).  A health-coded model (size:H<N> specs)
 * carries its health model embedded.  The module-level getMove / valueB use
 * a default instance built from plain env on first use.
 */

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;

const { extractFeatures, evaluateFeatures, deltaZ, loadWeights, prepareSpecs, makeWeights, needsGame3 } = _isNode ? require('../vpatterns.js') : window.VPatterns;
const { search: abSearch } = _isNode ? require('../ab-search.js') : window.ABSearch;
const Util = _isNode ? require('../util.js') : window.Util;
const { BLACK, PASS } = _isNode ? require('../game2.js') : window.game;
const { game3FromGame2 } = _isNode ? require('../game3.js') : window.Game3;

// ── Agent factory ─────────────────────────────────────────────────────────────

function create(cfg) {
  const MIN_LIBS     = cfg.int('MIN_LIBS', 1);
  const MAX_LIBS     = cfg.int('MAX_LIBS', 1);
  const SEARCH_DEPTH = cfg.int('SEARCH_DEPTH', 1);
  const DITHER       = cfg.float('DITHER', 0.002);
  const VPAT_DATA    = cfg.str('VPAT_DATA', 'out/ref13.js');

  let model;
  if (_isNode && VPAT_DATA) {
    model = loadWeights(VPAT_DATA);
  } else {
    const specs = [];
    for (let maxLibs = MIN_LIBS; maxLibs <= MAX_LIBS; maxLibs++)
      for (const size of [1, 2, 3]) specs.push({ size, maxLibs });
    model = { weights: makeWeights(), specs, preparedSpecs: prepareSpecs(specs) };
  }

  function getMove(game) {
    return { move: search(game, model, SEARCH_DEPTH, DITHER) };
  }

  // Position value oracle: the static evaluation itself — V(s) = P(BLACK wins).
  // Each call is an independent position with no Game3 in hand, so a ladder spec
  // must rebuild one — inherent here (game3RebuildOk), not a missed reuse.
  function valueB(game) {
    const g = game.cells ? game : game.toGame2();
    if (g.gameOver) return g.calcWinner() === BLACK ? 1 : 0;
    return evaluateFeatures(extractFeatures(g, model.preparedSpecs, false, undefined, true, undefined, true), model.weights);
  }

  return { getMove, valueB };
}

// ── Search ────────────────────────────────────────────────────────────────────

function search(game, m, depth = 1, dither = 0) {
  if (depth === 1) return search1(game, m, dither);
  const evaluate = g => evaluateFeatures(extractFeatures(g, m.preparedSpecs, false, undefined, true), m.weights);
  return abSearch(game, depth, evaluate, dither);
}

// Depth-1 fast path: one base extraction, then speculative-incremental
// deltaZ per candidate (V = sigma(zBase + dz)).  Captures fall back to full
// extraction on a SEPARATE prepSpecs, so the base planes deltaZ reads stay
// valid for the remaining candidates.  Ladder-coded (size:L), chain-
// attribute (C) and phase-binned (pN) specs have no incremental contract,
// so EVERY candidate takes the fallback path there — supported, several
// times slower per move.  Semantics mirror ab() at depth 1:
// board-index order, strict-improvement argmax, PASS considered last under
// the same conditions, terminal PASS scored exactly.
function search1(game, m, dither) {
  const prep = m.preparedSpecs;
  const incremental = !(prep.hasLadder || prep.hasPhasedPatterns || prep.hasHealth || prep.hasTurn);
  // Ladder specs need a Game3.  Build ONE synced to the base and reuse it for
  // the base extraction and every fallback candidate (advanced with play/undo),
  // instead of rebuilding a Game3 per extraction.  undefined for non-ladder.
  const useG3 = needsGame3(prep);
  const g3 = useG3 ? game3FromGame2(game) : undefined;
  const f = extractFeatures(game, prep, false, undefined, true, g3);
  evaluateFeatures(f, m.weights);
  const zBase = f.z;
  const cap = game.N * game.N;
  const isBlack = game.current === BLACK;
  let v = isBlack ? -Infinity : Infinity, best = PASS, found = false;
  for (let i = 0; i < cap; i++) {
    if (!game.isLegal(i) || game.isTrueEye(i)) continue;
    found = true;
    const d = incremental ? deltaZ(game, prep, m.weights, i) : NaN;
    let s;
    if (d === d) {
      s = 1 / (1 + Math.exp(-(zBase + d))) + (dither > 0 ? Math.random() * dither : 0);
    } else {
      // reuse the health model the caller already resolved onto m.preparedSpecs
      const fb = m._fbPrep || (m._fbPrep = prepareSpecs(m.specs,
        { health: m.preparedSpecs && m.preparedSpecs.healthModel,
          ladderMinChain: prep.ladderMinChain }));
      const g = game.clone();
      g.play(i);
      if (useG3) g3.play(i);
      s = evaluateFeatures(extractFeatures(g, fb, false, undefined, true, g3), m.weights) + (dither > 0 ? Math.random() * dither : 0);
      if (useG3) g3.undo();
    }
    if (isBlack ? s > v : s < v) { v = s; best = i; }
  }
  if (!found || game.consecutivePasses > 0 || game.emptyCount < cap / 2) {
    const g = game.clone();
    g.play(PASS);
    const s = g.gameOver ? (g.estimateWinner() === BLACK ? 1 : 0)
                         : 1 / (1 + Math.exp(-zBase)) + (dither > 0 ? Math.random() * dither : 0);
    if (isBlack ? s > v : s < v) best = PASS;
  }
  return best;
}

// ── Default instance (lazy) ───────────────────────────────────────────────────
// Built from plain env on first use, for callers that use the module directly
// (gen-agent-evals, evalagentvalues).  Two-agent callers (selfplay) use create.
let _default = null;
function _def() { return _default || (_default = create(Util.makeCfg())); }

// ── Exports ───────────────────────────────────────────────────────────────────

const PatternAgent = {
  create,
  search,
  getMove: (game) => _def().getMove(game),
  valueB:  (game) => _def().valueB(game),
};

if (typeof module !== 'undefined') module.exports = PatternAgent;
else window.PatternAgent = PatternAgent;

})();
