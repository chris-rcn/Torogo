'use strict';

// cascade: three increasingly expensive, increasingly accurate stages —
// ref-search-top2-fp taken one step further, with ppat simulations as the
// final decider.
//
//   1. featurepol proposes its FP_TOP moves (raw linear scores),
//   2. vlibpat's depth-1 value narrows them to VLIB_TOP,
//   3. PPAT_SIMS ppat playouts, split evenly across the finalists, decide:
//      highest mean result (from the mover's perspective) wins.
//
// Before SIM_PHASE (board fullness 1 − empty/area) the cascade stops at
// stage 2 with a single vlibpat pick and runs no simulations — early-game
// playouts are long and noisy, so the sim budget is saved for the phase
// where it discriminates.
//
// Each stage spends more per candidate than the one before, so the funnel
// puts the cheap sharp policy first and the expensive unbiased estimator
// last.  Stochastic via the playouts (and vlibpat dither on the narrowing).
//
// ── Factory ──
// The module exports create(cfg) → { getMove }.  cfg is a Util.makeCfg
// reader (slot-aware env).  Config:
//   FP_TOP      featurepol proposal count            (default 3)
//   VLIB_TOP    finalists kept by vlibpat value      (default 2)
//   PPAT_SIMS   total ppat playouts, split across finalists (default 100)
//   SIM_PHASE   below this phase, VLIB_TOP acts as 1 (no sims) (default 0.5)
//   FP_WEIGHTS / VLIB_WEIGHTS / PPAT_DATA  weight-file overrides

const path = require('path');
const Util = require('../util.js');
const { PASS, BLACK } = require('../game2.js');
const { makeRng } = require('../xorshift.js');
const { game3FromGame2 } = require('../game3.js');
const FeaturePol = require('../featurepol-lib.js');
const { extractFeatures: vExtract, evaluateFeatures: vEval, loadWeights: vLoadWeights } = require('../vlibpat.js');
const PPat = require('../ppat-lib.js');

const DITHER = 0.001;   // uniform noise on the vlibpat narrowing values

function create(cfg) {
  cfg = cfg || Util.makeCfg();

  const FP_TOP    = cfg.int('FP_TOP', 3);
  const VLIB_TOP  = cfg.int('VLIB_TOP', 2);
  const PPAT_SIMS = cfg.int('PPAT_SIMS', 100);
  const SIM_PHASE = cfg.float('SIM_PHASE', 0.5);

  const fpModel = FeaturePol.loadModel({ name: 'cascade',
    path: cfg.str('FP_WEIGHTS', path.join(__dirname, '..', 'ref', 'featurepol-6082.js')) });
  const fpWeights = fpModel.weights;
  const vModel  = vLoadWeights(cfg.str('VLIB_WEIGHTS', path.join(__dirname, '..', 'ref', 'vlibpat-4074.js')));
  const ppatModel = PPat.loadWeights(cfg.str('PPAT_DATA', path.join(__dirname, '..', 'ppat-data.js')));

  console.log(`cascade[${cfg.slot != null ? cfg.slot : '-'}]: fp=${fpWeights.size}w vlib=${vModel.weights.size}w ppat=${ppatModel.weights.length}w  ` +
              `top${FP_TOP} -> top${VLIB_TOP} -> ${PPAT_SIMS} sims (sims from phase ${SIM_PHASE})`);

  const rng = makeRng();
  let fpState = null, fpScores = null, ppatState = null;

  // Stage 1: featurepol's FP_TOP moves by raw score.
  function _fpCandidates(game, game3) {
    const N = game.N;
    if (!fpState || fpState.moves.length < N * N) {
      fpState  = FeaturePol.createState(N, fpWeights.spec);
      fpScores = new Float64Array(N * N + 1);
    }
    FeaturePol.extractFeatures(game, fpState, fpWeights, game3);
    const n = FeaturePol.scoreAll(fpState, fpWeights, fpScores);
    if (n === 0) return [];
    const k = Math.min(FP_TOP, n);
    const order = new Array(n);
    for (let i = 0; i < n; i++) order[i] = i;
    if (k < n) order.sort((a, b) => fpScores[b] - fpScores[a]);
    const out = new Array(k);
    for (let j = 0; j < k; j++) out[j] = fpState.moves[order[j]];
    return out;
  }

  // Stage 2: vlibpat depth-1 value (BLACK-perspective) narrows to `keep`.
  function _vlibNarrow(game3, candidates, moverIsBlack, keep) {
    if (candidates.length <= keep) return candidates;
    const scored = candidates.map(m => {
      game3.play(m);
      const f = vExtract(game3, vModel.preparedSpecs);
      vEval(f, vModel.weights);
      game3.undo();
      const v = f.val + DITHER * rng.random();
      return { m, v: moverIsBlack ? v : -v };
    });
    scored.sort((a, b) => b.v - a.v);
    return scored.slice(0, keep).map(s => s.m);
  }

  // Stage 3: mean of `nSims` ppat playouts from the position after `move`,
  // from the mover's perspective.
  function _simValue(game, move, mover, nSims) {
    const N = game.N;
    if (!ppatState || ppatState.moves.length < N * N) ppatState = PPat.createState(N);
    let wins = 0;
    for (let s = 0; s < nSims; s++) {
      const sim = game.clone();
      sim.play(move);
      const moveLimit = 3 * sim.emptyCount + 20;
      let moves = 0;
      while (!sim.gameOver && moves++ < moveLimit) {
        sim.play(PPat.ppatMove(sim, ppatState, ppatModel));
      }
      if (sim.estimateWinner() === mover) wins++;
    }
    return wins / nSims;
  }

  function getMove(game) {
    if (game.gameOver) return { move: PASS };
    const game3 = game3FromGame2(game);
    const mover = game.current;

    const proposals = _fpCandidates(game, game3);
    if (proposals.length === 0) return { move: PASS };

    // Below SIM_PHASE, vlibpat decides alone (single finalist, no sims).
    const keep = game.phase() < SIM_PHASE ? 1 : VLIB_TOP;

    const finalists = _vlibNarrow(game3, proposals, mover === BLACK, keep);
    if (finalists.length === 1) return { move: finalists[0] };

    // PPAT_SIMS is the TOTAL playout budget, split evenly across finalists.
    const perFinalist = Math.max(1, Math.floor(PPAT_SIMS / finalists.length));
    let best = finalists[0], bestV = -Infinity;
    for (const m of finalists) {
      const v = _simValue(game, m, mover, perFinalist);
      if (v > bestV) { bestV = v; best = m; }
    }
    return { move: best, info: `simWR=${bestV.toFixed(3)}` };
  }

  return { getMove };
}

// Lazy default instance for direct-require callers.
let _default = null;
function _def() { return _default || (_default = create(Util.makeCfg())); }

module.exports = { create, getMove: (g, b, o) => _def().getMove(g, b, o) };
