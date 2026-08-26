'use strict';

// mc-ppat: a deliberately minimal agent whose ONLY decision input is the
// ppat playout policy — built to measure playout quality directly.
//
// Each move: sample CANDIDATES legal non-eye moves uniformly at random, split
// the PLAYOUTS budget evenly between them, and play the one with the best mean
// result (from the mover's perspective).  No tree, no priors, no RAVE, no
// featurepol.  PLAYOUTS is the TOTAL per decision, as in every other agent, so
// changing CANDIDATES redistributes a fixed budget rather than inflating it.
//
// Why this rather than puct-ppat-fp: in the puct agents the featurepol priors
// and the tree make most of the decision, so swapping the ppat model barely
// moves the result — two different ppat checkpoints once produced identical
// movedetails output down to the same worst-case move choices.  Here the ppat
// policy decides every move, so an A/B between two models measures the models.
//
// Cost is fixed per decision (PLAYOUTS) and independent of the time budget,
// which is ignored.  Sized for fast head-to-heads on small boards.
//
// ── Factory ──
// create(cfg) -> { getMove }.  cfg is a Util.makeCfg reader (slot-aware env),
// so P1_/P2_ prefixes select a different model per side in selfplay:
//   P1_PPAT_DATA=a.js P2_PPAT_DATA=b.js node selfplay.js --p1 mc-ppat --p2 mc-ppat
//
// Config:
//   CANDIDATES    moves sampled per decision                     (default 2)
//   PLAYOUTS      TOTAL playouts per decision, split evenly     (default 100)
//   CAND_PLAYOUTS playouts PER CANDIDATE; overrides PLAYOUTS when > 0 (default 0)
//                 The two conventions allocate to opposite ends of the game: a
//                 fixed total gives each candidate more playouts as the move
//                 list shrinks (endgame-weighted), while a fixed per-candidate
//                 count spends more total playouts while the list is long
//                 (midgame-weighted).
//   PPAT_DATA     ppat weight file                 (default root ppat-data.js)
//   PPAT_UNIFORM_BELOW_PHASE  uniform playout moves below this board fullness
//                 (default 0 = ppat everywhere)

const path = require('path');
const Util = require('../util.js');
const { PASS, BLACK, EMPTY } = require('../game2.js');
const { makeRng } = require('../xorshift.js');
const PPat = require('../ppat-lib.js');

function create(cfg) {
  cfg = cfg || Util.makeCfg();

  const CANDIDATES = Math.max(1, cfg.int('CANDIDATES', 2));
  const PLAYOUTS   = Math.max(1, cfg.int('PLAYOUTS', 100));
  const CAND_PLAYOUTS = Math.max(0, cfg.int('CAND_PLAYOUTS', 0));

  const ppatPath = cfg.str('PPAT_DATA', path.join(__dirname, '..', 'ppat-data.js'));
  const model    = PPat.loadWeights(ppatPath);
  // Hard failure, not a fallback: this agent exists to measure a ppat model, so
  // silently running uniform playouts would produce a meaningless comparison.
  if (!model) throw new Error(`mc-ppat: cannot load ppat weights from ${ppatPath}`);
  model.uniformBelowPhase = cfg.float('PPAT_UNIFORM_BELOW_PHASE', 0);
  console.log(`mc-ppat[${cfg.slot != null ? cfg.slot : '-'}]: ${model.weights.length} ppat weights ` +
              `from ${path.basename(ppatPath)}, ` +
              (CAND_PLAYOUTS > 0 ? `${CAND_PLAYOUTS} playouts/candidate` : `${PLAYOUTS} playouts/move`) +
              ` over ${CANDIDATES} candidates`);

  const rng = makeRng();
  let ppatState = null;

  // ppat playout to the end of the game (mutates game2).  Returns 1 if BLACK
  // wins, else 0.
  function playout(game2) {
    if (ppatState === null || ppatState.moves.length < game2.N * game2.N)
      ppatState = PPat.createState(game2.N);
    const moveLimit = 3 * game2.emptyCount + 20;
    let moves = 0;
    while (!game2.gameOver && moves < moveLimit) {
      game2.play(PPat.ppatMove(game2, ppatState, model));
      moves++;
    }
    return game2.estimateWinner() === BLACK ? 1 : 0;
  }

  // Reservoir-sample k legal non-eye moves without building the full list:
  // one pass over the empty cells, O(area) with no allocation beyond `out`.
  function sampleMoves(game, k, out) {
    const area = game.N * game.N;
    let seen = 0;
    for (let i = 0; i < area; i++) {
      if (game.cells[i] !== EMPTY || game.isTrueEye(i) || !game.isLegal(i)) continue;
      if (seen < k) out[seen] = i;
      else { const j = rng.int(seen + 1); if (j < k) out[j] = i; }
      seen++;
    }
    return seen < k ? seen : k;
  }

  const _cand = new Int32Array(64);

  function getMove(game) {
    if (game.gameOver) return { move: PASS };
    const n = sampleMoves(game, Math.min(CANDIDATES, _cand.length), _cand);
    if (n === 0) return { move: PASS };

    // CAND_PLAYOUTS fixes the per-candidate count (total then scales with n);
    // otherwise split the total budget evenly across the candidates actually
    // found (n can be < CANDIDATES late), keeping per-decision cost at PLAYOUTS.
    const per = CAND_PLAYOUTS > 0 ? CAND_PLAYOUTS : Math.max(1, Math.round(PLAYOUTS / n));
    const mover = game.current;
    let best = _cand[0], bestV = -1;
    for (let c = 0; c < n; c++) {
      let wins = 0;
      for (let p = 0; p < per; p++) {
        const g = game.clone();
        g.play(_cand[c]);
        const r = playout(g);
        wins += mover === BLACK ? r : 1 - r;
      }
      // Fractional dither breaks exact ties randomly; it can never reorder
      // distinct means, which are multiples of 1/per.
      const v = wins / per + rng.random() * 1e-9;
      if (v > bestV) { bestV = v; best = _cand[c]; }
    }
    return { move: best, info: `wr=${bestV.toFixed(3)} of ${n}x${per}` };
  }

  return { getMove };
}

// Lazy default instance for direct-require callers.
let _default = null;
function _def() { return _default || (_default = create(Util.makeCfg())); }

module.exports = { create, getMove: (g, b, o) => _def().getMove(g, b, o) };
