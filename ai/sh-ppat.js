'use strict';

// sh-ppat: "unbiased" reference agent — successive halving over ALL legal
// moves, with ppat playouts as the only evaluator.
//
// Every legal (non-true-eye) move starts as a candidate.  Each round spends
// an equal share of the total playout budget, split evenly across the
// surviving candidates; the top half by mean playout result (from the
// mover's perspective) survives.  Halve until one move remains.
//
// No priors, no policy at the root, no RAVE, no pruning heuristics: every
// candidate gets identical treatment and the only evaluator is the ppat
// playout.  Cost is fixed at ~PLAYOUTS playouts per decision regardless of
// the time budget (the budget argument is ignored).
//
// ── Factory ──
// The module exports create(cfg) → { getMove }.  cfg is a Util.makeCfg
// reader (slot-aware env).  Config:
//   PLAYOUTS   total playout budget per decision            (default 1000)
//   PPAT_DATA  ppat weight file                (default root ppat-data.js)
//   PPAT_MIN_PHASE  uniform playout moves below this board
//              fullness — skips ppat extraction in the early game (default 0)

const path = require('path');
const Util = require('../util.js');
const { PASS, BLACK, EMPTY } = require('../game2.js');
const { makeRng } = require('../xorshift.js');
const PPat = require('../ppat-lib.js');

function create(cfg) {
  cfg = cfg || Util.makeCfg();

  const PLAYOUTS = cfg.int('PLAYOUTS', 1000);

  const model = PPat.loadWeights(cfg.str('PPAT_DATA', path.join(__dirname, '..', 'ppat-data.js')));
  model.uniformBelowPhase = cfg.float('PPAT_MIN_PHASE', 0);

  console.log(`sh-ppat[${cfg.slot != null ? cfg.slot : '-'}]: ${model.weights.length} ppat weights, ` +
              `${PLAYOUTS} playouts/decision, halving over all legal moves`);

  const rng = makeRng();
  let ppatState = null;

  // ppat playout to the end of the game (mutates game2).  Returns P(BLACK
  // wins) ∈ {0,1}.
  function playout(game2) {
    if (ppatState === null || ppatState.moves.length < game2.N * game2.N)
      ppatState = PPat.createState(game2.N);
    const moveLimit = 3 * game2.emptyCount + 20;
    let moves = 0;
    while (!game2.gameOver && moves < moveLimit) {
      game2.play(PPat.ppatMove(game2, ppatState, model, rng));
      moves++;
    }
    return game2.estimateWinner() === BLACK ? 1 : 0;
  }

  function getMove(game) {
    if (game.gameOver) return { move: PASS };
    const cap = game.N * game.N;
    let arms = [];
    for (let i = 0; i < cap; i++) {
      if (game.cells[i] === EMPTY && !game.isTrueEye(i) && game.isLegal(i))
        arms.push({ idx: i, wins: 0, n: 0 });
    }
    if (arms.length === 0) return { move: PASS };
    const mover  = game.current;
    const rounds = Math.ceil(Math.log2(arms.length));

    while (arms.length > 1) {
      const per = Math.max(1, Math.round(PLAYOUTS / (rounds * arms.length)));
      for (const a of arms) {
        for (let p = 0; p < per; p++) {
          const g = game.clone();
          g.play(a.idx);
          const r = playout(g);
          a.wins += mover === BLACK ? r : 1 - r;
          a.n    += 1;
        }
        // Fractional dither: breaks exact-mean ties randomly, can never
        // reorder distinct means (playout means are multiples of 1/n).
        a.key = a.wins / a.n + rng.random() * 1e-9;
      }
      arms.sort((x, y) => y.key - x.key);
      arms = arms.slice(0, Math.ceil(arms.length / 2));
    }

    const a = arms[0];
    return { move: a.idx, info: `simWR=${(a.wins / a.n).toFixed(3)} n=${a.n}` };
  }

  return { getMove };
}

// Lazy default instance for direct-require callers.
let _default = null;
function _def() { return _default || (_default = create(Util.makeCfg())); }

module.exports = { create, getMove: (g, b, o) => _def().getMove(g, b, o) };
