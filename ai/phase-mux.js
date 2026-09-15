'use strict';

// phase-mux: delegates each move to one of two agents by board fullness —
// agent A below the switch phase, agent B at and above it.  Both sub-agents
// are ordinary ai/ modules loaded with this instance's cfg, so per-slot env
// (P1_/P2_ prefixes) reaches them unchanged.
//
//   PHASE_MUX_A       agent below the switch (default ref-ab2-fp5-vpat)
//   PHASE_MUX_B       agent at/above the switch (default ref-puct-ppat-fp-e2-u6-300)
//   PHASE_MUX_THRESH  the switch phase (default 0.5)

const path = require('path');

function create(cfg) {
  const nameA = cfg.str('PHASE_MUX_A', 'ref-ab2-fp5-vpat');
  const nameB = cfg.str('PHASE_MUX_B', 'ref-puct-ppat-fp-e2-u6-300');
  const switchPhase = cfg.float('PHASE_MUX_THRESH', 0.5);

  function load(name) {
    const mod = require(path.join(__dirname, name + '.js'));
    return typeof mod.create === 'function' ? mod.create(cfg) : mod;
  }
  const a = load(nameA), b = load(nameB);

  function getMove(game, budgetMs, opts) {
    const phase = 1 - game.emptyCount / (game.N * game.N);
    return (phase < switchPhase ? a : b).getMove(game, budgetMs, opts);
  }

  return { getMove };
}

module.exports = { create };
