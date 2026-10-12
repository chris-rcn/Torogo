'use strict';

// ref-mux-ply8-e2u6-1k — frozen reference: phase-mux with
// PHASE_MUX_A=ref-ply8, PHASE_MUX_B=ref-puct-ppat-fp-e2-u6-1k and the switch at
// phase 0.6.  ref-ply8 plays every move below the switch,
// ref-puct-ppat-fp-e2-u6-1k every move at and above it.  A thin wrapper: both
// components are themselves frozen references, and the mux rule is copied
// here rather than loaded from ai/phase-mux.js, so neither can drift.
//
// Frozen once fielded: any strength-affecting change gets a new name.
// All parameters are hardcoded.  This script reads no environment variables.

const SWITCH_PHASE = 0.6;

function create() {
  const a = require('./ref-ply8.js').create();
  const b = require('./ref-puct-ppat-fp-e2-u6-1k.js').create();
  console.error(`ref-mux-ply8-e2u6-1k: ref-ply8 below phase ${SWITCH_PHASE}, ref-puct-ppat-fp-e2-u6-1k at and above it`);

  function getMove(game, budgetMs, opts) {
    const phase = 1 - game.emptyCount / (game.N * game.N);
    return (phase < SWITCH_PHASE ? a : b).getMove(game, budgetMs, opts);
  }

  return { getMove };
}

// Lazy default instance for direct-require callers.
let _default = null;
function _def() { return _default || (_default = create()); }

module.exports = { create, getMove: (g, b, o) => _def().getMove(g, b, o) };
