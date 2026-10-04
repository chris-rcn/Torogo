'use strict';

// ref-mux-fpheavy-puct300 — frozen reference: phase-mux with
// PHASE_MUX_A=ref-fp-heavy, PHASE_MUX_B=ref-puct-trunc-300 and the switch at
// phase 0.5 (phase-mux's default).  ref-fp-heavy plays every move below the
// switch, ref-puct-trunc-300 every move at and above it.  A thin wrapper: both
// components are themselves frozen references, and the mux rule is copied
// here rather than loaded from ai/phase-mux.js, so neither can drift.
//
// Frozen once fielded: any strength-affecting change gets a new name.
// All parameters are hardcoded.  This script reads no environment variables.

const SWITCH_PHASE = 0.5;

function create() {
  const a = require('./ref-fp-heavy.js');
  const b = require('./ref-puct-trunc-300.js').create();
  console.error(`ref-mux-fpheavy-puct300: ref-fp-heavy below phase ${SWITCH_PHASE}, ref-puct-trunc-300 at and above it`);

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
