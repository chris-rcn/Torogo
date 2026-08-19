'use strict';

// Fixed-config reference agent: mc (flat Monte Carlo) with exactly 200
// playouts per move.
//
// Playout-count budgets are machine-independent, so this engine's strength
// is reproducible on any hardware — use it as a weak ladder rung between
// random and the pattern policies.
//
// All parameters are hardcoded.  This script reads no environment variables.

const { getMove: mcMove } = require('./mc.js');

const PLAYOUTS = 200;

function getMove(game) {
  return mcMove(game, 1, { playoutLimit: PLAYOUTS });
}

module.exports = { getMove };
