'use strict';

// Fixed-config reference agent: prod (rave-npat-prune) with exactly 5000
// playouts per move — prod as of 2026-08-18, frozen as a ladder rung.
//
// Playout-count budgets are machine-independent, so this engine's strength
// is reproducible on any hardware — use it as a ladder reference.
//
// All parameters are hardcoded.  This script reads no environment variables.

const { getMove: prodMove } = require('./prod.js');

const PLAYOUTS = 5000;

function getMove(game) {
  return prodMove(game, 1, { playoutLimit: PLAYOUTS });
}

module.exports = { getMove };
