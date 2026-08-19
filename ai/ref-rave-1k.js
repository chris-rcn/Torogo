'use strict';

// Fixed-config reference agent: rave with exactly 1000 playouts per move.
//
// Playout-count budgets are machine-independent, so this engine's strength
// is reproducible on any hardware — use it as a ladder reference.
//
// All parameters are hardcoded.  This script reads no environment variables.

const { getMove: raveMove } = require('./rave.js');

const PLAYOUTS = 1000;

function getMove(game) {
  return raveMove(game, 1, { playoutLimit: PLAYOUTS });
}

module.exports = { getMove };
