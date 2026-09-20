'use strict';

// Fixed-config reference agent: prod-as-of-2026-08-18 (rave-npat-prune,
// now ai/prodOldC.js) with exactly 5000
// playouts per move — prod as of 2026-08-18, frozen as a ladder rung.
//
// Playout-count budgets are machine-independent, so this engine's strength
// is reproducible on any hardware — use it as a ladder reference.
//
// Also exports valueB(game, opts) -> P(BLACK wins): the same 5000-playout
// search's root visit-weighted win rate, mapped to BLACK's frame, so this
// engine can label positions via gen-agent-evals.  valueB does not touch
// getMove, so the frozen playing strength is unchanged.
//
// All parameters are hardcoded.  This script reads no environment variables.

const { getMove: prodMove } = require('./prodOldC.js');
const { BLACK } = require('../game2.js');

const PLAYOUTS = 5000;

function getMove(game) {
  return prodMove(game, 1, { playoutLimit: PLAYOUTS });
}

// rootWinRatio is P(side-to-move wins) from the search prodMove runs; map it to
// P(BLACK wins).  Terminal positions are scored directly.
function valueB(game, options = {}) {
  if (game.gameOver) return game.calcWinner() === BLACK ? 1 : 0;
  const r = prodMove(game, 1, { playoutLimit: PLAYOUTS, rng: options.rng });
  return game.current === BLACK ? r.rootWinRatio : 1 - r.rootWinRatio;
}

module.exports = { getMove, valueB };
