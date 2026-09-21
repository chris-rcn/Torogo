'use strict';

// mix: per-move random mixture of two agents.  Each move is played by MIX_B
// with probability MIX_B_RATIO, otherwise by MIX_A.  Both sub-agents are
// ordinary ai/ modules loaded with this instance's cfg, so per-slot env
// (P1_/P2_ prefixes) reaches them unchanged.
//
//   MIX_A         agent for the (1 - MIX_B_RATIO) fraction of moves (required)
//   MIX_B         agent for the MIX_B_RATIO fraction of moves (required)
//   MIX_B_RATIO   probability of using MIX_B each move, in [0, 1] (default 0.5)

const path = require('path');

function create(cfg) {
  const slot  = cfg.slot != null ? cfg.slot : '-';
  const nameA = cfg.str('MIX_A', '');
  const nameB = cfg.str('MIX_B', '');
  const ratio = cfg.float('MIX_B_RATIO', 0.5);
  if (!nameA || !nameB) throw new Error(`mix[${slot}]: MIX_A and MIX_B are both required`);
  if (!(ratio >= 0 && ratio <= 1)) throw new Error(`mix[${slot}]: MIX_B_RATIO (${ratio}) must be in [0, 1]`);

  function load(name) {
    const mod = require(path.join(__dirname, name + '.js'));
    return typeof mod.create === 'function' ? mod.create(cfg) : mod;
  }
  const a = load(nameA), b = load(nameB);

  console.log(`mix[${slot}]: ${nameB} with ratio ${ratio}, else ${nameA}`);

  // Per move, flip MIX_B_RATIO and delegate to the chosen agent (only the chosen
  // one is asked, so no wasted move generation).  Uses the caller's rng when
  // present so a seeded run is reproducible; the picked sub-agent gets the same
  // rng, one draw advanced.
  function getMove(game, budgetMs, opts) {
    const r = (opts && opts.rng) ? opts.rng.random() : Math.random();
    return (r < ratio ? b : a).getMove(game, budgetMs, opts);
  }

  return { getMove };
}

module.exports = { create };
