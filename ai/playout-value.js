'use strict';

// playout-value: a value oracle whose playouts are played by another agent.
// valueB(game, options) is the mean result of PLAYOUTS playouts from the
// position, every move of each (both sides) chosen by PO_AGENT's getMove.  A
// playout runs until the game ends or 3·empty + 20 moves (mc-ppat's cap) and
// is scored by estimateWinner.  Returns P(BLACK wins).  getMove throws: this
// agent is a value oracle only.  PO_AGENT's getMove is called without a time
// budget, so it should be a fixed-configuration agent.
//
// Config (cfg reader, slot-aware):
//   PO_AGENT    ai/<name>.js that plays the playouts; built from the same cfg,
//               so it reads its own knobs from the env too (a PLAYOUTS knob of
//               its own would see this agent's PLAYOUTS)   (default ref-ply8)
//   PLAYOUTS    playouts per valueB call                       (default 10)

const path = require('path');
const Util = require('../util.js');
const { BLACK } = require('../game2.js');
const { makeRng } = require('../xorshift.js');

function create(cfg) {
  cfg = cfg || Util.makeCfg();

  const PO_AGENT  = cfg.str('PO_AGENT', 'ref-ply8');
  const PLAYOUTS  = cfg.int('PLAYOUTS', 10);
  if (!(PLAYOUTS >= 1)) throw new Error(`playout-value: PLAYOUTS must be >= 1, got ${PLAYOUTS}`);

  const mod = require(path.join(__dirname, PO_AGENT + '.js'));
  const poGetMove = (typeof mod.create === 'function' ? mod.create(cfg) : mod).getMove;
  if (typeof poGetMove !== 'function') throw new Error(`playout-value: PO_AGENT ${PO_AGENT} has no getMove`);

  console.error(`playout-value[${cfg.slot != null ? cfg.slot : '-'}]: ${PLAYOUTS} playouts by ${PO_AGENT}`);

  const ownRng = makeRng();

  // One playout from `game` (cloned); 1 if BLACK wins.
  function playout(game, rng) {
    const g = game.clone();
    const moveLimit = 3 * g.emptyCount + 20;
    for (let moves = 0; !g.gameOver && moves < moveLimit; moves++) {
      const m = poGetMove(g, undefined, { rng }).move;
      if (!g.play(m)) throw new Error(`playout-value: ${PO_AGENT} returned an illegal move (${m})`);
    }
    return g.estimateWinner() === BLACK ? 1 : 0;
  }

  function valueB(game, options = {}) {
    const g = game.cells ? game : game.toGame2();
    if (g.gameOver) return g.calcWinner() === BLACK ? 1 : 0;
    const rng = options.rng || ownRng;
    let wins = 0;
    for (let p = 0; p < PLAYOUTS; p++) wins += playout(g, rng);
    return wins / PLAYOUTS;
  }

  function getMove() {
    throw new Error('playout-value: a value oracle only (valueB); it has no getMove');
  }

  return { getMove, valueB };
}

let _default = null;
function _def() { return _default || (_default = create(Util.makeCfg())); }
module.exports = { create, getMove: (g, b, o) => _def().getMove(g, b, o), valueB: (g, o) => _def().valueB(g, o) };
