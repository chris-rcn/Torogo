'use strict';

// mc-vlib — 1-ply argmax player over a per-position value that blends uniform
// Monte-Carlo rollouts with vlibpat, ramped by game phase.  Built to rank
// evaluators by playing strength (vs a fixed yardstick like ref-npat-softmax)
// and to run through evalmovedetails.
//
// For the side to move P, every legal non-eye move m is scored by evaluating
// the resulting child position (opponent to move):
//
//   u = fraction of MC_K uniform rollouts the child's side-to-move wins
//   v = vlibpat value of the child, side-to-move perspective
//   blend = (1-alpha)*u + alpha*v,   alpha = ALPHA_OPEN * (1 - phase_child)
//
// child blend is P(opponent wins after m), so P's value of m is 1 - blend; we
// pick the m maximising that (== minimising the child blend).  ALPHA_OPEN=0 is
// pure flat Monte-Carlo (vlibpat never touched); larger ALPHA_OPEN mixes in
// more vlibpat, fading to 0 by the endgame (alpha_end = 0 at phase 1).
//
// Config (env, since this is an experiment agent):
//   MC_K        uniform rollouts per candidate child   (default 100)
//   ALPHA_OPEN  vlibpat weight at phase 0              (default 0)

const path = require('path');
const { Game2, BLACK, PASS } = require('../game2.js');
const { makeRng } = require('../xorshift.js');
const RefVlib  = require('./ref-vlibpat.js');
const RefNpat  = require('./ref-npat-softmax.js');

const MC_K       = parseInt(process.env.MC_K || '100', 10);
const ALPHA_OPEN = parseFloat(process.env.ALPHA_OPEN || '0');
// Past this phase (board fullness), hand the move to ref-npat-softmax instead
// of the expensive uniform+vlibpat argmax — lets us test only the early game
// and speed through the late game.  Default 1 = never delegate.
const PHASE_CUTOFF = parseFloat(process.env.PHASE_CUTOFF || '1');

// Default rng for game play (evalmovedetails passes its own for reproducibility).
const defaultRng = makeRng((((Date.now() ^ (process.pid << 16)) >>> 0) || 1));

// Reused rollout scratch board (avoids per-rollout allocation).
let sim = null;

// Mean uniform-rollout value of `child` from its side-to-move's perspective.
function uniformValue(child, k, rng) {
  if (!sim || sim.N !== child.N) sim = new Game2(child.N);
  const player = child.current;
  const moveCap = 3 * (child.N * child.N) + 20;
  let wins = 0;
  for (let i = 0; i < k; i++) {
    child.cloneInto(sim);
    let moves = 0;
    while (!sim.gameOver && moves < moveCap) {
      sim.play(sim.randomLegalMove(rng));
      moves++;
    }
    if (sim.estimateWinner() === player) wins++;
  }
  return wins / k;
}

// vlibpat value of `g`, side-to-move perspective.
function vlibValue(g) {
  const pBlack = RefVlib.valueB(g);              // P(BLACK wins)
  return g.current === BLACK ? pBlack : 1 - pBlack;
}

// Blended value of `g` from its side-to-move's perspective:
//   (1-alpha)*uniform + alpha*vlibpat,   alpha = ALPHA_OPEN * (1 - phase).
function blendStm(g, rng) {
  const phase = 1 - g.emptyCount / (g.N * g.N);
  const alpha = ALPHA_OPEN * (1 - phase);
  const u = alpha < 1 ? uniformValue(g, MC_K, rng) : 0;
  const v = alpha > 0 ? vlibValue(g)              : 0;
  return (1 - alpha) * u + alpha * v;
}

function getMove(game, budgetMs, opts) {
  const rng  = (opts && opts.rng) || defaultRng;
  const area = game.N * game.N;

  // Past the cutoff, delegate to the cheap npat-softmax policy.
  const phase = 1 - game.emptyCount / area;
  if (phase > PHASE_CUTOFF) return RefNpat.getMove(game, budgetMs, opts);

  // Candidate moves: legal, not filling our own true eye.
  const candidates = [];
  for (let idx = 0; idx < area; idx++) {
    if (!game.isTrueEye(idx) && game.isLegal(idx)) candidates.push(idx);
  }
  if (candidates.length === 0) return { move: PASS, info: 'pass (no moves)' };

  // Score each child with value() = P(BLACK wins); BLACK maximises it, WHITE
  // minimises it.  Track the chosen move's win prob for the current player.
  const sign = game.current === BLACK ? 1 : -1;
  let bestMove = candidates[0], bestScore = -Infinity, bestPBlack = 0.5;
  for (const m of candidates) {
    const child = game.clone();
    child.play(m);
    const pBlack = value(child, rng);
    const score  = sign * pBlack;
    if (score > bestScore) { bestScore = score; bestMove = m; bestPBlack = pBlack; }
  }

  const myWin = game.current === BLACK ? bestPBlack : 1 - bestPBlack;
  return { move: bestMove, info: `val=${myWin.toFixed(3)} K=${MC_K} a0=${ALPHA_OPEN}` };
}

// Blended value of a Game2 position: P(BLACK wins) in [0,1], matching
// ref-vlibpat.valueB's convention (absolute, not side-to-move).  For use as an
// SB value oracle / for comparison against ref-vlibpat.
function valueB(game, options = {}) {
  const rng = options.rng || defaultRng;
  const stm = blendStm(game, rng);
  return game.current === BLACK ? stm : 1 - stm;
}

module.exports = { getMove, valueB };
