'use strict';

// Fixed-config reference agent: the raw SB-trained ppat playout policy
// (ref/ppat-3374337.js), sampling one move per call — no search.
//
// Simulation balancing trains ppat to make playout *outcomes* unbiased, not
// to pick strong moves, so this engine measures what the playout policy is
// worth as a standalone player — a diagnostic rung, not a contender.
//
// All parameters are hardcoded.  This script reads no environment variables.

const path = require('path');
const { PASS } = require('../game2.js');
const { createState, ppatMove, loadWeights } = require('../ppat-lib.js');

const model = loadWeights(path.join(__dirname, '..', 'ref', 'ppat-3374337.js'));
// Standard-playout phase split: uniform below 0.6, ppat above — ppat was
// band-trained and should not operate below its band (changed 2026-09-07;
// earlier this rung sampled ppat at every phase).
model.ppatMinPhase = 0.6;

console.error(`ref-ppat: ${model.weights.length} weights (${model.phaseCount} phase(s)) [policy sampling, no search]`);

let state = null;

function getMove(game) {
  if (game.gameOver) return { move: PASS };
  if (state === null || state.moves.length < game.N * game.N) state = createState(game.N);
  return { move: ppatMove(game, state, model) };
}

module.exports = { getMove };
