'use strict';

// ref-value — frozen value reference: a self-contained copy of the vpatsearch
// agent, pinned to out/vpat-pe-jdi54tel.js at vpatsearch's default config
// (alpha-beta depth 1, dither 0.002).  Exports getMove (search) and valueB
// (static V(s) = P(BLACK wins)), so it serves as a fixed valueB oracle (e.g.
// the --referee in compare-move-selection) or a fielded reference.
//
// Self-contained by design: the logic below is copied from ai/vpatsearch.js
// (not required from it) and frozen — it will not track later vpatsearch edits.
// Depends only on shared libraries.

const path = require('path');
const { extractFeatures, evaluateFeatures, deltaZ, loadWeights, prepareSpecs } = require('../vpatterns.js');
const { search: abSearch } = require('../ab-search.js');
const { BLACK, PASS } = require('../game2.js');

const SEARCH_DEPTH = 1;      // vpatsearch default
const DITHER       = 0.002;  // vpatsearch default

const model = loadWeights(path.join(__dirname, '..', 'out', 'vpat-pe-jdi54tel.js'), '');

console.error(`ref-value: ${model.weights.size} vpat weights from vpat-pe-jdi54tel.js [alpha-beta depth ${SEARCH_DEPTH}]`);

// ── Search (copied from vpatsearch.js) ─────────────────────────────────────────

function search(game, m, depth = 1, dither = 0) {
  if (depth === 1) return search1(game, m, dither);
  const evaluate = g => evaluateFeatures(extractFeatures(g, m.preparedSpecs), m.weights);
  return abSearch(game, depth, evaluate, dither);
}

// Depth-1 fast path: one base extraction, then speculative-incremental deltaZ
// per candidate (V = sigma(zBase + dz)); captures / non-incremental specs fall
// back to full extraction on a separate prepSpecs.  Semantics mirror ab() at
// depth 1: board-index order, strict-improvement argmax, PASS considered last
// under the same conditions, terminal PASS scored exactly.
function search1(game, m, dither) {
  const prep = m.preparedSpecs;
  const incremental = !(prep.hasLadder || prep.hasPhasedPatterns || prep.hasHealth || prep.hasTurn);
  const f = extractFeatures(game, prep);
  evaluateFeatures(f, m.weights);
  const zBase = f.z;
  const cap = game.N * game.N;
  const isBlack = game.current === BLACK;
  let v = isBlack ? -Infinity : Infinity, best = PASS, found = false;
  for (let i = 0; i < cap; i++) {
    if (!game.isLegal(i) || game.isTrueEye(i)) continue;
    found = true;
    const d = incremental ? deltaZ(game, prep, m.weights, i) : NaN;
    let s;
    if (d === d) {
      s = 1 / (1 + Math.exp(-(zBase + d))) + (dither > 0 ? Math.random() * dither : 0);
    } else {
      const fb = m._fbPrep || (m._fbPrep = prepareSpecs(m.specs,
        { health: m.preparedSpecs && m.preparedSpecs.healthModel }));
      const g = game.clone();
      g.play(i);
      s = evaluateFeatures(extractFeatures(g, fb), m.weights) + (dither > 0 ? Math.random() * dither : 0);
    }
    if (isBlack ? s > v : s < v) { v = s; best = i; }
  }
  if (!found || game.consecutivePasses > 0 || game.emptyCount < cap / 2) {
    const g = game.clone();
    g.play(PASS);
    const s = g.gameOver ? (g.estimateWinner() === BLACK ? 1 : 0)
                         : 1 / (1 + Math.exp(-zBase)) + (dither > 0 ? Math.random() * dither : 0);
    if (isBlack ? s > v : s < v) best = PASS;
  }
  return best;
}

function getMove(game) {
  return { move: search(game, model, SEARCH_DEPTH, DITHER) };
}

// Position value oracle: the static evaluation itself — V(s) = P(BLACK wins).
function valueB(game) {
  const g = game.cells ? game : game.toGame2();
  if (g.gameOver) return g.calcWinner() === BLACK ? 1 : 0;
  return evaluateFeatures(extractFeatures(g, model.preparedSpecs), model.weights);
}

module.exports = { getMove, valueB };
