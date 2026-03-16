'use strict';

/**
 * Pattern-based policy agent.
 *
 * Loads a pattern-statistics file produced by minepatterns.js, then selects
 * moves by sampling from a distribution proportional to each candidate's
 * selection ratio.  Patterns not present in the file receive a small uniform
 * prior weight (DEFAULT_WEIGHT) so that novel positions still produce legal
 * moves.  Falls back to pass when no legal non-eye move exists.
 *
 * Interface: makeAgent(patternFile) → getMove(game, timeBudgetMs)
 *   patternFile  - path to the file emitted by minepatterns.js
 *                  format per line: <hash>,<ratio>,<seen_count>
 *   getMove      - standard agent interface (timeBudgetMs is ignored)
 *
 * Usage (selfplay / recordgames):
 *   const getMove = require('./ai/pattern.js').makeAgent('patterns.csv');
 */

const fs = require('fs');
const { patternHash } = require('../patterns.JS');

// Weight given to patterns that were never observed in the training data.
const DEFAULT_WEIGHT = 0.01;

/**
 * Load a pattern-statistics file and return a Map<hash, ratio>.
 *
 * @param {string} filePath
 * @returns {Map<number, number>}
 */
function loadPatterns(filePath) {
  const table = new Map();
  const lines = fs.readFileSync(filePath, 'utf8').trim().split('\n');
  for (const line of lines) {
    if (!line.trim()) continue;
    const [hashStr, ratioStr] = line.split(',');
    const hash  = parseInt(hashStr,  10);
    const ratio = parseFloat(ratioStr);
    if (!Number.isNaN(hash) && !Number.isNaN(ratio)) {
      table.set(hash, ratio);
    }
  }
  return table;
}

/**
 * Create an agent function that uses the supplied pattern table.
 *
 * @param {string} patternFile
 * @returns {function(game, timeBudgetMs): {type: string, x?: number, y?: number}}
 */
function makeAgent(patternFile) {
  const table = loadPatterns(patternFile);

  return function getMove(game, _timeBudgetMs) {
    if (game.gameOver) return { type: 'pass' };

    const N     = game.boardSize;
    const color = game.current;
    const board = game.board;

    // Collect legal non-true-eye candidates with their pattern weights.
    const candidates = [];  // [x, y]
    const weights    = [];  // parallel array of selection-ratio weights

    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        if (board.get(x, y) !== null)          continue;
        if (board.isTrueEye(x, y, color))      continue;
        if (board.isSuicide(x, y, color))      continue;
        if (board.isKo(x, y, color, game.koFlag)) continue;

        const hash   = patternHash(game, x, y);
        const weight = table.has(hash) ? table.get(hash) : DEFAULT_WEIGHT;

        candidates.push([x, y]);
        weights.push(weight);
      }
    }

    if (candidates.length === 0) return { type: 'pass' };

    // Weighted random sample: pick one candidate with probability proportional
    // to its weight.
    let total = 0;
    for (const w of weights) total += w;

    let r = Math.random() * total;
    for (let i = 0; i < candidates.length; i++) {
      r -= weights[i];
      if (r <= 0) {
        const [x, y] = candidates[i];
        return { type: 'place', x, y };
      }
    }

    // Floating-point rounding safety: return the last candidate.
    const [x, y] = candidates[candidates.length - 1];
    return { type: 'place', x, y };
  };
}

module.exports = { makeAgent };
