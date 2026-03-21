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
const { patternHash } = require('../patterns.js');

// Weight given to patterns that were never observed in the training data.
const DEFAULT_WEIGHT = 0.01;

// Number of randomly sampled candidates to score per move.
const CANDIDATES = 8;

/**
 * Load a pattern-statistics file and return a Map<hash, weight>.
 *
 * @param {string} filePath
 * @returns {Map<number, number>}
 */
function loadPatterns(filePath) {
  const table = new Map();
  const lines = fs.readFileSync(filePath, 'utf8').trim().split('\n');
  for (const line of lines) {
    if (!line.trim()) continue;
    const parts = line.split(',');
    const hash  = parseInt(parts[0], 10);
    const value = parseFloat(parts[1]);
    if (!Number.isNaN(hash) && !Number.isNaN(value)) {
      table.set(hash, value);
    }
  }
  return table;
}

/**
 * Create a weighting function (game, x, y) → number from a pattern file.
 * Lighter than makeAgent: no candidate sampling, just the hash→ratio lookup.
 * Intended for callers (e.g. ravepat) that manage candidate selection themselves.
 *
 * @param {string} patternFile
 * @returns {function(game, x, y): number}
 */
function makeWeighter(patternFile) {
  const table = loadPatterns(patternFile);
  return function weight(game, x, y) {
    const hash = patternHash(game, x, y, game.current);
    return table.has(hash) ? table.get(hash) : DEFAULT_WEIGHT;
  };
}

// Default getMove for use with selfplay.js / recordgames.js:
//   node selfplay.js --p1 pattern --p2 random
// Looks for patterns.csv next to game.js (project root).
const path = require('path');
const _defaultFile   = path.join(__dirname, '..', 'patterns.csv');
const _defaultWeight  = makeWeighter(_defaultFile);

module.exports = { weight: _defaultWeight };

