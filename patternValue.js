'use strict';

/**
 * Loads a pattern-statistics file produced by minepatterns.js.  Patterns not 
 * present in the file receive a small uniform
 * prior weight (DEFAULT_WEIGHT) so that novel positions still produce legal
 * moves.  Falls back to pass when no legal non-eye move exists.
 *
 */

const fs = require('fs');
const { patternHash2 } = require('./patterns2.js');

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
 * Create a weighting function (game2, idx) → number from a pattern file.
 * Lighter than makeAgent: no candidate sampling, just the hash→ratio lookup.
 * Intended for callers (e.g. raveladpat2) that manage candidate selection themselves.
 *
 * @param {string} patternFile
 * @returns {function(game2, idx): number}
 */
function makeWeighter(patternFile) {
  const table = loadPatterns(patternFile);
  return function weight(game2, idx) {
    const hash = patternHash2(game2, idx, game2.current);
    return table.has(hash) ? table.get(hash) : DEFAULT_WEIGHT;
  };
}

const path = require('path');
const _defaultFile  = path.join(__dirname, process.env.PAT_DATA || 'patterns.csv');
const _defaultWeight = makeWeighter(_defaultFile);

module.exports = { weight: _defaultWeight };

