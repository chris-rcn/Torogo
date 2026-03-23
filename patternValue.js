'use strict';

// patternValue.js — pattern weight lookup.
// BROWSER-COMPATIBLE: no Node.js-only APIs at top level.  Keep it this way.

const PatternValue = (() => {
  if (typeof require === 'function') { var Patterns2 = require('./patterns2.js'); var Util = require('./util.js'); }

  const DEFAULT_WEIGHT = 0;
  const table = new Map();

  function parseCSV(text) {
    table.clear();
    for (const line of text.trim().split('\n')) {
      if (!line.trim()) continue;
      const parts = line.split(',');
      const hash  = parseInt(parts[0], 10);
      const value = parseFloat(parts[1]);
      if (!Number.isNaN(hash) && !Number.isNaN(value)) table.set(hash, value);
    }
  }

  // Load patterns from a file path (Node) or URL (browser).
  // Returns a Promise.
  function load(source) {
    if (typeof require === 'function') {
      parseCSV(require('fs').readFileSync(source, 'utf8'));
      return Promise.resolve();
    }
    return fetch(source).then(r => r.text()).then(parseCSV);
  }

  // Node: auto-load the default patterns file synchronously at require time.
  if (typeof require === 'function') {
    const path = require('path');
    load(path.join(__dirname, Util.envStr('PAT_DATA', 'patterns.csv')));
  }

  return {

    load: load,

    weight(game2, idx) {
      const hash = Patterns2.patternHash2(game2, idx, game2.current);
      return table.has(hash) ? table.get(hash) : DEFAULT_WEIGHT;
    },

  };
})();

if (typeof module !== 'undefined') module.exports = PatternValue;
