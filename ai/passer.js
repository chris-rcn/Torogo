'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs (require, process, etc.).
// Wrapped in an IIFE to avoid polluting the global namespace.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

/**
 * Pass agent — always passes.
 *
 * Interface: getMove(game, timeBudgetMs) → PASS
 *   game         - a live Game2 instance (read-only; do not mutate)
 *   timeBudgetMs - milliseconds allowed for this decision (required)
 */

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
const { PASS } = _isNode ? require('../game2.js') : window.Game2;

function getMove(game, timeBudgetMs) {
  return PASS;
}

if (typeof module !== 'undefined') module.exports = { getMove };
else window.getMove = getMove;

})();
