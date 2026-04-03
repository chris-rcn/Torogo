'use strict';

// BROWSER-COMPATIBLE: no Node.js-only APIs (require, process, etc.).
// Wrapped in an IIFE to avoid polluting the global namespace.
// Loaded as a plain <script> tag; do not add require/module/process at top level.

(function () {

const _isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
const { PASS } = _isNode ? require('../game2.js') : window.Game2;

function getMove(game) {
  const move = game.randomLegalMove();
  return { move };
}

if (typeof module !== 'undefined') module.exports = { getMove };
else window.getMove = getMove;

})();

