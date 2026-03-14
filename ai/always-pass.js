'use strict';

/**
 * Always-pass policy.
 *
 * Interface: getMove(game) → { type: 'pass' } | { type: 'place', x, y }
 *   game  - a live Game instance (read-only; do not mutate)
 */
module.exports = function getMove(_game) {
  return { type: 'pass' };
};
